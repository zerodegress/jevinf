"""jevinf command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import upstream
from .arch import ARCHITECTURES, DEFAULT_ARCH
from .backend import BACKENDS, DEFAULT_BACKEND
from .bench import compare, head_selfcheck, run_bench, summarize
from .engine import PrefixShareEngine
from .plan import build_plan


def _load_payload(path: str | None, split: str | None, states: int | None) -> dict:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if not split:
        raise SystemExit("give either --input or --split")
    return upstream.read_split(split, limit_states=states)


def _predictor(args):
    return upstream.load_predictor(
        checkpoint_dir=args.model, backend=args.backend, arch=args.arch, precision="fp32",
        crop_state=getattr(args, "crop_state", False),
    )


def _engine(pred, args):
    """Pick the engine for the loaded family. `arrangement` is the dispatch key, not the model name."""
    arrangement = pred.arch.arrangement
    if arrangement == "single-path":
        from .laya import LayaEngine

        return LayaEngine(pred, max_rows=args.max_rows, crop_state=args.crop_state)
    if arrangement == "state-fork":
        from .decider import DeciderEngine

        return DeciderEngine(pred, max_rows=args.max_rows, prefix_chunk=args.prefix_chunk,
                             max_state_tokens=args.max_state_tokens or None)
    return PrefixShareEngine(pred, strategy=args.strategy, head_chunk=args.head_chunk)


def _evaluate(engine, pred, args, payload):
    """Hand each family the knobs it has; a knob it does not have is refused inside the engine."""
    if pred.arch.arrangement == "three-stage":
        return engine.evaluate(payload)
    if pred.arch.arrangement == "state-fork":
        return engine.evaluate(payload, layout=args.layout, temperature=args.temperature or None)
    return engine.evaluate(payload, temperature=args.temperature or None)


def cmd_selfcheck(args) -> int:
    pred = _predictor(args)
    payload = _load_payload(args.input, args.split, args.states)
    if pred.arch.arrangement == "three-stage":
        print(json.dumps(head_selfcheck(pred, payload), ensure_ascii=False, indent=1))
        return 0
    print(json.dumps(_engine(pred, args).selfcheck(payload), ensure_ascii=False, indent=1))
    return 0


def cmd_bench(args) -> int:
    pred = _predictor(args)
    payload = _load_payload(args.input, args.split, args.states)
    reference = None if args.input else upstream.split_truth(args.split, limit_states=args.states)
    strategies = tuple(s for s in args.strategies.split(",") if s)
    report = run_bench(
        pred, payload, repeats=args.repeats, strategies=strategies,
        batch_questions=args.batch_questions, reference=reference, head_chunk=args.head_chunk,
        chunk_states=args.chunk_states, stage_a_group=args.stage_a_group,
    )
    print(summarize(report))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
        print(f"\nreport written to {args.out}")
    return 0


def cmd_eval(args) -> int:
    pred = _predictor(args)
    payload = _load_payload(args.input, args.split, args.states)
    out = _evaluate(_engine(pred, args), pred, args, payload)
    text = json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"output": args.out, "execution": out["execution"]},
                         ensure_ascii=False))
    else:
        print(text)
    return 0


def cmd_oracle(args) -> int:
    """Run the upstream baseline and write it to disk, so it can be compared later."""
    pred = _predictor(args)
    payload = _load_payload(args.input, args.split, args.states)
    out = pred.predict(payload, batch_questions=args.batch_questions, temperature=1.0)
    text = json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"output": args.out, "execution": out["execution"]}, ensure_ascii=False))
    else:
        print(text)
    return 0


def cmd_compare(args) -> int:
    a = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    b = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    ref = json.loads(Path(args.teacher).read_text(encoding="utf-8")) if args.teacher else None
    print(json.dumps(compare(a, b, reference=ref), ensure_ascii=False, indent=1))
    return 0


def cmd_serve(args) -> int:
    """Start the Jev-compatible service. Single process + single inference thread: the model requires single-threaded access, and a shared cache has to stay in one process."""
    import uvicorn

    from .server import STRATEGIES, EvaluateService, create_app

    service = EvaluateService(
        checkpoint_dir=args.model,
        backend=args.backend,
        arch=args.arch,
        strategy=args.strategy,
        max_rows=args.max_rows,
        max_state_tokens=args.max_state_tokens or None,
        crop_state=args.crop_state,
        enforce_limits=not args.no_limits,
        api_key=args.api_key,
    )
    app = create_app(service)
    knob = {"state-fork": "layout=state_first",
            "single-path": f"sequence_limit={service.predictor.max_len} crop_state={args.crop_state}"}.get(
        service.facts["arrangement"], f"strategies={','.join(STRATEGIES)}")
    print(f"jevinf serve: device={service.predictor.device} checkpoint={service.predictor.root} "
          f"arch={service.arch} arrangement={service.facts['arrangement']} {knob} "
          f"max_rows={args.max_rows} temperature={service.facts['temperature_default']} "
          f"limits={'on' if service.enforce_limits else 'off'} "
          f"auth={'required' if args.api_key else 'open'}", flush=True)
    print("  Jev-compatible: POST /v1/systemone  GET /v1/models"
          "   (native debug entry points: POST /api/evaluate, GET /health)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level=args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jevinf", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(BACKENDS),
                        help="compute backend; torch-mps (Apple silicon) and torch-cuda (NVIDIA) "
                             "are wired up, torch-cpu and torch-rocm refuse to run")
    common.add_argument("--arch", default=DEFAULT_ARCH, choices=list(ARCHITECTURES),
                        help="model architecture; nanojev, decider-2b and laya are wired up")

    family = argparse.ArgumentParser(add_help=False)
    family.add_argument("--max-rows", type=int, default=32,
                        help="rows per batched forward (decider-2b, laya)")
    family.add_argument("--prefix-chunk", type=int, default=1024,
                        help="decider-2b: prefix tokens per forward (0 = one shot)")
    family.add_argument("--crop-state", action="store_true",
                        help="laya: crop a state that does not fit the sequence budget instead of "
                             "refusing it; every answer from a cropped state says so")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selfcheck", parents=[common, family],
                       help="isolated check that the engine matches its reference")
    s.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    s.add_argument("--input", default=None)
    s.add_argument("--split", default=None, help="dev split (jsonl)")
    s.add_argument("--states", type=int, default=8)
    s.set_defaults(func=cmd_selfcheck)

    s = sub.add_parser("bench", parents=[common],
                       help="end-to-end comparison + timing on the dev split")
    s.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    s.add_argument("--input", default=None)
    s.add_argument("--split", default=None, help="dev split (jsonl)")
    s.add_argument("--states", type=int, default=None)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--strategies", default="fused_state,fused_q,two_stage")
    s.add_argument("--batch-questions", type=int, default=0)
    s.add_argument("--head-chunk", type=int, default=128)
    s.add_argument("--chunk-states", type=int, default=32,
                   help="run in chunks of N states (0=no chunking); 32 matches the Jev API limit")
    s.add_argument("--stage-a-group", type=int, default=8,
                   help="how many states to forward together per Stage A batch (1=one state at a time, no batching)")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("eval", parents=[common, family], help="run a single payload through the engine")
    s.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    s.add_argument("--input", default=None)
    s.add_argument("--split", default=None, help="dev split (jsonl)")
    s.add_argument("--states", type=int, default=8)
    s.add_argument("--strategy", default="fused")
    s.add_argument("--head-chunk", type=int, default=128)
    s.add_argument("--layout", default="state_first",
                   help="decider-2b: layout of one state's questions (state_first today)")
    s.add_argument("--max-state-tokens", type=int, default=0,
                   help="decider-2b: state token budget (0 = the model's own budget); a longer state is refused, not truncated")
    s.add_argument("--temperature", type=float, default=0.0,
                   help="decider-2b: override the calibrated readout temperature (0 = the card's value)")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("oracle", parents=[common],
                       help="run the upstream baseline and write it to disk")
    s.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    s.add_argument("--input", default=None)
    s.add_argument("--split", default=None, help="dev split (jsonl)")
    s.add_argument("--states", type=int, default=None)
    s.add_argument("--batch-questions", type=int, default=0)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_oracle)

    s = sub.add_parser("compare", help="compare two results already written to disk")
    s.add_argument("--reference", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--teacher", default=None)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("serve", parents=[common],
                       help="start the Jev-compatible service (POST /api/evaluate)")
    s.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8226)
    s.add_argument("--strategy", default="fused_state")
    s.add_argument("--max-rows", type=int, default=64)
    s.add_argument("--max-state-tokens", type=int, default=0,
                   help="decider-2b: state token budget for the whole service (0 = the model's own "
                        "budget); a state over it is refused with a reason, not truncated")
    s.add_argument("--crop-state", action="store_true",
                   help="laya: crop a state that does not fit the sequence budget instead of refusing "
                        "it; every answer from a cropped state says so")
    s.add_argument("--no-limits", action="store_true",
                   help="do not enforce the per-family request limits (states/questions/paths and body size)")
    s.add_argument("--api-key", default=None,
                   help="when set, /v1/* requires Authorization: Bearer ***")
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    if not hasattr(args, "input"):
        args.input = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
