"""jevinf command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import upstream
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
        checkpoint_dir=args.model, device_name=args.device, precision="fp32"
    )


def cmd_selfcheck(args) -> int:
    pred = _predictor(args)
    payload = _load_payload(args.input, args.split, args.states)
    print(json.dumps(head_selfcheck(pred, payload), ensure_ascii=False, indent=1))
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
    eng = PrefixShareEngine(pred, strategy=args.strategy, head_chunk=args.head_chunk)
    out = eng.evaluate(payload)
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
        strategy=args.strategy,
        max_rows=args.max_rows,
        enforce_limits=not args.no_limits,
        api_key=args.api_key,
    )
    app = create_app(service)
    print(f"jevinf serve: device={service.predictor.device} checkpoint={service.predictor.root} "
          f"strategy={args.strategy} max_rows={args.max_rows} "
          f"limits={'on' if service.enforce_limits else 'off'} "
          f"auth={'required' if args.api_key else 'open'} "
          f"strategies={','.join(STRATEGIES)}", flush=True)
    print("  Jev-compatible: POST /v1/systemone  GET /v1/models"
          "   (native debug entry points: POST /api/evaluate, GET /health)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level=args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jevinf", description=__doc__)
    parser.add_argument("--device", default="mps")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selfcheck", help="isolated check that the decision head matches upstream")
    s.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    s.add_argument("--input", default=None)
    s.add_argument("--split", default=None, help="dev split (jsonl)")
    s.add_argument("--states", type=int, default=8)
    s.set_defaults(func=cmd_selfcheck)

    s = sub.add_parser("bench", help="end-to-end comparison + timing on the dev split")
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

    s = sub.add_parser("eval", help="run a single payload through the engine")
    s.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    s.add_argument("--input", default=None)
    s.add_argument("--split", default=None, help="dev split (jsonl)")
    s.add_argument("--states", type=int, default=8)
    s.add_argument("--strategy", default="fused")
    s.add_argument("--head-chunk", type=int, default=128)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("oracle", help="run the upstream baseline and write it to disk")
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

    s = sub.add_parser("serve", help="start the Jev-compatible service (POST /api/evaluate)")
    s.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8226)
    s.add_argument("--strategy", default="fused_state")
    s.add_argument("--max-rows", type=int, default=64)
    s.add_argument("--no-limits", action="store_true",
                   help="do not enforce the 32 states / 96 questions / 256 paths / 2 MiB limits")
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
