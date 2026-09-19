"""Measure against the real Jev API calling caliber (with memory guardrails).

Caliber change: the previously reported 2.27× was measured over the whole dev split
(120 states / 898 paths), roughly equivalent to 4 full requests.
What the client actually feels is **single-request latency**. This script measures against the
API's own limits:

    POST /api/evaluate   {states:[{id,state,questions}]}
    per-request limits: 32 states / 96 questions / 256 paths / 2 MB

Guardrails (added after getting burned):
  * Before every measurement, read `sysctl vm.swapusage`; if free swap is below the threshold,
    **refuse to run** and say so.
  * Pre-flight shape check: `paths × batch width` is the dominant memory term; report it when it
    exceeds the threshold.
  * Output is **not piped** (`| tail` buffers all progress, equivalent to flying blind); flush
    line by line.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

from jevinf import upstream
from jevinf.bench import run_bench

SWAP_FLOOR_MB = 500  # do not start a run when free swap is below this value
MAX_POSITIONS = 60_000  # rough upper bound on paths × batch width (~2-3GB peak on a 16GB machine)


def swap_free_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"free = ([\d.]+)M", out)
    return float(m.group(1)) if m else -1.0


def guard(label: str) -> bool:
    free = swap_free_mb()
    print(f"    [guardrail] {label}: swap free {free:.0f} MB", flush=True)
    if 0 <= free < SWAP_FLOOR_MB:
        print(f"    [guardrail] below {SWAP_FLOOR_MB} MB, refusing to run. Free up memory first "
              f"(e.g. stop the resident omlx-server), then retry.",
              flush=True)
        return False
    return True


def estimate(payload: dict, tokenizer) -> tuple[int, int, int]:
    """Rough estimate of (paths, batch width, paths × batch width). batch width = token count of the longest path."""
    examples = upstream.prepare_examples(payload, tokenizer, 512)
    paths = sum(len(ex["leaf_tokens"]) for ex in examples)
    width = max(len(leaf) for ex in examples for leaf in ex["leaf_tokens"])
    return paths, width, paths * width


def api_request(split, n_states: int = 32) -> dict:
    payload = upstream.read_split(split)
    return {"states": payload["states"][:n_states]}


def stress_requests(n_cand: int = 255) -> list[tuple[str, dict]]:
    """Stress shape that hugs the API limits. Uses short state text -- a long state would hit the
    checkpoint's max_length=512 and be raised upstream, so we would not measure what we want and
    would burn memory for nothing."""
    short_state = ("Nav episode: blocked cells at (3,4),(7,1); heading north; 12 steps left. ")
    big = {f"opt{i}": f"candidate action {i}" for i in range(n_cand)}
    one_big = {"states": [{"id": "stress-big-choice", "state": short_state,
                           "questions": {
                               "pick": {"type": "choice",
                                        "instructions": "Pick the single best action.",
                                        "criteria": big},
                               "ok": {"type": "boolean", "instructions": "Is the goal reachable?"},
                               "sev": {"type": "score", "instructions": "How blocked?",
                                       "criteria": ["clear", "blocked", "trapped"]},
                           }}]}
    many = {"states": [
        {"id": f"stress-{i}", "state": short_state,
         "questions": {
             "pick": {"type": "choice", "instructions": "Pick the single best action.",
                      "criteria": {k: big[k] for k in list(big)[:3]}},
             "ok": {"type": "boolean", "instructions": "Is the goal reachable?"},
             "sev": {"type": "score", "instructions": "How blocked?",
                     "criteria": ["clear", "blocked", "trapped"]},
         }} for i in range(32)]
    }
    return [(f"stress A: single state × {n_cand} candidates (single-question limit)", one_big),
            ("stress B: 32 states × full request", many)]


def run_one(name: str, payload: dict, pred, strategies, repeats: int, chunk_states: int) -> dict | None:
    paths, width, positions = estimate(payload, pred.tokenizer)
    nq = sum(len(s["questions"]) for s in payload["states"])
    print(f"\n— {name}: {len(payload['states'])} states / {nq} questions / {paths} paths / "
          f"batch width {width} / positions {positions:,}", flush=True)
    if positions > MAX_POSITIONS:
        print(f"    [guardrail] positions exceed {MAX_POSITIONS:,}, refusing to run "
              f"(would not fit in 16GB).", flush=True)
        return None
    if not guard(name):
        return None
    rep = run_bench(pred, payload, repeats=repeats, strategies=strategies,
                    chunk_states=chunk_states)
    base = rep["baseline"]["median_s"]
    be = rep["baseline"]["execution"]
    print(f"    baseline upstream  {base:>8.2f}s  forwards={be.get('forward_passes')}", flush=True)
    for s in strategies:
        e, x, c = rep["engine"][s], rep["engine"][s]["execution"], rep["compare"][s]
        print(f"    {s:<16} {e['median_s']:>8.2f}s  {e['speedup_vs_baseline']:>5.2f}×  "
              f"forwards={x['forwards']:<4} tokens={x['computed_tokens']:<8,} "
              f"stateKV={x['state_kv_bytes']/1e6:>5.1f}MB  "
              f"agreement={c['argmax_agreement']*100:.2f}%  TV median={c['tv']['median']:.2e}",
              flush=True)
    sc = rep["selfcheck"]
    print(f"    head self-check: argmax_match={sc['argmax_match']} max|Δlogit|={sc['max_abs_logit_diff']}",
          flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["api", "stress", "all"], default="api")
    ap.add_argument("--states", type=int, default=32, help="number of states per API request (max 32)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--strategies", default="fused_state,two_stage")
    ap.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    ap.add_argument("--split", required=True, help="dev split (jsonl)")
    args = ap.parse_args()

    print(f"swap free at start: {swap_free_mb():.0f} MB", flush=True)
    pred = upstream.load_predictor()
    strategies = tuple(s for s in args.strategies.split(",") if s)

    if args.only in ("api", "all"):
        # one request = one call: no chunking
        run_one(f"real split · single request ({args.states} states, API limit)",
                api_request(args.states), pred, strategies, args.repeats, 0)
    if args.only in ("stress", "all"):
        for name, payload in stress_requests():
            run_one(name, payload, pred, strategies, 0, 0)
    print(f"\nswap free at end: {swap_free_mb():.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
