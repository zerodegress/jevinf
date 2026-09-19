"""Measure against the real Jev API calling caliber.

Motivation: the previously reported 2.27× was measured over the whole dev split
(120 states / 898 paths), roughly equivalent to 4 full requests.
What the client actually feels is **single-request latency**, and the shape of a single request
differs from that of the whole split.
This script measures against the API's own limits:

    POST /api/evaluate   {states:[{id,state,questions}]}
    per-request limits: 32 states / 96 questions / 256 paths / 2 MB

Three things:
  1. A 32-state single request from the real split (API limit): compare latency at equal size.
  2. Stress shape: a single state carrying 255 candidates (single-question limit), with the path
     count close to the 256 limit, to check whether `max_rows` sub-batching really contains memory
     -- the dev split has only 1-4 candidates per question, so this edge cannot be exercised there.
  3. Agreement must always be 100% (the criterion for an equivalent transform, independent of the
     question content).
"""
from __future__ import annotations

import argparse
import sys

from jevinf import upstream
from jevinf.bench import run_bench


def api_request(path=None, n_states: int = 32) -> dict:
    """Take the first n_states states of the real split to form one full API request."""
    payload = upstream.read_split(path)
    states = payload["states"][:n_states]
    return {"states": states}


def stress_requests() -> list[tuple[str, dict]]:
    """Stress shapes built right up against the API limits, aimed at max_rows and batch width."""
    big_choice = {f"opt{i}": f"candidate action {i} for the navigation step"
                  for i in range(255)}
    long_state = ("The grid reports blocked cells, visited cells, the current heading and the "
                  "remaining distance to the goal for the active navigation episode. ") * 12
    one_big = {
        "states": [{"id": "stress-big-choice", "state": long_state,
                    "questions": {
                        "pick": {"type": "choice",
                                 "instructions": "Pick the single best action.",
                                 "criteria": big_choice},
                        "ok": {"type": "boolean",
                               "instructions": "Is the goal reachable?"},
                        "sev": {"type": "score",
                                "instructions": "How blocked is the position?",
                                "criteria": ["clear", "blocked", "trapped"]},
                    }}]
    }
    many = {"states": [
        {"id": f"stress-{i}", "state": long_state,
         "questions": {
             "pick": {"type": "choice", "instructions": "Pick the single best action.",
                      "criteria": {k: big_choice[k] for k in list(big_choice)[:3]}},
             "ok": {"type": "boolean", "instructions": "Is the goal reachable?"},
             "sev": {"type": "score", "instructions": "How blocked?",
                     "criteria": ["clear", "blocked", "trapped"]},
         }} for i in range(32)]
    }
    return [("stress A: single state × 255 candidates", one_big),
            ("stress B: 32 states × full request", many)]


def report_line(name: str, rep: dict, strategies) -> None:
    base = rep["baseline"]["median_s"]
    print(f"\n— {name}")
    print(f"  baseline upstream    {base:>8.2f}s   forwards={rep['baseline']['execution'].get('forward_passes')}"
          f"  paths={rep['baseline']['execution'].get('candidate_paths')}")
    for s in strategies:
        e = rep["engine"][s]
        x = e["execution"]
        c = rep["compare"][s]
        print(f"  {s:<18} {e['median_s']:>8.2f}s   {e['speedup_vs_baseline']:>5.2f}×   "
              f"forwards={x['forwards']:<4} tokens={x['computed_tokens']:<8,} "
              f"stateKV={x['state_kv_bytes']/1e6:>5.1f}MB  "
              f"agreement={c['argmax_agreement']*100:.2f}%  TV median={c['tv']['median']:.2e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    ap.add_argument("--split", required=True, help="dev split (jsonl)")
    args = ap.parse_args()
    pred = upstream.load_predictor(args.model)
    strategies = ("fused_state", "two_stage")

    # 1) full API request: 32 states, no chunking (one request = one call)
    req = api_request(args.split, n_states=32)
    nq = sum(len(s["questions"]) for s in req["states"])
    print(f"full API request: {len(req['states'])} states / {nq} questions")
    rep = run_bench(pred, req, repeats=2, strategies=strategies, chunk_states=0)
    report_line("real split · single request (32 states)", rep, strategies)
    print(f"  head self-check: {rep['selfcheck']['argmax_match']} max|Δlogit|={rep['selfcheck']['max_abs_logit_diff']}")

    # 2) stress shapes
    for name, payload in stress_requests():
        try:
            rep = run_bench(pred, payload, repeats=0, strategies=strategies, chunk_states=0)
            report_line(name, rep, strategies)
        except Exception as exc:  # memory/topology failures must be surfaced, not hidden
            print(f"\n— {name}\n  FAILED {type(exc).__name__}: {str(exc)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
