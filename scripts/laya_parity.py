"""Does the laya family adapter agree with the checkpoint's own inference?

Three checks, in order of how much they would hurt to get wrong:

  1. sequence identity -- our token ids and marker positions equal the checkpoint's `build_sequence`
  2. answers           -- our probabilities equal `RLAgent.system_one`'s: argmax/choice and max |Δp|
                          per question, plus the score value and the noul probability
  3. batching          -- one row per forward against one padded batch. This is the check that matters
                          for a bidirectional family: padding is only harmless if the attention mask is
                          actually applied, and a masking bug moves short rows' answers while long rows
                          still look fine.

The checkpoint's entry points ship beside the weights (`rl_agent_api.py` over `rl_common.py`), so they
are imported from the model directory rather than vendored. Both sides run on the same device and in the
same precision: a cross-device comparison would measure the port, not the adapter.

    uv run python scripts/laya_parity.py --model /path/to/laya [--max-rows 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CASES = [
    ("email triage",
     {"from": "customer@acme.com", "subject": "Duplicate billing on March invoice #4411",
      "body": "Hi team, we were billed twice for March. Please refund the duplicate before Friday or we "
              "will cancel our plan."},
     {"department": {"type": "choice", "instructions": "Which department should handle this email?",
                     "criteria": {"billing": "invoices, payments, refunds",
                                  "technical": "bugs, outages, integrations",
                                  "sales": "pricing, contracts, demos",
                                  "other": "everything else"}},
      "urgency": {"type": "score", "instructions": "How urgent is this request?",
                  "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
      "churn_risk": {"type": "noul",
                     "instructions": "Does the user threaten to cancel or switch to a competitor?"},
      "is_phishing": {"type": "noul", "instructions": "Is this email a phishing or scam attempt?"}}),
    ("twelve options",
     "The nightly export wrote 14,000 rows to staging and the dashboard has been stale since 03:10.",
     {"route": {"type": "choice", "instructions": "Where should this be routed?",
                "criteria": {f"queue_{i}": None for i in range(12)}},
      "hosed": {"type": "noul", "instructions": "Is the pipeline broken right now?"}}),
    ("ten levels",
     "Latency on the checkout API went from 120ms to 2.4s after the last deploy; error rate is "
     "unchanged and the database looks healthy.",
     {"severity": {"type": "score", "instructions": "How severe is this?",
                   "criteria": [f"level {i}" for i in range(10)]},
      "regression": {"type": "noul", "instructions": "Did the deploy cause this?"}}),
    ("two options",
     "Employee asks whether they can expense a 40 EUR team lunch without a receipt.",
     {"approve": {"type": "choice", "instructions": "Approve or deny?",
                  "criteria": {"approve": None, "deny": None}},
      "policy": {"type": "score", "instructions": "How clear is the policy?",
                 "criteria": ["unclear", "mostly clear", "clear"]}}),
    # instructions that contain the mask literal: the checkpoint strips it, and so must we
    ("mask in instructions",
     "Ticket: the user pasted a redacted log line and asks what it means.",
     {"kind": {"type": "choice", "instructions": "Is the marker [MASK] relevant to the answer?",
               "criteria": {"relevant": "the marker is part of the question",
                            "noise": "the marker is a redaction artefact"}}}),
    # the question set the Jev conformance script posts, whose 3-way routing question is the one where
    # reversing the options moved the winner in the engine's own selfcheck
    ("conformance set",
     "Incident: error rate 3.4%, p99 latency 850ms, the last deploy was 40 minutes ago and the on-call "
     "engineer is paged. Customer says payouts have been failing for three days and nobody replied to "
     "their emails.",
     {"team": {"type": "choice", "instructions": "Which team should handle it?",
               "criteria": {"payments": "Payouts, refunds, charges",
                            "platform": "Latency, outages, errors", "other": "other"}},
      "billing": {"type": "noul", "instructions": "Is this message about billing?"},
      "tone": {"type": "score", "instructions": "How frustrated is the customer?",
               "criteria": ["calm", "frustrated", "very angry"]}}),
    # long state: near the 512-token budget once the options and markers are in
    ("long state",
     "Incident thread, newest last.\n" + "\n".join(
         f"[{i:02d}] tenant t{i:04d} reports the export failed again with error 500 after 14 minutes; "
         f"they ask whether the retry queue is stuck and whether the report can be read from a replica."
         for i in range(9)),
     {"sev": {"type": "choice", "instructions": "How severe is this incident?",
              "criteria": ["cosmetic", "degraded", "partial outage", "full outage"]},
      "comms": {"type": "noul", "instructions": "Does the customer need an update within the hour?"}}),
]


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({detail})" if detail else ""))
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.rows if not ok]


def compare_answers(mine: dict, theirs: dict) -> tuple[bool, float, str]:
    """(argmax/score agreement, max probability gap, detail) for one question."""
    if mine["type"] == "noul":
        gap = abs(float(mine["noul"]) - float(theirs["noul"]))
        return gap <= 2e-3, gap, f"noul {mine['noul']} vs {theirs['noul']}"
    pa, pb = mine["probabilities"], theirs["probabilities"]
    keys = [k for k in pa if k in pb]
    gap = max((abs(float(pa[k]) - float(pb[k])) for k in keys), default=0.0)
    if mine["type"] == "choice":
        ok = mine["choice"] == theirs["choice"]
        return ok, gap, f"choice {mine['choice']} vs {theirs['choice']}"
    ok = abs(float(mine["score"]) - float(theirs["score"])) <= 1e-3
    return ok, gap, f"score {mine['score']} vs {theirs['score']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="torch-mps")
    ap.add_argument("--max-rows", type=int, default=16)
    ap.add_argument("--tolerance", type=float, default=2e-3)
    args = ap.parse_args()

    model_dir = Path(args.model)
    sys.path.insert(0, str(model_dir))
    import rl_agent_api  # the checkpoint's own entry point, beside the weights
    import rl_common

    from jevinf.arch import resolve
    from jevinf.laya import LayaEngine, build_sequence, load_predictor, question_spec, render_options

    print(f"reference: {model_dir}/rl_agent_api.py (imported in place)")
    t0 = time.perf_counter()
    ref = rl_agent_api.RLAgent(str(model_dir), device="mps")
    print(f"  loaded reference in {time.perf_counter() - t0:.1f}s")
    pred = load_predictor(str(model_dir), backend=args.backend, arch=resolve("laya"), precision="fp32")
    engine = LayaEngine(pred, max_rows=args.max_rows)
    report = Report()
    print(f"  weights: {len(pred.missing_keys)} missing / {len(pred.unexpected_keys)} unexpected keys "
          f"(0/0 means the strict load matched the checkpoint)")

    # ---------------------------------------------------------------- 1) sequences
    same_ids, same_markers, checked = 0, 0, 0
    for name, state, questions in CASES:
        for qid, q in questions.items():
            theirs_ids, theirs_markers = rl_common.build_sequence(
                ref.tok, state, rl_agent_api.RLAgent._to_internal(q),
                ref.cfg["max_len"], ref.cfg["head_max_len"],
            )
            ours_ids, ours_markers, _info = build_sequence(
                pred.tok, state, question_spec(q), pred.max_len, pred.head_max_len,
            )
            checked += 1
            same_ids += int(ours_ids == theirs_ids)
            same_markers += int(ours_markers == list(theirs_markers))
            if ours_ids != theirs_ids or ours_markers != list(theirs_markers):
                print(f"    mismatch in {name}/{qid}: ids {len(ours_ids)} vs {len(theirs_ids)}, "
                      f"markers {ours_markers} vs {theirs_markers}")
    report.check("sequence identity (token ids equal the checkpoint's builder)",
                 same_ids == checked and checked > 0, f"{same_ids}/{checked} questions")
    report.check("marker positions identical", same_markers == checked and checked > 0,
                 f"{same_markers}/{checked} questions")

    # ---------------------------------------------------------------- 2) answers
    agree, worst, total = 0, 0.0, 0
    rows_total, q_total = 0, 0
    for name, state, questions in CASES:
        theirs = ref.system_one(state, questions)["answers"]
        started = time.perf_counter()
        mine = engine.evaluate({"states": [{"id": "s", "state": state, "questions": questions}]})
        elapsed = time.perf_counter() - started
        answers = mine["states"][0]["answers"]
        rows_total += mine["execution"]["rows"]
        q_total += len(questions)
        for qid in questions:
            ok, gap, detail = compare_answers(answers[qid], theirs[qid])
            total += 1
            agree += int(ok)
            worst = max(worst, gap)
            print(f"    {name}/{qid}: {detail}  Δ={gap:.4f}  ({elapsed:.2f}s for {len(questions)} questions)")
    report.check("answers agree with RLAgent.system_one", agree == total and total > 0,
                 f"{agree}/{total} questions, max Δp {worst:.4f}")
    report.check("probability gap within tolerance", worst <= args.tolerance,
                 f"max Δp {worst:.6f} <= {args.tolerance}")

    # ---------------------------------------------------------------- 3) batching
    worst_batch, batch_disagree = 0.0, 0
    for name, state, questions in CASES:
        saved, engine.max_rows = engine.max_rows, max(1, len(questions))
        try:
            batched = engine.evaluate({"states": [{"id": "s", "state": state, "questions": questions}]})
        finally:
            engine.max_rows = saved
        saved, engine.max_rows = engine.max_rows, 1
        try:
            single = engine.evaluate({"states": [{"id": "s", "state": state, "questions": questions}]})
        finally:
            engine.max_rows = saved
        for qid in questions:
            ok, gap, _detail = compare_answers(single["states"][0]["answers"][qid],
                                               batched["states"][0]["answers"][qid])
            worst_batch = max(worst_batch, gap)
            batch_disagree += int(not ok)
    report.check("padding does not move an answer (1 row per forward vs one batch)",
                 batch_disagree == 0 and worst_batch <= args.tolerance,
                 f"{batch_disagree} disagreements, max Δp {worst_batch:.6f}")

    # ---------------------------------------------------------------- 4) option order
    # This family scores each option at its own marker inside one sequence, so which position an option
    # sits in is part of the input: reversing a list of options moves their probabilities and, on a close
    # question, can move which one wins. The question that matters is not "does it move" but "does it move
    # the way the checkpoint's own code moves" -- that is what separates a port artefact from the model's
    # behaviour, so the reference supplies the expected movement for every choice question in CASES.
    flips, mismatches, compared = [], 0, 0
    for name, state, questions in CASES:
        for qid, q in questions.items():
            if q.get("type") != "choice" or len(q.get("criteria") or {}) < 2:
                continue
            # criteria may be a dict (named options, optionally described) or a list (names only)
            criteria = q["criteria"]
            flipped = (list(reversed(criteria)) if isinstance(criteria, list)
                       else dict(reversed(list(criteria.items()))))
            ours = [engine.evaluate({"states": [{"id": "s", "state": state, "questions": {qid: spec}}]})
                    ["states"][0]["answers"][qid] for spec in (q, {**q, "criteria": flipped})]
            refa = [ref.system_one(state, {qid: spec})["answers"][qid] for spec in (q, {**q, "criteria": flipped})]
            ours_gap = max(abs(ours[0]["probabilities"][k] - ours[1]["probabilities"][k])
                           for k in ours[0]["probabilities"])
            ref_gap = max(abs(refa[0]["probabilities"][k] - refa[1]["probabilities"][k])
                          for k in refa[0]["probabilities"])
            ours_flip, ref_flip = ours[0]["choice"] != ours[1]["choice"], refa[0]["choice"] != refa[1]["choice"]
            compared += 1
            mismatches += int(ours_flip != ref_flip or abs(ours_gap - ref_gap) > 0.01)
            flips.append(f"{name}/{qid}")
            print(f"    {name}/{qid}: reversing options moves {ours_gap:.4f} (reference {ref_gap:.4f}), "
                  f"winner flips: ours={ours_flip} reference={ref_flip}")
    report.check("option order moves the answer like the checkpoint's own code (same gap, same flips)",
                 compared > 0 and mismatches == 0, f"{compared} choice questions, {mismatches} mismatches")

    # ---------------------------------------------------------------- informational
    wide = CASES[1][2]["route"]
    out = engine.evaluate({"states": [{"id": "s", "state": CASES[1][1], "questions": {"route": wide}}]})
    ans = out["states"][0]["answers"]["route"]
    top = max(ans["probabilities"].values())
    ext = ans["rl_agent"]
    print(f"\n  · 12-option choice: temperature {ext['temperature']} ({ext['bucket']}), "
          f"top-1 probability {top} — the shipped bucket value is below 1.0, so this bucket reports a "
          f"sharpened distribution rather than the measured odds")
    print(f"  · {rows_total} rows / {q_total} questions, "
          f"{out['execution']['sequence_limit']} token sequence budget, "
          f"prefix_sharing={out['execution']['prefix_sharing']}")

    print(f"\n{json.dumps({'pass': not report.failed, 'failed': report.failed})}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
