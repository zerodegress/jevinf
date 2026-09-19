"""Does the decider family adapter agree with the decider package?

Three checks, in order of how much they would hurt to get wrong:

  1. prompt identity  -- our row token ids and slot positions equal the package's `prompt.build` output
  2. answers          -- our probabilities equal the package's `Decider.system_one(independent=True)`,
                         per question: argmax, and max |Δp| over that question's options
  3. isolation        -- reordering or adding questions must not move another question's answer

The package is imported from the checkpoint directory (it ships beside the weights), so nothing is
vendored here. Usage:

    uv run python scripts/decider_parity.py --model /path/to/decider-2b [--max-state-tokens 1536]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CASES = [
    ("support ticket", "My card was charged twice for the same purchase and I want the extra charge refunded.",
     {"dept": {"type": "choice", "instructions": "Which department should handle this?",
               "criteria": ["billing", "technical support", "sales"]},
      "refund": {"type": "noul", "instructions": "Does this need a refund action?"},
      "urgency": {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["none", "slight", "moderate", "high"]}}),
    ("broad labels", "The nightly batch job wrote 14,000 rows to the staging table and the dashboard is stale.",
     {"route": {"type": "choice", "instructions": "Where should this be routed?",
                "criteria": {f"queue_{i}": None for i in range(12)}},
      "abstain": {"type": "choice", "instructions": "Which team owns the pipeline?",
                  "criteria": ["data-platform", "analytics", "infra", "none of the above"]}}),
    ("descriptions", "Latency on the checkout API went from 120ms to 2.4s after the last deploy; "
                     "error rate is unchanged and the database looks healthy.",
     {"cause": {"type": "choice", "instructions": "What is the most likely cause?",
                "criteria": {"cache": "cold caches or cache stampede",
                             "code": "a regression in the deployed code path",
                             "db": "database contention or a missing index",
                             "net": "network or upstream dependency latency"}}}),
    # A state long enough that the state prefix itself clears MIN_PREFIX, which is what makes the fork
    # path (and not just the batched path) run; the short states above only exercise full rows.
    ("long state", "Incident thread, newest last.\n"
                   + "\n".join(f"[{i:02d}] customer reports that the export job for tenant t{i:04d} failed again with "
                               f"error 500 after 14 minutes; they asked whether the retry queue is stuck and whether "
                               f"the report can be pulled from a replica instead."
                               for i in range(12)),
     {"severity": {"type": "choice", "instructions": "How severe is this incident?",
                   "criteria": ["cosmetic", "degraded", "partial outage", "full outage"]},
      "owner": {"type": "choice", "instructions": "Which team should own the fix?",
                "criteria": {"platform": "the job runner and retry queue",
                             "data": "the export pipeline and replicas",
                             "support": "customer communication only"}},
      "comms": {"type": "noul", "instructions": "Does the customer need an update within the hour?"},
      "confidence": {"type": "score", "instructions": "How confident can we be in a same-day fix?",
                     "criteria": ["no chance", "unlikely", "possible", "likely", "very likely"]}}),
]


def our_rows(module, tok, state, questions, limit):
    rendered = {qid: module.render_question(spec) for qid, spec in questions.items()}
    rows, index = module.plan_rows(rendered, False)
    state_ids = tok.encode("Context:\n" + module.render_state(state), add_special_tokens=False)[:limit]
    built = []
    for k, row in enumerate(rows):
        options = module.neutralize_options(row["options"])
        ids, slot = module.row_ids(tok, state_ids, dict(row, options=options), False, k)
        built.append((ids, slot))
    return rendered, index, built


def official_rows(prompt_module, tok, state, questions, limit):
    from decider.infer import Example, Q
    from decider.systemone import render_question as official_render

    rqs = {qid: official_render(spec) for qid, spec in questions.items()}
    from decider.systemone import plan_rows as official_plan

    rows, _index = official_plan(rqs, False)
    from decider.infer import neutralize_options

    ctx = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    built = []
    for row in rows:
        options = neutralize_options(row["options"])[0]
        item = prompt_module.build(Example(ctx, [Q(row["question"], options, 0)]), tok,
                                   _Keep(), max_options=prompt_module.MAX_OPTIONS,
                                   max_ctx_tokens=limit, layout="state_first")
        built.append((item["ids"], item["slots"][0]))
    return built


class _Keep:
    def shuffle(self, x):
        pass

    def sample(self, xs, k):
        return xs[:k]


def compare_answers(official, ours, label):
    """Per question: argmax agreement and the largest probability gap."""
    agree, worst, keys = 0, 0.0, []
    for qid, ref in official.items():
        mine = ours.get(qid)
        if mine is None:
            print(f"    [{label}:{qid}] MISSING from our output")
            continue
        kind = ref["type"]
        if kind == "noul":
            ref_p, my_p = [1 - ref["noul"], ref["noul"]], [1 - mine["noul"], mine["noul"]]
        elif kind == "choice":
            names = list(ref["probabilities"])
            ref_p = [ref["probabilities"][n] for n in names]
            my_p = [mine["probabilities"][n] for n in names]
        else:
            levels = list(ref["probabilities"])
            ref_p = [ref["probabilities"][l] for l in levels]
            my_p = [mine["probabilities"][l] for l in levels]
        delta = max(abs(a - b) for a, b in zip(ref_p, my_p))
        top = (max(range(len(ref_p)), key=ref_p.__getitem__) == max(range(len(my_p)), key=my_p.__getitem__))
        agree += int(top)
        worst = max(worst, delta)
        keys.append(qid)
        flag = "=" if top else "!"
        print(f"    [{label}:{qid}] type={kind:<6} argmax {flag} ref={_top(ref, kind)} ours={_top(mine, kind)} "
              f"max|dp|={delta:.4f}")
    return agree, len(keys), worst


def _top(answer, kind):
    if kind == "noul":
        return f"noul={answer['noul']}"
    if kind == "choice":
        return f"{answer['choice']}({answer['confidence']})"
    return f"level {answer['score']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-state-tokens", type=int, default=1536)
    ap.add_argument("--max-rows", type=int, default=32,
                    help="rows per batched forward; 1 makes every row's numbers shape-independent")
    # The gate is argmax agreement; the probability gap is reported next to it. Continuing a row from a
    # forked prefix is not bit-identical to one-shot computation (chunk phase of the recurrence), and the
    # gap grows with prefix length: 0.007 at ~50-token states, 0.011 at a 637-token state.
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()

    model = Path(args.model).resolve()
    sys.path.insert(0, str(model))
    import decider.prompt as prompt_module  # the package that ships beside the weights
    from decider.infer import Decider

    print(f"checkpoint {model}")
    print("[1] prompt identity")
    from jevinf import decider as ours_module
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model))
    prompt_ok = True
    for label, state, questions in CASES:
        _rendered, _index, ours = our_rows(ours_module, tok, state, questions, args.max_state_tokens)
        theirs = official_rows(prompt_module, tok, state, questions, args.max_state_tokens)
        same_ids = all(a[0] == b[0] for a, b in zip(ours, theirs))
        same_slots = all(a[1] == b[1] for a, b in zip(ours, theirs))
        prompt_ok &= same_ids and same_slots and len(ours) == len(theirs)
        first_diff = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a[0] != b[0]), None)
        print(f"    [{label}] rows={len(ours)} ids {'==' if same_ids else '!='} slots "
              f"{'==' if same_slots else '!='}" + (f"  first differing row {first_diff}" if first_diff is not None else ""))
        if not same_ids:
            ours_ids, their_ids = ours[first_diff or 0][0], theirs[first_diff or 0][0]
            n = min(len(ours_ids), len(their_ids))
            k = next((i for i in range(n) if ours_ids[i] != their_ids[i]), n)
            print(f"        first differing token at {k}: ours={ours_ids[k:k+4]} theirs={their_ids[k:k+4]}")
    print(f"    prompt identity: {'PASS' if prompt_ok else 'FAIL'}")

    print("[2] answers vs the package")
    # The reference is produced first and then released: two bf16 copies of this checkpoint plus
    # activations is more residency than the machine comfortably holds at once.
    d = Decider(str(model), device="mps", use_graphs=False)
    refs = {label: d.system_one(state, questions, independent=True, max_state_tokens=args.max_state_tokens)["answers"]
            for label, state, questions in CASES}
    del d
    import gc
    import torch

    gc.collect()
    torch.mps.empty_cache()

    agree = total = 0
    worst = 0.0
    shared_cases = 0
    for label, state, questions in CASES:
        a, n, w = compare_answers(refs[label], _ours_answers(model, state, questions, args), label)
        agree += a
        total += n
        worst = max(worst, w)
        ex = _CACHE["last_execution"]
        shared_cases += 1 if ex["prefix_sharing"] else 0
        print(f"    [{label}] rows={ex['rows']} shared={ex['prefix_sharing']} prefix_len={ex['prefix_len']} "
              f"forked={ex['forked_rows']} saved={ex['prefix_tokens_saved']} tok")
    print(f"    argmax agreement {agree}/{total}; max |Δp| = {worst:.4f} (tolerance {args.tolerance})")
    if not shared_cases:
        print("    the fork path never ran: no case cleared MIN_PREFIX, so this check proved less than it looks")
        return 1

    print("[3] question isolation")
    # Each row's prompt is the same whatever else is in the request, so the only thing that can move an
    # answer here is the batch shape (right padding, which is a few bits of bf16, not information).
    # With --max-rows 1 every row is forwarded on its own and the answers are bit-identical.
    isolation_ok, isolation_worst = True, 0.0
    label, state, questions = CASES[0]
    base = _ours_answers(model, state, questions, args)
    reordered = {k: questions[k] for k in list(questions)[::-1]}
    extra = dict(questions, added={"type": "choice", "instructions": "Is this a duplicate report?",
                                   "criteria": ["no", "yes"]})
    for name, variant in (("reordered", reordered), ("with an extra question", extra)):
        got = _ours_answers(model, state, variant, args)
        for qid in questions:
            if got[qid] == base[qid]:
                continue
            delta = _answer_delta(base[qid], got[qid])
            isolation_worst = max(isolation_worst, delta)
            if delta > args.tolerance:
                isolation_ok = False
            print(f"    {name}: {qid} moved by {delta:.4f} "
                  f"({'within tolerance, batch shape' if delta <= args.tolerance else 'TOO FAR'})")
    print(f"    isolation: {'PASS' if isolation_ok else 'FAIL'}; max move {isolation_worst:.4f} "
          f"(tolerance {args.tolerance}); max_rows={args.max_rows}")

    ok = prompt_ok and agree == total and worst <= args.tolerance and isolation_ok
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _answer_delta(a: dict, b: dict) -> float:
    """Largest probability gap between two answers to the same question."""
    if a.get("type") == "noul":
        return abs(a["noul"] - b["noul"])
    keys = [k for k in a.get("probabilities", {}) if k in b.get("probabilities", {})]
    if not keys:
        return 1.0
    return max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in keys)


_CACHE: dict = {}


def _ours_answers(model, state, questions, args):
    """Answers from our engine, cached so checks 2 and 3 do not reload the weights."""
    engine = _CACHE.get("engine")
    if engine is None:
        from jevinf import upstream

        pred = upstream.load_predictor(str(model), arch="decider-2b")
        from jevinf.decider import DeciderEngine

        engine = DeciderEngine(pred, max_rows=args.max_rows, max_ctx_tokens=args.max_state_tokens)
        _CACHE["engine"] = engine
    out = engine.evaluate({"states": [{"id": "s", "state": state, "questions": questions}]},
                          max_ctx_tokens=args.max_state_tokens)
    _CACHE["last_execution"] = out["execution"]
    return out["states"][0]["answers"]


if __name__ == "__main__":
    sys.exit(main())
