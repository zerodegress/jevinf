"""Shape robustness sweep: swap the dev split's shape and see whether the speedup still holds.

Motivation: 2.27× was measured on a split with a **fixed shape** (3 questions per state, 1–4
candidates per question, state segment ~125 token, question segment 18–36 token). If the gain
depends on that shape, then it is "a property of these questions" rather than "a property of the
engine".

Sweep three axes:
  * state segment length (hard-limited by checkpoint max_length=512, state segment at most ~450 token)
  * number of questions per state (affects the gain from reusing the prefix across questions)
  * number of candidates for choice questions (affects the batch width of per-question batching)

Print the wall time and speedup of upstream / fused_state / two_stage for each shape, plus the
argmax agreement rate (agreement must always be 100% -- it is an equivalent transform, independent
of the question content).
"""
from __future__ import annotations

import argparse
import sys

from jevinf import upstream
from jevinf.bench import run_bench

# (name, target state-segment tokens, questions per state, number of choice candidates)
SHAPES = [
    ("dev-like(125/3q/4c)", 125, 3, 4),
    ("short state(50/3q/4c)", 50, 3, 4),
    ("long state(450/3q/4c)", 450, 3, 4),
    ("many candidates(125/3q/16c)", 125, 3, 16),
    ("few candidates(125/3q/2c)", 125, 3, 2),
    ("single-question state(125/1q/4c)", 125, 1, 4),
    ("many-question state(125/12q/4c)", 125, 12, 4),
    ("no choice(125/3q/-)", 125, 3, 0),
]

FILLER = ("The grid reports blocked cells, visited cells, the current heading and the "
          "remaining distance to the goal for the active navigation episode. ")


def state_text(tokenizer, target_tokens: int) -> str:
    """Build a state text of about target_tokens (calibrated by measuring with the tokenizer)."""
    text = FILLER
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += FILLER
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_payload(tokenizer, target_state_tokens: int, n_questions: int, n_choices: int,
                  n_states: int = 8) -> dict:
    state = state_text(tokenizer, target_state_tokens)
    criteria = {f"opt{i}": f"candidate action number {i} for the navigation step"
                for i in range(n_choices)} if n_choices else None
    states = []
    for s in range(n_states):
        qs = {}
        for j in range(n_questions):
            qid = f"q{j}"
            if n_choices and j % 3 == 0:
                qs[qid] = {"type": "choice",
                           "instructions": "Which next action is most appropriate?",
                           "criteria": criteria}
            elif j % 3 == 1:
                qs[qid] = {"type": "boolean",
                           "instructions": "Is the goal reachable from the current cell?"}
            else:
                qs[qid] = {"type": "score",
                           "instructions": "How blocked is the current position?",
                           "criteria": ["clear", "lightly blocked", "blocked",
                                        "heavily blocked", "trapped"]}
        states.append({"id": f"st{s}", "state": state, "questions": qs})
    return {"states": states}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    args = ap.parse_args()
    pred = upstream.load_predictor(args.model)
    tok = pred.tokenizer
    print(f"{'shape':<24} {'base s':>8} {'fused_state':>12} {'two_stage':>11} "
          f"{'factor':>7} {'agree':>8} {'tok ratio':>8}")
    rows = []
    for name, st_tok, nq, nc in SHAPES:
        payload = build_payload(tok, st_tok, nq, nc)
        rep = run_bench(pred, payload, repeats=0,
                        strategies=("fused_state", "two_stage"), chunk_states=8)
        base = rep["baseline"]["median_s"]
        fs = rep["engine"]["fused_state"]
        ts = rep["engine"]["two_stage"]
        cx = rep["compare"]["fused_state"]
        lex = fs["execution"]
        tok_ratio = lex["baseline_tokens"] / lex["computed_tokens"]
        print(f"{name:<24} {base:>8.2f} {fs['median_s']:>12.2f} {ts['median_s']:>11.2f} "
              f"{fs['speedup_vs_baseline']:>7.2f} "
              f"{cx['argmax_agreement']*100:>7.2f}% {tok_ratio:>7.2f}×")
        rows.append((name, base, fs["median_s"], ts["median_s"],
                     fs["speedup_vs_baseline"], ts["speedup_vs_baseline"],
                     cx["argmax_agreement"], tok_ratio))
    print("\nstrategy comparison (two_stage/fused_state):")
    for r in rows:
        name, base, fs, ts, rfs, rts, ag, tr = r
        print(f"  {name:<24} fused_state {rfs:.2f}× vs two_stage {rts:.2f}× "
              f"→ {'two_stage wins' if rts > rfs else 'fused_state leads'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
