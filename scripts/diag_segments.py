"""Segment and KV size diagnostics -- no performance conclusions, facts only.

Purpose: confirm the real token counts of the state/question/suffix segments, and the KV bytes
resident for a single state segment (this decides whether "lazy residency per state" holds).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jevinf import upstream
from jevinf.engine import PrefixShareEngine
from jevinf.plan import build_plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    ap.add_argument("--split", required=True, help="dev split (jsonl)")
    ap.add_argument("--states", type=int, default=3)
    args = ap.parse_args()
    payload = upstream.read_split(args.split, limit_states=args.states)
    pred = upstream.load_predictor(args.model)
    plan = build_plan(payload, pred.tokenizer, pred.limit)

    print(f"states={len(plan.state_ids)} questions={len(plan.questions)} "
          f"paths={sum(q.n_paths for q in plan.questions)}")
    for i, (sid, seg) in enumerate(zip(plan.state_ids, plan.state_segments)):
        print(f"  state[{i}] {sid}: state segment {len(seg)} tok")
    for q in plan.questions:
        print(f"  {q.ex_id}: q segment {len(q.question_seg)} tok | "
              f"candidate suffixes {[len(s) for s in q.suffix_ids]} | path lengths {q.path_lengths}")

    eng = PrefixShareEngine(pred)
    n_bytes = {}
    for sid, seg in zip(plan.state_ids, plan.state_segments):
        _, kv, _ = eng.forward_rows([seg])
        n_bytes[sid] = eng.cache_bytes(kv)
    k, L = pred.model.backbone.config.num_key_value_heads, pred.model.backbone.config.head_dim
    layers = pred.model.backbone.config.num_hidden_layers
    per_tok = 2 * k * L * 4 * layers
    print(f"\ntheoretical KV/token = 2 × {k} × {L} × 4B × {layers} layers = {per_tok} B "
          f"({per_tok/1024:.0f} KiB)")
    for sid, b in n_bytes.items():
        seg = plan.state_segments[plan.state_ids.index(sid)]
        print(f"  state {sid}: KV {b/1e6:.1f} MB / {len(seg)} tok = {b/len(seg)/1024:.0f} KiB/tok")
    total_all = sum(n_bytes.values())
    print(f"\nlazy residency per state: peak = largest single state {max(n_bytes.values())/1e6:.1f} MB")
    print(f"if the whole split were resident (extrapolated from 120 states): {total_all/len(n_bytes)*120/1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
