"""Probe v2: assemble single-row caches layer by layer via the official `DynamicCache.update()`
and verify numerical equivalence.

v1's conclusion (scripts/probe_cache_select.py): assigning `DynamicLayer.keys/values` directly still
leaves `get_seq_length()` reading 0, so the model computes the target length as the "new segment
length", which conflicts with the full mask and raises a shape error.
v2 gets `get_seq_length()` right after `update()`, but v2's first version wrote the reference
baseline wrong (`torch.cat(rows, dim=1)` concatenates the three rows into one 18-token sequence
instead of three independent rows) --
this version changes the reference to a per-row independent forward, and compares hidden for
"continuation from the extracted cache" against "one full-length forward".

If it passes, Stage A can be compressed from "one forward per state" to "one forward per batch".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, DynamicCache




def build_via_update(kv, row: int, length: int) -> DynamicCache:
    hand = DynamicCache()
    for idx, layer in enumerate(kv.layers):
        hand.update(
            layer.keys[row : row + 1, :, :length, :].clone(),
            layer.values[row : row + 1, :, :length, :].clone(),
            idx,
        )
    return hand


def main(model: str) -> int:
    cfg = AutoConfig.from_pretrained(str(Path(model) / "backbone_config"), local_files_only=True)
    cfg.use_cache = True
    model = AutoModel.from_config(cfg, attn_implementation="sdpa").float().eval().to("mps")
    torch.manual_seed(0)

    lens = [7, 5, 6]
    tails = [torch.randint(0, 1000, (1, 2), device="mps") for _ in lens]
    rows = [torch.randint(0, 1000, (1, n), device="mps") for n in lens]
    width = max(lens)
    tokens = torch.full((len(rows), width), 0, dtype=torch.long, device="mps")
    for i, r in enumerate(rows):
        tokens[i, : r.shape[1]] = r
    attn = torch.arange(width, device="mps")[None, :] < torch.tensor(lens, device="mps")[:, None]

    with torch.no_grad():
        # reference: run the "whole path" per row independently = that row's tokens + tail, take the last position
        ref = [model(input_ids=torch.cat([rows[i], tails[i]], dim=1)).last_hidden_state[:, -1]
               for i in range(len(rows))]

        kv = model(input_ids=tokens, attention_mask=attn, use_cache=True).past_key_values
        for i in range(len(rows)):
            hand = build_via_update(kv, i, lens[i])
            seq = hand.get_seq_length()
            mask = torch.ones((1, seq + 2), dtype=torch.bool, device="mps")
            try:
                got = model(input_ids=tails[i], attention_mask=mask, past_key_values=hand,
                            use_cache=True).last_hidden_state[:, -1]
                d = float((got - ref[i]).abs().max())
                print(f"row{i} len={lens[i]} seq_len={seq} max|Δ|={d:.3e}  "
                      f"{'OK' if d < 1e-4 else 'MISMATCH'}")
            except Exception as exc:
                print(f"row{i} len={lens[i]} seq_len={seq} RAISED "
                      f"{type(exc).__name__}: {str(exc)[:110]}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    raise SystemExit(main(ap.parse_args().model))
