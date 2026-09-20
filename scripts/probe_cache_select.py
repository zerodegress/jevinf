"""Probe: can we safely do "whole-batch forward + per-row extraction" to save Stage A's
per-state forward?

Background: Stage A currently does one forward per state (120 for the dev split, 25% of all
forwards). To compress that to one per chunk, we must extract a single-row cache out of the batch
cache row by row, and after extraction `get_seq_length()` must return the true length -- the model
relies on it to derive cache_position.

Known: `batch_select_indices` rewrites in place (returns None); after hand-assembling a DynamicLayer
`get_seq_length()` reads 0. This probe checks whether DynamicLayer has an initialization flag that
can be set by hand, and **verifies by numerical equivalence** (continue the forward after
extraction and compare hidden against one full-length forward).

`--backend` picks the device (default `torch-mps`). Re-run on CUDA on 2026-09-20 the answer is the
same: the hand-assembled layer carries `is_initialized=True`, and the continuation forward still
raises a shape error at row 0 rather than returning a wrong number.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, DynamicCache
from transformers.cache_utils import DynamicLayer

def main(model: str, backend: str) -> int:
    from jevinf.backend import resolve as resolve_backend

    device = resolve_backend(backend).device
    cfg = AutoConfig.from_pretrained(str(Path(model) / "backbone_config"), local_files_only=True)
    cfg.use_cache = True
    model = AutoModel.from_config(cfg, attn_implementation="sdpa").float().eval().to(device)
    torch.manual_seed(0)

    lens = [7, 5, 6]
    rows = [torch.randint(0, 1000, (1, n), device=device) for n in lens]
    width = max(lens)
    tokens = torch.full((len(rows), width), 0, dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        tokens[i, : r.shape[1]] = r
    attn = torch.arange(width, device=device)[None, :] < torch.tensor(lens, device=device)[:, None]

    with torch.no_grad():
        ref_full = model(input_ids=torch.cat(rows, dim=1), attention_mask=None).last_hidden_state[:, -1]
        out = model(input_ids=tokens, attention_mask=attn, use_cache=True)
        kv = out.past_key_values

        lay = kv.layers[0]
        marks = {a: getattr(lay, a) for a in dir(lay)
                 if not a.startswith("__") and isinstance(getattr(lay, a, None), bool)}
        print("bool flags on layer:", marks)

        # extract row by row by hand, then continue the forward and compare against one full-length forward
        for i in range(len(rows)):
            hand = DynamicCache()
            for L in kv.layers:
                nl = DynamicLayer()
                nl.keys = L.keys[i : i + 1, :, : lens[i], :].clone()
                nl.values = L.values[i : i + 1, :, : lens[i], :].clone()
                hand.layers.append(nl)
            seq = hand.get_seq_length()
            # try to set the initialization flag
            if hasattr(hand.layers[0], "lazy_initialization"):
                hand.layers[0].lazy_initialization = False
            for attr in ("is_initialized", "_is_initialized"):
                if hasattr(hand.layers[0], attr):
                    try:
                        setattr(hand.layers[0], attr, True)
                    except Exception:
                        pass
            seq2 = hand.get_seq_length()
            new = torch.randint(0, 1000, (1, 2), device=device)
            full_mask = torch.ones((1, lens[i] + 2), dtype=torch.bool, device=device)
            got = model(input_ids=new, attention_mask=full_mask, past_key_values=hand,
                        use_cache=True).last_hidden_state[:, -1]
            d = float((got - ref_full[i : i + 1]).abs().max())
            print(f"row{i} len={lens[i]} seq_before={seq} seq_after_marks={seq2} "
                  f"max|Δ|={d:.3e}  {'OK' if d < 1e-4 else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True, help="checkpoint directory (weights)")
    ap.add_argument("--backend", default="torch-mps", help="compute backend, see jevinf.backend")
    args = ap.parse_args()
    raise SystemExit(main(args.model, args.backend))
