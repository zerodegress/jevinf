"""Decision-head math -- a line-by-line mirror of the latter half of the upstream DecisionModel.forward.

It lives in its own file so it can be self-checked in isolation: feed in the leaves computed
upstream and compare element-wise against the logits the upstream produces. The head takes only
0.3% of the time, but its equivalence must be proven.

**Both kmax and the width of `valid` come from `len(candidate_ids)`.**
A boolean question has only one semantic path (one leaf) yet two candidate ids ("false"/"true"),
and upstream marks both slots as valid, with the second slot filled in by `pad([0, z])`. Marking
by leaf count would leave one slot short, and the self-check would expose it at once as
valid_match=false.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def head_logits(model, leaves: list[torch.Tensor], qtypes: list[str],
                n_candidates: list[int] | None = None):
    """leaves: one (n_leaf_i, H) tensor per question; n_candidates: number of candidate ids per question.

    Returns (logits (n, kmax), valid (n, kmax)), isomorphic to the upstream forward return.
    """
    if not leaves:
        raise ValueError("leaves must not be empty")
    n_candidates = list(n_candidates) if n_candidates is not None else [t.shape[0] for t in leaves]
    device = leaves[0].device
    n = len(leaves)
    kmax = max(n_candidates)
    hidden = leaves[0].shape[-1]

    h = leaves[0].new_zeros((n, kmax, hidden))
    valid = torch.zeros((n, kmax), dtype=torch.bool, device=device)
    for i, t in enumerate(leaves):
        h[i, : t.shape[0]] = t
        valid[i, : n_candidates[i]] = True

    h = model.norm(h)
    z = model.scalar(h).squeeze(-1).float()

    choice = torch.tensor([i for i, t in enumerate(qtypes) if t == "choice"], device=device)
    if model.set_head == "attention" and len(choice):
        log_k = valid[choice].sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
        u = model.set_project(torch.cat([h[choice], log_k.to(h.dtype)], dim=-1))
        mixed, _ = model.set_attention(
            u, u, u, key_padding_mask=~valid[choice], need_weights=False
        )
        delta = model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        z = z.index_add(0, choice, delta)

    # boolean has only one semantic path: the two logits are built from [0, z].
    if kmax < 2:
        if any(t != "boolean" for t in qtypes):
            raise ValueError("kmax<2 can only be an all-boolean batch")
        return torch.stack([torch.stack([z[i, 0] * 0, z[i, 0]]) for i in range(n)]), valid
    rows = []
    for i, t in enumerate(qtypes):
        if t == "boolean":
            rows.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax - 2)))
        else:
            rows.append(z[i])
    return torch.stack(rows).masked_fill(~valid, -1e9), valid
