"""Comparison and timing: upstream per-path baseline vs jevinf prefix sharing.

Comparison protocol: pick the answer as the argmax over candidate_ids for each
question and compare agreement rates; also report TV / KL.
Timing protocol: end-to-end wall-clock on the dev split, timed after device
synchronization, with one warm-up run and then the median of the repeats.
"""
from __future__ import annotations

import copy
import json
import math
import statistics
import time

from . import upstream
from .device import empty_cache, synchronize
from .engine import PrefixShareEngine
from .plan import build_plan


def _answers_by_q(result: dict) -> dict[str, dict]:
    out = {}
    for st in result["states"]:
        for qid, ans in st["answers"].items():
            out[f"{st['id']}:{qid}"] = ans
    return out


def _chunk_payload(payload: dict, size: int | None):
    if not size:
        return [payload]
    states = payload["states"]
    return [{"states": states[i : i + size]} for i in range(0, len(states), size)]


_MAX_KEYS = {"state_kv_bytes", "peak_resident_bytes"}


def _merge_results(results: list[dict]) -> dict:
    """Merge the per-chunk results into one. Integer execution fields are summed,
    but "peak"-type fields take the max (summing would fabricate a single-state KV
    of 485MB out of 15 chunks); strings take the first value, floats are dropped
    (total elapsed time comes from the timer)."""
    states, ex = [], {}
    for r in results:
        states.extend(r["states"])
        for k, v in r.get("execution", {}).items():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                # Floats are summed too (the per-stage timings rely on this); the end-to-end value comes from the timer.
                ex[k] = max(ex.get(k, 0), v) if k in _MAX_KEYS else ex.get(k, 0) + v
            elif isinstance(v, str):
                ex.setdefault(k, v)
    return {"states": states, "execution": ex}


def compare(a: dict, b: dict, reference: dict | None = None) -> dict:
    """a = reference (upstream), b = candidate under test (jevinf). Compare the distributions over the same candidate_ids order."""
    A, B = _answers_by_q(a), _answers_by_q(b)
    if set(A) != set(B):
        raise AssertionError("question sets differ between the two sides")
    n = len(A)
    agree = 0
    tv, kl = [], []
    by_type: dict[str, list[int]] = {}
    flips = []
    ref_hit_a = ref_hit_b = ref_n = 0
    for key in sorted(A):
        qa, qb = A[key], B[key]
        ids = list(qa["probabilities"])
        if list(qb["probabilities"]) != ids:
            raise AssertionError(f"{key}: candidate order mismatch")
        pa = [qa["probabilities"][k] for k in ids]
        pb = [qb["probabilities"][k] for k in ids]
        ia = max(range(len(pa)), key=pa.__getitem__)
        ib = max(range(len(pb)), key=pb.__getitem__)
        same = ia == ib
        agree += int(same)
        by_type.setdefault(qa["type"], [0, 0])
        by_type[qa["type"]][0] += int(same)
        by_type[qa["type"]][1] += 1
        if not same:
            flips.append({"key": key, "type": qa["type"],
                          "upstream": ids[ia], "jevinf": ids[ib],
                          "p_upstream": round(pa[ia], 6), "p_jevinf": round(pb[ib], 6),
                          "margin_upstream": round(pa[ia] - sorted(pa)[-2], 6)})
        tv.append(0.5 * sum(abs(x - y) for x, y in zip(pa, pb)))
        kl.append(sum(x * math.log(x / y) for x, y in zip(pa, pb) if x > 0 and y > 0))
        if reference and key in reference:
            ref = reference[key]
            ra = [ref[k] for k in ids]
            ref_n += 1
            ref_hit_a += int(ia == max(range(len(ra)), key=ra.__getitem__))
            ref_hit_b += int(ib == max(range(len(ra)), key=ra.__getitem__))
    out = {
        "questions": n,
        "argmax_agreement": agree / n if n else None,
        "argmax_agreement_by_type": {
            t: {"hit": h, "n": m, "rate": h / m} for t, (h, m) in sorted(by_type.items())
        },
        "tv": {"median": statistics.median(tv), "mean": statistics.fmean(tv), "max": max(tv)},
        "kl_mean": statistics.fmean(kl),
        "flips": flips,
    }
    if ref_n:
        out["teacher_argmax_hit"] = {
            "upstream": ref_hit_a / ref_n, "jevinf": ref_hit_b / ref_n, "n": ref_n
        }
    return out


def timed(fn, repeats: int, device):
    """Return (first-run time, median of repeats, list of all run times, result). With repeats=0 it runs only once."""
    t0 = time.perf_counter()
    r0 = fn()
    synchronize(device)
    first = time.perf_counter() - t0
    times, result = [], r0
    for _ in range(repeats):
        empty_cache(device)
        t0 = time.perf_counter()
        result = fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return first, (statistics.median(times) if times else first), times, result


def head_selfcheck(predictor, payload, limit=None):
    """Isolated check of the decision head: feed upstream-computed leaves into my head and compare element-wise with the upstream logits."""
    import torch

    from .head import head_logits

    model, tok = predictor.model, predictor.tokenizer
    limit = limit or predictor.limit
    examples = upstream.prepare_examples(payload, tok, limit)
    with torch.inference_mode():
        paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
        lengths = torch.tensor([len(p) for p in paths], device=predictor.device)
        width = int(lengths.max())
        tokens = torch.full((len(paths), width), tok.pad_token_id, dtype=torch.long,
                            device=predictor.device)
        for i, ids in enumerate(paths):
            tokens[i, : len(ids)] = torch.tensor(ids, device=predictor.device)
        attn = torch.arange(width, device=predictor.device)[None, :] < lengths[:, None]
        hidden = model.backbone(input_ids=tokens, attention_mask=attn,
                                use_cache=False).last_hidden_state
        leaves = hidden[torch.arange(len(paths), device=predictor.device), lengths - 1]
        del hidden

        offset, mine_leaves, types, n_cands = 0, [], [], []
        for ex in examples:
            n = len(ex["leaf_tokens"])
            mine_leaves.append(leaves[offset : offset + n])
            types.append(ex["type"])
            n_cands.append(len(ex["candidate_ids"]))
            offset += n

        # The full upstream chain (including its own backbone forward), compared against my head on the same leaves.
        up_logits, up_valid = model(examples, tok.pad_token_id)
        my_logits, my_valid = head_logits(model, mine_leaves, types, n_cands)
    ok_valid = bool(torch.equal(up_valid, my_valid))
    same_shape = tuple(up_logits.shape) == tuple(my_logits.shape)
    diff = float((up_logits - my_logits).abs().max()) if same_shape else None
    same_argmax = None
    if same_shape:
        k_mask = up_valid
        a = up_logits.masked_fill(~k_mask, -1e9).argmax(-1)
        b = my_logits.masked_fill(~k_mask, -1e9).argmax(-1)
        same_argmax = bool(torch.equal(a, b))
    return {
        "examples": len(examples),
        "shape_match": same_shape,
        "valid_match": ok_valid,
        "max_abs_logit_diff": diff,
        "argmax_match": same_argmax,
        "upstream_shape": tuple(up_logits.shape),
        "mine_shape": tuple(my_logits.shape),
    }


def run_bench(predictor, payload, *, repeats=3, strategies=("fused", "two_stage"),
              batch_questions=0, reference=None, head_chunk=128, chunk_states=32,
              stage_a_group=8):
    device = predictor.device
    report = {"baseline": {}, "engine": {}, "compare": {}, "selfcheck": {},
              "chunk_states": chunk_states}

    report["selfcheck"] = None

    chunks = _chunk_payload(payload, chunk_states)

    # The self-check only needs a small slice: a single forward over all 898 paths would blow past 16GB in intermediate tensors (measured: the MLP alone needs 5GB).
    report["selfcheck"] = head_selfcheck(predictor, chunks[0])
    empty_cache(device)

    def run_upstream():
        outs = []
        for c in chunks:
            outs.append(predictor.predict(c, batch_questions=batch_questions, temperature=1.0))
            empty_cache(device)
        return _merge_results(outs)

    first, med, times, res_up = timed(run_upstream, repeats, device)
    report["baseline"] = {
        "label": f"upstream(batch_questions={batch_questions or 'all'}, chunk_states={chunk_states})",
        "first_s": first, "median_s": med, "runs_s": times,
        "execution": res_up["execution"],
    }

    for strat in strategies:
        empty_cache(device)
        eng = PrefixShareEngine(predictor, strategy=strat, head_chunk=head_chunk,
                                stage_a_batch=stage_a_group > 1,
                                stage_a_group=max(1, stage_a_group))
        plans = [build_plan(c, predictor.tokenizer, predictor.limit) for c in chunks]

        def run_engine():
            outs = []
            for c, p in zip(chunks, plans):
                outs.append(eng.evaluate(c, plan=p))
                empty_cache(device)
            return _merge_results(outs)

        first, med, times, res_eng = timed(run_engine, repeats, device)
        report["engine"][strat] = {
            "first_s": first, "median_s": med, "runs_s": times,
            "execution": res_eng["execution"],
            "speedup_vs_baseline": report["baseline"]["median_s"] / med if med else None,
        }
        report["compare"][strat] = compare(res_up, res_eng, reference=reference)
    return report


def summarize(report: dict) -> str:
    b = report["baseline"]
    lines = []
    sc = report["selfcheck"]
    lines.append(
        f"Head self-check: shape_match={sc['shape_match']} valid_match={sc['valid_match']} "
        f"argmax_match={sc['argmax_match']} max|Δlogit|={sc['max_abs_logit_diff']}"
    )
    ex = b["execution"]
    lines.append(
        f"baseline {b['label']}: median {b['median_s']:.3f}s (first {b['first_s']:.3f}s) "
        f"forwards={ex['forward_passes']} paths={ex['candidate_paths']} "
        f"tokens={b['median_s'] and ''}"
    )
    for strat, e in report["engine"].items():
        x = e["execution"]
        c = report["compare"][strat]
        lines.append(
            f"[{strat}] median {e['median_s']:.3f}s → {e['speedup_vs_baseline']:.2f}× | "
            f"forward={x['forwards']} (A{x['stage_a_forwards']}/C{x['stage_c_forwards']}"
            f"/B{x['stage_b_forwards']}) | tokens {x['computed_tokens']:,} vs baseline "
            f"{x['baseline_tokens']:,} | state KV {x['state_kv_bytes']/1e6:.1f}MB | "
            f"agreement {c['argmax_agreement']*100:.2f}% | TV median {c['tv']['median']:.2e} | "
            f"KL mean {c['kl_mean']:.2e}"
        )
        phases = sum(x.get(k, 0.0) for k in ("t_stage_a", "t_stage_b", "t_stage_c",
                                             "t_copy", "t_head"))
        wall = x.get("wall_s", 0.0)
        resid = wall - phases
        flag = "  ⚠ attribution gap" if wall and abs(resid) > 0.05 * wall else ""
        lines.append(
            f"         per-stage (with sync, attribution only): stageA {x.get('t_stage_a', 0):.2f}s | "
            f"stageB {x.get('t_stage_b', 0):.2f}s | stageC {x.get('t_stage_c', 0):.2f}s | "
            f"deepcopy+broadcast {x.get('t_copy', 0):.2f}s | head {x.get('t_head', 0):.3f}s | "
            f"wall {wall:.2f}s | unattributed {resid:+.2f}s{flag}"
        )
        if "teacher_argmax_hit" in c:
            t = c["teacher_argmax_hit"]
            lines.append(
                f"         argmax hits vs teacher: upstream {t['upstream']*100:.2f}% vs "
                f"jevinf {t['jevinf']*100:.2f}% (n={t['n']})"
            )
        for t, s in c["argmax_agreement_by_type"].items():
            lines.append(f"         {t}: {s['hit']}/{s['n']} = {s['rate']*100:.2f}%")
    return "\n".join(lines)
