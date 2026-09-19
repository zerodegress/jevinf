"""NanoJev prefix-sharing engine.

Which model families this arrangement is defined for is data, not code: `arch.py` holds the
structural facts (how attention runs, where answers are read from, what stays resident between
segments) and `_check_arch` refuses anything whose facts do not fit the three stages below.

Three-stage forward, all reusing the same upstream backbone object:

  Stage A  state segment       once per state (once per whole batch after uniquing)
  Stage B  question segment    with the state's KV as past, once per question  ┐ pick one of
  Stage C  candidate suffix    candidates batched                              ┘ the two strategies

  strategy="two_stage": A → B → C   -- more forwards, fewest computed tokens
  strategy="fused"    : A → (question segment + suffix) one row per candidate -- half the
                        forwards, at the cost of repeating the question segment once per
                        candidate (+39% tokens)

Which one wins is decided by measurement.

Equivalence argument (verified with a probe: whole path in one forward vs two-stage forwards,
max|Δ| = 0.0):
under a causal decoder a token's hidden depends only on the tokens before it, so computing the
prefix once and then continuing with the suffix is mathematically equivalent to computing the
whole path in one shot. When past is supplied, **attention_mask must cover the full extent**
(previous segment + new segment); supplying only the new segment computes it wrong (measured
deviation 1.58).

KV residency granularity (decided by this prototype):
  KV/token = 2 × n_kv_heads(8) × head_dim(128) × 4B = 8 KiB (fp32, measured shape
  [B, 8, L, 128]). The 120 state segments of the dev split total about 130 MB -- entirely
  affordable, so **the number of forwards is the constraint**. Hence "keep state-level KV
  resident for the whole split": streaming state by state would squeeze the batch width down to
  one state's questions, 3 on average, which hurts batch efficiency.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

from transformers import DynamicCache

from .head import head_logits
from .plan import Plan, build_plan


@dataclass
class EngineStats:
    strategy: str = ""
    states: int = 0
    questions: int = 0
    paths: int = 0
    forwards: int = 0
    stage_a_forwards: int = 0
    stage_b_forwards: int = 0
    stage_c_forwards: int = 0
    computed_tokens: int = 0  # tokens this engine really pushed through a forward (padding excluded)
    padded_tokens: int = 0  # includes in-batch padding
    baseline_tokens: int = 0  # upstream per-path accounting: Σ path lengths
    state_kv_bytes: int = 0  # KV bytes of a single state segment (peak residency)
    peak_resident_bytes: int = 0
    wall_s: float = 0.0
    # per-stage timing (MPS is asynchronous; these numbers all carry synchronize, used for attribution only)
    t_stage_a: float = 0.0
    t_stage_b: float = 0.0
    t_stage_c: float = 0.0
    t_copy: float = 0.0
    t_head: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class PrefixShareEngine:
    def __init__(self, predictor, *, strategy: str = "fused", head_chunk: int = 128,
                 stage_a_batch: bool = True, stage_a_group: int = 8, max_rows: int = 64):
        if strategy not in {"two_stage", "fused", "fused_q", "fused_state", "per_question"}:
            raise ValueError("strategy must be two_stage / fused_state / fused_q / per_question")
        if strategy == "fused":
            strategy = "fused_state"  # default mode: batch all paths of the same state together
        self.p = predictor
        self.arch = self._check_arch(predictor)
        self.model = predictor.model
        self.tok = predictor.tokenizer
        self.device = predictor.device
        self.limit = predictor.limit
        self.torch = predictor._torch
        self.pad_id = predictor.tokenizer.pad_token_id
        # upstream DecisionModel.forward explicitly passes use_cache=False, so enabling the cache here does not affect the oracle.
        self.model.backbone.config.use_cache = True
        self.strategy = strategy
        self.head_chunk = head_chunk
        self.stage_a_batch = stage_a_batch
        self.stage_a_group = stage_a_group
        self.max_rows = max(1, max_rows)

    # ---------------------------------------------------------------- low level
    @staticmethod
    def _check_arch(predictor):
        """The three-stage arrangement below is only defined for a causal backbone with per-token KV.

        Which families satisfy that is recorded in `arch.py`; anything else is refused here rather
        than arranged anyway. A predictor that carries no tag was not loaded through
        `upstream.load_predictor`, so the engine cannot tell what it is holding.
        """
        spec = getattr(predictor, "arch", None)
        if spec is None:
            raise ValueError(
                "predictor carries no architecture tag; load it through upstream.load_predictor"
            )
        if spec.attention != "causal" or spec.cache != "kv":
            raise ValueError(
                f"architecture {spec.name!r} is not one this engine arranges yet: prefix sharing "
                f"needs a causal backbone with per-token KV (got attention={spec.attention!r}, "
                f"cache={spec.cache!r})"
            )
        return spec

    @staticmethod
    def cache_bytes(kv) -> int:
        total = 0
        for layer in getattr(kv, "layers", []):
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if t is not None:
                    total += t.numel() * t.element_size()
        return total

    def forward_rows(self, rows: list[list[int]], past=None):
        """One right-padded batched forward. Returns (hidden at the last real position of each row, new cache, batch width)."""
        torch = self.torch
        if not rows:
            raise ValueError("rows must not be empty")
        lengths = [len(r) for r in rows]
        if min(lengths) == 0:
            raise ValueError("empty row present")
        width = max(lengths)
        tokens = torch.full((len(rows), width), self.pad_id, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            tokens[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=self.device)
        new_mask = torch.arange(width, device=self.device)[None, :] < torch.tensor(
            lengths, device=self.device
        )[:, None]
        if past is None:
            attn = new_mask
        else:
            plen = past.get_seq_length()
            attn = torch.cat(
                [torch.ones((len(rows), plen), dtype=torch.bool, device=self.device), new_mask],
                dim=1,
            )
        out = self.model.backbone(
            input_ids=tokens, attention_mask=attn, past_key_values=past, use_cache=True
        )
        kv = out.past_key_values
        if kv is None or len(kv) == 0:
            raise RuntimeError("backbone did not return a KV cache: use_cache did not take effect")
        idx = torch.arange(len(rows), device=self.device)
        last = out.last_hidden_state[idx, torch.tensor(lengths, device=self.device) - 1]
        return last, kv, width

    @staticmethod
    def _repeat_cache(cache, n: int):
        """Broadcast single-row prefix KV into n rows (required when batching candidates: the number of cache rows must equal the number of input rows)."""
        if n == 1:
            return cache
        out = cache.batch_repeat_interleave(n)
        if out is None:  # in-place rewrite API shape
            out = cache
        batch = out.layers[0].keys.shape[0]
        if batch != n:
            raise AssertionError(f"cache batch size after broadcast {batch} != {n}")
        return out

    def _state_cache(self, seg: list[int], stats: EngineStats):
        """Stage A (per-state variant): one state segment → one KV."""
        _, kv, _ = self.forward_rows([seg])
        stats.stage_a_forwards += 1
        stats.forwards += 1
        nbytes = self.cache_bytes(kv)
        stats.state_kv_bytes = max(stats.state_kv_bytes, nbytes)
        stats.peak_resident_bytes = max(stats.peak_resident_bytes, nbytes)
        return kv

    def _state_caches_batched(self, segments: list[list[int]], stats: EngineStats):
        """Stage A (batched variant): forward the whole group of state segments at once, then pull each row out.

        Extraction **must go through the official `DynamicCache.update()`**: assigning
        `DynamicLayer.keys/values` directly gives the right shape, but `get_seq_length()`
        then reads 0, and the model derives `cache_position` from it -- RoPE positions are
        recomputed from 0 and it **computes wrong silently** (scripts/probe_cache_select.py
        measured a shape error; scripts/probe_cache_update.py after calling update() gives
        max|Δ| = 5e-06, which is fp32 reduction noise).

        The cost: the whole group's cache and the extracted copies are resident at the same
        time (8 states per group, about 480 MB), in exchange for cutting Stage A forwards
        from "once per state" to "once per group".
        """
        _, kv, _ = self.forward_rows(segments)
        stats.stage_a_forwards += 1
        stats.forwards += 1
        parent_bytes = self.cache_bytes(kv)
        out = []
        for i, seg in enumerate(segments):
            length = len(seg)
            hand = DynamicCache()
            for idx, layer in enumerate(kv.layers):
                hand.update(
                    layer.keys[i : i + 1, :, :length, :].clone(),
                    layer.values[i : i + 1, :, :length, :].clone(),
                    idx,
                )
            if hand.get_seq_length() != length:
                raise AssertionError(
                    f"cache length after extraction {hand.get_seq_length()} != {length}"
                )
            out.append(hand)
        one = self.cache_bytes(out[0]) if out else 0
        stats.state_kv_bytes = max(stats.state_kv_bytes, one)
        stats.peak_resident_bytes = max(
            stats.peak_resident_bytes, parent_bytes + one * len(out)
        )
        return out

    def _forward_rows_capped(self, rows: list[list[int]], base, stats: EngineStats):
        """Run the same prefix in batches split by `max_rows`, concatenating the hidden at the last position of each row.

        Why the cap is mandatory: the Jev API allows 255 candidates for a single question and
        256 paths per request, and those paths may all belong to the same state. One forward
        over 256 rows × ~200 tokens of intermediate tensors runs to hundreds of MB, which is
        unstable on a 16GB machine. The dev split has only 1-4 candidates per question, so it
        cannot exercise this boundary.
        """
        torch = self.torch
        outs = []
        for i in range(0, len(rows), self.max_rows):
            chunk_rows = rows[i : i + self.max_rows]
            past = None
            if base is not None:
                self._sync()
                _t = time.perf_counter()
                cloned = copy.deepcopy(base)
                self._sync()
                stats.t_copy += time.perf_counter() - _t
                _t = time.perf_counter()
                past = self._repeat_cache(cloned, len(chunk_rows))
                self._sync()
                stats.t_copy += time.perf_counter() - _t
            self._sync()
            _t = time.perf_counter()
            last, _, width = self.forward_rows(chunk_rows, past=past)
            self._sync()
            stats.t_stage_c += time.perf_counter() - _t
            stats.stage_c_forwards += 1
            stats.forwards += 1
            stats.padded_tokens += width * len(chunk_rows)
            outs.append(last)
        return torch.cat(outs, 0) if len(outs) > 1 else outs[0]

    def _head(self, leaves, qtypes, n_candidates):
        torch = self.torch
        outs, valids = [], []
        for i in range(0, len(leaves), self.head_chunk):
            lg, vd = head_logits(self.model, leaves[i : i + self.head_chunk],
                                 qtypes[i : i + self.head_chunk],
                                 n_candidates[i : i + self.head_chunk])
            outs.append(lg)
            valids.append(vd)
        if len(outs) == 1:
            return outs[0], valids[0]
        ks = max(t.shape[0] for t in leaves)
        lg = torch.zeros((len(leaves), ks), device=self.device)
        vd = torch.zeros((len(leaves), ks), dtype=torch.bool, device=self.device)
        off = 0
        for a, b in zip(outs, valids):
            lg[off : off + a.shape[0], : a.shape[1]] = a
            vd[off : off + b.shape[0], : b.shape[1]] = b
            off += a.shape[0]
        return lg, vd

    # ---------------------------------------------------------------- main path
    def _sync(self):
        if self.device.type == "mps":
            self.torch.mps.synchronize()

    def evaluate(self, payload, plan: Plan | None = None, temperature: float = 1.0):
        from .upstream import predict_module

        answer_from_probabilities = predict_module().answer_from_probabilities
        torch = self.torch
        t0 = time.perf_counter()
        plan = plan or build_plan(payload, self.tok, self.limit)
        stats = EngineStats(
            strategy=self.strategy,
            states=len(plan.state_ids),
            questions=len(plan.questions),
            paths=sum(q.n_paths for q in plan.questions),
            baseline_tokens=sum(sum(q.path_lengths) for q in plan.questions),
        )
        suffix_tokens = sum(sum(len(s) for s in q.suffix_ids) for q in plan.questions)
        seg_tokens = sum(len(s) for s in plan.state_segments)
        q_seg_total = sum(len(q.question_seg) for q in plan.questions)
        q_seg_repeated = sum(q.n_paths * len(q.question_seg) for q in plan.questions)
        if self.strategy == "per_question":
            stats.computed_tokens = 0  # accumulated per row inside the loop
        elif self.strategy in {"fused", "fused_state", "fused_q"}:
            # the question segment repeats per candidate and must be counted here -- otherwise it would be underestimated as "cheaper than two_stage".
            stats.computed_tokens = seg_tokens + suffix_tokens + q_seg_repeated
        else:
            stats.computed_tokens = seg_tokens + suffix_tokens + q_seg_total

        state_index = {sid: i for i, sid in enumerate(plan.state_ids)}
        seg_of = {sid: seg for sid, seg in zip(plan.state_ids, plan.state_segments)}
        # group by state (questions are already contiguous by state in the plan); with batched Stage A the whole group's KV is fetched at once.
        groups: list[tuple[str, list]] = []
        for q in plan.questions:
            if groups and groups[-1][0] == q.state_id:
                groups[-1][1].append(q)
            else:
                groups.append((q.state_id, [q]))

        leaves, qtypes = [], []
        # inference_mode throughout: weights require grad by default, and omitting it builds an autograd graph that keeps all 28 layers of activations alive.
        with torch.inference_mode():
            gi = 0
            while gi < len(groups):
                span = self.stage_a_group if self.stage_a_batch else 1
                batch = groups[gi : gi + span]
                gi += len(batch)
                if self.strategy == "per_question":
                    caches = [None] * len(batch)
                elif self.stage_a_batch:
                    self._sync()
                    _t = time.perf_counter()
                    caches = self._state_caches_batched(
                        [seg_of[sid] for sid, _ in batch], stats
                    )
                    self._sync()
                    stats.t_stage_a += time.perf_counter() - _t
                else:
                    self._sync()
                    _t = time.perf_counter()
                    caches = [self._state_cache(seg_of[sid], stats) for sid, _ in batch]
                    self._sync()
                    stats.t_stage_a += time.perf_counter() - _t

                for (sid, qs), base in zip(batch, caches):
                    if self.strategy == "fused_state":
                        # merge every candidate path of the same state into one batch (they share the same state KV),
                        # so forwards are once per state per sub-batch (per question would be 3x more).
                        rows, spans = [], []
                        for q in qs:
                            start = len(rows)
                            rows.extend(q.question_seg + s for s in q.suffix_ids)
                            spans.append((q, start, len(rows)))
                        last = self._forward_rows_capped(rows, base, stats)
                        for q, a, b in spans:
                            leaves.append(last[a:b])
                            qtypes.append(q.qtype)
                        continue
                    for q in qs:
                        if self.strategy == "per_question":
                            rows = [seg_of[sid] + q.question_seg + s for s in q.suffix_ids]
                            last = self._forward_rows_capped(rows, None, stats)
                            stats.computed_tokens += sum(len(r) for r in rows)
                        elif self.strategy == "two_stage":
                            self._sync()
                            _t = time.perf_counter()
                            _, kvq, _ = self.forward_rows(
                                [q.question_seg], past=copy.deepcopy(base)
                            )
                            self._sync()
                            stats.t_stage_b += time.perf_counter() - _t
                            stats.stage_b_forwards += 1
                            stats.forwards += 1
                            last = self._forward_rows_capped(
                                q.suffix_ids, self._repeat_cache(kvq, 1), stats
                            )
                        else:  # fused_q: the question segment and the candidate suffix form one row, batched per question
                            rows = [q.question_seg + s for s in q.suffix_ids]
                            last = self._forward_rows_capped(rows, base, stats)
                        leaves.append(last)
                        qtypes.append(q.qtype)
            self._sync()
            _t = time.perf_counter()
            logits, _valid = self._head(
                leaves, qtypes, [len(q.candidate_ids) for q in plan.questions]
            )
            self._sync()
            stats.t_head += time.perf_counter() - _t

        outputs = {sid: {"id": sid, "answers": {}} for sid in plan.state_ids}
        for i, q in enumerate(plan.questions):
            # k is len(candidate_ids): a boolean question has one semantic path (one leaf)
            # while carrying two candidate ids, and the head builds the two [0, z] logit slots.
            k = len(q.candidate_ids)
            scores = logits[i, :k].float()
            if not torch.isfinite(scores).all():
                raise ValueError("model produced non-finite logits")
            probabilities = (scores / temperature).softmax(-1).cpu().tolist()
            ex_like = {"candidate_ids": q.candidate_ids, "type": q.qtype}
            outputs[q.state_id]["answers"][q.qid] = answer_from_probabilities(ex_like, probabilities)

        stats.wall_s = time.perf_counter() - t0
        return {
            "schema_version": "jevinf-v0",
            "execution": {
                "engine": "jevinf",
                "architecture": self.arch.name,
                "strategy": self.strategy,
                "device": str(self.device),
                "parameter_storage": "float32",
                "prefix_sharing": True,
                "autoregressive_decode_steps": 0,
                **stats.as_dict(),
            },
            "states": list(outputs.values()),
        }
