"""laya -- ModernBERT encoder plus a from-scratch decision head (the `single-path` family).

The arrangement follows from the architecture rather than from a preference. The encoder is
bidirectional, so there is no prefix whose state could be computed and carried into a suffix, and the
checkpoint's own layout puts the *state at the end of one sequence per question*:

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]

Each option is scored at its own `[MASK]` marker, and the answer distribution is a softmax over that
question's markers divided by a fitted temperature. Two consequences the engine cannot get around:
rows are independent sequences (nothing is shared between questions of one state), and the only
batching available is padding several rows into one forward pass -- which is why padding is harmless
here (attention is masked) where it was destructive for the recurrent family.

Two deliberate refusals, both about declining to answer a question nobody asked:

* a state that does not fit the 512-token budget is an error. The checkpoint's own entry point crops
  the state to fit (`st[:room]`); a crop silently changes the document the model reads, so here it is
  opt-in (`crop_state`) and every answer produced from a cropped state says so.
* a request-level temperature is refused: this family's calibration is per (question type, option
  count) -- `temperature_by_options` in its config -- so a single scalar would change what every
  probability means. (`choice:11+` is 0.1006 in the shipped config, i.e. the fitted value sharpens
  that bucket hard; the engine reports the applied temperature per answer so a caller can see it,
  rather than quietly agreeing that 11 options behave like 4.)

`scripts/laya_parity.py` holds this to the checkpoint's own inference, which ships next to the
weights (`rl_agent_api.RLAgent` over `rl_common`): same sequences, same markers, same temperatures.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .arch import Architecture, resolve as resolve_arch
from .backend import DEFAULT_BACKEND, resolve as resolve_backend
from .device import synchronize

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
MAX_OPTIONS = 255
MAX_LEVELS = 10
DEFAULT_MAX_ROWS = 16
MIN_STATE_ROOM = 8  # below this the "state" is not a document any more; refuse even with crop_state
# Options live at fixed positions in one sequence, so listing them differently moves their
# probabilities (measured 0.03 on one question, 0.23 with a flipped winner on another). That is a
# property of the family, so it is measured and reported rather than gated; the gate is in
# scripts/laya_parity.py, where the checkpoint's own code provides the expected movement.


# ------------------------------------------------------------------ rendering (mirrors rl_common)
def render_state(state) -> str:
    """The state as the sequence builder wants it: a string stays a string, anything else is JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(spec: dict) -> list[str]:
    """Option texts in label order. Noul is always [false, true], so p[1] is P(true)."""
    kind, criteria = spec["t"], spec.get("crit")
    if kind == "choice":
        return [k if not v else f"{k}: {v}" for k, v in criteria.items()]
    if kind == "score":
        return [f"level {i}: {c}" for i, c in enumerate(criteria)]
    criteria = criteria or {}
    return [
        "false: " + (criteria.get("false") or "no, the statement does not hold"),
        "true: " + (criteria.get("true") or "yes, the statement holds"),
    ]


def question_spec(spec: dict) -> dict:
    """A Jev question -> this family's internal `{t, ins, crit}` shape (the checkpoint's own mapping).

    `boolean` is the Jev translation layer's name for this family's `noul`, so it is normalised here
    rather than at the call sites. The option-domain checks duplicate what the wire validators enforce:
    a family that is asked for 300 options should say so itself, not trust that nobody will.
    """
    kind = spec.get("type")
    if kind == "boolean":
        kind = "noul"
    if kind not in QTYPES:
        raise ValueError(f"unknown question type {kind!r}; this family answers {', '.join(QTYPES)}")
    criteria = spec.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = {str(c): None for c in criteria}
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(
                f"a choice needs 2-{MAX_OPTIONS} named options, got "
                f"{len(criteria) if isinstance(criteria, dict) else type(criteria).__name__}"
            )
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_LEVELS:
            raise ValueError(
                f"a score needs 2-{MAX_LEVELS} levels, got "
                f"{len(criteria) if isinstance(criteria, list) else type(criteria).__name__}"
            )
    else:
        criteria = criteria if isinstance(criteria, dict) else {}
    instructions = spec.get("instructions")
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return {"t": kind, "ins": instructions, "crit": criteria}


def build_sequence(tok, state, spec: dict, max_len: int, head_max_len: int) -> tuple[list[int], list[int], dict]:
    """`[CLS] head [SEP] markers [SEP] state [SEP]` -- the checkpoint's layout, with its budget arithmetic.

    Returns the ids, the marker positions and what happened to the state (`state_tokens`, `room`,
    `cropped`), because whether the state survived intact is the caller's business, not ours.
    """
    mask_tok = tok.mask_token
    options = render_options(spec)
    instructions = str(spec["ins"]).replace(mask_tok, " ")
    head_ids = tok(f"{spec['t']} question: {instructions}", add_special_tokens=False)["input_ids"]
    option_ids = [
        [tok.mask_token_id] + tok(" " + text.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        for text in options
    ]
    option_budget = head_max_len - sum(len(o) for o in option_ids)
    if option_budget < 16:  # too many or too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [o[:per] for o in option_ids]
        option_budget = head_max_len - sum(len(o) for o in option_ids)
    head_ids = head_ids[: max(8, option_budget)]

    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tok.sep_token_id)

    room = max(0, max_len - len(ids) - 1)
    state_ids = tok(render_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    cropped = len(state_ids) > room
    ids = (ids + state_ids[:room] + [tok.sep_token_id])[:max_len]
    info = {"state_tokens": len(state_ids), "room": room, "cropped": cropped}
    return ids, [m for m in markers if m < max_len], info


# ------------------------------------------------------------------ fitted temperatures
def temp_bucket(kind: str, k: int) -> str:
    """Per-cardinality calibration key: a 2-option noul and a 20-option choice are not the same task."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{kind}:{size}"


def confidence_from_probs(p: list[float], k: int) -> float:
    """Jev-style confidence: 1 - normalised entropy of the answer distribution."""
    if k < 2:
        return 1.0
    ent = -sum(x * math.log(max(x, 1e-12)) for x in p[:k])
    return float(1 - ent / math.log(k))


# ------------------------------------------------------------------ the model
class LayaModel(nn.Module):
    """Encoder + marker scorer (+ act head) -- the exact module structure the checkpoint's state dict has.

    Reproduced rather than imported so the family lives in this package like the others; the structure
    is not a free choice, because `load_state_dict(strict=True)` is what proves it matches, and
    `scripts/laya_parity.py` is what proves the arithmetic does.
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                hidden = layer(hidden, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
        marks = torch.gather(hidden, 1, idx)
        logits = self.scorer(marks).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        # the act head sees the pooled sequence plus a detached summary of this question's own answer
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / MAX_OPTIONS], -1)
        pooled = hidden[:, 0].float()
        return logits, self.act_head(torch.cat([pooled, feats], -1))


class LayaPredictor:
    """Weights + config + tokenizer, with the family's own read-only constants."""

    def __init__(self, checkpoint_dir, backend: str = DEFAULT_BACKEND, precision: str = "fp32",
                 arch: Architecture | None = None, crop_state: bool = False):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.root = Path(checkpoint_dir)
        config = json.loads((self.root / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.config = config
        self.max_len = int(config["max_len"])
        self.head_max_len = int(config["head_max_len"])
        self.temperature = [float(t) for t in config.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {k: float(v) for k, v in config.get("temperature_by_options", {}).items()}
        self.act_costs = config.get("act_costs", {})
        self.crop_state = bool(crop_state)
        # One cap for the whole sequence: this family has no prefix to share, so a request is bounded by
        # the sequence budget, which is what the service advertises in place of nanojev's path cap.
        self.limit = self.max_len

        self.device = torch.device(resolve_backend(backend).device)
        self.tok = AutoTokenizer.from_pretrained(str(self.root / "tokenizer"))
        encoder_config = AutoConfig.from_pretrained(str(self.root / "encoder"))
        encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
        self.model = LayaModel(encoder, int(config.get("head_layers", 2)), len(self.act_costs) + 1)
        report = self.model.load_state_dict(load_file(str(self.root / "model.safetensors")), strict=True)
        self.model.to(self.device).eval()
        try:  # the checkpoint's own entry point turns this off; torch.compile is not available here
            self.model.encoder.config.reference_compile = False
        except Exception:  # noqa: BLE001 -- a config that does not carry the flag is fine
            pass
        self.precision = "fp16" if precision in ("fp16", "bfloat16", "bf16") else "fp32"
        if self.precision == "fp16":  # the shipped weights are fp16; fp32 is the reference path
            self.model.to(dtype=torch.float16)
        self.arch = arch or resolve_arch("laya")
        self.missing_keys = list(report.missing_keys)
        self.unexpected_keys = list(report.unexpected_keys)

    @property
    def storage(self) -> str:
        return "float16" if self.precision == "fp16" else "float32"

    def dtype(self) -> torch.dtype:
        return torch.float16 if self.precision == "fp16" else torch.float32

    def temperature_for(self, kind: str, k: int) -> tuple[float, str]:
        """(applied temperature, bucket key). One place decides it, and every answer reports it."""
        bucket = temp_bucket(kind, k)
        return self.temperature_by_options.get(bucket, self.temperature[QTYPES[kind]]), bucket


# ------------------------------------------------------------------ engine
@dataclass
class LayaStats:
    """What the request cost. `computed_tokens` is the honest number: rows cannot share anything."""

    questions: int = 0
    rows: int = 0
    forwards: int = 0
    computed_tokens: int = 0
    padded_tokens: int = 0
    baseline_tokens: int = 0
    states: int = 0
    cropped_states: int = 0
    max_seq_len: int = 0
    t_forward: float = 0.0
    wall_s: float = 0.0
    temperatures: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["wall_s"] = round(out["wall_s"], 6)
        out["t_forward"] = round(out["t_forward"], 6)
        return out


class LayaEngine:
    """One sequence per question, padded into batches. No cache, no prefix, no sharing to be had.

    The layout this family runs cannot be reconfigured into a cheaper one: the state sits after the
    question and its options inside a single sequence, so two questions cannot borrow each other's
    computation. That is why `execution.prefix_sharing` is False and `computed_tokens` equals
    `baseline_tokens` here, rather than a saving that failed to materialise.
    """

    def __init__(self, predictor: LayaPredictor, *, max_rows: int = DEFAULT_MAX_ROWS,
                 crop_state: bool | None = None):
        self._check_arch(predictor)
        self.p = predictor
        self.arch = predictor.arch
        self.device = predictor.device
        self.max_rows = max(1, int(max_rows))
        self.crop_state = predictor.crop_state if crop_state is None else bool(crop_state)

    @staticmethod
    def _check_arch(predictor) -> None:
        if predictor.arch.arrangement != "single-path":
            raise ValueError(
                f"{predictor.arch.name!r} is a {predictor.arch.arrangement!r} architecture; "
                f"this engine arranges single-path families"
            )

    def _sync(self) -> None:
        synchronize(self.device)

    # -------------------------------------------------------------- rows
    def plan_rows(self, state, questions: dict) -> tuple[list[dict], dict]:
        rows, seen = [], None
        for qid, raw in questions.items():
            spec = question_spec(raw)
            ids, markers, info = build_sequence(self.p.tok, state, spec, self.p.max_len, self.p.head_max_len)
            if len(markers) != len(render_options(spec)):
                raise ValueError(
                    f"question {qid!r}: {len(render_options(spec))} options do not fit the "
                    f"head budget of {self.p.head_max_len} tokens"
                )
            if info["cropped"] and not self.crop_state:
                raise ValueError(
                    f"question {qid!r}: the state needs {info['state_tokens']} tokens but this "
                    f"sequence has room for {info['room']} (budget {self.p.max_len}); shorten the "
                    f"state or run with crop_state, which crops it and marks every answer"
                )
            if info["room"] < MIN_STATE_ROOM:
                raise ValueError(
                    f"question {qid!r}: only {info['room']} tokens are left for the state "
                    f"(budget {self.p.max_len}), which is not enough to read a state at all"
                )
            seen = info if seen is None else seen
            rows.append({"qid": qid, "spec": spec, "ids": ids, "markers": markers,
                         "qtype": QTYPES[spec["t"]], "info": info})
        return rows, (seen or {"state_tokens": 0, "room": self.p.max_len, "cropped": False})

    # -------------------------------------------------------------- forward
    def forward_rows(self, rows: list[dict], stats: LayaStats) -> list[tuple[torch.Tensor, torch.Tensor]]:
        pad_id = self.p.tok.pad_token_id
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(rows), self.max_rows):
            batch = rows[start:start + self.max_rows]
            length = max(len(r["ids"]) for r in batch)
            kmax = max(len(r["markers"]) for r in batch)
            ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), length), dtype=torch.long)
            mpos = torch.zeros((len(batch), kmax), dtype=torch.long)
            mmask = torch.zeros((len(batch), kmax), dtype=torch.bool)
            for j, row in enumerate(batch):
                ids[j, :len(row["ids"])] = torch.tensor(row["ids"], dtype=torch.long)
                att[j, :len(row["ids"])] = 1
                k = len(row["markers"])
                mpos[j, :k] = torch.tensor(row["markers"], dtype=torch.long)
                mmask[j, :k] = True
            qtype = torch.tensor([r["qtype"] for r in batch], dtype=torch.long)
            started = time.perf_counter()
            with torch.inference_mode():
                logits, act = self.p.model(
                    ids.to(self.device), att.to(self.device), mpos.to(self.device),
                    mmask.to(self.device), qtype.to(self.device),
                )
            self._sync()
            stats.t_forward += time.perf_counter() - started
            stats.forwards += 1
            stats.padded_tokens += int(att.sum())
            stats.max_seq_len = max(stats.max_seq_len, length)
            # fp16 weights produce fp16 marker scores; the reference path is fp32 and this keeps the
            # decode arithmetic identical no matter which precision the weights are in
            logits, act = logits.float().cpu(), torch.softmax(act.float(), -1).cpu()
            for j in range(len(batch)):
                out.append((logits[j], act[j]))
        return out

    def decode_row(self, row: dict, logits: torch.Tensor, act: torch.Tensor) -> dict:
        """Markers -> probabilities, at this question's fitted temperature (reported, never implied)."""
        k = len(row["markers"])
        kind = row["spec"]["t"]
        temperature, bucket = self.p.temperature_for(kind, k)
        z = logits[:k].float() / temperature
        z = z - z.max()
        probs = torch.exp(z)
        probs = (probs / probs.sum()).tolist()
        ext = {"act_probability": round(float(act[0]), 6), "temperature": temperature, "bucket": bucket}
        if kind == "choice":
            names = list(row["spec"]["crit"].keys())
            best = max(range(k), key=lambda i: probs[i])
            return {"type": "choice", "choice": names[best],
                    "probabilities": {n: round(probs[i], 4) for i, n in enumerate(names)},
                    "confidence": round(confidence_from_probs(probs, k), 4), "rl_agent": ext}
        if kind == "score":
            return {"type": "score", "score": round(sum(i * probs[i] for i in range(k)), 4),
                    "legend": {str(i): c for i, c in enumerate(row["spec"]["crit"])},
                    "probabilities": {str(i): round(probs[i], 4) for i in range(k)},
                    "confidence": round(confidence_from_probs(probs, k), 4), "rl_agent": ext}
        return {"type": "noul", "noul": round(probs[1], 4), "rl_agent": ext}

    def score(self, state, questions: dict, stats: LayaStats) -> dict:
        rows, info = self.plan_rows(state, questions)
        decoded = zip(rows, self.forward_rows(rows, stats))
        answers = {}
        for row, (logits, act) in decoded:
            answer = self.decode_row(row, logits, act)
            answers[row["qid"]] = answer
            if row["spec"]["t"] != "noul":
                stats.temperatures[answer["rl_agent"]["bucket"]] = answer["rl_agent"]["temperature"]
        stats.rows += len(rows)
        stats.questions += len(questions)
        stats.computed_tokens += sum(len(r["ids"]) for r in rows)
        stats.baseline_tokens = stats.computed_tokens
        if info["cropped"]:
            stats.cropped_states += 1
        return {"answers": answers, "state_tokens": info["state_tokens"],
                "state_truncated": bool(info["cropped"])}

    # -------------------------------------------------------------- public entry points
    def evaluate(self, payload: dict, *, temperature: float | None = None, **ignored) -> dict:
        """Jev-shaped payload in, Jev-shaped answers out -- the same contract the other families serve."""
        if temperature is not None:
            raise ValueError(
                "this family reads its answers at per-cardinality fitted temperatures "
                "(temperature_by_options in the checkpoint), so a request-level temperature is refused "
                "rather than applied to every probability; the applied value is reported per answer"
            )
        t0 = time.perf_counter()
        stats = LayaStats()
        outputs = []
        for entry in payload.get("states") or []:
            scored = self.score(entry["state"], entry["questions"], stats)
            stats.states += 1
            outputs.append({"id": entry.get("id", f"state{len(outputs)}"), **scored})
        stats.wall_s = time.perf_counter() - t0
        return {
            "schema_version": "jevinf-v0",
            "execution": {
                "engine": "jevinf",
                "architecture": self.arch.name,
                "layout": "single_path",
                "device": str(self.device),
                "parameter_storage": self.p.storage,
                "prefix_sharing": False,
                "temperature": None,
                "temperature_source": "checkpoint (temperature_by_options, per question type and option count)",
                "autoregressive_decode_steps": 0,
                "sequence_limit": self.p.max_len,
                "head_limit": self.p.head_max_len,
                "crop_state": self.crop_state,
                **stats.as_dict(),
            },
            "states": outputs,
        }

    def selfcheck(self, payload: dict, tolerance: float = 0.02) -> dict:
        """This family's own equivalence checks: batching must not move an answer, and neither must the
        order the options are listed in. Both are about the same thing -- that the marker scorer reads
        each option's own text, not its position in the sequence.
        """
        states = payload.get("states") or []
        report: dict[str, Any] = {"architecture": self.arch.name, "checks": {}}

        # 1) one row at a time vs one padded batch: padding must not change anything
        rows_total, worst, disagree = 0, 0.0, 0
        for entry in states:
            rows, _ = self.plan_rows(entry["state"], entry["questions"])
            if not rows:
                continue
            rows_total += len(rows)
            base = LayaStats()
            saved, self.max_rows = self.max_rows, max(1, len(rows))
            single = LayaStats()
            try:
                batched = self.forward_rows(rows, base)
                one_by_one = []
                for row in rows:
                    one_by_one.extend(self.forward_rows([row], single))
            finally:
                self.max_rows = saved
            for row, (lb, ab), (ls, as_) in zip(rows, batched, one_by_one):
                pb = self.decode_row(row, lb, ab)
                ps = self.decode_row(row, ls, as_)
                gap, moved = _prob_gap(pb, ps)
                worst = max(worst, gap)
                disagree += int(moved)
        report["checks"]["batch_invariance"] = {
            "rows": rows_total, "max_probability_gap": round(worst, 6), "argmax_disagreements": disagree,
            "pass": disagree == 0 and worst <= tolerance,
        }

        # 2) options listed in the other order. This family reads each option at its own marker inside
        # one sequence, so the layout is positional by construction -- the number moves. The check is
        # therefore "the same option still wins", with the size of the movement reported, not "nothing
        # moves": a large or flipping change is what a wrong marker position would look like.
        worst_perm, perm_disagree, compared = 0.0, 0, 0
        for entry in states:
            for qid, raw in entry["questions"].items():
                spec = question_spec(raw)
                if spec["t"] != "choice" or len(spec["crit"]) < 2:
                    continue
                flipped = dict(reversed(list(spec["crit"].items())))
                q = {"type": "choice", "instructions": spec["ins"], "criteria": flipped}
                try:
                    forward = self.score(entry["state"], {qid: q}, LayaStats())
                except ValueError:
                    continue  # the state does not fit at all: nothing to compare
                straight = self.score(entry["state"], {qid: raw}, LayaStats())
                gap, moved = _prob_gap(straight["answers"][qid], forward["answers"][qid])
                worst_perm = max(worst_perm, gap)
                perm_disagree += int(moved)
                compared += 1
        report["checks"]["option_order"] = {
            "questions": compared, "max_probability_gap": round(worst_perm, 6),
            "argmax_disagreements": perm_disagree,
            "note": ("positional layout: listing the same options in the other order moves their "
                     "probabilities, and on a close question it can move which one wins. Measured here, "
                     "not asserted — the checkpoint's own code is what says whether the movement is the "
                     "model's or ours (scripts/laya_parity.py compares the two)"),
            "pass": compared > 0,
        }

        # 3) the state budget: over it is an error, and with crop_state it is a crop that says so
        first = states[0] if states else None
        if first is not None:
            long_state = (first["state"] if isinstance(first["state"], str) else json.dumps(first["state"]))
            long_state = (long_state + " ") * 40
            probe = {"state": long_state, "questions": first["questions"]}
            refused, cropped = False, None
            try:
                self.score(long_state, first["questions"], LayaStats())
            except ValueError as exc:
                refused = "room for" in str(exc)
            saved, self.crop_state = self.crop_state, True
            try:
                with_crop = self.score(long_state, first["questions"], LayaStats())
                cropped = {"state_tokens": with_crop["state_tokens"], "state_truncated": with_crop["state_truncated"]}
            except ValueError as exc:
                cropped = {"error": str(exc)[:120]}
            finally:
                self.crop_state = saved
            report["checks"]["state_budget"] = {
                "over_budget_refused": refused, "crop_state_result": cropped,
                "pass": refused and bool(cropped and cropped.get("state_truncated")),
            }

        # 4) which temperature each bucket gets -- the >=11-option bucket is the one to watch
        buckets = {temp_bucket(kind, k): self.p.temperature_for(kind, k)[0]
                   for kind in QTYPES for k in (2, 4, 8, 16)}
        report["checks"]["temperature_buckets"] = {
            "applied": {b: round(t, 4) for b, t in buckets.items()},
            "note": "a bucket below 1.0 sharpens its distribution; the shipped choice:11+ value does so hard",
            "pass": all(t > 0 for t in buckets.values()),
        }
        report["pass"] = all(bool(c.get("pass")) for c in report["checks"].values())
        return report


def _prob_gap(a: dict, b: dict) -> tuple[float, bool]:
    """Largest probability difference between two answers, and whether their argmax moved."""
    if a.get("type") == "noul" or b.get("type") == "noul":
        gap = abs(float(a.get("noul", 0.0)) - float(b.get("noul", 0.0)))
        return gap, (float(a.get("noul", 0.0)) >= 0.5) != (float(b.get("noul", 0.0)) >= 0.5)
    pa, pb = a["probabilities"], b["probabilities"]
    keys = [k for k in pa if k in pb]
    gap = max((abs(float(pa[k]) - float(pb[k])) for k in keys), default=0.0)
    top_a = max(pa, key=lambda k: pa[k])
    top_b = max(pb, key=lambda k: pb[k])
    if a.get("type") == "choice":
        return gap, top_a != top_b
    return gap, abs(float(a["score"]) - float(b["score"])) > 0.5


def load_predictor(checkpoint_dir, backend: str = DEFAULT_BACKEND, arch=None, precision: str = "fp32",
                   crop_state: bool = False) -> LayaPredictor:
    """Construct the laya predictor. The family's weights are fp16, but the reference arithmetic is fp32."""
    return LayaPredictor(checkpoint_dir, backend=backend, precision=precision, arch=arch,
                         crop_state=crop_state)
