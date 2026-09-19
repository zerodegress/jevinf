"""decider-2b family: Jev-shaped requests on a gated-delta-net backbone.

The structural facts live in `arch.py` (causal attention, language-model-head readout, a cache that is
part per-token KV and part recurrent state). This module is the arrangement those facts call for:

    Stage A  state prefix        one row, unpadded, optionally chunked
    Stage B  fork               `reorder_cache` copies that row's cache to N question rows
    Stage C  question suffixes  one right-padded batch; every answer slot is read out of it

Why the fork is worth it: with `independent=True` a request is one row per question behind the *same*
state, so without sharing the state is prefilled N times. Measured on the same model, forking is
exact for this cache type, and the copy it costs is the only price (see docs/strategies.md).

Prompt layout and readout mirror the decider package (Apache-2.0, Mapika/decider) closely enough to be
numerically comparable: the same token ids, the same slot positions, the same LM-head letter rows, the
same temperature. `scripts/decider_parity.py` is the check that says so out loud.

The one thing not to do with this cache: keep using a prefix cache object after a suffix has been run
through it. The suffix is appended in place, so the second reader sees the first reader's suffix.
Fork first, then run each suffix against its own copy.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

# Prompt constants, mirroring decider.prompt (narrow rendering is unchanged since v1).
LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)
MAX_OPTIONS = 255
MAX_LEVELS = 10
MIN_PREFIX = 192  # below this the fork copy costs more than re-prefilling; decider uses the same floor
DEFAULT_TEMPERATURE = 1.3
DEFAULT_MAX_CTX_TOKENS = 1536  # decider's own API default; the model's budget is 32768
ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"
NEUTRAL_NONE = "not listed here"

__all__ = [
    "DeciderEngine",
    "DeciderPredictor",
    "load_predictor",
    "render_question",
    "render_state",
    "plan_rows",
    "assemble",
]


# --------------------------------------------------------------------------- prompt
_LABELS: dict[int, tuple[list[str], list[int], list[int]]] = {}
_OPT_CACHE: dict[tuple[int, str], list[int]] = {}


def label_table(tok):
    """(label names, label token ids, ids of "\\n("), MAX_OPTIONS entries; the first NARROW are A..J."""
    key = id(tok)
    if key not in _LABELS:
        import string

        upper = string.ascii_uppercase
        out: list[tuple[str, int]] = []
        for name in list(upper) + [a + b for a in upper for b in upper]:
            ids = tok.encode(name, add_special_tokens=False)
            if len(ids) == 1:
                out.append((name, ids[0]))
            if len(out) == MAX_OPTIONS:
                break
        if len(out) != MAX_OPTIONS or len({i for _, i in out}) != MAX_OPTIONS:
            raise RuntimeError("could not build a 255-entry single-token label table")
        _LABELS[key] = ([n for n, _ in out], [i for _, i in out], tok.encode("\n(", add_special_tokens=False))
    return _LABELS[key]


def letter_ids(tok) -> list[int]:
    """Label token ids for the readout: the whole MAX_OPTIONS-wide table.

    Wide questions (more than NARROW options) use A..Z and then two-letter labels, so restricting this
    to the first NARROW entries would silently truncate every such question to ten options. The narrow
    rendering additionally relies on A..J being single tokens, which the table guarantees by
    construction and this checks directly.
    """
    ids = label_table(tok)[1]
    for j, letter in enumerate(LETTERS):
        if tok.encode(letter, add_special_tokens=False) != [ids[j]]:
            raise RuntimeError(f"label {letter!r} is not a single token")
    return ids


def _enc_option(tok, text: str) -> list[int]:
    """Ids of ") <option text>" (cached: fixed label sets repeat the same strings)."""
    key = (id(tok), text)
    ids = _OPT_CACHE.get(key)
    if ids is None:
        ids = tok.encode(f") {text}", add_special_tokens=False)
        _OPT_CACHE[key] = ids
    return ids


def option_ids(tok, options: list[str]) -> list[int]:
    """Option block ids: <= 10 options are the "(A) .. (J)" string; more use one label token each."""
    if len(options) <= NARROW:
        return tok.encode(
            "".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)), add_special_tokens=False
        )
    _, lab_ids, open_ids = label_table(tok)
    out: list[int] = []
    for j, o in enumerate(options):
        out += open_ids + [lab_ids[j]] + _enc_option(tok, o)
    return out


def _txt(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


ANNOTATE_MIN = 8


def annotate_indices(x, min_len: int = ANNOTATE_MIN):
    """Write each element's position into long arrays, so a path like `records[47].text` is a lookup."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [
                ({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                 else {"_index": i, "value": annotate_indices(v, min_len)})
                for i, v in enumerate(x)
            ]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state) -> str:
    """A string state is used as it is; anything else is compact JSON with array indices annotated."""
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def render_question(spec: dict) -> dict:
    """Jev-shaped question -> decider row question. Returns dict(question, options, type, names, legend)."""
    qtype = spec.get("type", "choice")
    instructions = _txt(spec.get("instructions", spec.get("question", "")))
    criteria = spec.get("criteria", spec.get("options"))
    if not instructions:
        raise ValueError("question without instructions")
    if qtype == "choice":
        if isinstance(criteria, (list, tuple)):
            criteria = {str(c): None for c in criteria}
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(f"choice criteria: a map of 2..{MAX_OPTIONS} options")
        names = list(criteria)
        options = [n if criteria[n] in (None, "") else f"{n}: {_txt(criteria[n])}" for n in names]
    elif qtype == "score":
        if isinstance(criteria, dict):
            criteria = [criteria[k] for k in sorted(criteria, key=float)]
        if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= MAX_LEVELS:
            raise ValueError(f"score criteria: an ordered list of 2..{MAX_LEVELS} level descriptions")
        names = list(range(len(criteria)))
        options = [f"{i}: {_txt(c)}" for i, c in enumerate(criteria)]
    elif qtype in ("noul", "bool"):
        names = [False, True]
        c = criteria or {}
        f, t = c.get("false", c.get(False)), c.get("true", c.get(True))
        options = ["no" if f in (None, "") else f"no: {_txt(f)}",
                   "yes" if t in (None, "") else f"yes: {_txt(t)}"]
    else:
        raise ValueError(f"unknown question type {qtype!r}")
    return dict(
        question=instructions,
        options=options,
        type="noul" if qtype == "bool" else qtype,
        names=names,
        legend=[_txt(c) for c in criteria] if qtype == "score" else None,
        isolated=bool(spec.get("isolated", True)),
    )


def strip_level_number(text: str) -> str:
    import re

    return re.sub(r"^\s*-?\d+\s*:\s*", "", text)


def isolated_rows(question: str, levels: list[str]) -> list[tuple[str, list[str]]]:
    """One yes/no row per level: each level is judged without seeing its number or its neighbours."""
    return [(ISOLATED.format(q=question, level=strip_level_number(l)), ["no", "yes"]) for l in levels]


def plan_rows(rendered: dict, isolated: bool = True):
    """One scoring row per question; a Score question with isolated levels becomes one row per level.

    Returns (rows, index) where index entries are (question id, "iso" | "list", first row, n rows).
    """
    rows, index = [], []
    for qid, r in rendered.items():
        if isolated and r["type"] == "score" and r.get("isolated", True):
            rws = isolated_rows(r["question"], r["legend"])
            index.append((qid, "iso", len(rows), len(rws)))
            rows += [dict(question=t, options=o) for t, o in rws]
        else:
            index.append((qid, "list", len(rows), 1))
            rows.append(dict(question=r["question"], options=r["options"]))
    return rows, index


def certainty(p: list[float]) -> float:
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, 1.0 - h / math.log(len(p))) if len(p) > 1 else 1.0


def format_answer(rendered: dict, p, nd: int = 4) -> dict:
    p = [float(x) for x in p[: len(rendered["options"])]]
    total = sum(p) or 1.0
    p = [x / total for x in p]
    j = max(range(len(p)), key=p.__getitem__)
    if rendered["type"] == "noul":
        return {"type": "noul", "noul": round(p[1], nd)}
    if rendered["type"] == "choice":
        return {"type": "choice", "choice": rendered["names"][j], "confidence": round(p[j], nd),
                "certainty": round(certainty(p), nd),
                "probabilities": {n: round(x, nd) for n, x in zip(rendered["names"], p)}}
    return {"type": "score", "score": round(sum(i * x for i, x in enumerate(p)), 2),
            "confidence": round(p[j], nd), "certainty": round(certainty(p), nd),
            "legend": {str(i): d for i, d in enumerate(rendered["legend"])},
            "probabilities": {str(i): round(x, nd) for i, x in enumerate(p)}}


def combine_isolated(p_yes: list[float]):
    """Per-level P(fits) -> a distribution over levels, plus the unnormalised mass."""
    total = sum(p_yes) or 1e-9
    return [x / total for x in p_yes], total


def assemble(rendered: dict, index, probs: list[list[float]]) -> dict:
    out = {}
    for qid, kind, start, n in index:
        if kind == "iso":
            fit = [float(probs[start + j][1]) for j in range(n)]
            p, mass = combine_isolated(fit)
            answer = format_answer(rendered[qid], p)
            answer["level_fit"] = {str(j): round(x, 4) for j, x in enumerate(fit)}
            answer["fit_mass"] = round(mass, 4)
            out[qid] = answer
        else:
            out[qid] = format_answer(rendered[qid], probs[start])
    return out


def neutralize_options(options: list[str]) -> list[str]:
    """Rewrite abstain-style options to a neutral phrasing.

    The training augmentation used the literal "none of the above" and the model learned that exact
    string as an abstain signal, so it abstains even on clear cases when the string is offered. The
    answer still reports the original option: only what the model reads changes.
    """
    out = []
    for option in options:
        key = option.strip().lower()
        if key.startswith("none of the above") or key in ("none of the above", "none", "n/a", "none of these"):
            out.append(NEUTRAL_NONE)
        else:
            out.append(option)
    return out


def row_ids(tok, state_ids: list[int], row: dict, multi: bool, k: int) -> tuple[list[int], int]:
    """Ids of one scoring row and the position of its answer slot.

    `multi`/`k` only vary the numbering, and numbering only appears when a row carries several
    questions; this prototype serves one question per row (decider's `independent=True`).
    """
    head = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {row['question']}\nOptions:"
    tail = f"\nAnswer{' ' + str(k + 1) if multi else ''}: ("
    if len(row["options"]) <= NARROW:
        piece = tok.encode("".join([head] + [f"\n({LETTERS[j]}) {o}" for j, o in enumerate(row["options"])])
                           + tail, add_special_tokens=False)
    else:
        piece = tok.encode(head, add_special_tokens=False) + option_ids(tok, row["options"])
        piece += tok.encode(tail, add_special_tokens=False)
    ids = list(state_ids) + piece
    return ids, len(ids) - 1


def unique_tokens(rows: list[list[int]]) -> int:
    """Input tokens of a request whose rows share a prefix: the prefix counts once."""
    if len(rows) < 2:
        return sum(len(r) for r in rows)
    lcp = 0
    short = min(len(r) for r in rows)
    while lcp < short and all(r[lcp] == rows[0][lcp] for r in rows):
        lcp += 1
    return lcp + sum(len(r) - lcp for r in rows)


# --------------------------------------------------------------------------- model
class DeciderPredictor:
    """Weights plus the two things the readout needs: the backbone and the option-letter rows."""

    def __init__(self, checkpoint_dir, backend: str = "torch-mps", precision: str = "bf16",
                 arch=None, max_ctx_tokens: int = DEFAULT_MAX_CTX_TOKENS, temperature: float | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .backend import DEFAULT_BACKEND, resolve as resolve_backend

        self._torch = torch
        self.root = str(checkpoint_dir)
        self.device = resolve_backend(backend or DEFAULT_BACKEND).device
        self.arch = arch
        cfg = {}
        try:
            cfg = json.loads((__import__("pathlib").Path(self.root) / "decider_config.json").read_text())
        except Exception:  # a checkout without the config file still runs, with the defaults below
            pass
        self.config = cfg
        self.temperature = float(temperature if temperature is not None else cfg.get("temperature", DEFAULT_TEMPERATURE))
        self.temperature_schema_first = float(cfg.get("temperature_schema_first", self.temperature))
        self.max_options = int(cfg.get("max_options", MAX_OPTIONS))
        self.max_state_tokens = int(cfg.get("max_state_tokens", 32768))
        self.isolated_levels = bool(cfg.get("isolated_levels", False))
        self.neutralize_none = bool(cfg.get("neutralize_none", True))
        self.max_ctx_tokens = int(max_ctx_tokens)
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.root)
        lm = AutoModelForCausalLM.from_pretrained(self.root, dtype=dtype).to(self.device).eval()
        for param in lm.parameters():
            param.requires_grad_(False)
        self.lm = lm
        self.backbone = lm.model
        letters = letter_ids(self.tokenizer)
        self.letters = torch.tensor(letters, dtype=torch.long)
        # The readout only ever addresses these rows of the head, so they are copied out once (255 x
        # hidden, about a megabyte) and the readout stops caring about the head's layout. The head's
        # full weight matrix stays resident regardless: this checkpoint ties it to the token embedding.
        self.W = lm.lm_head.weight[self.letters].detach().clone()
        self.limit = self.max_state_tokens  # name kept so both families answer the same way


# --------------------------------------------------------------------------- engine
@dataclass
class DeciderStats:
    layout: str = "state_first"
    states: int = 0
    questions: int = 0
    rows: int = 0
    forwards: int = 0
    prefix_forwards: int = 0
    suffix_forwards: int = 0
    shared: bool = False
    prefix_len: int = 0
    forked_rows: int = 0
    computed_tokens: int = 0
    baseline_tokens: int = 0  # rows counted in full, i.e. what a per-row recompute would push
    prefix_tokens_saved: int = 0
    cache_bytes: int = 0
    wall_s: float = 0.0
    t_prefix: float = 0.0
    t_fork: float = 0.0
    t_suffix: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class DeciderEngine:
    """The arrangement in this module's docstring, for a predictor whose cache is hybrid."""

    def __init__(self, predictor: DeciderPredictor, *, max_rows: int = 32, prefix_chunk: int = 1024,
                 min_prefix: int = MIN_PREFIX, max_ctx_tokens: int | None = None):
        self.p = predictor
        self.arch = self._check_arch(predictor)
        self.torch = predictor._torch
        self.backbone = predictor.backbone
        self.W = predictor.W
        self.tok = predictor.tokenizer
        self.device = predictor.device
        self.max_rows = max(1, max_rows)
        self.prefix_chunk = max(0, prefix_chunk)  # 0 = one shot
        self.min_prefix = min_prefix
        self.max_ctx_tokens = int(max_ctx_tokens or predictor.max_ctx_tokens)
        self.temperature = predictor.temperature

    @staticmethod
    def _check_arch(predictor):
        """This arrangement needs causal attention and a cache that can be forked with reorder_cache."""
        spec = getattr(predictor, "arch", None)
        if spec is None:
            raise ValueError("predictor carries no architecture tag; load it through upstream.load_predictor")
        if spec.attention != "causal" or spec.cache != "hybrid":
            raise ValueError(
                f"architecture {spec.name!r} is not one this engine arranges: it needs causal attention "
                f"with a hybrid (KV + recurrent) cache, got attention={spec.attention!r}, cache={spec.cache!r}"
            )
        return spec

    # ---------------------------------------------------------------- cache primitives
    @staticmethod
    def cache_bytes(cache) -> int:
        """Every tensor the cache holds, whatever the layer type calls it (keys/values, conv_states, ...)."""
        total = 0
        for layer in getattr(cache, "layers", []):
            for value in vars(layer).values():
                if hasattr(value, "numel") and hasattr(value, "element_size"):
                    total += value.numel() * value.element_size()
        return total

    @staticmethod
    def cache_rows(cache) -> int:
        """Batch width of a cache row-wise, read off whichever tensor each layer type keeps."""
        for layer in getattr(cache, "layers", []):
            for value in vars(layer).values():
                if hasattr(value, "dim") and value.dim() >= 1:
                    return int(value.shape[0])
        raise RuntimeError("could not read the row count of the cache")

    def fork(self, cache, n: int):
        """Copy row 0 of `cache` into n rows.

        `reorder_cache` is the one primitive that works for both halves of this cache: per-token KV
        layers and the linear layers' recurrent state. `batch_repeat_interleave` exists only on the KV
        layers, so it raises here. Measured exact: forking to 3 rows and reading rows 0 and 2 back gives
        a state difference of 0.0 (docs/strategies.md).
        """
        if n == 1:
            return cache
        index = self.torch.zeros(n, dtype=self.torch.long, device=self.device)  # every row reads row 0
        cache.reorder_cache(index)
        rows = self.cache_rows(cache)
        if rows != n:
            raise AssertionError(f"cache row count after fork {rows} != {n}")
        return cache

    def _sync(self):
        if str(self.device).startswith("mps"):
            self.torch.mps.synchronize()

    # ---------------------------------------------------------------- forwards
    def forward_prefix(self, ids: list[int], stats: DeciderStats | None = None):
        """Stage A: one unpadded row, chunked if asked. Returns (last hidden for all positions, cache).

        Chunking exists for memory, not speed: on the reference machine a 32k prefix is 10.3 GiB in one
        shot and 5.2 GiB in 1024-token chunks, at the same wall time and bit-identical (docs/internals.md).
        """
        torch = self.torch
        chunk = self.prefix_chunk or len(ids)
        cache = None
        out = None
        for start in range(0, len(ids), chunk):
            piece = ids[start : start + chunk]
            tokens = torch.tensor([piece], dtype=torch.long, device=self.device)
            if cache is None:
                out = self.backbone(input_ids=tokens, use_cache=True)
            else:
                attn = torch.ones((1, start + len(piece)), dtype=torch.bool, device=self.device)
                out = self.backbone(input_ids=tokens, attention_mask=attn, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if stats is not None:
                stats.prefix_forwards += 1
                stats.forwards += 1
        if cache is None or len(cache) == 0:
            raise RuntimeError("backbone did not return a cache: use_cache did not take effect")
        return out.last_hidden_state[0], cache

    def forward_suffix(self, rows: list[list[int]], cache, stats: DeciderStats | None = None):
        """Stage C: one right-padded batch of suffixes. Returns (hidden at each row's slot, rows, width)."""
        torch = self.torch
        if not rows:
            raise ValueError("rows must not be empty")
        lengths = [len(r) for r in rows]
        if min(lengths) == 0:
            raise ValueError("empty suffix row present")
        width = max(lengths)
        tokens = torch.full((len(rows), width), self.tok.pad_token_id, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            tokens[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=self.device)
        if stats is not None:
            stats.suffix_forwards += 1
            stats.forwards += 1
        # No attention mask, on purpose: right padding after a causal position cannot reach an earlier
        # slot, and this is what the reference engine does, so the numbers stay comparable.
        out = self.backbone(input_ids=tokens, past_key_values=cache, use_cache=True)
        return out.last_hidden_state

    def slot_probabilities(self, hidden, rows_slot: list[tuple[int, int]], nopts: list[int], temperature: float):
        """Gather each slot's hidden, project onto the option-letter rows, mask and softmax."""
        torch = self.torch
        import torch.nn.functional as F

        index = torch.tensor(rows_slot, dtype=torch.long, device=self.device)
        hs = hidden[index[:, 0], index[:, 1]]
        logits = F.linear(hs, self.W).float()  # float() after the linear, as the reference readout does
        ar = torch.arange(logits.shape[1], device=self.device)[None, :]
        logits = logits.masked_fill(ar >= torch.tensor(nopts, device=self.device)[:, None], float("-inf"))
        if not torch.isfinite(logits).any(1).all():
            raise ValueError("a slot had every option masked out")
        return torch.softmax(logits / temperature, -1)

    def _read(self, hidden, slots: list[tuple[int, int]], nopts: list[int], temperature: float):
        """Read one batch's slots. Returns one 1-D tensor per slot, in the order the slots were given.

        Flat on purpose: `assemble` indexes into this list by row offset, so a per-item grouping here
        would only have to be undone by every caller.
        """
        probs = self.slot_probabilities(hidden, slots, nopts, temperature)
        return [probs[j].cpu() for j in range(probs.shape[0])]

    # ---------------------------------------------------------------- scoring
    def score_plain(self, items: list[dict], temperature: float, stats: DeciderStats):
        """Every row whole, batched in `max_rows` groups: no sharing, but no fork copy either."""
        torch = self.torch
        hidden_out = []
        for i in range(0, len(items), self.max_rows):
            group = items[i : i + self.max_rows]
            flat = [pair for it in group for pair in it["rows"]]  # (ids, slot), one per row
            rows = [ids for ids, _slot in flat]
            slots = [(j, slot) for j, (_ids, slot) in enumerate(flat)]
            nopts = [n for it in group for n in it["nopts"]]
            hidden = self.forward_suffix(rows, None, stats)
            hidden_out.append(self._read(hidden, slots, nopts, temperature))
        if len(hidden_out) == 1:
            return hidden_out[0]
        return [p for chunk in hidden_out for p in chunk]

    def score_shared(self, items: list[dict], temperature: float, stats: DeciderStats):
        """Prefix once, fork, suffixes batched. Items must share one prefix (they do: the state)."""
        torch = self.torch
        ids = [it["rows"][0][0] for it in items]  # the first row of each item is the whole prefix + suffix
        prefix_len = 0
        short = min(len(x) for x in ids)
        while prefix_len < short and all(x[prefix_len] == ids[0][prefix_len] for x in ids):
            prefix_len += 1
        if len(items) < 2 or prefix_len < self.min_prefix:
            stats.shared = False
            return self.score_plain(items, temperature, stats)
        stats.shared = True
        stats.prefix_len = prefix_len
        self._sync()
        t = time.perf_counter()
        _, cache = self.forward_prefix(ids[0][:prefix_len], stats)
        self._sync()
        stats.t_prefix += time.perf_counter() - t
        stats.cache_bytes = self.cache_bytes(cache)
        t = time.perf_counter()
        self.fork(cache, len(items))
        self._sync()
        stats.t_fork += time.perf_counter() - t
        stats.forked_rows = len(items)

        # One suffix batch for every row; slots are relative to the end of the shared prefix.
        flat = [pair for it in items for pair in it["rows"]]  # (ids, slot), one per row
        suffixes = [ids[prefix_len:] for ids, _slot in flat]
        slots = [(j, slot - prefix_len) for j, (_ids, slot) in enumerate(flat)]
        nopts = [n for it in items for n in it["nopts"]]
        t = time.perf_counter()
        hidden = self.forward_suffix(suffixes, cache, stats)
        probs = self._read(hidden, slots, nopts, temperature)
        self._sync()
        stats.t_suffix += time.perf_counter() - t
        return probs

    # ---------------------------------------------------------------- top level
    def score_rows(self, state, rows: list[dict], *, layout: str = "state_first",
                   temperature: float | None = None, max_ctx_tokens: int | None = None):
        """Score a list of (question, options) rows against one state. Returns probabilities per row."""
        torch = self.torch
        temperature = self.temperature if temperature is None else temperature
        limit = int(max_ctx_tokens or self.max_ctx_tokens)
        state_ids = self.tok.encode("Context:\n" + render_state(state), add_special_tokens=False)[:limit]
        items, all_rows = [], []
        for k, row in enumerate(rows):
            options = neutralize_options(row["options"]) if self.p.neutralize_none else list(row["options"])
            # One question per row, so no numbering: this mirrors decider's independent=True requests,
            # where every row is the state plus exactly one question.
            ids, slot = row_ids(self.tok, state_ids, dict(row, options=options), False, k)
            items.append({"rows": [(ids, slot)], "nopts": [len(options)]})
            all_rows.append(ids)
        stats = DeciderStats(layout=layout, rows=len(rows), baseline_tokens=sum(len(r) for r in all_rows))
        with torch.inference_mode():
            if layout == "state_first":
                probs = self.score_shared(items, temperature, stats) if len(items) > 1 else self.score_plain(items, temperature, stats)
            else:
                raise ValueError("only the state_first layout is implemented for this family")
        stats.computed_tokens = stats.baseline_tokens if not stats.shared else unique_tokens(all_rows)
        stats.prefix_tokens_saved = stats.baseline_tokens - stats.computed_tokens
        return probs, stats

    def evaluate(self, payload: dict, *, layout: str = "state_first", isolated: bool | None = None,
                 independent: bool = True, max_ctx_tokens: int | None = None, temperature: float | None = None):
        """Jev-shaped payload in, Jev-shaped answers out -- the same contract the nanojev path serves."""
        t0 = time.perf_counter()
        isolated = self.p.isolated_levels if isolated is None else isolated
        if not independent:
            raise ValueError(
                "packing several questions into one row is not implemented for this family; "
                "requests are served one question per row (decider's independent=True)"
            )
        states = payload.get("states") or []
        outputs = []
        totals = DeciderStats(layout=layout)
        for entry in states:
            rendered = {qid: render_question(spec) for qid, spec in entry["questions"].items()}
            rows, index = plan_rows(rendered, isolated and independent)
            probs, stats = self.score_rows(entry["state"], rows, layout=layout,
                                           temperature=temperature, max_ctx_tokens=max_ctx_tokens)
            totals.questions += len(rendered)
            totals.rows += stats.rows
            totals.forwards += stats.forwards
            totals.prefix_forwards += stats.prefix_forwards
            totals.suffix_forwards += stats.suffix_forwards
            totals.computed_tokens += stats.computed_tokens
            totals.baseline_tokens += stats.baseline_tokens
            totals.prefix_tokens_saved += stats.prefix_tokens_saved
            totals.shared = totals.shared or stats.shared
            totals.prefix_len = max(totals.prefix_len, stats.prefix_len)
            totals.forked_rows = max(totals.forked_rows, stats.forked_rows)
            totals.cache_bytes = max(totals.cache_bytes, stats.cache_bytes)
            for name in ("t_prefix", "t_fork", "t_suffix"):
                setattr(totals, name, getattr(totals, name) + getattr(stats, name))
            totals.states += 1
            outputs.append({"id": entry.get("id", f"state{len(outputs)}"),
                            "answers": assemble(rendered, index, [p.tolist() for p in probs])})
        totals.wall_s = time.perf_counter() - t0
        return {
            "schema_version": "jevinf-v0",
            "execution": {
                "engine": "jevinf",
                "architecture": self.arch.name,
                "layout": layout,
                "device": str(self.device),
                "parameter_storage": "bfloat16",
                "prefix_sharing": totals.shared,
                "isolated_levels": bool(isolated and independent),
                "temperature": float(self.temperature if temperature is None else temperature),
                "autoregressive_decode_steps": 0,
                **totals.as_dict(),
            },
            "states": outputs,
        }

    def selfcheck(self, payload: dict, tolerance: float = 0.02, isolated: bool | None = None):
        """Shared-prefix path against full recompute on the same rows: the family's own equivalence check.

        The gate is argmax agreement; the probability gap is reported because it is not zero and should
        not be papered over. Continuing a row from a forked prefix differs from computing it in one shot
        by a few thousandths of a probability at a 600-token prefix: the recurrence inside the linear
        layers is chunked, so the suffix's tokens land in different chunk phases than they would in a
        single forward. It grows with prefix length, which is what `tolerance` is for.
        """
        isolated = self.p.isolated_levels if isolated is None else isolated
        compared, agree, worst, shared_states = 0, 0, 0.0, 0
        for entry in payload.get("states") or []:
            rendered = {qid: render_question(spec) for qid, spec in entry["questions"].items()}
            rows, _index = plan_rows(rendered, isolated)
            if len(rows) < 2:
                continue
            saved = self.min_prefix
            try:
                self.min_prefix = 0
                shared, stats = self.score_rows(entry["state"], rows)
                self.min_prefix = 10**9
                plain, _ = self.score_rows(entry["state"], rows)
            finally:
                self.min_prefix = saved
            shared_states += 1 if stats.shared else 0
            for a, b in zip(shared, plain):
                a, b = a.tolist(), b.tolist()
                if len(a) != len(b):
                    raise AssertionError("row option counts differ between paths")
                compared += 1
                agree += int(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__))
                worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
        return {
            "architecture": self.arch.name,
            "rows_compared": compared,
            "states_with_shared_prefix": shared_states,
            "argmax_agreement": (agree / compared) if compared else None,
            "max_abs_probability_delta": worst,
            "tolerance": tolerance,
            "pass": bool(compared and agree == compared and worst <= tolerance),
        }


# --------------------------------------------------------------------------- loading
def load_predictor(checkpoint_dir, backend: str = "torch-mps", arch=None, precision: str = "bf16",
                   max_ctx_tokens: int = DEFAULT_MAX_CTX_TOKENS, temperature: float | None = None):
    """Build the decider predictor. `upstream.load_predictor` is the way in; this is its decider branch."""
    return DeciderPredictor(checkpoint_dir, backend=backend, precision=precision, arch=arch,
                            max_ctx_tokens=max_ctx_tokens, temperature=temperature)
