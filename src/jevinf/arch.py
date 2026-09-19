"""Model architectures.

Three families are wired up (`nanojev`, `decider-2b`, `laya`); an architecture that is declared but
not wired up refuses to run rather than being loaded as if it were one of the working ones.

The fields are structural, because that is what the engine's arrangement depends on:

* `attention` -- `causal`: a token's hidden depends only on the tokens before it, which is what makes
  "compute the prefix once, continue into the suffix" exact (see engine.py and docs/internals.md).
  `bidirectional`: no prefix exists, so a question is one path.
* `readout` -- where answers come from: a decision head applied to the backbone's hidden states
  (mirrored in head.py), or probabilities taken off the language-model head (decider.py).
* `cache` -- what has to stay resident between segments: `kv` for per-token keys/values, `hybrid` when
  part of the layers keep a recurrent state instead. `hybrid` decides which fork primitive is legal
  (`reorder_cache`; the KV-only broadcast does not exist on those layers), not whether sharing works.
* `arrangement` -- which of the two stage layouts the family is served by: `three-stage` (state →
  question → candidate suffix, engine.py) or `state-fork` (state prefix once, forked to one row per
  question, decider.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    """A decision-model family the engine can be pointed at."""

    name: str
    implemented: bool
    attention: str
    readout: str
    cache: str
    arrangement: str
    note: str


ARCHITECTURES: dict[str, Architecture] = {
    "nanojev": Architecture(
        "nanojev", True, "causal", "decision-head", "kv", "three-stage",
        "Qwen3-0.6B causal backbone plus a trained decision head; state, question and candidate "
        "suffix form three segments of one causal path",
    ),
    "laya": Architecture(
        "laya", True, "bidirectional", "marker-scores", "none", "single-path",
        "ModernBERT encoder: state, question and every option share one sequence, and each option is "
        "scored at its own marker token, so a question is a single path",
    ),
    "decider-2b": Architecture(
        "decider-2b", True, "causal", "lm-head", "hybrid", "state-fork",
        "Qwen3.5-2B causal backbone whose layers alternate full attention with linear attention "
        "(3 of every 4, so the cache is KV plus a recurrent state); options are scored off the "
        "language-model head at one slot per question, and one state prefix is forked to every "
        "question row",
    ),
}
DEFAULT_ARCH = "nanojev"
NAMES: tuple[str, ...] = tuple(ARCHITECTURES)


def resolve(name: str = DEFAULT_ARCH) -> Architecture:
    """Look an architecture up, refusing the ones that are declared but not wired up."""
    arch = ARCHITECTURES.get(name)
    if arch is None:
        raise SystemExit(f"unknown architecture {name!r}; choices: {', '.join(NAMES)}")
    if not arch.implemented:
        raise SystemExit(
            f"architecture {name!r} is declared but not wired up yet; run with --arch {DEFAULT_ARCH}"
        )
    return arch
