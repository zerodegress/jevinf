"""Model architectures.

`nanojev` is the one the engine is wired for today. The other two are declared so the option surface
is stable, and so an architecture that is not wired up refuses to run rather than being loaded as if
it were the wired-up one.

The fields are structural, because that is what the engine's arrangement depends on:

* `attention` -- `causal`: a token's hidden depends only on the tokens before it, which is what makes
  "compute the prefix once, continue into the suffix" exact (see engine.py and docs/internals.md).
  `bidirectional`: no prefix exists, so a question is one path.
* `readout` -- where answers come from: a decision head applied to the backbone's hidden states
  (mirrored in head.py), or probabilities taken off the language-model head.
* `cache` -- what has to stay resident between segments: per-token keys/values, or a recurrent state
  that is not addressable per token.
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
    note: str


ARCHITECTURES: dict[str, Architecture] = {
    "nanojev": Architecture(
        "nanojev", True, "causal", "decision-head", "kv",
        "Qwen3-0.6B causal backbone plus a trained decision head; state, question and candidate "
        "suffix form three segments of one causal path",
    ),
    "laya": Architecture(
        "laya", False, "bidirectional", "marker-scores", "none",
        "ModernBERT encoder: state, question and every option share one sequence, and each option is "
        "scored at its own marker token, so a question is a single path",
    ),
    "decider-2b": Architecture(
        "decider-2b", False, "causal", "lm-head", "recurrent",
        "Qwen3.5-2B causal backbone whose layers alternate full attention with linear attention "
        "(3 of every 4); options are scored off the language-model head",
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
