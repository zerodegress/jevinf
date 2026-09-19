"""Jev (TypeSafe System One) wire contract + translation layer to and from the local NanoJev engine.

**Contract sources (all first-hand, no restated docs):**
  * official OpenAPI 3.1 — https://api.typesafe.ai/openapi.json (info.version 0.2.0)
  * official API docs — https://docs.typesafe.ai/api
  * official Python SDK typesafe-sdk 0.7.0 (repo typesafe-ai/typesafe-sdk-python,
    whose _schemas/models.py is generated from the official OpenAPI) — it **validates
    responses with its own pydantic models**, so hitting our service with it is the
    strictest consistency test we have.

The contract has only two paths: `POST /v1/systemone` (200/422), `GET /v1/models` (200/422), security = HTTPBearer.

**This file translates between the two interfaces:** there are six known gaps between the
NanoJev internal interface and the Jev wire, all filled in here (see the comments in
to_nano_payload / to_jev_response):
  noul→boolean, null/non-string normalization of criteria, confidence (computed here),
  the legend for score, the usage convention, and how state is serialized.
"""
from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------- types
JsonContent = Union[str, dict[str, Any], list[Any]]

# NanoJev checkpoint hard constraints (from its config: max_length=512; the caps on the number of options/levels come from its validator)
NANO_MAX_PATH_TOKENS = 512
NANO_CHOICE_MIN, NANO_CHOICE_MAX = 2, 255
NANO_SCORE_MIN, NANO_SCORE_MAX = 2, 10

# Jev's model aliases. We accept any non-empty string (proxy semantics) but advertise the usable names on /v1/models.
ALIAS_JEV = "jev-latest"
ALIAS_LOCAL = "nanojev-0.6b"


class JevRequestError(ValueError):
    """The request itself is valid (Jev contract), but the local backend cannot serve it → mapped to 422."""


# ---------------------------------------------------------------- request models
class NoulCriteria(BaseModel):
    true: JsonContent | None = None
    false: JsonContent | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JsonContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JsonContent | None = None
    criteria: dict[str, JsonContent | None]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JsonContent | None = None
    criteria: list[JsonContent] = Field(min_length=1)


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")
]


class SystemOneRequest(BaseModel):
    state: JsonContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)


# ---------------------------------------------------------------- response models
class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, JsonContent]
    probabilities: dict[str, float]


Answer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")
]


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]


# ---------------------------------------------------------------- confidence
CONFIDENCE_FORMULA = "1 - H(p)/ln(k)"


def confidence_from_probs(probs: list[float]) -> float:
    """**This engine's own definition**.

    TypeSafe explicitly does not publish its algorithm (official docs, verbatim:
    confidence is a statistic computed from the probability distribution,
    "you are never locked into our definition"); it only requires 0–1 and that a
    flatter distribution means less certainty. This fills the gap with normalized
    entropy: 1 - H(p)/ln(k), taking 1.0 when k=1.
    The full `probabilities` are returned alongside, so a caller who wants a different
    measure can compute it themselves.
    """
    k = len(probs)
    if k <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


# ---------------------------------------------------------------- translation: Jev → engine
def _as_text(value: Any, *, fallback: str = "") -> str:
    """Jev allows instructions/criteria to be str|object|array; the NanoJev validator only accepts non-empty strings."""
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _question_to_nano(qid: str, q: Question) -> dict[str, Any]:
    instructions = _as_text(q.instructions)
    if not instructions:
        # The NanoJev validator requires instructions to be a non-empty string; Jev allows omitting them.
        raise JevRequestError(
            f"question {qid!r}: the local backend requires instructions to be non-empty text, "
            f"Jev allows omitting them but NanoJev has no instruction-less samples in training"
        )

    if isinstance(q, NoulQuestion):
        criteria: dict[str, str] = {}
        if q.criteria is not None:
            for key, label in (("true", q.criteria.true), ("false", q.criteria.false)):
                text = _as_text(label)
                if text:
                    criteria[key] = text
        out: dict[str, Any] = {"type": "boolean", "instructions": instructions}
        if criteria:
            out["criteria"] = criteria  # NanoJev only recognizes the false/true keys
        return out

    if isinstance(q, ChoiceQuestion):
        if not (NANO_CHOICE_MIN <= len(q.criteria) <= NANO_CHOICE_MAX):
            raise JevRequestError(
                f"question {qid!r}: choice has {len(q.criteria)} candidates, "
                f"the local backend supports only {NANO_CHOICE_MIN}–{NANO_CHOICE_MAX} (Jev allows 2–255)"
            )
        # Jev docs: a candidate with no description is "understood by its name alone" → null maps to the key.
        # NanoJev requires both the key and the description to be non-empty strings.
        criteria = {}
        for key, label in q.criteria.items():
            name = (key or "").strip()
            if not name:
                raise JevRequestError(f"question {qid!r}: choice candidate name must not be empty")
            criteria[name] = _as_text(label, fallback=name)
        return {"type": "choice", "instructions": instructions, "criteria": criteria}

    levels = [_as_text(v, fallback=f"level {i}") for i, v in enumerate(q.criteria)]
    if not (NANO_SCORE_MIN <= len(levels) <= NANO_SCORE_MAX):
        # Jev's spec says minItems=1, while NanoJev's score needs at least two levels — that is a capability difference, so report it honestly as 422.
        raise JevRequestError(
            f"question {qid!r}: score has {len(levels)} levels, "
            f"the local backend supports only {NANO_SCORE_MIN}–{NANO_SCORE_MAX} (Jev spec allows ≥1)"
        )
    return {"type": "score", "instructions": instructions, "criteria": levels}


def to_nano_payload(req: SystemOneRequest) -> dict[str, Any]:
    """Jev request → NanoJev engine payload (a single state).

    `state` is **passed through verbatim**: upstream NanoJev was trained on
    `f"State:\\n{state}\\n"`, so dict/list go through Python repr, the format the model saw
    during training. Switching to json.dumps here would move it away from that distribution.
    """
    questions = {qid: _question_to_nano(qid, q) for qid, q in req.questions.items()}
    return {"states": [{"id": "state", "state": req.state, "questions": questions}]}


# ---------------------------------------------------------------- translation: engine → Jev
def _round_probs(probs: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in probs.items()}


def to_jev_response(
    req: SystemOneRequest,
    nano_out: dict[str, Any],
    execution: dict[str, Any],
    *,
    model: str | None = None,
) -> SystemOneResponse:
    """Engine output → Jev response. Built via pydantic, so a malformed shape raises right here, which keeps it from leaving the process."""
    state_out = nano_out["states"][0]
    answers: dict[str, Answer] = {}

    for qid, q in req.questions.items():
        raw = state_out["answers"][qid]
        probs = list(raw["probabilities"].values())

        if isinstance(q, NoulQuestion):
            # Jev's noul answer carries no confidence (official docs are explicit: "Noul answers don't carry one")
            answers[qid] = NoulAnswer(noul=round(float(raw["p_true"]), 6))
            continue

        conf = round(confidence_from_probs(probs), 6)
        if isinstance(q, ChoiceQuestion):
            answers[qid] = ChoiceAnswer(
                choice=str(raw["choice"]),
                probabilities=_round_probs(raw["probabilities"]),
                confidence=conf,
            )
            continue

        # score: fill in the legend (Jev requires it); the keys are the level index as a string
        legend = {str(i): _as_text(v, fallback=f"level {i}") for i, v in enumerate(q.criteria)}
        answers[qid] = ScoreAnswer(
            score=round(float(raw["score"]), 6),
            legend=legend,
            probabilities=_round_probs(raw["probabilities"]),
            confidence=conf,
        )

    return SystemOneResponse(
        # spec: model is "the name of the model that answered, which may differ from the alias in the request".
        # We echo the requested value back (proxy semantics); the real identity is advertised on /v1/models and in the response headers.
        model=model or req.model,
        answers=answers,
        usage=Usage(
            # Jev convention: input is billed, output is free. We report the tokens we actually computed,
            # and output is always 0 — a decision model generates no tokens.
            input_tokens=int(execution.get("computed_tokens", 0)),
            output_tokens=0,
        ),
    )


# ---------------------------------------------------------------- /v1/models
def model_list() -> ModelMetadataList:
    return ModelMetadataList(
        models=[
            ModelMetadata(
                name=ALIAS_JEV,
                description=(
                    "Alias served by the local jevinf engine: NanoJev (Qwen3-0.6B backbone "
                    "+ decision head, 596M params, fp32 on Apple MPS). Wire-compatible with "
                    "TypeSafe's Jev API; this backend runs a different model with its own calibration. "
                    "Per-request hard limits: single state, every candidate path must fit in "
                    f"{NANO_MAX_PATH_TOKENS} tokens (no silent truncation)."
                ),
                release_date="2026-09-19",
            ),
            ModelMetadata(
                name=ALIAS_LOCAL,
                description=(
                    "The underlying local decision model. Deterministic, no autoregressive "
                    "decoding: one batched forward pass returns a probability distribution per "
                    f"question. Path limit {NANO_MAX_PATH_TOKENS} tokens, choice 2–{NANO_CHOICE_MAX} "
                    f"options, score {NANO_SCORE_MIN}–{NANO_SCORE_MAX} levels."
                ),
                release_date="2026-09-18",
            ),
        ]
    )
