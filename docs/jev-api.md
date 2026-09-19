# Jev-compatible service layer (`POST /v1/systemone`)

[Back to README](../README.md)

**Every contract detail comes from a primary source**: the official OpenAPI 3.1 spec
(`https://api.typesafe.ai/openapi.json`, info.version 0.2.0), the official docs `docs.typesafe.ai/api`,
and the official Python SDK `typesafe-sdk 0.7.0` (whose `_schemas/models.py` is generated from that
OpenAPI spec). The official contract has exactly two paths, `security: HTTPBearer`, and a validation
failure is **422** (body `{"detail":[...]}`):

- `POST /v1/systemone` → `{state, model, questions}` → `{model, answers, usage}`
- `GET /v1/models` → `{models:[{name, description, release_date}]}`

```bash
uv run jevinf serve -m models/NanoJev --port 8226        # default fused_state; Jev alias jev-latest
uv run jevinf serve -m models/NanoJev --api-key sk-local # enforce HTTPBearer on /v1/*
uv run jevinf serve --arch decider-2b -m models/decider-2b --port 8226   # same endpoints, other family

# Point a client at it and it just works (official SDK / any Jev client)
TYPESAFE_BASE_URL=http://127.0.0.1:8226 TYPESAFE_API_KEY=sk-any uv run python your_jev_client.py
```

## Which family is behind the contract

`--arch` picks the model family and the service states what that family can serve (in `/health` and in
`/v1/models`); it does not silently apply one family's numbers to the other.

| | `nanojev` | `decider-2b` |
|---|---|---|
| arrangement | three-stage prefix sharing, `x-jevinf-strategy` tiers | state prefix forked to one row per question |
| state / path budget | **512 tokens per candidate path**; over that is 422 | **32768 tokens per state**, and a state over it is **refused** (422), never truncated |
| choice / score | 2–255 / 2–10 | 2–255 / 2–10 |
| request temperature default | 1.0 | the model's fitted value (1.3) |
| per-request knobs | strategy, max-rows, temperature, cold | max-rows, temperature, cold (a strategy header is refused with 400) |

## Translation layer (`jev_api.py`) — every interface gap is filled here

| Gap | Handling |
|---|---|
| `noul` vs `boolean` | question-type mapping; `criteria{true,false}` maps directly, and both families accept the `boolean` spelling the layer emits |
| `noul` answer key | the wire has only `noul`; nanojev's engine calls it `p_true` and decider's calls it `noul`, so the layer reads either |
| criteria may be `null` / non-string | `null` → use the candidate name (Jev docs: when undescribed, read it by its name); dict/list → normalized JSON text |
| `confidence` missing | **computed here**: normalized entropy, `1 - H(p)/ln(k)`. TypeSafe keeps its formula unpublished (their words: "you are never locked into our definition"); the formula is published at `/health` and in the `x-jevinf-confidence-formula` response header |
| `legend` missing | filled from the request criteria, keyed by the **string** form of the level index (matching the official OpenAPI's `dict[str,…]`) |
| `usage` caliber | `input_tokens` = tokens actually computed; `output_tokens` = **0** (a decision model generates no tokens) |
| `state` serialization | nanojev: **passed through verbatim** (it was trained on `f"State:\n{state}\n"`, dict/list through Python repr); decider: compact JSON with array indices annotated, per its own `render_state` |

## Conformance (`uv run python scripts/jev_conformance.py`, **drives the local service with the official SDK**)

The chain of evidence: the official SDK validates responses with its own pydantic models → if it does
not raise `TypeSafeAPIResponseValidationError`, the wire layer passes. The script reads `/health`
first and adapts the two family-dependent checks, so it is one script for both families.

- `GET /v1/models` reads; `system_one` with all three question types mixed: SDK validation passes,
  the `.nouls/.choices/.scores` typed accessors read, probabilities sum to ≈1, `noul` carries no `confidence`
- Raw wire (bypassing the SDK models): keys are **strings**, the answer field sets are exact (four for
  choice, two for noul), and the response carries `x-typesafe-request-id`
- The same state sent through `/v1/systemone` and `/api/evaluate`, question by question: noul/choice/score
  **agree exactly** → the translation layer does not alter answer semantics
- Error contract, all 422: score with only 1 level, choice with only 1 candidate, noul missing
  instructions, unknown question type — plus, on `nanojev` only, a path over 512 tokens. On `decider-2b`
  that same over-512 request must **succeed** (its state budget is 32768), so the script asserts the
  opposite there instead: a long state is served and billed at its real length rather than truncated.

## ⚠️ Wire compatibility ≠ capability compatibility (must read)

| Dimension | Jev | `nanojev` | `decider-2b` |
|---|---|---|---|
| Context | 64k (state + longest single question ≤32k) | **≤512 tokens per candidate path; over that is 422** | **≤32768 tokens per state; over that is 422** |
| state truncation | n/a | none (path over limit is refused) | none (a state over budget is refused, never silently cut) |
| score levels | ≥1 | 2–10 | 2–10 |
| choice candidates | 2–255 | 2–255 | 2–255 |
| questions / candidate paths per request | not published | ≤96 / ≤256 (local memory limit) | ≤96 (options are free here: a state is one row of questions) |

Every limit is reported explicitly with 422 and a reason, so an over-limit request reads as a limit
error.
