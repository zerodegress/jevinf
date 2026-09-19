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

# Point a client at it and it just works (official SDK / any Jev client)
TYPESAFE_BASE_URL=http://127.0.0.1:8226 TYPESAFE_API_KEY=sk-any uv run python your_jev_client.py
```

## Translation layer (`jev_api.py`) — every NanoJev interface gap is filled here

| Gap | Handling |
|---|---|
| `noul` vs `boolean` | question-type mapping; `criteria{true,false}` maps directly |
| criteria may be `null` / non-string | `null` → use the candidate name (Jev docs: when undescribed, read it by its name); dict/list → normalized JSON text |
| `confidence` missing | **computed here**: normalized entropy, `1 - H(p)/ln(k)`. TypeSafe keeps its formula unpublished (their words: "you are never locked into our definition"); the formula is published at `/health` and in the `x-jevinf-confidence-formula` response header |
| `legend` missing | filled from the request criteria, keyed by the **string** form of the level index (matching the official OpenAPI's `dict[str,…]`) |
| `usage` caliber | `input_tokens` = tokens actually computed; `output_tokens` = **0** (a decision model generates no tokens) |
| `state` serialization | **passed through verbatim**. NanoJev was trained on `f"State:\n{state}\n"` (dict/list go through Python repr); normalizing it to canonical JSON would move away from the training distribution |

## Conformance (`uv run python scripts/jev_conformance.py`, **drives the local service with the official SDK**)

The chain of evidence: the official SDK validates responses with its own pydantic models → if it does
not raise `TypeSafeAPIResponseValidationError`, the wire layer passes.

- `GET /v1/models` reads; `system_one` with all three question types mixed: SDK validation passes,
  the `.nouls/.choices/.scores` typed accessors read, probabilities sum to ≈1, `noul` carries no `confidence`
- Raw wire (bypassing the SDK models): keys are **strings**, the answer field sets are exact (four for
  choice, two for noul), and the response carries `x-typesafe-request-id`
- The same state sent through `/v1/systemone` and `/api/evaluate`, question by question: noul/choice/score
  **agree exactly** → the translation layer does not alter answer semantics
- Error contract, all 422: path over 512 tokens, score with only 1 level, choice with only 1 candidate,
  noul missing instructions, unknown question type

## ⚠️ Wire compatibility ≠ capability compatibility (must read)

| Dimension | Jev | This backend |
|---|---|---|
| Context | 64k (state + longest single question ≤32k) | **≤512 tokens per candidate path; over that is 422** |
| score levels | ≥1 | 2–10 |
| choice candidates | 2–255 | 2–255 |
| questions / candidate paths per request | not published | ≤96 / ≤256 (local memory limit) |

Every limit is reported explicitly with 422 and a reason, so an over-limit request reads as a limit
error.
