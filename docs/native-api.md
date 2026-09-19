# Native debug entry point (`POST /api/evaluate`)

[Back to README](../README.md)

The engine-native shape, isomorphic to the upstream toy entry point. Useful for debugging and for
cross-checking that the Jev translation layer changes nothing: the same state sent to both endpoints
must agree question by question.

```bash
uv run jevinf serve -m models/NanoJev --port 8226   # default fused_state; caps 32/96/256/2MiB
uv run jevinf serve -m models/NanoJev --no-limits   # disable limit enforcement
uv run jevinf serve --arch decider-2b -m models/decider-2b --port 8226   # other family, same endpoints
curl -sS -X POST http://127.0.0.1:8226/api/evaluate \
  -H 'content-type: application/json' \
  -d '{"states":[{"id":"s","state":"The service is up.","questions":{"q":{"type":"boolean","instructions":"Is the service up?"}}}]}'
```

`GET /health` reports device / arrangement / caps / knobs / resident memory / the request temperature
default, all read off the loaded family rather than hard-coded.

## Contract (unchanged from upstream)

- `body` and response are isomorphic to upstream `predict_toy_decisions.predict()`:
  request `{"states":[{id,state,questions}]}`; response `{"schema_version","checkpoint","temperature","execution","states"}`.
- **The validator is upstream's own** (`upstream.validate_request`), used exactly as it ships
  → one extra field in `state` yields 400.
- **No engine hints can travel in the request body** (a hard constraint imposed by the upstream
  validator). Knobs travel only in HTTP headers:
  - `x-jevinf-strategy`: `fused_state` (default) / `fused_q` / `two_stage` / `per_question` — pure
    optimization tiers that **do not change answer semantics**. `nanojev` only: on `decider-2b` a
    strategy header is a 400, because that family is arranged by layout and has no tiers.
  - `x-jevinf-max-rows`: row cap per forward (default 64), to control memory at high candidate counts
  - `x-jevinf-temperature`: same meaning as upstream `predict(temperature=)`; the default is the loaded
    model's own value (1.0 nanojev, the fitted 1.3 for decider-2b), not a hard-coded 1.0
  - `x-jevinf-cold`: force a cold computation; there is no cross-request cache today so it is always
    cold, the switch is kept for future use
- **Statistics travel in response headers**, per family: `x-jevinf-architecture` and `wall-ms` always;
  `strategy`/`paths`/`state-kv-bytes`/`path-limit` for `nanojev`; `layout`/`rows`/`prefix-sharing`/
  `prefix-len`/`state-limit` for `decider-2b`. A statistic that does not apply to the loaded family is
  left out rather than sent as `None`.
- Caps match the Jev API: 32 states / 96 questions / 2 MiB for both families, plus 256 paths on
  `nanojev` (there, candidate count drives memory; on `decider-2b` options are free — one state is one
  row of questions — so no path cap is enforced). Over-limit → 413.
- **A state over the family's token budget is refused (413) with the two numbers in the message**, never
  silently truncated: `nanojev` caps a candidate path at 512 tokens, `decider-2b` caps a state at 32768
  (raise it at startup with `--max-state-tokens`).
- **Single process + single inference thread** (global lock). The model requires single-threaded
  access, and multiple workers would split a shared cache into N copies.

## Verified (`uv run python scripts/api_smoke.py`, compared against the offline `jevinf eval` golden)

- 32 states / 96 questions: HTTP 200, **argmax agreement 100.00%** with the offline golden, median TV 0
  → the service layer forwards engine output unchanged
- Switching to `x-jevinf-strategy: two_stage`: 100.00% agreement with the default tier
- Boundaries: extra `state` field 400, unknown question type 400, invalid JSON 400, invalid strategy
  header 400, >32 states 413, >256 paths 413, >2 MiB 413
