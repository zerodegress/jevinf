"""Jev-compatible API service layer — the contract stays untouched, we only attach the engine behind it.

The external contract is **isomorphic** to upstream `predict_toy_decisions.predict()`:

    POST /api/evaluate
      body  {"states": [{"id": str, "state": str|dict|list,
                         "questions": {qid: {"type","instructions","criteria"}}}]}
      resp  {"schema_version","checkpoint","temperature","execution","states":[{"id","answers"}]}

Three rules of discipline:

1. **Use the upstream validator** (`upstream.validate_request`), exactly as it ships.
   state only allows `id/state/questions`, a question only allows `type/instructions/criteria`;
   one extra field is a 400.
2. **No engine hint can be smuggled into the request body** — that is a hard constraint coming
   from the upstream validator. So the knobs can only travel in
   HTTP headers (see HEADER_KNOBS); the body stays byte-compatible.
3. **Stats travel in response headers**, keeping the body in its upstream shape. In the body,
   `execution`
   only fills in keys upstream already has (`forward_passes`/`candidate_paths`/`prefix_sharing`
   etc.), so a client written against upstream can read them without code changes.

Limits match the Jev API: 32 states / 96 questions / 256 paths / 2 MiB.

Process model: **one process + one inference thread** (a global lock). The model is not
thread-safe, and if cross-request caching is added later, multiple workers would split the
cache into N copies. Keep uvicorn at its default of 1 worker.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from . import __version__, upstream
from .arch import DEFAULT_ARCH
from .backend import DEFAULT_BACKEND
from .engine import PrefixShareEngine
from .jev_api import (
    CONFIDENCE_FORMULA,
    ALIAS_JEV,
    JevRequestError,
    ModelMetadataList,
    SystemOneRequest,
    SystemOneResponse,
    model_list,
    to_jev_response,
    to_nano_payload,
)

MAX_STATES = 32
MAX_QUESTIONS = 96
MAX_PATHS = 256
MAX_BODY_BYTES = 2 * 1024 * 1024

STRATEGIES = ("fused_state", "fused_q", "two_stage", "per_question")

HEADER_KNOBS = {
    "x-jevinf-strategy": "engine strategy (default fused_state); pure optimization tier, does not change answer semantics",
    "x-jevinf-max-rows": "row cap for a single forward pass (default 64); bounds memory when there are many candidates",
    "x-jevinf-temperature": "softmax temperature (default 1.0); same meaning as upstream predict(temperature=)",
    "x-jevinf-cold": "1 = force a cold compute. There is no cross-request cache today, so it is always cold; the switch is kept for future use",
}

RESPONSE_STAT_HEADERS = (
    "x-jevinf-strategy",
    "x-jevinf-forwards",
    "x-jevinf-computed-tokens",
    "x-jevinf-paths",
    "x-jevinf-state-kv-bytes",
    "x-jevinf-wall-ms",
)


class ApiError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message


def count_paths(states: list[dict]) -> tuple[int, int]:
    """(question count, candidate path count). boolean has only one semantic path, the same convention as upstream execution.candidate_paths."""
    questions = paths = 0
    for row in states:
        for q in row["questions"].values():
            questions += 1
            paths += 1 if q["type"] == "boolean" else len(q["criteria"])
    return questions, paths


class EvaluateService:
    """Wrap the engine as a service: validation + limits + knobs + stats. The engine itself knows nothing about HTTP."""

    def __init__(self, checkpoint_dir, backend: str = DEFAULT_BACKEND, arch: str = DEFAULT_ARCH,
                 strategy: str = "fused_state",
                 max_rows: int = 64, enforce_limits: bool = True, api_key: str | None = None):
        self.predictor = upstream.load_predictor(checkpoint_dir, backend=backend, arch=arch)
        self.arch = self.predictor.arch.name
        self.default_strategy = strategy
        self.default_max_rows = max_rows
        self.enforce_limits = enforce_limits
        self.api_key = api_key
        self.lock = threading.Lock()
        self.calls = 0
        self._engines: dict[tuple[str, int], PrefixShareEngine] = {}

    # ------------------------------------------------------- Jev-compatible endpoints
    def system_one(self, req: SystemOneRequest) -> tuple[SystemOneResponse, dict]:
        """Jev `/v1/systemone` → engine → Jev response. The translation layer lives in jev_api.py."""
        nano_payload = to_nano_payload(req)  # contract-valid but the backend cannot serve it → JevRequestError
        questions, paths = count_paths(nano_payload["states"])
        if self.enforce_limits:
            if questions > MAX_QUESTIONS:
                raise JevRequestError(
                    f"questions {questions} > {MAX_QUESTIONS} (local backend limit)"
                )
            if paths > MAX_PATHS:
                raise JevRequestError(
                    f"candidate paths {paths} > {MAX_PATHS} (local backend limit; "
                    f"Jev itself allows more, but memory runs out here first)"
                )
        with self.lock:
            self.calls += 1
            engine = self._engine(self.default_strategy, self.default_max_rows)
            started = time.perf_counter()
            try:
                out = engine.evaluate(nano_payload, temperature=1.0)
            except (ValueError, AssertionError) as exc:
                # upstream validator / paths over 512 tokens etc. → map to 422 per the Jev contract
                raise JevRequestError(str(exc)) from None
            wall = time.perf_counter() - started
        ex = dict(out["execution"])
        ex["wall_s"] = round(wall, 6)
        return to_jev_response(req, out, ex), ex

    # ------------------------------------------------------------------ engine
    def _engine(self, strategy: str, max_rows: int) -> PrefixShareEngine:
        key = (strategy, max_rows)
        if key not in self._engines:
            self._engines[key] = PrefixShareEngine(
                self.predictor, strategy=strategy, max_rows=max_rows
            )
        return self._engines[key]

    # ------------------------------------------------------------------ main path
    def evaluate(self, payload: Any, *, strategy: str | None = None, max_rows: int | None = None,
                 temperature: float = 1.0) -> tuple[dict, dict]:
        # 1) upstream validator (strict)
        try:
            states = upstream.validate_request(payload)
        except (ValueError, TypeError) as exc:
            raise ApiError(400, type(exc).__name__, str(exc)) from None

        # 2) limits (matching the Jev API)
        questions, paths = count_paths(states)
        if self.enforce_limits:
            if len(states) > MAX_STATES:
                raise ApiError(413, "LimitExceeded",
                               f"states {len(states)} > {MAX_STATES}")
            if questions > MAX_QUESTIONS:
                raise ApiError(413, "LimitExceeded",
                               f"questions {questions} > {MAX_QUESTIONS}")
            if paths > MAX_PATHS:
                raise ApiError(413, "LimitExceeded", f"paths {paths} > {MAX_PATHS}")

        strat = (strategy or self.default_strategy).strip()
        if strat not in STRATEGIES:
            raise ApiError(400, "BadStrategy",
                           f"strategy must be one of {'/'.join(STRATEGIES)}, got {strat!r}")
        rows = int(max_rows) if max_rows else self.default_max_rows
        if rows < 1:
            raise ApiError(400, "BadMaxRows", "max-rows must be >= 1")
        if not isinstance(temperature, (int, float)) or not (temperature > 0):
            raise ApiError(400, "BadTemperature", "temperature must be a positive number")

        # 3) inference (single-threaded, serialized)
        with self.lock:
            self.calls += 1
            engine = self._engine(strat, rows)
            started = time.perf_counter()
            out = engine.evaluate(payload, temperature=float(temperature))
            wall = time.perf_counter() - started

        ex = dict(out["execution"])
        # fill in the keys upstream predict() already has, so clients written against it can read them without code changes
        ex.update({
            "precision": "fp32",
            "forward_autocast": "disabled",
            "candidate_paths": paths,
            "forward_passes": ex["forwards"],
            "batch_questions_limit": "all",
            "max_length": self.predictor.limit,
            "disable_native_triton": False,
            "network_model_calls": 0,
            "persistent_model_load_count": 1,
            "inference_call_index": self.calls,
            "batch_questions": 0,
            "wall_s": round(wall, 6),
        })
        body = {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {
                "directory": str(self.predictor.root),
                "base_model": self.predictor.run_config.get("model"),
                "base_revision": self.predictor.run_config.get("resolved_model_revision"),
                "set_head": self.predictor.run_config["set_head"],
            },
            "temperature": {
                "value": float(temperature),
                "fitted_by_this_command": False,
                "note": "applies the given scalar explicitly; a default of 1 does not mean the model is calibrated.",
            },
            "execution": ex,
            "states": out["states"],
        }
        return body, ex


def create_app(service: EvaluateService):
    """Build the ASGI app.

    Note: `Request` must come from a **module-level** import — this module enables
    `from __future__ import annotations`, so the annotations are strings and FastAPI
    evaluates them in the module's global namespace. If `Request` were imported only
    inside the function, FastAPI could not resolve the type and would treat `request`
    as a **required query parameter**, returning 422.
    """
    app = FastAPI(title="jevinf", version=__version__,
                  description="Jev-compatible decision inference service (prefix-sharing engine)")

    @app.get("/health")
    async def health():
        import torch

        return {
            "status": "ok",
            "engine": "jevinf",
            "version": __version__,
            "architecture": service.arch,
            "device": str(service.predictor.device),
            "checkpoint": str(service.predictor.root),
            "default_strategy": service.default_strategy,
            "default_max_rows": service.default_max_rows,
            "limits": {"states": MAX_STATES, "questions": MAX_QUESTIONS,
                       "paths": MAX_PATHS, "body_bytes": MAX_BODY_BYTES},
            "enforce_limits": service.enforce_limits,
            "calls": service.calls,
            "auth": "required" if service.api_key else "open",
            "jev_compat": {
                "endpoints": ["POST /v1/systemone", "GET /v1/models"],
                "alias": ALIAS_JEV,
                "contract": "official OpenAPI 3.1 (api.typesafe.ai/openapi.json) + typesafe-sdk 0.7.0",
                "confidence_formula": CONFIDENCE_FORMULA,
                "backend_limits": {
                    "path_tokens": 512,
                    "choice_options": "2-255",
                    "score_levels": "2-10",
                },
            },
            "mps_allocated_bytes": (torch.mps.current_allocated_memory()
                                    if service.predictor.device.type == "mps" else None),
            "knobs": HEADER_KNOBS,
        }

    bearer = HTTPBearer(auto_error=False)

    async def _check_auth(creds: HTTPAuthorizationCredentials | None) -> None:
        if service.api_key is None:
            return  # auth is off by default (loopback service); only --api-key forces it
        if creds is None or creds.credentials != service.api_key:
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def system_one_endpoint(req: SystemOneRequest, response: Response,
                                  creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
        """Jev-compatible endpoint. Contract in jev_api.py (first-hand sources: official OpenAPI 3.1 + official SDK)."""
        await _check_auth(creds)
        try:
            body, ex = await run_in_threadpool(service.system_one, req)
        except JevRequestError as err:
            # in the Jev contract, validation / unserviceable errors are all 422, with FastAPI's {"detail":[...]} body shape
            return JSONResponse(
                {"detail": [{"type": "value_error", "loc": ["body"], "msg": str(err)}]},
                status_code=422,
            )
        except Exception as exc:
            return JSONResponse(
                {"detail": [{"type": "internal_error", "loc": ["body"],
                             "msg": f"{type(exc).__name__}: {exc}"}]},
                status_code=500,
            )
        response.headers["x-typesafe-request-id"] = uuid.uuid4().hex
        for k, v in (
            ("x-jevinf-strategy", str(ex.get("strategy"))),
            ("x-jevinf-forwards", str(ex.get("forwards"))),
            ("x-jevinf-computed-tokens", str(ex.get("computed_tokens"))),
            ("x-jevinf-padded-tokens", str(ex.get("padded_tokens"))),
            ("x-jevinf-path-tokens", str(ex.get("baseline_tokens"))),
            ("x-jevinf-path-limit", "512"),
            ("x-jevinf-confidence-formula", CONFIDENCE_FORMULA),
            ("x-jevinf-wall-ms", f"{ex.get('wall_s', 0.0) * 1000:.1f}"),
        ):
            response.headers[k] = v
        return body

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def models_endpoint(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
        await _check_auth(creds)
        return model_list()

    @app.post("/api/evaluate")
    async def evaluate(request: Request):
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return JSONResponse(
                {"error": "PayloadTooLarge",
                 "message": f"body {len(raw)} B > {MAX_BODY_BYTES} B"},
                status_code=413,
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": type(exc).__name__, "message": str(exc)},
                                status_code=400)

        hdr = request.headers
        try:
            temperature = float(hdr.get("x-jevinf-temperature", 1.0))
        except ValueError:
            return JSONResponse({"error": "BadTemperature",
                                 "message": "x-jevinf-temperature must be numeric"},
                                status_code=400)
        rows_raw = hdr.get("x-jevinf-max-rows")
        try:
            max_rows = int(rows_raw) if rows_raw else None
        except ValueError:
            return JSONResponse({"error": "BadMaxRows",
                                 "message": "x-jevinf-max-rows must be an integer"},
                                status_code=400)

        try:
            # inference is blocking, so hand it to the threadpool to avoid stalling the event loop (the lock guarantees serialization)
            body, ex = await run_in_threadpool(
                service.evaluate, payload,
                strategy=hdr.get("x-jevinf-strategy"),
                max_rows=max_rows,
                temperature=temperature,
            )
        except ApiError as err:
            return JSONResponse({"error": err.kind, "message": err.message},
                                status_code=err.status)
        except Exception as exc:  # don't swallow: keep the type and message for easier diffing
            return JSONResponse({"error": type(exc).__name__, "message": str(exc)},
                                status_code=500)

        headers = {
            "x-jevinf-strategy": str(ex.get("strategy")),
            "x-jevinf-forwards": str(ex.get("forwards")),
            "x-jevinf-computed-tokens": str(ex.get("computed_tokens")),
            "x-jevinf-paths": str(ex.get("paths")),
            "x-jevinf-state-kv-bytes": str(ex.get("state_kv_bytes")),
            "x-jevinf-wall-ms": f"{ex.get('wall_s', 0.0) * 1000:.1f}",
            "x-jevinf-cold": "1",  # no cross-request cache today, so it is always a cold compute
        }
        return JSONResponse(body, headers=headers)

    return app
