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
    ALIAS_LOCAL,
    NANO_CHOICE_MAX,
    NANO_MAX_PATH_TOKENS,
    NANO_SCORE_MAX,
    NANO_SCORE_MIN,
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
    "x-jevinf-temperature": "softmax temperature (default: the loaded model's own value); same meaning as upstream predict(temperature=)",
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


def family_facts(predictor, max_state_tokens: int | None = None, crop_state: bool = False) -> dict:
    """What the loaded architecture can serve: its own name, limits, knobs and default temperature.

    The two families differ in ways the service has to state rather than assume: which knob applies
    (a strategy tier only exists for the three-stage arrangement), how large a request may be, and
    what temperature a request gets when the caller does not ask for one. The last one matters:
    nanojev was trained to be read at 1.0, while decider ships a fitted value and silently reading it
    at 1.0 would de-calibrate every answer.
    """
    arch = predictor.arch
    shared_limits = {"states": MAX_STATES, "questions": MAX_QUESTIONS}
    if arch.arrangement == "single-path":
        return {
            "arrangement": arch.arrangement,
            "model_name": "laya",
            # No single temperature exists for this family: see the refusal in EvaluateService.
            "temperature_default": None,
            "precision": predictor.storage,
            "crop_state": bool(crop_state),
            "limits": {**shared_limits, "paths": None},
            "backend_limits": {
                "sequence_tokens": predictor.max_len,
                "head_tokens": predictor.head_max_len,
                "choice_options": f"2-{NANO_CHOICE_MAX}",
                "score_levels": f"{NANO_SCORE_MIN}-{NANO_SCORE_MAX}",
            },
            "knobs": {k: v for k, v in HEADER_KNOBS.items()
                      if k not in ("x-jevinf-strategy", "x-jevinf-temperature")},
            "description": (
                f"Local decision model ({arch.name}): ModernBERT-large encoder (bidirectional, 421M) plus a "
                f"from-scratch decision head. Each question is one sequence of at most {predictor.max_len} "
                f"tokens (question and options within {predictor.head_max_len}) with one [MASK] marker per "
                f"option, scored at its own marker and read at a fitted per-cardinality temperature, so "
                f"nothing is shared between questions. "
                + ("A state that does not fit is cropped and every answer says so. " if crop_state
                   else "A state that does not fit is refused, not cropped. ")
                + f"choice 2-{NANO_CHOICE_MAX} options, score {NANO_SCORE_MIN}-{NANO_SCORE_MAX} levels, noul "
                f"P(true). Note the >=11-option choice bucket is fitted at "
                f"{predictor.temperature_by_options.get('choice:11+')}, below 1.0, i.e. it sharpens its "
                f"distribution rather than reporting the odds as measured."
            ),
        }
    if arch.arrangement == "state-fork":
        return {
            "arrangement": arch.arrangement,
            "model_name": "decider-2b",
            "temperature_default": float(predictor.temperature),
            "precision": "bfloat16",
            "limits": {**shared_limits, "paths": None},
            "backend_limits": {
                "state_tokens": int(max_state_tokens or predictor.max_state_tokens),
                "choice_options": f"2-{predictor.max_options}",
                "score_levels": f"{NANO_SCORE_MIN}-{NANO_SCORE_MAX}",
                "temperature": predictor.temperature,
            },
            "knobs": {k: v for k, v in HEADER_KNOBS.items() if k != "x-jevinf-strategy"},
            "description": (
                f"Local decision model ({arch.name}): Qwen3.5-2B causal backbone with 18 of 24 layers "
                f"linear attention, so its cache is part KV and part recurrent state; one forward pass "
                f"scores every question. The state prefix is computed once and forked to one row per "
                f"question. State budget {int(max_state_tokens or predictor.max_state_tokens)} tokens, "
                f"and a state over it is refused rather than truncated; choice "
                f"2-{predictor.max_options} options, score "
                f"{NANO_SCORE_MIN}-{NANO_SCORE_MAX} levels, readout temperature {predictor.temperature}."
            ),
        }
    return {
        "arrangement": arch.arrangement,
        "model_name": ALIAS_LOCAL,
        "temperature_default": 1.0,
        "precision": "fp32",
        "limits": {**shared_limits, "paths": MAX_PATHS},
        "backend_limits": {
            "path_tokens": NANO_MAX_PATH_TOKENS,
            "choice_options": f"2-{NANO_CHOICE_MAX}",
            "score_levels": f"{NANO_SCORE_MIN}-{NANO_SCORE_MAX}",
        },
        "knobs": HEADER_KNOBS,
        "description": (
            "The underlying local decision model: deterministic, no autoregressive decoding, one "
            f"batched forward pass returns a probability distribution per question. Path limit "
            f"{NANO_MAX_PATH_TOKENS} tokens, choice 2-{NANO_CHOICE_MAX} options, score "
            f"{NANO_SCORE_MIN}-{NANO_SCORE_MAX} levels."
        ),
    }


def stat_headers(ex: dict, facts: dict) -> dict:
    """Response stat headers, per family.

    Only the numbers the loaded family actually reports are sent: a decider response has no strategy
    tier or path count, and a nanojev response has no layout or fork statistics. Sending `None` for a
    statistic that does not apply would be worse than sending nothing, because a client would read it
    as data.
    """
    out = {
        "x-jevinf-architecture": str(ex.get("architecture")),
        "x-jevinf-wall-ms": f"{ex.get('wall_s', 0.0) * 1000:.1f}",
    }
    if facts["arrangement"] == "single-path":
        out.update({
            "x-jevinf-layout": str(ex.get("layout")),
            "x-jevinf-rows": str(ex.get("rows")),
            "x-jevinf-prefix-sharing": "0",
            "x-jevinf-sequence-limit": str(facts["backend_limits"]["sequence_tokens"]),
            "x-jevinf-crop-state": "1" if ex.get("crop_state") else "0",
        })
    elif facts["arrangement"] == "state-fork":
        out.update({
            "x-jevinf-layout": str(ex.get("layout")),
            "x-jevinf-rows": str(ex.get("rows")),
            "x-jevinf-prefix-sharing": "1" if ex.get("prefix_sharing") else "0",
            "x-jevinf-prefix-len": str(ex.get("prefix_len")),
            "x-jevinf-state-limit": str(facts["backend_limits"]["state_tokens"]),
        })
    else:
        out.update({
            "x-jevinf-strategy": str(ex.get("strategy")),
            "x-jevinf-paths": str(ex.get("paths")),
            "x-jevinf-state-kv-bytes": str(ex.get("state_kv_bytes")),
            "x-jevinf-path-limit": str(facts["backend_limits"]["path_tokens"]),
        })
    for key, name in (("forwards", "x-jevinf-forwards"),
                      ("computed_tokens", "x-jevinf-computed-tokens"),
                      ("padded_tokens", "x-jevinf-padded-tokens"),
                      ("baseline_tokens", "x-jevinf-path-tokens")):
        if ex.get(key) is not None:
            out[name] = str(ex[key])
    return out


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
                 max_rows: int = 64, max_state_tokens: int | None = None, crop_state: bool = False,
                 enforce_limits: bool = True, api_key: str | None = None):
        self.predictor = upstream.load_predictor(checkpoint_dir, backend=backend, arch=arch,
                                                 crop_state=crop_state)
        self.arch = self.predictor.arch.name
        if max_state_tokens and self.predictor.arch.arrangement != "state-fork":
            raise ValueError(
                f"--max-state-tokens only applies to the state-fork family; {self.arch!r} bounds a "
                f"request by its candidate-path cap ({self.predictor.limit} tokens) instead"
            )
        if crop_state and self.predictor.arch.arrangement != "single-path":
            raise ValueError(
                f"--crop-state only applies to the single-path family; {self.arch!r} has no fixed "
                f"sequence budget to crop against"
            )
        self.facts = family_facts(self.predictor, max_state_tokens=max_state_tokens, crop_state=crop_state)
        self.default_strategy = strategy
        self.default_max_rows = max_rows
        self.max_state_tokens = max_state_tokens
        self.enforce_limits = enforce_limits
        self.api_key = api_key
        self.lock = threading.Lock()
        self.calls = 0
        self._engines: dict[tuple[str, int], Any] = {}

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
            if self.facts["limits"]["paths"] is not None and paths > MAX_PATHS:
                raise JevRequestError(
                    f"candidate paths {paths} > {MAX_PATHS} (local backend limit; "
                    f"Jev itself allows more, but memory runs out here first)"
                )
        with self.lock:
            self.calls += 1
            engine = self._engine(self.default_strategy, self.default_max_rows)
            started = time.perf_counter()
            try:
                out = engine.evaluate(nano_payload, temperature=self.facts["temperature_default"])
            except (ValueError, AssertionError) as exc:
                # upstream validator / paths over 512 tokens etc. → map to 422 per the Jev contract
                raise JevRequestError(str(exc)) from None
            wall = time.perf_counter() - started
        ex = dict(out["execution"])
        ex["wall_s"] = round(wall, 6)
        return to_jev_response(req, out, ex), ex

    # ------------------------------------------------------------------ engine
    def _engine(self, strategy: str, max_rows: int):
        """One engine per (strategy, rows) pair. Which arrangement is built is data, not a flag here."""
        if self.facts["arrangement"] == "single-path":
            from .laya import LayaEngine

            key = ("single-path", max_rows)
            if key not in self._engines:
                self._engines[key] = LayaEngine(self.predictor, max_rows=max_rows,
                                                crop_state=self.facts["crop_state"])
            return self._engines[key]
        if self.facts["arrangement"] == "state-fork":
            from .decider import DeciderEngine

            key = ("state-fork", max_rows)
            if key not in self._engines:
                self._engines[key] = DeciderEngine(self.predictor, max_rows=max_rows,
                                                   max_state_tokens=self.max_state_tokens)
            return self._engines[key]
        key = (strategy, max_rows)
        if key not in self._engines:
            self._engines[key] = PrefixShareEngine(
                self.predictor, strategy=strategy, max_rows=max_rows
            )
        return self._engines[key]

    # ------------------------------------------------------------------ main path
    def evaluate(self, payload: Any, *, strategy: str | None = None, max_rows: int | None = None,
                 temperature: float | None = None) -> tuple[dict, dict]:
        # 1) upstream validator (strict)
        try:
            states = upstream.validate_request(payload)
        except (ValueError, TypeError) as exc:
            raise ApiError(400, type(exc).__name__, str(exc)) from None

        # 2) limits (matching the Jev API; the family decides which of them can bite)
        questions, paths = count_paths(states)
        limits = self.facts["limits"]
        if self.enforce_limits:
            if limits["states"] is not None and len(states) > limits["states"]:
                raise ApiError(413, "LimitExceeded",
                               f"states {len(states)} > {limits['states']}")
            if limits["questions"] is not None and questions > limits["questions"]:
                raise ApiError(413, "LimitExceeded",
                               f"questions {questions} > {limits['questions']}")
            if limits["paths"] is not None and paths > limits["paths"]:
                raise ApiError(413, "LimitExceeded", f"paths {paths} > {limits['paths']}")

        # 3) knobs. A strategy tier only exists for the three-stage arrangement: asking for one on
        # another family is refused rather than silently ignored, so nobody believes they set it.
        if self.facts["arrangement"] != "three-stage":
            if strategy:
                raise ApiError(
                    400, "BadStrategy",
                    f"architecture {self.arch!r} is arranged by "
                    f"{self.facts['arrangement']!r}, not by strategy; x-jevinf-strategy does not apply",
                )
            strat = self.facts["arrangement"]
        else:
            strat = (strategy or self.default_strategy).strip()
            if strat not in STRATEGIES:
                raise ApiError(400, "BadStrategy",
                               f"strategy must be one of {'/'.join(STRATEGIES)}, got {strat!r}")
        rows = int(max_rows) if max_rows else self.default_max_rows
        if rows < 1:
            raise ApiError(400, "BadMaxRows", "max-rows must be >= 1")
        if self.facts["temperature_default"] is None:
            # This family has no single readout temperature: its calibration is per (question type,
            # option count). Substituting one scalar would silently change every probability, so a
            # request that asks for one is refused; the applied value rides along in every answer.
            if temperature is not None:
                raise ApiError(
                    400, "BadTemperature",
                    f"architecture {self.arch!r} reads its answers at per-cardinality fitted "
                    f"temperatures (temperature_by_options in the checkpoint), so a request-level "
                    f"temperature is refused; the applied value is reported as rl_agent.temperature",
                )
            temp = None
        else:
            temp = self.facts["temperature_default"] if temperature is None else float(temperature)
            if not (temp > 0):
                raise ApiError(400, "BadTemperature", "temperature must be a positive number")

        # 4) inference (single-threaded, serialized)
        with self.lock:
            self.calls += 1
            engine = self._engine(strat, rows)
            started = time.perf_counter()
            try:
                out = engine.evaluate(payload, temperature=temp)
            except ValueError as exc:
                # A family-level refusal (a state over its budget, a slot with every option masked out,
                # ...) is the client's input being out of range, not a server fault.
                raise ApiError(413, "LimitExceeded", str(exc)) from None
            wall = time.perf_counter() - started

        ex = dict(out["execution"])
        # fill in the keys upstream predict() already has, so clients written against it can read them without code changes
        ex.update({
            "precision": self.facts["precision"],
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
                **self._checkpoint_facts(),
            },
            "temperature": (
                {
                    "value": float(temp),
                    "fitted_by_this_command": False,
                    "note": ("applies the given scalar explicitly; the default is the loaded model's own "
                             "value, which for this architecture is a fitted one"),
                }
                if temp is not None else
                {
                    "value": None,
                    "fitted_by_this_command": False,
                    "note": ("this architecture has no single readout temperature: it is fitted per "
                             "question type and option count, and the value applied to each answer is "
                             "reported as rl_agent.temperature with its bucket"),
                }
            ),
            "execution": ex,
            "states": out["states"],
        }
        return body, ex

    def _checkpoint_facts(self) -> dict:
        """Provenance keys upstream predict() reports. Only filled in when the family actually knows them."""
        run = getattr(self.predictor, "run_config", None)
        if not run:
            return {}
        return {
            "base_model": run.get("model"),
            "base_revision": run.get("resolved_model_revision"),
            "set_head": run.get("set_head"),
        }


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
            "arrangement": service.facts["arrangement"],
            "device": str(service.predictor.device),
            "checkpoint": str(service.predictor.root),
            "precision": service.facts["precision"],
            "default_strategy": (service.default_strategy
                                 if service.facts["arrangement"] != "state-fork" else None),
            "default_max_rows": service.default_max_rows,
            "default_temperature": service.facts["temperature_default"],
            "crop_state": service.facts.get("crop_state"),
            "limits": {**service.facts["limits"], "body_bytes": MAX_BODY_BYTES},
            "enforce_limits": service.enforce_limits,
            "calls": service.calls,
            "auth": "required" if service.api_key else "open",
            "jev_compat": {
                "endpoints": ["POST /v1/systemone", "GET /v1/models"],
                "alias": ALIAS_JEV,
                "local_model": service.facts["model_name"],
                "contract": "official OpenAPI 3.1 (api.typesafe.ai/openapi.json) + typesafe-sdk 0.7.0",
                "confidence_formula": CONFIDENCE_FORMULA,
                "backend_limits": service.facts["backend_limits"],
            },
            "mps_allocated_bytes": (torch.mps.current_allocated_memory()
                                    if service.predictor.device.type == "mps" else None),
            "knobs": service.facts["knobs"],
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
        response.headers["x-jevinf-confidence-formula"] = CONFIDENCE_FORMULA
        for k, v in stat_headers(ex, service.facts).items():
            response.headers[k] = v
        return body

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def models_endpoint(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
        await _check_auth(creds)
        return model_list(local_name=service.facts["model_name"],
                          local_description=service.facts["description"])

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
        raw_temp = hdr.get("x-jevinf-temperature")
        temperature = None  # absent = the loaded model's own value, not a hard-coded 1.0
        if raw_temp is not None:
            try:
                temperature = float(raw_temp)
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

        headers = stat_headers(ex, service.facts)
        headers["x-jevinf-cold"] = "1"  # no cross-request cache today, so it is always a cold compute
        return JSONResponse(body, headers=headers)

    return app
