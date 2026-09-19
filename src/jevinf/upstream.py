"""Load the upstream NanoJev code -- vendored under vendor/nanojev/ (MIT; see its README).

Which weights and which data to use comes from the caller: every command takes -m/--model for the
checkpoint directory and --split for the evaluation split.
The upstream code is the vendored copy under vendor/nanojev/, which carries one local patch allowing
MPS/CPU devices -- numerically neutral in fp32, the only precision this prototype runs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .arch import DEFAULT_ARCH, resolve as resolve_arch
from .backend import DEFAULT_BACKEND, resolve as resolve_backend

DEFAULT_SRC = Path(__file__).resolve().parents[2] / "vendor" / "nanojev"


def _load(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not path.is_file():
        raise FileNotFoundError(
            f"upstream file not found: {path} "
            f"(the vendored copy under vendor/nanojev/ looks incomplete)"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def predict_module(src: Path = DEFAULT_SRC):
    """Upstream inference entry module (prepare_examples / validate_request / DecisionPredictor)."""
    return _load("jevinf_upstream_predict", Path(src) / "scripts" / "predict_toy_decisions.py")


def decision_model_class(src: Path = DEFAULT_SRC):
    """Upstream DecisionModel definition (loaded via upstream's own importlib path)."""
    return predict_module(src).load_decision_model_class()


def validate_request(payload):
    """Upstream payload validator (strict: a state may only contain id/state/questions)."""
    return predict_module().validate_request(payload)


def prepare_examples(payload, tokenizer, max_length):
    """Upstream segmented-encode entry point -- the authoritative source of segmented token ids."""
    return predict_module().prepare_examples(payload, tokenizer, max_length)


def load_predictor(checkpoint_dir, backend: str = DEFAULT_BACKEND, arch: str = DEFAULT_ARCH,
                   precision="fp32", src: Path = DEFAULT_SRC):
    """Construct the predictor for the requested architecture. It is this prototype's oracle and weight carrier.

    The architecture tag is resolved here and stamped on the predictor, so the engine can tell which
    family it is arranging without being told twice. Which loader runs is read off `arrangement`, not
    off a branch on the name.
    """
    spec = resolve_arch(arch)
    if spec.arrangement == "state-fork":
        from .decider import load_predictor as load_decider

        # This checkpoint is bf16 and its readout is bf16, so fp32 would double the resident weights for
        # nothing; the precision argument is accepted and mapped rather than rejected.
        return load_decider(checkpoint_dir, backend=backend, arch=spec,
                            precision=precision if precision in ("bf16", "fp16") else "bf16")
    module = predict_module(src)
    predictor = module.DecisionPredictor(
        str(checkpoint_dir), device_name=resolve_backend(backend).device, precision=precision
    )
    predictor.arch = spec
    return predictor


def read_split(path, limit_states: int | None = None) -> dict:
    """Read the upstream dev split (jsonl) and trim it to {states:[{id,state,questions}]}.

    Upstream data rows contain gold/teacher, while the inference entry point only
    accepts these three keys, so they are trimmed away here.
    """
    import json

    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit_states is not None:
        rows = rows[:limit_states]
    return {
        "states": [
            {"id": r["id"], "state": r["state"], "questions": r["questions"]} for r in rows
        ]
    }


def split_truth(path, limit_states: int | None = None) -> dict:
    """Teacher native probabilities (reference) from the same split, used as the comparison reference."""
    import json

    path = Path(path)
    out = {}
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit_states is not None and i >= limit_states:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for qid, probs in row.get("teacher", {}).get("native_probs", {}).items():
                out[f"{row['id']}:{qid}"] = probs
    return out
