"""Compute backends.

`torch-mps` is the one wired up today. The other three are declared so the option surface is stable
and so an unimplemented backend refuses to run rather than quietly landing somewhere else.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """A named compute backend and the torch device name it runs on."""

    name: str
    device: str
    implemented: bool
    note: str


BACKENDS: dict[str, Backend] = {
    "torch-mps": Backend("torch-mps", "mps", True, "Apple silicon, Metal through torch MPS"),
    "torch-cpu": Backend("torch-cpu", "cpu", False, "CPU only"),
    "torch-cuda": Backend("torch-cuda", "cuda:0", False, "NVIDIA GPUs through CUDA"),
    "torch-rocm": Backend("torch-rocm", "cuda:0", False,
                          "AMD GPUs through ROCm (torch exposes them as cuda)"),
}
DEFAULT_BACKEND = "torch-mps"
NAMES: tuple[str, ...] = tuple(BACKENDS)


def resolve(name: str = DEFAULT_BACKEND) -> Backend:
    """Look a backend up, refusing the ones that are declared but not implemented."""
    backend = BACKENDS.get(name)
    if backend is None:
        raise SystemExit(f"unknown backend {name!r}; choices: {', '.join(NAMES)}")
    if not backend.implemented:
        raise SystemExit(
            f"backend {name!r} ({backend.note}) is declared but not implemented yet; "
            f"run with --backend {DEFAULT_BACKEND}"
        )
    return backend
