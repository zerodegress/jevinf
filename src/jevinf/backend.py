"""Compute backends.

`torch-mps` and `torch-cuda` are wired up today. `torch-cpu` and `torch-rocm` are declared so the
option surface is stable and so an unimplemented backend refuses to run rather than quietly landing
somewhere else.
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
    "torch-cuda": Backend("torch-cuda", "cuda:0", True, "NVIDIA GPUs through CUDA"),
    "torch-rocm": Backend("torch-rocm", "cuda:0", False,
                          "AMD GPUs through ROCm (torch exposes them as cuda)"),
}
DEFAULT_BACKEND = "torch-mps"
NAMES: tuple[str, ...] = tuple(BACKENDS)


def _require_device(backend: Backend) -> None:
    """Refuse an implemented backend whose device this machine does not actually have.

    `implemented` is a statement about this repository -- the code path exists. This is a statement
    about the machine -- the hardware is there. Keeping the two apart stops "we wired it up" from
    being read as "it will run here", which is the confusion the unimplemented backends exist to
    prevent.
    """
    import torch

    if backend.device.startswith("cuda"):
        present = torch.cuda.is_available()
    elif backend.device == "mps":
        present = getattr(torch.backends.mps, "is_available", lambda: False)()
    else:  # CPU needs nothing from a driver
        present = True
    if not present:
        raise SystemExit(
            f"backend {backend.name!r} is implemented but torch {torch.__version__} cannot see a "
            f"{backend.device} device on this machine; check the driver, or pick another --backend"
        )


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
    _require_device(backend)
    return backend
