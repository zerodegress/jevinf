"""Device runtime primitives the engines share.

Two device-specific operations show up in every engine, and both fail *silently* when they are wrong,
which is why they live in one place instead of being re-derived per family:

* `synchronize` -- CUDA and MPS queue work asynchronously, so a `time.perf_counter()` taken around a
  forward measures how long it took to *enqueue* the kernels, not how long they ran. Every stage
  timing, and every speedup derived from one, is fiction without this call. CPU is synchronous and
  needs nothing.
* `empty_cache` -- MPS and CUDA both keep freed blocks in an allocator pool. Running a payload chunk by
  chunk, the pool grows with every chunk: on MPS the 15-chunk dev split climbed from a ~1.3GB peak to
  19.5GB and OOM'd (measured; see `bench.py`). CUDA's caching allocator holds freed blocks the same
  way, so the chunked paths call this too.

`torch.cuda.*` does not exist in a CPU-only build, hence the dispatch on `device.type` rather than a
bare call.
"""
from __future__ import annotations


def synchronize(device) -> None:
    """Block until everything queued on `device` has finished. A no-op on CPU."""
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache(device) -> None:
    """Return the allocator's cached free blocks to the driver. A no-op on CPU."""
    import torch

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def allocated_bytes(device) -> int | None:
    """Bytes held for live tensors, on the devices that report it; None where they do not."""
    import torch

    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return None
