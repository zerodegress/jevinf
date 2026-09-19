# jevinf

An inference engine for decision models of the Jev kind: each candidate path runs as segmented
forwards with prefix reuse, and the Jev wire contract is served on top. NanoJev is the backend wired
up today.

> Chinese version: [README_CN.md](README_CN.md)

Measured throughput: **2.57×** on a single API-sized request (25.80 s → 10.06 s) and **2.27×** end to
end on the dev split (84.2 s → 37.0 s), at 100% argmax agreement. How the forwards are arranged, and
the shape dependence behind those numbers, is in
[Strategies and measurements](docs/strategies.md).

## Quick start

Every command that loads weights takes `-m/--model`, the checkpoint directory; commands that read
the evaluation split also take `--split`, a jsonl.

```bash
uv sync
uv run jevinf selfcheck -m models/NanoJev --split data/dev.jsonl --states 4
uv run jevinf bench -m models/NanoJev --split data/dev.jsonl --states 8
uv run jevinf bench -m models/NanoJev --split data/dev.jsonl --out report.json
uv run jevinf eval -m models/NanoJev --input request.json
uv run jevinf oracle -m models/NanoJev --split data/dev.jsonl --out base.json
uv run jevinf compare --reference base.json --candidate mine.json
```

Two service entry points, both served by `jevinf serve`:

```bash
uv run jevinf serve -m models/NanoJev --port 8226            # /v1/systemone + /api/evaluate
uv run jevinf serve -m models/NanoJev --api-key sk-local     # enforce HTTPBearer on /v1/*
```

```bash
# Point any Jev client at it and it just works (the official SDK included)
TYPESAFE_BASE_URL=http://127.0.0.1:8226 TYPESAFE_API_KEY=sk-any uv run python your_jev_client.py
```

## Development environment

Requires Python ≥3.14 and [uv](https://docs.astral.sh/uv/). Dependencies are declared in
`pyproject.toml` and pinned in `uv.lock` (torch, transformers, fastapi, uvicorn, safetensors;
`typesafe-sdk` is a dev-only dependency for conformance testing).

```bash
uv sync                # build .venv from uv.lock
uv run jevinf --help
```

**Keep the project and `.venv` on a filesystem that supports symlinks and POSIX permissions.** A uv
`.venv` on a filesystem that lacks them (ExFAT, for instance) breaks; keep build artifacts on a
normal local disk. The prototype is tuned for Apple silicon (MPS) with unified memory.

Backends are chosen with `--backend`:

| Backend | Runs on | Status |
|---|---|---|
| `torch-mps` | Apple silicon through Metal | default |
| `torch-cpu` | CPU only | declared |
| `torch-cuda` | NVIDIA GPUs through CUDA | declared |
| `torch-rocm` | AMD GPUs through ROCm | declared |

Only `torch-mps` is wired up; the other three refuse to run.

Checks, cheapest first:

```bash
uv run jevinf selfcheck -m models/NanoJev --split data/dev.jsonl --states 4   # decision head only
uv run jevinf serve -m models/NanoJev --port 8226 &                          # needs a live server
uv run python scripts/jev_conformance.py --base http://127.0.0.1:8226        # official SDK conformance
uv run python scripts/api_smoke.py --split data/dev.jsonl                    # golden cross-check
```

## Documentation

- [Strategies and measurements](docs/strategies.md) — how the forwards are arranged, speedups by
  strategy, shape dependence, cost model
- [Jev-compatible service layer](docs/jev-api.md) — wire contract, translation layer, conformance
- [Native debug entry point](docs/native-api.md) — `/api/evaluate` contract, knobs, limits
- [Internals](docs/internals.md) — design disciplines, equivalence evidence, environment facts, known pitfalls
