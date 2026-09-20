# jevinf

An inference engine for decision models of the Jev kind: each candidate path runs as segmented
forwards with prefix reuse, and the Jev wire contract is served on top. NanoJev is the backend wired
up today.

> Chinese version: [README_CN.md](README_CN.md)

Measured throughput on MPS: **2.57×** on a single API-sized request (25.80 s → 10.06 s) and **2.27×**
end to end on the dev split (84.2 s → 37.0 s), at 100% argmax agreement. The same two shapes measure
**2.15×** and **2.07×** on CUDA — the speedup comes from the arrangement, not the device. How the
forwards are arranged, and the shape dependence behind those numbers, is in
[Strategies and measurements](docs/strategies.md).

## Quick start

Every command that loads weights takes `-m/--model`, the checkpoint directory; commands that read
the evaluation split also take `--split`, a jsonl.

Neither is in this repository. [AGENTS.md](AGENTS.md#inputs-that-are-not-in-the-repo) pins the exact
download commands and the sha256 of every file they produce.

```bash
uv sync
uv run jevinf selfcheck -m models/NanoJev --split data/dev.jsonl --states 4
uv run jevinf bench -m models/NanoJev --split data/dev.jsonl --states 8
uv run jevinf bench -m models/NanoJev --split data/dev.jsonl --out data/report.json
uv run jevinf eval -m models/NanoJev --input data/request.json
uv run jevinf oracle -m models/NanoJev --split data/dev.jsonl --out data/base.json
uv run jevinf eval   -m models/NanoJev --split data/dev.jsonl --out data/mine.json
uv run jevinf compare --reference data/base.json --candidate data/mine.json
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

**Optional: faster `decider-2b`.** Three of its four layers are linear attention, and transformers
falls back to reference PyTorch kernels for them unless `flash-linear-attention` is installed. It is
not a runtime dependency:

```bash
uv sync --group decider-kernels
```

[AGENTS.md](AGENTS.md#environment-and-setup) records why `causal-conv1d` is not in that group.

**Keep the project and `.venv` on a filesystem that supports symlinks and POSIX permissions.** A uv
`.venv` on a filesystem that lacks them (ExFAT, for instance) breaks; keep build artifacts on a
normal local disk. The prototype is tuned for Apple silicon (MPS) with unified memory.

Backends are chosen with `--backend`:

| Backend | Runs on | Status |
|---|---|---|
| `torch-mps` | Apple silicon through Metal | default |
| `torch-cpu` | CPU only | declared |
| `torch-cuda` | NVIDIA GPUs through CUDA | wired up |
| `torch-rocm` | AMD GPUs through ROCm | declared |

`torch-mps` and `torch-cuda` are wired up; `torch-cpu` and `torch-rocm` refuse to run. Every measured
number in these docs was taken on MPS — a CUDA host has to re-measure them for itself.

The model family is chosen with `--arch`:

| Architecture | Backbone | Status |
|---|---|---|
| `nanojev` | Qwen3-0.6B causal decoder plus a trained decision head | default, three-stage prefix sharing |
| `decider-2b` | Qwen3.5-2B causal decoder, 3 of every 4 layers linear attention (KV + recurrent cache) | wired up, one state prefix forked to a row per question |
| `laya` | ModernBERT-large encoder plus a decision head, one sequence per question | wired up, one row per question (`laya.py`) |

All three are wired up. `src/jevinf/arch.py` records what each
family is structurally — how attention runs, where answers are read from, what has to stay resident
between segments — because those facts are what decide which arrangement applies (`engine.py` for the
three-stage one, `decider.py` for the fork, `laya.py` for the single sequence per question).

Each family also has an adapter check against its own reference implementation, which ships beside the
weights: `scripts/decider_parity.py` against the `decider` package, `scripts/laya_parity.py` against
`rl_agent_api.RLAgent`. Both must agree with their reference question by question.

Checks, cheapest first:

```bash
uv run jevinf selfcheck -m models/NanoJev --split data/dev.jsonl --states 4   # decision head only
uv run python scripts/decider_parity.py --model models/decider-2b             # vs the decider package
uv run python scripts/laya_parity.py --model models/laya                      # vs the checkpoint's own code
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
