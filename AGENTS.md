# AGENTS.md

Working notes for agents and contributors. `README.md` is the user-facing document; this file is the
operational contract: how to build, how to verify, what the code is not allowed to break, and where
the evidence for each claim lives. Read [docs/internals.md](docs/internals.md) before touching the
engine.

## What this project is

`jevinf` is an inference engine for decision models of the Jev kind. Instead of recomputing a full
`[state] + [question] + [candidate] + eos` path per candidate, it splits the path into segments and
computes each prefix once. On top of that it serves the Jev wire contract (`POST /v1/systemone`) and
an engine-native debug endpoint (`POST /api/evaluate`).

Measured result: **2.57×** on a single API-sized request, **2.27×** end to end on the dev split, at
**100% argmax agreement**. Numbers and the shape dependence behind them: [docs/strategies.md](docs/strategies.md).

Three model families are wired up, dispatched on `arch.arrangement` — never on the model name:

| `--arch` | Backbone | Arrangement | Engine |
|---|---|---|---|
| `nanojev` (default) | Qwen3-0.6B causal decoder + trained decision head | `three-stage` | [engine.py](src/jevinf/engine.py) |
| `decider-2b` | Qwen3.5-2B, 3 of 4 layers linear attention (KV + recurrent) | `state-fork` | [decider.py](src/jevinf/decider.py) |
| `laya` | ModernBERT-large encoder + decision head | `single-path` | [laya.py](src/jevinf/laya.py) |

## Environment and setup

Requires Python ≥3.14 and [uv](https://docs.astral.sh/uv/). `.venv` is not committed; create it:

```bash
uv sync                       # builds .venv from uv.lock
uv run jevinf --help
```

**Platform constraint.** `torch-mps` (Apple silicon) and `torch-cuda` (NVIDIA) are implemented;
`torch-cpu` and `torch-rocm` are declared in [backend.py](src/jevinf/backend.py) but refuse to run, by
design — an unimplemented backend must fail loudly rather than quietly land somewhere else. `resolve()`
also checks that the machine actually has the device, so a wired-up backend on the wrong hardware
reports that reason instead of a raw torch error. Do not "fix" the CPU hole as a convenience; that is a
deliberate scope boundary.

Keep the project and `.venv` on a filesystem with symlinks and POSIX permissions (ExFAT breaks uv).
Keep build artifacts on a normal local disk.

## Inputs that are not in the repo

Neither the weights nor the evaluation split is committed. Every command needs them:

- `-m/--model <dir>` — checkpoint directory, e.g. `models/NanoJev`, `models/decider-2b`, `models/laya`
- `--split <jsonl>` — evaluation split, e.g. `data/dev.jsonl` (rows: `id`, `state`, `questions`, plus
  `teacher` gold that [upstream.py](src/jevinf/upstream.py) trims away before inference)

`models/` and `data/` are ignored at the repository root, so nothing fetched here can be committed.
Fetch the published artifacts rather than inventing fixtures — every agreement number in
[docs/](docs/) is measured against the real split, so a synthesised one proves nothing.

Nothing weight-like may be committed, and not only under those two directories — [.gitignore](.gitignore)
also blocks the weight formats (`*.safetensors`, `*.gguf`, `*.pt`, …) and the checkpoint accessory
filenames (`tokenizer.json`, `chat_template.jinja`, …) wherever they land, because the snapshotter
refuses only files over 1 MiB and would otherwise commit a 313 KiB tokenizer silently. Audit the whole
invariant with

```bash
git ls-files | git check-ignore --stdin --no-index     # must print nothing
```

which reports any tracked path matching any ignore rule. Measured 2026-09-20: 98 blobs totalling
996 KB, largest 128 KB (`uv.lock`) — this repository stores no weights and no checkpoint accessories.

**NanoJev stage 2.** The root of the published checkpoint is stage 2, and that is the pair these docs
were measured on; `stage1/` is a different model, whose dev split is 184 states / 552 questions rather
than 120 / 360. Identity, verified 2026-09-20 — the LFS tensors against the dataset manifest's declared
digests, the rest byte for byte against the model repo:

| File | sha256 |
|---|---|
| `models/NanoJev/best.safetensors` | `fff62d1412685c1714eaa386acb603f9690371fb3cc8ad03dc41319302597c28` |
| `models/NanoJev/config.json` | `8139aff38992e200c92de038169b1924412dc7a73c72fc90d86d3db4734fc6c1` |
| `models/NanoJev/backbone_config/config.json` | `30c011854471509747858ac07ffb8a4dab2bc6c04035b184729b69500c557c54` |
| `models/NanoJev/tokenizer/tokenizer.json` | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |
| `models/NanoJev/tokenizer/tokenizer_config.json` | `1cc816812993bff176eb4f7495433b736f06fba9b6e7b05cac7b4a1780650c95` |
| `models/NanoJev/tokenizer/chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `data/dev.jsonl` (320394 B) | `e7fb075f4d5a27f2c7675dabd6404c1b0faff4d8615a18b5ac1f3a74ca1ff760` |

```bash
uv run hf download C-Tianyu/NanoJev --revision 4a19595eada0857133c0d2be024f879a4077054b \
  config.json best.safetensors backbone_config/config.json \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json tokenizer/chat_template.jinja \
  --local-dir models/NanoJev
curl -sSL -o data/dev.jsonl \
  "https://huggingface.co/datasets/C-Tianyu/NanoJev-Data/resolve/87061eb91e8fc687e9b046454afdcc5551e3eff7/stage2/dev.jsonl"
```

`config.json` doubles as the training-run record: its `data_sha256` names the stage-2 training input,
and its own digest equals the dataset manifest's `training_config_sha256`. The split is 313 KiB —
**under** the 1 MiB snapshot guard — which is why `data/` is ignored at the root rather than by a rule
inside the directory. `decider-2b` and `laya` are not covered here; ask the user for those.

Upstream NanoJev is vendored at [vendor/nanojev/](vendor/nanojev/) (MIT, pinned commit `71a513b`) and
loaded **by path via importlib** in [upstream.py](src/jevinf/upstream.py). It carries one local patch,
[local-mps.patch](vendor/nanojev/local-mps.patch), that lets it run on MPS/CPU. Never copy upstream
code into `src/jevinf/`, and never edit the vendored files in place — amend the patch instead.

## Verification: there is no test suite

No pytest, no linter, no CI. Correctness is established by the checks below, in ascending cost. Run
the cheapest one that covers your change, and report the actual output.

| # | Check | Command | Needs |
|---|---|---|---|
| 1 | Decision-head math equals upstream | `uv run jevinf selfcheck -m models/NanoJev --split data/dev.jsonl --states 4` | weights |
| 2 | `decider` family matches its reference package | `uv run python scripts/decider_parity.py --model models/decider-2b` | weights + `decider` pkg |
| 3 | `laya` family matches the checkpoint's own code | `uv run python scripts/laya_parity.py --model models/laya` | weights |
| 4 | Official SDK conformance | `uv run jevinf serve -m models/NanoJev --port 8226 &` then `uv run python scripts/jev_conformance.py --base http://127.0.0.1:8226` | live server |
| 5 | Service layer forwards engine output verbatim | `uv run python scripts/api_smoke.py --split data/dev.jsonl` | live server + golden |
| 6 | Equivalence + timing on the dev split | `uv run jevinf bench -m models/NanoJev --split data/dev.jsonl --out report.json` | weights |

**On a CUDA host, add `--backend torch-cuda` to any of these.** The checkpoint and split are the same;
the numbers are not — re-measure rather than compare against the MPS figures.

Check 5 compares against an offline golden at `/tmp/golden32.json`; generate it with
`uv run jevinf eval -m models/NanoJev --split data/dev.jsonl --states 32 --out /tmp/golden32.json`, or
pass `--golden`. A missing golden only skips that comparison.

The equivalence protocol is always the same: run the upstream oracle and your candidate, then compare.

```bash
uv run jevinf oracle -m models/NanoJev --split data/dev.jsonl --out base.json
uv run jevinf eval   -m models/NanoJev --split data/dev.jsonl --out mine.json
uv run jevinf compare --reference base.json --candidate mine.json
```

**Pass criteria: `argmax_agreement` 100%, TV median 0.** The primary performance metric is
end-to-end wall-clock on the dev split; `argmax + TV/KL` is the comparison caliber. Any change to
forward arrangement must be justified with these numbers — a speedup claim without an agreement
number is not a result.

Config sweeps and probes live in `scripts/` and are facts-only or measurement-only; they are not part
of the gate. `scripts/api_smoke.py` and `scripts/jev_conformance.py` assume the `nanojev` family in
places — read the module docstring before pointing them at another `--arch`.

## Repository map

```
src/jevinf/
  cli.py        argparse entry point; subcommands selfcheck/bench/eval/oracle/compare/serve
  arch.py       the three families as structural facts; the dispatch key is `arrangement`
  backend.py    named compute backends; torch-mps + torch-cuda are wired up
  device.py     device runtime primitives the engines share (sync, allocator cache, memory)
  upstream.py   loads vendored upstream by path; predictor factory; split readers
  plan.py       reverse-derives segmented token ids from upstream `leaf_tokens`, self-checked
  head.py       decision-head math, a mirror of the latter half of upstream DecisionModel.forward
  engine.py     three-stage prefix sharing (nanojev) + strategies + stats
  decider.py    prefix → reorder_cache fork → padded suffix batch (decider-2b)
  laya.py       one sequence per question, bidirectional, nothing shared (laya)
  jev_api.py    the Jev translation layer; every interface gap is filled here
  server.py     FastAPI app: /v1/systemone, /v1/models, /api/evaluate, /health
  bench.py      compare / timing / head_selfcheck
docs/           strategies, jev-api, native-api, internals — see "Documentation" below
vendor/nanojev/ pinned upstream, unmodified except for the recorded local patch
```

## House style

- **`from __future__ import annotations`** in every module; PEP 604 unions (`X | None`) and builtin
  generics.
- **Module docstrings are the design record.** They are narrative and long, they name the primary
  sources a contract came from, and they explain *why* a thing is refused rather than silently
  handled. Follow that when you add a module.
- **Comments carry measurements.** The codebase states the number that justifies a decision
  ("measured deviation 1.58", "the MLP alone needs 5GB"). If you add a non-obvious constant, guard, or
  ordering constraint, record the measurement behind it, not an opinion.
- `_`-prefixed private helpers grouped under `# ---- section` banner comments; dataclass stats objects
  with `as_dict()`; a `_check_arch` gate and a `selfcheck` per family.
- No formatter or linter is configured, so match the surrounding file. Prose wraps near 100 columns;
  some signature lines exceed that, which is tolerated.
- **Refuse, don't guess.** Unknown architecture, unimplemented backend, over-budget state, a knob a
  family does not have — all raise or return the documented error. Never add a silent fallback,
  truncation, or default that papers over a mismatch.
- Error types matter: `SystemExit` for CLI misuse, `ValueError` for engine misuse, `JevRequestError`
  for the Jev wire, `ApiError(status, kind, message)` in the server.

## Hard invariants

These were each paid for in debugging time. [docs/internals.md](docs/internals.md) has the full list
and the evidence; the ones most likely to bite:

- **The whole pipeline runs under `torch.inference_mode()`.** Weights carry `requires_grad` by
  default; without it you build an autograd graph holding all 28 layers — measured past 20 GiB.
- **With a `past`, `attention_mask` must carry the full length** (prefix + new segment). Masking only
  the new segment puts the deviation at 1.58; it does not error, it computes wrong.
- **Extract per-row caches only via `DynamicCache.update()`.** `batch_select_indices` mutates in place
  and returns `None`; hand-assembling a `DynamicLayer` makes `get_seq_length()` read 0, so RoPE
  restarts at 0 and **the model silently computes wrong results**. There is a probe for exactly this
  in `scripts/probe_cache_update.py`.
- **A single-row prefix KV paired with multi-row input must be broadcast** (`batch_repeat_interleave`);
  otherwise you get `Sizes of tensors must match`.
- **`len(candidate_ids)` drives kmax, the width of `valid`, and the probability length.** A boolean
  question has one semantic path but two candidate ids. Using leaf count leaves a slot short.
- **Segmented token ids are reverse-derived from upstream `leaf_tokens` and cross-validated** in
  [plan.py](src/jevinf/plan.py) (three self-checks, any mismatch raises). That cross-check is the only
  thing keeping the two implementations in step — do not loosen it.
- **Never reuse a prefix cache after a suffix has run through it.** Fork first, then run each suffix
  against its own copy.
- **`max_rows` is mandatory** at high candidate counts (the Jev API allows 255 candidates/question);
  the row cap is what keeps peak memory bounded.
- **Fused strategies must count the repeated question segment**, or they look cheaper than
  `two_stage` in the report while doing the same work.
- Precision is a per-family decision: nanojev runs fp32, decider-2b's bf16 checkpoint is mapped to
  bf16 rather than doubled, laya keeps `logits.float()` to stay arithmetically identical to its fp32
  reference. Do not "unify" these.
- **Service layer:** single process + single inference thread (`threading.Lock`, uvicorn 1 worker) —
  the model is not thread-safe and multiple workers would split a future shared cache. Engine knobs
  travel **only in HTTP headers** (`x-jevinf-*`), never in the body, because the upstream validator
  rejects extra fields. Keep `Request` a module-level import in [server.py](src/jevinf/server.py) —
  with `from __future__ import annotations`, a function-local import makes FastAPI read it as a
  required query param (422).

## Adding a new model family

The pattern the three existing families follow, in order:

1. Add the family to `ARCHITECTURES` in [arch.py](src/jevinf/arch.py) with its structural facts
   (`implemented=False` until it works) and an `arrangement`.
2. Write a new module owning one `load_predictor`, one engine class with `_check_arch`, `evaluate`,
   `selfcheck`, and a stats dataclass.
3. Route it in `upstream.load_predictor` and `cli._engine`/`_evaluate` **by `arrangement`**, not by
   name.
4. Add a parity script against the family's own reference implementation; it must agree question by
   question.
5. Extend [jev_api.py](src/jevinf/jev_api.py) for whatever the contract needs that the family does not
   do natively, and update the family tables in `README.md`, [docs/jev-api.md](docs/jev-api.md) and
   [docs/native-api.md](docs/native-api.md).

## Version control

The repo is **jujutsu co-located with git** (both `.jj/` and `.git/` exist). Use `jj` for every
mutation — `git commit`, `git add`, `git reset`, `git switch`, `git rebase`, `git push` and friends
will corrupt jj state. Read-only git (`git log`, `git show`, `git diff`, `git blame`) is fine, but
prefer the `jj` equivalents.

```bash
jj st                          # status; working copy is a commit, snapshotted automatically
jj desc -m "feat: ..."         # describe the change BEFORE writing code
jj diff                        # review
jj log                         # history
jj undo                        # the first thing to reach for when something looks wrong
```

The working copy `@` is a commit. Describe it first, then edit; there is no staging area and no
`jj commit` needed. Keep one logical change per commit and leave `@` for the next task rather than
starting a new empty commit when you finish.

The `main` bookmark tracks the published line. Commit messages in history are lowercase
`type: imperative summary` on a single line (`feat: wire up laya as a third architecture`). Do not
push unless the user asks.

## Documentation to keep in sync

`README.md` is canonical and `README_CN.md` is its Chinese translation — a user-visible change lands
in both. `docs/` carries the evidence:

- [docs/strategies.md](docs/strategies.md) — forward arrangement, speedups by strategy, shape
  dependence, cost model
- [docs/jev-api.md](docs/jev-api.md) — Jev wire contract, translation layer, per-family limits
- [docs/native-api.md](docs/native-api.md) — `/api/evaluate` contract, knobs, caps
- [docs/internals.md](docs/internals.md) — design disciplines, equivalence evidence, memory facts,
  known pitfalls

A change that alters a measured number, a limit, a refusal, or a family's capability is not done until
these are updated. Quote measured values, not estimates.
