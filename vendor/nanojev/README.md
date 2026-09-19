# Vendored upstream: NanoJev

Two upstream files, copied verbatim so that jevinf's oracle does not depend on a local working copy.

- Upstream: `https://github.com/TianyuCodings/NanoJev`
- Pinned commit: `71a513bb0163b5634467842b523ee0c0ed6fb1c7`
- License: MIT — see [LICENSE](LICENSE), "Copyright (c) 2026 OpenJev contributors"
- Vendored: 2026-09-19

| File | Role | sha256 |
|---|---|---|
| `scripts/predict_toy_decisions.py` | inference entry point: `validate_request`, `prepare_examples`, `DecisionPredictor` | `758f8838ea96663671c59444b83370491a42b37cedd516ee03e9109a453206a3` |
| `scripts/train_toy_decisions.py` | defines the `DecisionModel` class, loaded by `predict_toy_decisions.py` itself | `4f39babd5575e43d7acf3eca357329a93c60041f334bc2bd7424929bf9c0da86` |

## Local modification

`predict_toy_decisions.py` carries one local patch, kept as [local-mps.patch](local-mps.patch):

* upstream hard-requires a CUDA device and raises otherwise; the patch also accepts `mps` and `cpu`,
  keeping the original CUDA checks inside the `cuda` branch, and casts weights to bf16 directly on
  non-CUDA devices instead of using autocast.
* the patch is numerically neutral in fp32, the only precision this prototype runs.

`train_toy_decisions.py` is unmodified relative to the pinned commit.
