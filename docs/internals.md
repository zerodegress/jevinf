# Internals

[Back to README](../README.md)

Design disciplines, the evidence that prefix sharing is exact, measured memory facts, and the
library pitfalls that cost real debugging time.

## Disciplines

* **Upstream is vendored.** `vendor/nanojev/` holds the two MIT-licensed upstream files the oracle
  loads, pinned at upstream commit `71a513b`, plus upstream's `LICENSE`. One local patch lets them run
  on MPS/CPU devices.
* **Segmented token ids are reverse-derived from the upstream-produced `leaf_tokens`** and
  cross-checked; any mismatch raises (`plan.py` has three self-checks). That cross-check is what keeps
  the two in step.
* **The decision-head math lives in its own file and is compared against upstream**
  (`bench.head_selfcheck`), measured `max|Δlogit| = 0.0`.
* Comparison calibers: **argmax agreement + TV/KL**; the primary metric is **end-to-end wall-clock on
  the dev split**.
* The whole pipeline runs under `torch.inference_mode()` — weights carry `requires_grad` by default and
  missing it builds an autograd graph that retains all 28 layers' activations, measured to blow past
  20 GiB.

## Equivalence evidence

1. **Localization probe**: whole path in one forward vs two segmented forwards gives
   `max|Δhidden| = 0.000e+00`; masking only the new segment while supplying a past puts the deviation
   at 1.58 → **with a past, attention_mask must carry the full length**.
2. **Head self-check**: `max|Δlogit| = 0.0`, `valid` and argmax all agree.
3. **End-to-end comparison**: see `argmax_agreement` in the `result` report.

## Memory facts

* Target platform: Apple silicon Mac with unified memory, MPS backend.
* Measured KV footprint: `2 × n_kv_heads(8) × head_dim(128) × 4B × 28 layers = 224 KiB/token` (fp32);
  roughly 30 MB per state, kept resident lazily per state so the peak holds only one copy.

## Known pitfalls (transformers 5.17)

* `DynamicCache.batch_select_indices` **mutates in place** (returns `None`); calling it twice destroys
  the original cache. Hand-assembling a `DynamicLayer` instead makes `get_seq_length()` read 0 — the
  model derives `cache_position` from it, so a wrong read recomputes RoPE positions from 0 and
  **silently computes wrong results**. Stage A therefore runs one forward per state.
* A single-row prefix KV paired with multi-row input is rejected (`Sizes of tensors must match`); it
  must first be broadcast with `batch_repeat_interleave(n)`.
* `crop(positive)` is deprecated as of 5.18; use a negative value (removing from the tail).
* A boolean question has only 1 semantic path while carrying 2 candidate ids: log width, `valid`
  width, and probability length **all take `len(candidate_ids)`**.
