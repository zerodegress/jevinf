# Strategies and measurements

[Back to README](../README.md)

Every timing is MPS (Apple silicon) unless it is marked CUDA. A speedup is a property of the
arrangement *and* the device, so a host of either kind must re-measure rather than quote the other's
figures.

## How the forwards are arranged

Upstream, every candidate path is `[state segment] + [question segment] + [candidate suffix] + eos`,
and the full prefill runs again for each candidate, so the prefix is recomputed every time. jevinf
splits that path into three stages, which lets each prefix be computed once:

```
Stage A  State segment      once per state  → KV resident (only the current state is kept)
Stage B  Question segment   with the state's KV as past, once per question
Stage C  Candidate suffix   candidates batched, with the question-level KV as past
```

In a causal decoder a token's hidden state depends only on the tokens before it, which makes
"compute the prefix once, then continue into the suffix" agree with "compute the entire path in one
pass" to `max|Δhidden| = 0.000e+00` (see [Internals](internals.md#equivalence-evidence)).

**Counts are single-request latency, within the API cap of 32 states / 96 questions / 256 paths**,
which is what a client actually feels. Engine efficiency is identical under both calibers
(41 ms/path); the difference comes entirely from the baseline's chunking efficiency.

**Single request (32 states / 96 questions / 245 paths):**

| Strategy | wall | speedup | forwards | computed tokens | agreement |
|---|---|---|---|---|---|
| upstream baseline | 25.80 s | 1.00× | 1 | 43,164 | — |
| **`fused_state` (default)** | **10.06 s** | **2.57×** | 36 | 15,807 | 100.00% |
| `two_stage` | 11.37 s | 2.27× | 196 | 10,731 | 100.00% |

**Full dev split (120 states / 360 questions / 898 paths):** baseline 84.2 s | `fused_state` 37.0 s →
2.27× | `two_stage` 40.9 s → 2.06× | `per_question` 77.5 s → 1.05×.

**The same dev split on CUDA** (fp32, 2026-09-20): baseline 35.93 s |
`fused_state` 17.36 s → 2.07× | `fused_q` 19.71 s → 1.82× | `two_stage` 24.44 s → 1.47×, agreement
100.00% on all three (TV median 7.5e-08 … 8.6e-08, KL mean ≤ 2.6e-09). Every strategy still beats the
baseline, `fused_state` still leads and `two_stage` still trails — but the magnitude does not carry
over: `fused_state` is 2.27× on MPS and 2.07× here. (That run measured `fused_q` in place of
`per_question`.)

**Single request on CUDA**, the same shape as the table above (32 states / 96 questions / 245 paths):
baseline 9.30 s | `fused_state` 4.33 s → 2.15× | `fused_q` 5.00 s → 1.86× | `two_stage` 6.30 s →
1.48×, agreement 100.00%. `two_stage` trails furthest here; its 360 stage-B forwards account for
2.31 s of its 5.91 s wall, where `fused_state` runs no stage B at all.

**The speedup depends strongly on shape** (8-shape sweep, agreement always 100%): from 1.98× for
single-question states up to 5.69× for long states. The larger the reuse radius (longer state
segment, more questions per state), the higher the speedup; **once the candidate count grows,
`two_stage` pulls ahead** (16 candidates: 3.54× vs 3.45×; 255 candidates: 2.46× vs 1.76×).

⚠️ **Pick the strategy per shape**: with 32 states of short segments, `two_stage` scores only
**0.73×** — slower than leaving the forwards alone. Short states drive prefix-reuse gains toward
zero, while the two extra small forwards per question outweigh the savings. Choosing a strategy
requires reading candidate count, questions per state, and state segment length together.

**Cost model**: the main axis is the **number of tokens actually computed** (~0.5–0.6 ms/token).
Small forwards cost real time: moving Stage C from one batch per question to one batch per state
(360 → 120 forwards) is **worth only 1%**, while `two_stage` computes 32% fewer tokens yet ends up
slower because it adds 360 tiny "1 row × 29 tokens" forwards (~45 ms each).
→ **Save tokens in whole batches.**

`max_rows` (default 64) caps the sub-batch size: the Jev API allows up to 255 candidates per
question, far from the dev split's 1–4. A 259-path payload was verified to split correctly into 5
sub-batches with agreement still at 100%.
