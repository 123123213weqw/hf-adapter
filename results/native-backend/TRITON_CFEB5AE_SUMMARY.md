# Forced-Triton cross-GPU summary (`cfeb5ae`)

All rows use code `cfeb5aeca860ce444ebb3515a20cc22f7e2b090b`,
`RWKV7_BACKEND=optimized`, `RWKV7_KERNEL_IMPL=triton`, and observed route
`native-triton-rank1-scan-v1` for supported inference requests.

| Gate | V100 | RTX 4080 | RTX 4090 |
|---|---:|---:|---:|
| Formal `lm_eval` | 48/48 | 48/48 | 48/48 |
| Batch 1/8 metric validator | Pass | Pass | Pass |
| HF ecosystem | Pass | Pass | Pass |
| Training reference fallback | Pass | Pass | Pass |
| LoRA SFT/DPO/GRPO | Pass | Pass | Pass |
| Speed job | Pass | Pass | Pass |
| FLA speed | Fused recurrent pass | Pass | Pass |
| FLA operator/state/greedy | Fused recurrent pass | Pass | Pass |
| Strict FP16 model-logit diagnostic | Fail | Fail | Fail |
| Final numerical promotion | Not promoted | Not promoted | Not promoted |

The three cards establish broad HF execution compatibility and reproducible
metric stability. They do **not** establish a clean numerical promotion for
the Triton lane: selected full-model FP16 logits exceed `max_abs=0.15`, while
finite, cosine, cache, state, and greedy checks pass. This distinction is
retained in every `final-corrected-exit-status.json`.

V100 cannot compile the pinned FLA chunk/backward implementation on sm70 with
the tested Triton toolchains; its fused-recurrent inference comparison passes.
RTX 4080 and RTX 4090 complete the pinned FLA speed job, while clean-vs-FLA
misses only the full-model logits gate.

Bundles:

- [`v100-cfeb5ae-triton`](v100-cfeb5ae-triton/README.md)
- [`4080-cfeb5ae-triton`](4080-cfeb5ae-triton/README.md)
- [`4090-cfeb5ae-triton`](4090-cfeb5ae-triton/README.md)

Kernel wheel SHA256:
`544b08225351d9658a1941c260a70710d47bb677d2976d730cec9e4f34771560`.
