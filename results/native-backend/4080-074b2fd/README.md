# RTX 4080 optional-backend acceptance

- Source: `074b2fdc1707216fe14ef458e9ca5aff01b41874`
- GPU: NVIDIA GeForce RTX 4080 (16 GB)
- PyTorch: 2.11.0+cu130
- Validation: FP16 optimized 0.1B/0.4B/1.5B passed; BF16 and FP32 auto-fallback passed.
- Coverage: B=1/4, T=1/17/128, padded logits/cache, teacher-forced cached decode, 64-token greedy, and gradient-checkpointed backward.
- Numerical result: promoted FP16 CUDA-graph recurrence is bit-exact against reference in the operator and model matrix.

## 0.4B paired speed (FP16, CUDA graph and torch.compile disabled at model level)

| case | reference ms | auto ms | speedup |
|---|---:|---:|---:|
| `cached_decode_b1` | 1403.488 | 1332.146 | 1.054x |
| `cached_decode_b4` | 1434.035 | 1362.752 | 1.052x |
| `generation_prefill_b1_t1` | 22.661 | 21.395 | 1.059x |
| `generation_prefill_b1_t128` | 228.465 | 92.338 | 2.474x |
| `generation_prefill_b1_t2048` | 3394.582 | 1207.065 | 2.812x |
| `generation_prefill_b1_t512` | 863.696 | 307.959 | 2.805x |
| `generation_prefill_b4_t128` | 296.324 | 151.747 | 1.953x |
| `generation_prefill_b4_t512` | 1132.100 | 565.599 | 2.002x |

Raw JSON files in this directory are the source of truth.
