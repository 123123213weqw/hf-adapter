# RTX 4090 optional-backend acceptance

- Source: `530d935b9b95d5ec4f09d04eb164a38c8669a1ec`
- GPU: NVIDIA GeForce RTX 4090 (24 GB)
- PyTorch / CUDA: 2.5.1+cu124 / 12.4
- Transformers: 4.49.0
- Wheels:
  - `rwkv7_hf-0.10.0.dev0-py3-none-any.whl`: `607a5d36dd4e57fc6e9e7417f6368ab702da44da428fdef45e1d29af3086999f`
  - `rwkv7_kernels-0.10.0.dev0-py3-none-any.whl`: `f67b5a03f733336364921a79274172c86fab37f9443f7937181f736124d0e938`
- Validation: FP16 optimized 0.1B/0.4B/1.5B passed; BF16 and FP32 reference fallback passed.
- Coverage: B=1/4, T=1/17/128, left/right padding, cache, teacher-forced decode,
  64-token greedy generation, package-free AutoModel loading, beam generation,
  save/reload, gradient checkpointing, backward, and optimizer update.
- PyTorch 2.5 checkpointing regression: fixed. Training now receives an explicit
  semantic training hint and never enters the inference-only v1 backend; graph
  replay mutations remain inside `torch.inference_mode()`.
- `lm_eval==0.4.9.1` PR smoke: 8/8 tasks passed on 0.1B, batch 1, limit 2.
  `lm-eval-smoke/manifest.jsonl` retains both the first invalid `~` path attempt
  and the successful absolute-path retry; the last record for every unit passed.

## 0.4B paired speed (FP16)

Model-level CUDA graph and `torch.compile` are disabled in both paths. The
`auto` path may use the optional recurrence graph defined by protocol v1.

| case | reference ms | auto ms | speedup |
|---|---:|---:|---:|
| `cached_decode_b1` | 4826.295 | 4602.295 | 1.049x |
| `cached_decode_b4` | 4899.716 | 4707.462 | 1.041x |
| `generation_prefill_b1_t1` | 72.834 | 73.398 | 0.992x |
| `generation_prefill_b1_t128` | 877.683 | 131.959 | 6.651x |
| `generation_prefill_b1_t512` | 3425.506 | 392.861 | 8.718x |
| `generation_prefill_b1_t2048` | 13603.495 | 1461.692 | 9.306x |
| `generation_prefill_b4_t128` | 1142.335 | 217.611 | 5.249x |
| `generation_prefill_b4_t512` | 4447.364 | 768.826 | 5.785x |

Raw JSON files in this directory are the source of truth.
