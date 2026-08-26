# Changelog

## 0.10.0.dev0

- Added a versioned optional recurrence protocol without moving hardware
  policy into config or modeling.
- Added a separate `rwkv7-kernels` companion whose promoted CUDA-graph v1
  backend preserves bit-exact reference operation order.
- Added fail-closed forced routing and automatic fallback for training, BF16,
  FP32, CPU, unsupported shapes, missing packages, and protocol mismatches.
- Added operator/model/cache/padding/teacher-decode/greedy/training validation
  bundles and paired reference-versus-auto speed measurement.
- Kept re-entrant gradient-checkpointing passes on the reference backend by
  propagating the model's semantic training mode instead of inferring it from
  `torch.is_grad_enabled()`; also made CUDA graph replay safe for inference
  tensors on PyTorch 2.5.
- Applied `logits_to_keep` before the vocabulary projection in label-free
  inference, matching the current Mamba-style HF contract.

## 0.9.0

- Replaced the default runtime with a readable pure-PyTorch HF reference model.
- Added `RWKV7Cache` with canonical `[B,H,K,V]` state and no persisted
  `v_first`.
- Made the self-contained `reference` conversion layout the default.
- Added standard cache, padding, generation, loss, gradient-checkpointing and
  package-free AutoModel tests.
- Added non-blocking FLA backend diagnostics, formal lm_eval matrix tooling, and direct LoRA
  SFT/DPO/GRPO examples with local reproducibility artifacts.
- Preserved `NativeRWKV7*` names as compatibility aliases.
- Moved CUDA/JIT/graph/quantization/hardware-specialized work to
  `perf/native-kernels-v0.8`.

Older release history is retained in Git tags.
