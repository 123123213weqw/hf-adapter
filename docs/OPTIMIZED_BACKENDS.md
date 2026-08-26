# Optional optimized backends

`modeling_rwkv7.py` is the single architectural source of truth.  It exposes
TimeMix, ChannelMix, blocks, cache lifecycle, loss, and the Hugging Face
contract directly.  Optimized code may replace only the versioned recurrence
boundary in `ops_rwkv7.py`; it must not replace the model or persist hardware
policy in `config.json`.

## Runtime modes

The default is `auto`:

- `reference` always executes the readable PyTorch recurrence;
- `optimized` requires an installed backend to accept the full request and
  fails closed otherwise;
- `auto` selects an accepted optimized request and falls back to reference for
  unsupported devices, dtypes, shapes, masks, or autograd requirements.

Use a scoped override for validation:

```python
from rwkv7_hf import use_rwkv7_backend

with use_rwkv7_backend("reference"):
    reference = model(input_ids)

with use_rwkv7_backend("optimized"):
    optimized = model(input_ids)
```

`RWKV7_BACKEND=reference|optimized|auto` is available for unmodified framework
entry points such as `lm_eval`.  It is runtime-only and is never written into a
checkpoint.

## Versioned package protocol

The optional `rwkv7-kernels` distribution exposes protocol v1:

```python
RWKV7_KERNEL_API_VERSION = 1
probe_recurrent_v1(...)
recurrent_v1(...)
```

The probe returns a support decision, implementation name, and reason.  Once a
backend claims support, runtime failures propagate instead of being hidden by
a retry.  This makes forced-backend validation meaningful and prevents broken
kernel installations from silently producing misleading performance results.

The promoted v1 lane captures the same readable PyTorch recurrence in reusable
CUDA graphs.  It is bit-exact because it preserves operation order, tensor
strides, cache layout, and mixed-precision boundaries while removing most
per-token launch overhead.  It accepts CUDA FP16 inference with `K=V=64` and
FP32 recurrent state.  More aggressive Triton scans stay on
`perf/native-kernels-v0.8`: several 1.5B logits missed the strict
`max_abs <= 0.15` gate despite passing cosine and greedy checks, so that code
is not shipped in the clean companion wheel.  Training, BF16, FP32, and
unsupported requests intentionally remain on the reference path.

## Acceptance rule

Installing `rwkv7-kernels` must not remove any reference capability.  The
complete AutoModel/generation, cache, padding, official-checkpoint, `lm_eval`,
Trainer, Accelerate, PEFT, and TRL matrix runs with `auto`.  Supported lanes
also run with forced `optimized`; unsupported lanes must produce an explicit
reason and pass through reference in `auto`.
