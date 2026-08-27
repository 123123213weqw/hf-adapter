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

The package contains two explicitly selectable implementations:

- `RWKV7_KERNEL_IMPL=graph` captures the readable PyTorch recurrence in
  reusable CUDA graphs. This is the exact compatibility baseline, not the
  native performance claim.
- `RWKV7_KERNEL_IMPL=triton` selects the candidate
  `native-triton-rank1-scan-v1` implementation.

`RWKV7_KERNEL_IMPL=auto` currently resolves to the conservative graph lane.
The Triton lane accepts CUDA FP16 inference with `K=V=64` and FP32 recurrent
state. Training, BF16, FP32, unsupported shapes, and requests that require
autograd remain on the readable reference path in backend `auto`.

The Triton implementation is present for reproducible testing but is not yet
the default release lane. V100, RTX 4080, and RTX 4090 all completed the
forced-Triton 48-unit `lm_eval` matrix, while selected full-model FP16 logits
still miss the strict `max_abs <= 0.15` diagnostic. Cache, state, finite,
cosine, and 64-token greedy checks pass. Promotion therefore remains blocked
on an explicit numerical-policy decision or a kernel correction; these
reports must not be described as a clean numerical pass.

## Acceptance rule

Installing `rwkv7-kernels` must not remove any reference capability.  The
complete AutoModel/generation, cache, padding, official-checkpoint, `lm_eval`,
Trainer, Accelerate, PEFT, and TRL matrix runs with `auto`.  Supported lanes
also run with forced `optimized`; unsupported lanes must produce an explicit
reason and pass through reference in `auto`.

The formal optimized-backend gate repeats the same 48-unit
`lm_eval==0.4.9.1` matrix used by the reference line (three model sizes,
batch 1/8, and eight tasks) with `RWKV7_BACKEND=optimized`.  Forced mode makes
an unsupported shape or hidden fallback fail immediately.  Compare the two
completed bundles with:

```bash
python evaluation/compare_lm_eval_matrices.py \
  --reference-dir results/lm-eval-reference/merged \
  --candidate-dir results/lm-eval-optimized/merged \
  --output results/lm-eval-optimized/parity.json
```

The comparison requires all 48 formal units, identical dataset fingerprints
and metric sets, finite values, and exact metric equality by default.  The v1
wheel is inference-only, so training checks use `auto`: Trainer, Accelerate,
PEFT, and TRL then execute the same readable autograd path rather than an
unvalidated backward kernel.

The release bundle additionally runs one artifact-producing step of each
public LoRA SFT, DPO, and GRPO example with both distributions installed.
`evaluation/validate_finetune_smoke_runs.py` requires finite loss, nonzero
gradients, changed adapter parameters, exact adapter save/reload, complete
source/model provenance, and recorded `rwkv7-hf` / `rwkv7-kernels` versions.

## Current forced-Triton evidence

The compact RTX 4090 bundle is committed at
[`results/native-backend/4090-cfeb5ae-triton`](../results/native-backend/4090-cfeb5ae-triton/README.md).
It records the exact `cfeb5ae` code revision and
`native-triton-rank1-scan-v1` route:

- formal `lm_eval`: 48/48 units, batch 1/8 stability validator passed;
- HF AutoModel/generation/save-reload and training fallback: passed;
- one-step LoRA SFT, DPO, and GRPO: passed with exact adapter reload;
- FLA operator/state/greedy gates: passed; model-logit diagnostic: outside
  threshold;
- 0.4B prefill versus the readable reference: 9.39x--54.99x faster for the
  measured B1/B4, T128--2048 cases;
- FLA remains faster at whole-model prefill/decode for the measured cases,
  even though the native scan is faster than FLA at several short operator
  shapes.

The committed bundle excludes checkpoints, model weights, samples, W&B data,
and large logs. `MANIFEST.sha256` covers every retained artifact.
