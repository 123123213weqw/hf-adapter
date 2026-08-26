# Native backend validation

`validate.py` runs the same readable model in `reference` and `auto` modes and
records operator output/state, padded model logits/cache, teacher-forced cached
decode, and 64-token greedy parity.  Run it with the optional package on
`PYTHONPATH` or installed into the environment:

```bash
PYTHONPATH="$PWD:$PWD/kernel_wheel" python benchmarks/native_backend/validate.py \
  --model 0.4b=/models/rwkv7-g1d-0.4b-hf \
  --dtype fp16 \
  --device cuda \
  --output results/native-backend/4080
```

Remote source mirrors normally exclude `.git`; pass `--code-sha "$(git
rev-parse HEAD)"` from the controlling checkout so every result remains tied
to an immutable revision.

FP16 requests supported by protocol v1 must select the optimized recurrent
scan.  BF16, FP32, training, and unsupported shapes currently select the
reference fallback; the complete HF ecosystem is then validated in `auto`
mode so installing the package cannot reduce compatibility.

`train_smoke.py` runs the same real checkpoint with forced `reference` and
`auto`, gradient checkpointing, causal loss, backward, and an optimizer step.
It requires the installed inference backend to select reference for autograd
and checks bit-equal loss, logits, and a representative projection gradient.

`speed.py` measures the same loaded model with `reference` and `auto` routing.
It records generation-prefill and cached-decode latency, raw samples, memory,
route decisions, and the complete environment without enabling CUDA graphs or
`torch.compile`:

```bash
PYTHONPATH="$PWD:$PWD/kernel_wheel" python benchmarks/native_backend/speed.py \
  --model /models/rwkv7-g1d-0.4b-hf \
  --dtype fp16 \
  --output results/native-backend/4080/speed-fp16.json
```
