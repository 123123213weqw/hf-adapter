# RTX 4090 forced-Triton acceptance bundle

This is the compact, immutable evidence bundle for:

- code: `cfeb5aeca860ce444ebb3515a20cc22f7e2b090b`;
- GPU: NVIDIA GeForce RTX 4090;
- backend: `RWKV7_BACKEND=optimized`;
- kernel selector: `RWKV7_KERNEL_IMPL=triton`;
- observed implementation: `native-triton-rank1-scan-v1`;
- kernel wheel SHA256:
  `544b08225351d9658a1941c260a70710d47bb677d2976d730cec9e4f34771560`.

## Result

Functional HF acceptance passed:

- formal `lm_eval==0.4.9.1`: 48/48 units exited zero;
- batch 1/8 metric stability and Wikitext perplexity validator: passed;
- HF AutoModel, generation, beam, save/reload, and cache smoke: passed;
- training reference fallback: passed;
- LoRA SFT, DPO, and GRPO: passed at step 1;
- adapter save/reload: exact (`max_abs=0.0`) for all three runs;
- FLA speed job: passed.

The bundle is **not** a clean numerical promotion pass. The final report keeps
`passed: false` because:

- selected forced-Triton full-model FP16 logits exceed the strict
  `max_abs <= 0.15` diagnostic;
- clean-vs-FLA model logits are outside threshold.

The clean-vs-FLA operator, recurrent state, and 64-token greedy gates pass.
All reported values are finite and cosine remains high. See
[`final-corrected-exit-status.json`](final-corrected-exit-status.json) and
[`fla/clean-vs-fla-rwkv7_04b_hf-fp16.md`](fla/clean-vs-fla-rwkv7_04b_hf-fp16.md).

## Performance snapshot

For the 0.4B model, forced Triton is 9.39x--54.99x faster than the readable
reference for the measured B1/B4 prefill cases from T128 through T2048. The
native scan is faster than FLA for several short operator shapes, but FLA is
still faster for the measured whole-model prefill and cached-decode cases.
Raw timings and environment metadata are in
[`fla/three-way-speed.json`](fla/three-way-speed.json) and
[`speed-0.4b-fp16.json`](speed-0.4b-fp16.json).

## Contents

- `validation/`: forced-Triton reference parity and route evidence;
- `lm_eval/`: pool status plus merged 48-unit validator;
- `fla/`: pinned-FLA parity and three-way speed summaries;
- `finetune/`: configs, revisions, metrics, checks, and validation summaries;
- `hf-ecosystem-fp16.json` and `training-fp16.json`: HF contract evidence;
- `MANIFEST.sha256`: checksums for every retained artifact.

No samples, model or adapter weights, checkpoints, W&B directories, or large
logs are committed.
