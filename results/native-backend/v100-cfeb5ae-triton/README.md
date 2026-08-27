# V100 forced-Triton acceptance bundle

Compact evidence for code
`cfeb5aeca860ce444ebb3515a20cc22f7e2b090b`, forced backend `optimized`,
selector `triton`, and observed route `native-triton-rank1-scan-v1`.

Functional HF acceptance passed: formal `lm_eval==0.4.9.1` completed 48/48
units and its merged batch 1/8 validator passed; HF ecosystem, training
fallback, LoRA SFT/DPO/GRPO, adapter reload, speed, and FLA fused-recurrent
inference parity/speed passed.

The final report remains `passed: false` because selected full-model FP16
logits exceed the strict `max_abs <= 0.15` diagnostic. Values are finite and
cache, state, cosine, and greedy checks pass. Pinned FLA chunk/backward cannot
compile on sm70 with the tested Triton toolchains, so this bundle records the
successful fused-recurrent inference comparison and preserves that external
limitation explicitly.

No samples, weights, checkpoints, W&B directories, or large logs are included.
`MANIFEST.sha256` covers every retained file.
