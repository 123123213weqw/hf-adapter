# RTX 4080 forced-Triton acceptance bundle

Compact evidence for code
`cfeb5aeca860ce444ebb3515a20cc22f7e2b090b`, forced backend `optimized`,
selector `triton`, and observed route `native-triton-rank1-scan-v1`.

Functional HF acceptance passed: formal `lm_eval==0.4.9.1` completed 48/48
units and its merged batch 1/8 validator passed; HF ecosystem, training
fallback, speed, and current-revision LoRA SFT/DPO/GRPO passed. All three
adapters reload exactly (`max_abs=0.0`).

The final report remains `passed: false`. Selected optimized-vs-reference
full-model FP16 logits exceed the strict `max_abs <= 0.15` diagnostic, and the
clean-vs-FLA comparison misses only its model-logits gate. FLA operator,
recurrent state, and 64-token greedy gates pass; values are finite and cosine
remains high.

No samples, weights, checkpoints, W&B directories, or large logs are included.
`MANIFEST.sha256` covers every retained file.
