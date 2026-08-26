# RTX 4080 self-contained HF ecosystem smoke

- Source: `485f00efdfa39411a47f56c96e40970fd3190986`
- Loaded through `AutoModelForCausalLM(..., trust_remote_code=True)` from a self-contained model directory.
- Passed padded prefill/cache, cached decode, beam generation, `save_pretrained` reload, gradient checkpointing, causal loss, and backward.
- The dynamic HF module selected the optimized backend for FP16 inference and the reference fallback for autograd.

The JSON file is the source of truth.
