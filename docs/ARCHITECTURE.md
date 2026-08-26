# Architecture

The reference line copies five architecture/runtime files into every converted
model repository:

1. `configuration_rwkv7.py` — architecture-only configuration.
2. `cache_rwkv7.py` — recurrent cache lifecycle and batch operations.
3. `kernel_bridge.py` — versioned optional-backend discovery and diagnostics.
4. `ops_rwkv7.py` — readable PyTorch WKV plus one dispatch boundary.
5. `modeling_rwkv7.py` — TMix, CMix, blocks, backbone and causal LM.

`RWKV7TimeMix` computes shifts and R/W/K/V/A/G projections, calls
`rwkv7_recurrent`, applies GroupNorm and the direct RKV term, then projects
back to the residual width. `RWKV7ChannelMix` is the shifted squared-ReLU
FFN. `RWKV7Block` exposes both residual paths directly.

## State

For each layer `RWKV7Cache` stores recurrent state
`[batch, heads, key_dim, value_dim]`, attention shift
`[batch, hidden]` and FFN shift `[batch, hidden]`.

`v_first` is passed from layer zero to later layers during one forward call
and is never persisted. False positions in a 2-D attention mask do not update
any recurrent or shift state, so left and right padding are deterministic.

The cache implements sequence length, reorder, select, repeat, reset, detach and
device/dtype conversion without graph runners, counters, hardware policy or
layout routing.

## Numerical reproducibility

The model still uses ordinary `torch.nn.functional.linear` and PyTorch matrix
multiplication. To make FP16 scores independent of how an evaluation or
training framework regroups the same examples, `modeling_rwkv7.py` tiles batch
and time into a fixed 128-row linear shape. Padding rows are discarded before
the recurrent boundary; `ops_rwkv7.py` remains the direct batched equation.
This changes neither the RWKV equations nor checkpoint keys and has no
device-specific route or custom kernel inside modeling.  `kernel_bridge.py`
may call an independently installed backend after an explicit protocol probe;
otherwise `ops_rwkv7.py` executes the same reference equation.  Runtime
routing is never serialized into a checkpoint.

## HF contract

`RWKV7ForCausalLM.forward` accepts `input_ids`, `inputs_embeds`,
a 2-D `attention_mask`, `past_key_values`, `labels`,
`use_cache`, `output_hidden_states`, `return_dict`,
`cache_position` and `logits_to_keep`. Labels are shifted internally
and -100 is ignored. Cache is disabled while training and gradient
checkpointing. `output_attentions=True` raises `NotImplementedError`
because RWKV has no Transformer attention matrix to return.

## Mamba-style separation

The separation mirrors the current Transformers Mamba2 pattern without
copying its SSM implementation:

| responsibility | Mamba2 | RWKV7 |
|---|---|---|
| readable architecture | `modeling_mamba2.py` mixer/model/LM | `modeling_rwkv7.py` TMix/CMix/model/LM |
| readable math fallback | decorated PyTorch kernel functions | `rwkv7_recurrent_reference` |
| optimized boundary | kernelized function decorators | versioned `kernel_bridge.py` protocol |
| state lifecycle | HF cache layers | `RWKV7Cache` canonical state and shifts |
| unsupported requests | PyTorch fallback | PyTorch fallback with a recorded reason |

The optional package never defines a model class, config class, cache class,
loss, layer loop, or generation method.  Tests reject CUDA, Triton, graph,
device-capability, environment-route, and companion-package policy inside
`modeling_rwkv7.py`, `configuration_rwkv7.py`, and the readable operator.
