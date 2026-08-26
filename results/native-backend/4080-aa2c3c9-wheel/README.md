# RTX 4080 clean-wheel acceptance (aa2c3c9)

This bundle verifies the final `0.10.0.dev0` artifacts from outside the source checkout.

- Source revision: `aa2c3c9e576419291afef86c295c122c0f243117`
- GPU: NVIDIA GeForce RTX 4080
- PyTorch: `2.11.0+cu130`
- Python: `3.12.2`
- `rwkv7_hf-0.10.0.dev0-py3-none-any.whl`: `40f90fa63bb5cca2a2bbced107c2b407afb23cecc77f8d3f7bc397f58404a8bf`
- `rwkv7_kernels-0.10.0.dev0-py3-none-any.whl`: `d90e8bceef2fecab076f611956be0b9fa70e3d6e3a386193e8990c91b4be43fa`
- Install mode: clean virtual environment with system CUDA/PyTorch, both wheels installed with `--no-deps`, tests launched from `/tmp` with `PYTHONPATH` unset.

## Results

- Exact backend parity: pass for B=`1/4`, T=`1/17/128`; output/state/logit max-abs error `0.0`.
- Cached teacher-forced decode: pass.
- Greedy generation: 64/64 tokens identical.
- `AutoModelForCausalLM(..., trust_remote_code=True)`: pass from a self-contained model directory.
- Padded cache, cached decode, beam generation, save/reload, gradient checkpointing, backward: pass.
- Inference selected `torch-cuda-graph-reference-v1`; training safely fell back to the readable PyTorch reference path.

See `validation-fp16.json` and `hf-ecosystem-smoke.json` for the machine-readable records.
