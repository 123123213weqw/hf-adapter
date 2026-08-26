# RWKV-7 Hugging Face 参考实现

[English](README.md) | [中文](README_ZH.md)

0.9 是兼容性优先、可阅读、可复现的纯 PyTorch RWKV-7 HF 实现。TMix、
CMix、残差、归一化、层循环、loss 与 cache 都能直接从
`rwkv7_hf/modeling_rwkv7.py` 看懂；WKV 递推只通过
`rwkv7_hf/ops_rwkv7.py` 这一处明确边界。激进的 CUDA/JIT/量化实验完整保存在
`perf/native-kernels-v0.8`；独立、带版本协议的算子包只能替换这一处递推边界，
不会替换 modeling。

## 直接使用 HF 模型

```bash
python -m pip install "torch" "transformers>=4.48,<6"
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wangyue114514/rwkv7-g1d-0.1b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.float16
).cuda().eval()

inputs = tokenizer("User: Hello! Assistant:", return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

新版模型仓库自带 config、cache、PyTorch 算子、modeling、tokenizer、词表和
safetensors。普通加载不要求安装 `rwkv7-hf`、FLA、Triton、编译器或
kernel wheel。

## 转换官方 .pth

```bash
python -m pip install "rwkv7-hf==0.9.0"
rwkv7-hf convert \
  --input /path/to/model.pth \
  --output ./rwkv7-model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --adapter-layout reference \
  --no-fuse-norm \
  --low-memory
```

`reference` 是新默认布局；`thin` 仅作为旧部署流程所需的包依赖布局保留。

## 公开结构

- `RWKV7Config`，`model_type = "rwkv7"`
- `RWKV7Cache`：每层 `[B,H,K,V]` state、TMix shift、CMix shift
- `RWKV7TimeMix`、`RWKV7ChannelMix`、`RWKV7Block`
- `RWKV7PreTrainedModel`、`RWKV7Model`、`RWKV7ForCausalLM`
- 标准 loss、cache、generation、save/reload、gradient checkpointing、PEFT、
  Trainer 与 TRL 接口

旧的 `NativeRWKV7*` 类名作为 0.9 兼容别名保留。

## 可选的精确性能后端

基础模型永远不依赖算子 wheel。安装独立的 `rwkv7-kernels` 后，`auto` 只在
已经验收的 CUDA FP16 推理形状上启用优化。v1 用 CUDA Graph 重放完全相同的
可读递推，因此 RTX 4080 数值逐位一致；0.4B 实测 B1/B4 prefill 加速
1.9–2.8 倍。训练、BF16、FP32、CPU 和不支持的形状自动保留参考路径。

modeling 仍完整包含 TMix、CMix、block、cache 和 loss；算子包只实现
`probe_recurrent_v1` / `recurrent_v1`。具体边界和验收命令见
[可选性能后端](docs/OPTIMIZED_BACKENDS.md) 与
[验证脚本](benchmarks/native_backend/README.md)。

- [架构](docs/ARCHITECTURE.md)
- [转换](docs/CONVERSION.md)
- [评测](docs/EVALUATION.md)
- [SFT / DPO / GRPO](docs/FINETUNING.md)
- [可复现产物](docs/REPRODUCIBILITY.md)
- [模型列表](docs/PUBLISHED_MODELS.md)
