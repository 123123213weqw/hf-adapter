# RWKV-7 vs Qwen3.5：统一 HF 快速路径测试协议

更新日期：**2026-08-12**。[English version](QWEN35_SPEED_COMPARISON.md)

## 当前状态

此前的跨卡 Qwen3.5 baseline **不能进入新的统一主表**。旧结果混用了不同的
causal-convolution 实现、运行时版本和 RWKV CUDA Graph 设置；即使命令写着
FLA，实际也可能绑定仓库 Triton convolution 或静默回退到慢速 PyTorch 路径。
这些证据只保留为历史复现和回归资料。

新的主表按卡逐张写入。RTX 4090 已于 2026-08-12 使用 RWKV
`best_optimized_hf` 路线完成严格一致的 `hf_fast_path_v1` 形状协议：96/96
行通过且没有 fallback；RTX 3090 和 RTX 5090 仍待按新协议重测。不完整卡、
后端回退和旧结果都不能混入主表。

## 固定协议（`hf_fast_path_v1`）

| 项目 | 固定设置 |
|---|---|
| GPU | RTX 3090 / 4090 / 5090，分别单卡 |
| 模型对 | RWKV 0.4/1.5/2.9/7.2B 对 Qwen3.5 0.8/2/4/9B |
| 精度 | Dense FP16；关闭量化、MTP 和 speculative decode |
| Batch | 1、8 |
| Prompt | 128、512、2048 token |
| Decode | 128、512 token |
| Prefill chunk | 512 token |
| 统计 | warmup 3 次、正式 7 次、每格取中位数 |
| Qwen | Transformers FLA 快速路径 + 官方 Dao-AILab `causal_conv1d` |
| RWKV 性能线 | 精确显卡 `best_optimized_hf`；开启 CUDA Graph 和已验证融合 |

每个模型侧 `4 × 2 × 3 × 2 = 48` 格；每张卡 96 行，三张卡共 288 行。
如果只重测 Qwen baseline，则三张卡共 144 行。

### 统一主表状态

| GPU | 行数 | Qwen 官方快速路径 | RWKV 性能线 | 参数校正 Prefill 超过 Qwen | 原始 / 参数校正 Decode 超过 Qwen | 证据 |
|---|---:|---|---|---:|---:|---|
| RTX 4090 | 96/96 | 48/48 通过，无 fallback | 48/48 `best_optimized_hf`；Decode Graph 开启 | 48/48 | 48/48 / 48/48 | [不可变证据](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 3090 | 待测 | 待测 | 待测 | — | — | — |
| RTX 5090 | 待测 | 待测 | 待测 | — | — | — |

RTX 4090 现在每一格都超过 Qwen：原始 Prefill 最小值/中位数为
`1.361373x/2.315043x`，参数校正后为 `1.060506x/1.549011x`；原始 Decode
为 `2.275368x/5.871032x`，参数校正后为 `1.829468x/4.468521x`。旧的
`native_jit` 无 Graph 矩阵保留作诊断附表，不再作为极限性能主结论。

RTX 4090 上校正后的 Qwen Decode 中位数为：

| Qwen3.5 | B1 | B8 |
|---|---:|---:|
| 0.8B | 35.5 tok/s | 269 tok/s |
| 2B | 35.1 tok/s | 268 tok/s |
| 4B | 25.4 tok/s | 196 tok/s |
| 9B | 25.5 tok/s | 197 tok/s |

环境也是验收的一部分：三张卡必须使用相同 Python、PyTorch+CUDA build、
Transformers、FLA、`causal-conv1d` 和仓库提交。产物保存运行时锁、
`pip freeze`、Docker digest（若存在）、仓库提交以及模型 config/safetensors
的 SHA256。

### Qwen 每行强制验收

```text
status=pass
qwen_fast_path_available=true
qwen_fast_path_verified=true
qwen_full_fused_contract_pass=true
qwen_causal_conv1d_importable=true
qwen_conv_backend_effective=causal_conv1d
qwen_force_torch=false
```

runner 现在会把请求的 convolution 后端与每一层 Qwen GatedDeltaNet 的实时
算子绑定逐一核对；环境、绑定、结果行三层都 fail-closed。RTX 5090 不允许
自动改用仓库 `fla_triton` convolution。如果 SM120 无法通过官方路径，则只记为
**“SM120 官方 HF fast path 未验证”**，并从统一主表排除。

### RWKV 最佳优化线

```bash
export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_MODEL_BACKEND=native_graph
```

runner 要求每行记录 `optimization_lane=best_optimized_hf`、
`rwkv_optimization_contract=exact_card_best_optimized_hf`，并确认 Decode
实际走 `native_graph`。所有精确显卡 Graph、融合和 block accumulation 路线
都必须写入遥测并通过 Prompt/缓存/greedy 正确性。仅 7.2B B8/P2048 为控制
24 GiB 显存关闭 Prefill Graph，Decode 仍然使用 Graph。无 Graph `native_jit`
结果不能混入这条性能主线。

### 单卡复现

准备八个本地模型目录并使用同一个运行时锁：

```bash
export GPU_MODEL=4090
export OUT_DIR=/path/to/hf-fast-path-v1-4090
export PYTHON_BIN=/path/to/locked-python
export RUNTIME_LOCK=/path/to/hf-fast-path-v1-runtime-lock.json
export FLA_SOURCE_COMMIT=2e38c1fab332174d056928feaf29f8c5fd5ac550
export CAUSAL_CONV1D_SOURCE_COMMIT=4f6ae4e26ae5fe8af9372f8d312ab25cc4595223

export RWKV_04_MODEL=/models/rwkv-0.4b
export RWKV_15_MODEL=/models/rwkv-1.5b
export RWKV_29_MODEL=/models/rwkv-2.9b
export RWKV_72_MODEL=/models/rwkv-7.2b
export QWEN_08_MODEL=/models/Qwen3.5-0.8B
export QWEN_2_MODEL=/models/Qwen3.5-2B
export QWEN_4_MODEL=/models/Qwen3.5-4B
export QWEN_9_MODEL=/models/Qwen3.5-9B

bash bench/run_hf_fast_path_v1.sh
```

只有第一张建立锁的卡使用 `WRITE_RUNTIME_LOCK=/path/to/lock.json`；后续卡必须
使用 `RUNTIME_LOCK`。脚本先跑 Qwen；官方快速路径失败时不会继续跑 RWKV，
也不会生成 `main_table.jsonl`。
两项扩展需通过 `bench/build_hf_fast_path_v1_extensions.sh` 从上述精确提交编译；
脚本要求 CUDA developer image，并强制
`TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"`。

## 参数口径

模型名是发布档位；下表把 benchmark 遥测的活跃参数按十亿参数（B）保留三位小数：

| 模型对（RWKV / Qwen3.5） | RWKV 活跃参数 | Qwen 活跃参数 |
|---|---:|---:|
| 0.4B / 0.8B | `0.451B` | `0.752B` |
| 1.5B / 2B | `1.527B` | `1.882B` |
| 2.9B / 4B | `2.948B` | `4.206B` |
| 7.2B / 9B | `7.199B` | `8.954B` |

- **原始速度比** = RWKV tok/s ÷ Qwen tok/s，代表用户实际拿到的吞吐。
- **参数规模校正速度比** = 原始速度比 × RWKV 活跃参数 ÷ Qwen 活跃参数，
  用于扣除“小模型本来就更快”的天然优势；精确参数数值保留在证据中。
- 例如本次 RTX 4090 的 0.4B/0.8B B8：原始 Prefill 中位值
  `2.173516x`，参数校正后为 `1.302180x`。

## 历史非统一 NVIDIA 证据

> 下表仅保留用于审计和回归。它混用了旧运行时与后端协议，**不是**
> `hf_fast_path_v1` 统一主表，不能再作为新的 3090/4090/5090 速度结论。

下面不再筛选代表项，而是逐行列出当前正式 optimized-Qwen 对照中所有
GPU、模型对和 Batch。行顺序统一为模型尺寸、显卡、B1/B8。`RWKV P / D
tok/s`与`Qwen P / D tok/s`是分别对声明范围内各格吞吐取中位数后的具体数值：
大于等于 100 tok/s 不保留小数，小于 100 tok/s 保留 1 位小数。`原始 P / D`
与`参数规模校正 P / D`仍是配对逐格速度比的中位数，因此不一定等于两列吞吐
中位数直接相除。RTX 4090 行已用最新统一正式产物刷新；本节其他 GPU 行仍保留
各自历史协议。

RTX 4080 现已通过更严格的逐格门槛：**参数校正 Prefill 36/36、Decode
36/36 全部超过**，全矩阵最小值为 `1.068520x / 1.140700x`。

RTX 4090 的最新严格门槛通过：**参数校正 Prefill 48/48、Decode 48/48
全部超过**，最小值为 `1.060506x / 1.829468x`。

| GPU | 模型对 | Batch | 范围 | RWKV 活跃参数 | Qwen 活跃参数 | RWKV P / D tok/s | Qwen P / D tok/s | 原始 P / D | 参数规模校正 P / D | 证据 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RTX 3090 | 0.4B / 0.8B | B1 | 3格 | 0.451B | 0.752B | **29,368 / 293** | **7,155 / 26.5** | **4.10x / 11.05x** | **2.46x / 6.62x** | [3090 极限性能](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 0.4B / 0.8B | B8 | 3格 | 0.451B | 0.752B | **78,949 / 1,692** | **32,678 / 213** | **2.47x / 7.93x** | **1.48x / 4.75x** | [3090 极限性能](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4080 | 0.4B / 0.8B | B1 | 6格，全过 | 0.451B | 0.752B | **45,538 / 492** | **24,889 / 100** | **1.83x / 4.91x** | **1.10x / 2.94x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 0.4B / 0.8B | B8 | 6格，全过 | 0.451B | 0.752B | **103,571 / 3,206** | **50,004 / 768** | **1.98x / 4.17x** | **1.19x / 2.50x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 0.4B / 0.8B | B1 | 6格，全过 | 0.451B | 0.752B | **63,487 / 585** | **10,779 / 35.5** | **6.26x / 16.49x** | **3.75x / 9.88x** | [4090 最佳优化](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 4090 | 0.4B / 0.8B | B8 | 6格，全过 | 0.451B | 0.752B | **147,413 / 3,845** | **68,760 / 269** | **2.17x / 14.30x** | **1.30x / 8.57x** | [4090 最佳优化](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 5090 | 0.4B / 0.8B | B1 | 3格 | 0.451B | 0.752B | **58,105 / 1,121** | **15,886 / 56.7** | **3.86x / 19.79x** | **2.31x / 11.85x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 0.4B / 0.8B | B8 | 3格 | 0.451B | 0.752B | **206,364 / 3,432** | **93,886 / 429** | **2.24x / 7.99x** | **1.34x / 4.79x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| V100 32GB | 1.5B / 2B | B1 | P512/D64 | 1.527B | 1.882B | **10,426 / 151** | **3,702 / 25.6** | **2.82x / 5.91x** | **2.29x / 4.80x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| V100 32GB | 1.5B / 2B | B8 | P512/D64 | 1.527B | 1.882B | **20,729 / 817** | **3,833 / 155** | **5.41x / 5.27x** | **4.39x / 4.28x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| RTX 3090 | 1.5B / 2B | B1 | 3格 | 1.527B | 1.882B | **17,641 / 164** | **8,529 / 28.5** | **2.12x / 5.75x** | **1.72x / 4.67x** | [3090 极限性能](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 1.5B / 2B | B8 | 3格 | 1.527B | 1.882B | **29,163 / 985** | **16,416 / 220** | **1.66x / 4.47x** | **1.34x / 3.63x** | [3090 极限性能](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4080 | 1.5B / 2B | B1 | 6格，全过 | 1.527B | 1.882B | **30,858 / 194** | **19,871 / 102** | **1.55x / 1.90x** | **1.26x / 1.55x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B8 | 6格，全过 | 1.527B | 1.882B | **38,144 / 1,356** | **21,602 / 765** | **1.76x / 1.77x** | **1.43x / 1.44x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 1.5B / 2B | B1 | 6格，全过 | 1.527B | 1.882B | **36,381 / 251** | **10,557 / 35.1** | **3.51x / 7.14x** | **2.85x / 5.79x** | [4090 最佳优化](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 4090 | 1.5B / 2B | B8 | 6格，全过 | 1.527B | 1.882B | **56,564 / 1,717** | **37,337 / 268** | **1.52x / 6.40x** | **1.23x / 5.20x** | [4090 最佳优化](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 5070 Laptop | 1.5B / 2B | B8 | 6格 | 1.527B | 1.882B | **10,770 / 690** | **8,239 / 269** | **1.33x / 2.62x** | **1.08x / 2.13x** | [5070](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | 1.5B / 2B | B1 | 3格 | 1.527B | 1.882B | **33,698 / 547** | **15,795 / 56.7** | **2.16x / 9.63x** | **1.75x / 7.82x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 1.5B / 2B | B8 | 3格 | 1.527B | 1.882B | **82,339 / 2,061** | **50,353 / 434** | **1.43x / 4.77x** | **1.16x / 3.87x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 3090 | 2.9B / 4B | B1 | 3格 | 2.948B | 4.206B | **11,774 / 88.7** | **5,657 / 19.2** | **2.08x / 4.61x** | **1.46x / 3.23x** | [3090 极限性能](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 2.9B / 4B | B8 | 3格 | 2.948B | 4.206B | **15,776 / 596** | **7,094 / 151** | **2.14x / 3.96x** | **1.50x / 2.78x** | [3090 极限性能](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4080 | 2.9B / 4B | B1 | 6格，全过 | 2.948B | 4.206B | **14,276 / 103** | **8,819 / 62.8** | **1.75x / 1.63x** | **1.22x / 1.15x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B8 | 6格，全过 | 2.948B | 4.206B | **19,517 / 729** | **9,824 / 416** | **1.99x / 1.75x** | **1.40x / 1.23x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 2.9B / 4B | B1 | 6格，全过 | 2.948B | 4.206B | **18,773 / 136** | **7,627 / 25.4** | **2.49x / 5.34x** | **1.74x / 3.74x** | [4090 最佳优化](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 4090 | 2.9B / 4B | B8 | 6格，全过 | 2.948B | 4.206B | **28,520 / 953** | **15,026 / 196** | **1.91x / 4.86x** | **1.34x / 3.41x** | [4090 最佳优化](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 5090 | 2.9B / 4B | B1 | 3格 | 2.948B | 4.206B | **21,787 / 309** | **11,795 / 41.3** | **1.87x / 7.49x** | **1.31x / 5.25x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 2.9B / 4B | B8 | 3格 | 2.948B | 4.206B | **37,326 / 1,247** | **22,253 / 317** | **1.69x / 3.92x** | **1.19x / 2.75x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 3090 | 7.2B / 9B | B1 | 3格 | 7.199B | 8.954B | **5,764 / 46.4** | **3,616 / 19.7** | **1.63x / 2.35x** | **1.31x / 1.89x** | [3090 极限性能](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 7.2B / 9B | B8 | 3格 | 7.199B | 8.954B | **6,633 / 342** | **4,156 / 164** | **1.60x / 2.08x** | **1.28x / 1.67x** | [3090 极限性能](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4090 | 7.2B / 9B | B1 | 6格，全过 | 7.199B | 8.954B | **10,842 / 61.6** | **7,476 / 25.5** | **1.45x / 2.42x** | **1.17x / 1.94x** | [4090 最佳优化](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 4090 | 7.2B / 9B | B8 | 6格，全过 | 7.199B | 8.954B | **13,836 / 450** | **8,525 / 197** | **1.62x / 2.28x** | **1.30x / 1.83x** | [4090 最佳优化](../bench/4090_hf_best_optimized_v1_20260812/README.md) |
| RTX 5090 | 7.2B / 9B | B1 | 3格 | 7.199B | 8.954B | **14,876 / 146** | **10,652 / 41.7** | **1.42x / 3.50x** | **1.14x / 2.81x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 7.2B / 9B | B8 | 3格 | 7.199B | 8.954B | **19,624 / 867** | **12,262 / 319** | **1.54x / 2.72x** | **1.24x / 2.19x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |

这张长表按模型尺寸、显卡和 B1/B8 保留全部正式对照，便于直接查看不同参数档位的
原始吞吐和参数规模校正速度。

### RTX 3090 最新检查点严格门槛

最新 RTX 3090 证据使用 RWKV-7 g1d 0.4B 和 2026-08-05 g1i
1.5B/2.9B/7.2B，对照官方 Qwen3.5 0.8B/2B/4B/9B，并逐格检查
B1/B8、P128/P512/P2048、D128。全部 `24/24` 个 Qwen 参考格均验证
FLA、Triton causal convolution、实时 fused bindings 和 full-fused contract。

严格 Prefill 门槛 `24/24` 全部通过：原始 Prefill 最低/中位为
`1.531589x/2.076170x`，参数规模校正 Prefill 最低/中位为
`1.227477x/1.467758x`；原始 Decode 最低/中位为
`2.069838x/4.524636x`，参数规模校正 Decode 最低/中位为
`1.664218x/3.433680x`。校正后最窄格是 0.4B/0.8B B8/P512：RWKV 为
`78,949 tok/s`，Qwen 为 `38,534 tok/s`，参数校正后为
`1.227477x`。

精确形状 FP16 accumulation 的正确性门槛覆盖全部直接调用与分块携带形状，
`25/25` 个 Prompt/缓存交接行都达到 cosine `>=0.9999`，greedy token
完全一致。提升策略只适用于实测 RTX 3090 的模型、Batch 和 token-block
形状。完整数据见
[不可变证据](../bench/3090_g1i_qwen35_maxperf_20260812/README.md)。

### RTX 4090 最新检查点严格门槛

最新 RTX 4090 证据使用 RWKV-7 g1d 0.4B 和 g1i 1.5B/2.9B/7.2B，对照
官方 Qwen3.5 0.8B/2B/4B/9B，覆盖 B1/B8、P128/P512/P2048 和
D128/D512。全部 `48/48` 个 Qwen 参考格都验证 FLA chunk Gated DeltaNet、
fused-recurrent Decode、fused gated normalization 和官方 Dao-AILab
causal convolution。

两项严格门槛均逐格通过：参数校正 Prefill 为 `48/48`，全局最低/中位
`1.060506x/1.549011x`；参数校正 Decode 为 `48/48`，全局最低/中位
`1.829468x/4.468521x`。RWKV Decode 全部走 `native_graph`。仅
7.2B B8/P2048 为控制 24 GiB 显存关闭 Prefill Graph；该分块路径的
Prompt/缓存/greedy 正确性通过，精确显卡 FP16 block accumulation 相对本地
control 提升 `1.422952x-1.436970x`。完整数据见
[不可变证据](../bench/4090_hf_best_optimized_v1_20260812/README.md)。

### RTX 5090 最新检查点严格门槛

最新 RTX 5090 行使用 RWKV-7 g1d 0.4B 和 2026-08-05 g1i
1.5B/2.9B/7.2B，对照官方 Qwen3.5 0.8B/2B/4B/9B。全部 24 个 Qwen
参考单元均验证 FLA、Triton causal convolution、实时 fused bindings 和
full-fused contract。

上面的主表展示每组中位值；严格门槛则逐个检查 B1/B8、P128/P512/P2048。
最终 `24/24` 全部通过：原始 Prefill 最低/中位为
`1.347871x/1.819072x`，参数规模校正 Prefill 最低/中位为
`1.072987x/1.317515x`；原始 Decode 最低/中位为
`2.710952x/6.104568x`，参数规模校正 Decode 最低/中位为
`2.179692x/4.330813x`。

0.4B/B1/P2048 达到 `61,344 tok/s`，是上一候选行的 `2.2495x`。
P2048 graph 对 eager 的正确性门槛在四组模型、B1/B8 上 `8/8` 通过，
Prompt/缓存交接后 cosine 最低为 `0.99999988/0.99999994`，greedy token
全部一致。移除负收益的 7.2B stacked-RKV 路径后，其候选峰值显存从
`17.4-18.6 GiB` 降到 `14.3-15.5 GiB`。完整数据见
[不可变证据](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md)。

### Apple M5：全部正式 target-only W4 对照

Apple MLX W4 单独成表，以保持每张表内部的后端和精度一致；具体吞吐为
aggregate tok/s 中位数，并沿用“`>=100` 不保留小数、`<100` 保留 1 位”规则：

| 模型对 | Batch / 形状 | RWKV 活跃参数 | Qwen 活跃参数 | RWKV P / D tok/s | Qwen P / D tok/s | 原始 P / D | 参数规模校正 P / D | 证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.4B / 0.8B | B8，cold，P512 字符/D64 | 0.451B | 0.752B | **11,650 / 992** | **5,702 / 487** | **2.04x / 2.04x** | **1.22x / 1.22x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |
| 1.5B / 2B | B1，P512 字符/D64 | 1.527B | 1.882B | **2,126 / 129** | **1,273 / 89.9** | **1.67x / 1.44x** | **1.36x / 1.17x** | [M5 B1](../bench/apple_bsz1_active_m5_20260715/README.md) |
| 1.5B / 2B | B8，cold，P512 字符/D64 | 1.527B | 1.882B | **2,249 / 186** | **1,601 / 132** | **1.41x / 1.40x** | **1.14x / 1.14x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |

## AMD 和其他硬件

除上面的 NVIDIA 与 Apple 对照外，仓库还覆盖 AMD ROCm、Turing/Ampere/Hopper
以及国产加速器。这里集中展示已实现能力、已验证路径和可直接执行的入口。

| 平台 | 已实现与已验证能力 | 入口 |
|---|---|---|
| AMD Navi 31 / `gfx1100` | Native HF、缓存、分块 Prefill、PEFT、BF16 Trainer、0.1B–13.3B 融合 Decode、40/40 output-head W8/W4 Decode 行 | [AMD ROCm 验证](validation/AMD_ROCM_HF_VALIDATION.md) |
| AMD MI 系列、`gfx1101/gfx1102` | ROCm/HIP 通用执行路径、架构识别与 portable dispatch | [硬件矩阵](HARDWARE_MATRIX.md) |
| Apple M5 | 0.4B/0.8B 与 1.5B/2B target-only W4 对照，MLX、MPS、CoreML 工作流 | [Apple 指南](APPLE_USAGE.md) |
| NVIDIA T4 | Native HF、量化、训练与 production-close 验证 | [T4 证据](../bench/t4_production_close_20260720/README.md) |
| NVIDIA A100/A800 | Ampere CUDA、训练、并行与 HF 工作流 | [硬件矩阵](HARDWARE_MATRIX.md) |
| NVIDIA H100 | Hopper CUDA、Transformers/HF 与 benchmark 执行路径 | [性能指南](PERFORMANCE.md) |
| Ascend、Biren、MetaX、MUSA | 专用后端、运行时适配和兼容性验证 | [硬件矩阵](HARDWARE_MATRIX.md) |

### AMD `gfx1100` 已有的直观数字

下表是 FP16、P128、cached decode；`加速`列展示同一 RWKV 的融合策略相对
通用策略的提升：

| RWKV-7 | B1 Decode | B8 Aggregate Decode | 融合策略 / 通用策略（B1 / B8） |
|---|---:|---:|---:|
| 0.1B | 347 tok/s | 2,667 tok/s | `1.88x / 2.04x` |
| 0.4B | 142 tok/s | 1,073 tok/s | `1.75x / 1.74x` |
| 1.5B | 71.3 tok/s | 514 tok/s | `1.40x / 1.47x` |
| 2.9B | 47.7 tok/s | 353 tok/s | `1.37x / 1.41x` |
| 7.2B | 29.7 tok/s | 214 tok/s | `1.23x / 1.29x` |
| 13.3B | 15.5 tok/s | 113 tok/s | `1.21x / 1.29x` |

此外，gfx1100 的 output-head W8/W4 在 0.4B–13.3B、B1/B2/B4/B8 的
40/40 Decode 行中都快于对应 RWKV FP16。详见
[AMD 验证说明](validation/AMD_ROCM_HF_VALIDATION.md)、
[融合 Decode 证据](../bench/amd_gfx1100_fused_decode_20260728/README.md)和
[0.4B–13.3B 回归证据](../bench/amd_gfx1100_rebase_validation_20260728/README.md)。

## 对比口径

- NVIDIA 表中两边使用同一张卡、相同 Batch、Prompt/Decode 长度和 FP16
  精度；Apple 表中两边使用各自正式的 MLX W4 路线。
- NVIDIA 的 Qwen3.5 全部记录并验证 **FLA + Triton causal-conv** 优化路径和
  fused operator 绑定。
- NVIDIA 的 RWKV 使用仓库的 Native prefill 与 native-graph cached decode；
  Apple 两边均为 MLX W4 target-only 路线。
- RTX 4080 采用双方各自已验证的优化运行时：RWKV 使用 PyTorch 2.11 的精确
  形状 FP16 accumulation，Qwen 使用 PyTorch 2.6 的 full-FLA 路线；版本和
  后端均记录在证据中，GPU、形状、Batch 和 FP16 精度保持一致。
- 模型按发布档位配对，例如 7.2B 对 9B；表中同时展示原始 tok/s、精确活跃
  参数和参数规模校正速度。
- NVIDIA 主表统一使用 dense FP16；Apple 表统一使用双方正式的 MLX W4，
  每张表内部保持一致口径。

## GPU 实测复现

下面的命令会在 GPU 上重新加载 RWKV-7 与 Qwen3.5、执行 warmup 和正式计时，
并重新生成 Prefill、Decode、参数规模校正速度和后端绑定结果。

### 1. 准备环境和模型

先使用对应证据目录里的 `environment.json`/`environment.txt` 对齐 PyTorch、
CUDA、Triton、Transformers、FLA 和 bitsandbytes 版本；每张卡按其已验证版本
创建独立环境。

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
git checkout 28f724259f8438cfcc71de40cf33889c6cf2396e

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[cuda,fla-reference,quant,torchao]"
```

准备两个本地目录：

1. 官方 RWKV-7 `.pth` 经
   [`scripts/convert_rwkv7_to_hf.py`](../scripts/convert_rwkv7_to_hf.py)
   转换后的 HF 模型目录；
2. 官方 Qwen3.5 HF 模型目录。

转换方法见[中文用户指南](USER_GUIDE_ZH.md#2-下载并转换模型)。先检查 RWKV：

```bash
python examples/check_environment.py --model /path/to/rwkv7-model-hf
```

必须看到 `RESULT: READY` 和 `[PASS] Model directory`。

### 2. 快速完整实测：RTX 4090、1.5B 对 2B、B8

```bash
OUT=/tmp/rwkv-qwen35-4090-b8

PYTHON_BIN=python \
BATCH_SIZES=8 \
PREFILL_CHUNK_SIZE=512 \
  bench/run_4090_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1h-1.5b-hf \
  /path/to/Qwen3.5-2B \
  "$OUT"

test "$(cat "$OUT/pipeline_exit_code.txt")" = 0
python - "$OUT/summary_active_work.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
speed = summary["speed"]
adjusted = summary["active_parameter_work"]
print(
    "raw prefill/decode median:",
    speed["median_prefill_speedup"],
    speed["median_decode_speedup"],
)
print(
    "parameter-adjusted prefill/decode:",
    adjusted["median_prefill_throughput_ratio"],
    adjusted["median_decode_throughput_ratio"],
)
print("red cells:", len(summary["red_cells"]))
PY
```

成功标准：脚本退出码为 0、`pipeline_exit_code.txt=0`、`red cells: 0`，并且
Qwen 行显示 full-FLA 优化路径；结果按中位值和两位小数进行对照。

### 3. 其他显卡入口

| GPU | 实测入口 | 说明 |
|---|---|---|
| V100 | [V100 证据中的命令](../bench/v100_active_b1b8_20260715/README.md#reproduce) | 1.5B/2B，B1/B8 |
| RTX 3090 最新检查点 | [`bench/run_3090_adjusted_prefill_pd.sh`](../bench/run_3090_adjusted_prefill_pd.sh) | 四个模型对、B1/B8、P128/512/2048、D128；逐格校正 Prefill 门槛与 25 格正确性门禁 |
| RTX 4080 | [`bench/run_4080_adjusted_pd.sh`](../bench/run_4080_adjusted_pd.sh) | 一次运行 3 个模型对、B1/B8 全部 36 格，并强制每格参数校正 P/D 均 `>1.00x` |
| RTX 4090 最新检查点 | [`bench/run_4090_adjusted_pd.sh`](../bench/run_4090_adjusted_pd.sh) | 三个模型对、B1/B8、P128/512/2048、D128/512；强制全部 36 格参数校正 P/D 均 `>1.00x` |
| RTX 5070 Laptop | [`bench/run_5070_qwen35_full_fla_bsz8.ps1`](../bench/run_5070_qwen35_full_fla_bsz8.ps1) | Windows PowerShell；通过 `-RwkvModel`、`-QwenModel`、`-OutDir` 传路径 |
| RTX 5090 | [`bench/run_5090_qwen35_full_matrix.sh`](../bench/run_5090_qwen35_full_matrix.sh) | 四个模型对、B1/B8 的完整矩阵 |
| RTX 5090 最新检查点 | [严格门槛证据中的命令](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md#reproduce-the-gate) | 四个模型对、B1/B8、P128/512/2048、D128 |

这些入口都会检查精确 GPU、后端绑定、矩阵覆盖和验收门槛，并在输出目录生成
`pipeline_exit_code.txt`、`matrix_failures.txt`、`summary*.json` 和完整日志。

### 4. Apple M5 GPU 实测

使用 MLX 环境和本地 W4 模型目录，执行 B8 target-only 正式对照：

```bash
PYTHON_BIN=/path/to/python \
MODEL_ROOT=/path/to/models \
COOLDOWN_SECONDS=30 \
INITIAL_COOLDOWN_SECONDS=60 \
  scripts/run_apple_bsz8_target_only_acceptance.sh
```

脚本会在 Apple GPU 上分别运行 RWKV-7 0.4B/1.5B 与 Qwen3.5 0.8B/2B，输出
原始 Prefill/Decode、参数规模校正口径、峰值内存和 token 一致性结果。

### 5. AMD `gfx1100` GPU 实测

使用 ROCm 7.2.1 对应的 PyTorch 环境，并确认用户可访问 `/dev/kfd` 和
`/dev/dri/render*`：

```bash
OUT=/tmp/rwkv7-amd-gfx1100
mkdir -p "$OUT"
set -o pipefail

bash bench/run_amd_rocm_hf_validation.sh \
  HF_DIR=/path/to/rwkv7-g1d-0.1b-hf \
  OUT_DIR="$OUT" |& tee "$OUT/console.log"

grep -F "AMD ROCm HF VALIDATION PASS" "$OUT/console.log"
```

脚本会验证 HIP 可见性和 `gcnArchName`；精确 `gfx1100` 会启用对应的融合
策略和架构调优。若只复现融合 Decode A/B，使用
[AMD 证据中的精简命令](../bench/amd_gfx1100_fused_decode_20260728/README.md#reproduce)。
