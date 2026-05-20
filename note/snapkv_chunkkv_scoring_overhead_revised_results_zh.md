# SnapKV / ChunkKV 打分开销修订结果

## 1. 本轮改动

这轮结果修正了上一版实验的两个问题：

1. `SnapKV` / `ChunkKV` 不再计入 K/V `gather` 开销，只计 attention score 与 top-k/index 构建。
2. attention kernel 对照不再只用完整 `L x L` prefill，而是让 prefill/decode 的 attention pair 数尽量匹配 `SnapKV` scoring 的 `W * L`。

修订后的实验方案见：

```text
note/snapkv_chunkkv_scoring_overhead_revised_plan_zh.md
```

## 2. 运行命令

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python evaluation/bench_scoring_overhead_snapkv_chunkkv.py \
  --lengths 2048 4096 8192 16384 \
  --window-sizes 64 \
  --chunk-lengths 20 \
  --warmup 10 \
  --repeat 50 \
  --dtype float16
```

结果路径：

- `evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/scoring_overhead_summary.csv`
- `evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/metadata.json`
- `figure/experiments/scoring_overhead_snapkv_chunkkv/`

## 3. 环境状态

实验设备：

```text
NVIDIA GeForce RTX 3090
```

当前 `.venv` 中原来的 `flash_attn 2.8.3` wheel 无法导入：

```text
ImportError: GLIBC_2.32 not found
host glibc = 2.31
```

我尝试过用本机 `nvcc 12.4` 从源码重编译 `flash-attn 2.8.3`：

```bash
CUDA_HOME=/home/bzx/local/cuda \
MAX_JOBS=2 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
uv pip install --python .venv/bin/python \
  --reinstall --no-deps --no-cache --no-binary :all: \
  --no-build-isolation flash-attn==2.8.3
```

但完整源码编译会编译大量 backward / 多 head-dim kernel，耗时过长，不适合作为本轮实验阻塞项。为了避免 `transformers.utils.is_flash_attn_2_available()` 误判为可用，我已经卸载这个坏 wheel。当前状态是：

```text
flash_attn_func = unavailable: ModuleNotFoundError("No module named 'flash_attn'")
is_flash_attn_2_available = False
```

因此本轮 benchmark 已改成：

1. 优先尝试 `flash_attn.flash_attn_func`；
2. 如果不可用，自动 fallback 到 PyTorch `scaled_dot_product_attention` 并强制 `FLASH_ATTENTION` backend；
3. fallback 原因写入 `metadata.json`。

本轮实际使用：

```text
torch.nn.functional.scaled_dot_product_attention forced FLASH_ATTENTION
```

这不是 `flash_attn` Python 包的直接调用，但仍是 fused flash attention backend。最终论文级结果如果必须写作 FlashAttention-2 包，应在容器或 glibc 更高的环境中复跑确认。

## 4. 关键结果

配置：

- `W = 64`
- `chunk_length = 20`
- `compression_ratio = 0.5`
- `dtype = float16`
- `query_heads = 32`
- `kv_heads = 8`
- `head_dim = 128`

| L | score pairs | fair prefill ms | fair decode ms | SnapKV no-gather ms | ChunkKV no-gather ms | Snap / fair prefill | Snap / fair decode | Chunk / fair decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 131072 | 0.112 | 0.110 | 0.444 | 2.394 | 3.98 | 4.05 | 21.84 |
| 4096 | 262144 | 0.174 | 0.148 | 0.706 | 4.395 | 4.06 | 4.78 | 29.71 |
| 8192 | 524288 | 0.262 | 0.228 | 1.548 | 8.686 | 5.91 | 6.78 | 38.04 |
| 16384 | 1048576 | 0.538 | 0.393 | 3.030 | 15.358 | 5.63 | 7.71 | 39.06 |

## 5. 结论

### 5.1 排除 gather 后，SnapKV scoring + top-k 仍显著慢于匹配计算量的 attention kernel

`SnapKV no-gather / fair decode attention` 从 `4.05x` 增长到 `7.71x`。

这说明核心瓶颈不是 K/V gather 造成的假象。即使只看 attention score 和 top-k index，当前 SnapKV 路径仍然比计算量相近的 fused attention kernel 慢很多。

### 5.2 ChunkKV 的 no-gather 开销更高，主要来自 chunk index 构建路径

`ChunkKV no-gather / fair decode attention` 达到 `21.84x ~ 39.06x`。

这里已经排除了 K/V gather，因此 ChunkKV 的额外成本来自：

- 继承 SnapKV token-level score；
- token score 到 chunk score 的聚合；
- top-k chunk；
- chunk index 展开成 token index 并排序。

下一步如果要进一步优化 ChunkKV，需要继续把 `chunk_index_ms` 拆成：

```text
chunk_score_agg_ms
chunk_topk_ms
token_index_construction_ms
```

### 5.3 fair prefill / fair decode 对照比 full prefill 更适合支撑论点

完整 prefill 的计算量是 `O(L^2)`，而 SnapKV scoring 是 `O(WL)`。如果只和 full prefill 比，长上下文下会弱化打分开销问题。

本轮用 `W * L` 匹配 attention pair 后，结论更清楚：

> 现有 SnapKV / ChunkKV 的重要性索引构建路径，比同等 attention pair 数的 fused attention kernel 更贵。这说明低开销 block-native scoring、score reuse、lazy refresh 是系统上必要的，而不是单纯的工程优化。

## 6. 推荐使用的图

优先用于汇报：

```text
figure/experiments/scoring_overhead_snapkv_chunkkv/scoring_overhead_presentation_summary.png
```

完整结果图：

```text
figure/experiments/scoring_overhead_snapkv_chunkkv/scoring_overhead_absolute_time.png
figure/experiments/scoring_overhead_snapkv_chunkkv/scoring_overhead_ratios.png
figure/experiments/scoring_overhead_snapkv_chunkkv/chunkkv_overhead_breakdown.png
```
