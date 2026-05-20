# SnapKV / ChunkKV 打分开销初步结果

## 1. 实验状态

本轮已经在物理 `nvidia-smi index 1` 的 `NVIDIA GeForce RTX 3090` 上完成第一组 synthetic per-layer microbenchmark。

运行命令：

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

- 汇总表：`evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/scoring_overhead_summary.csv`
- 元数据：`evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/metadata.json`
- raw json：`evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/raw/`
- benchmark 脚本：`evaluation/bench_scoring_overhead_snapkv_chunkkv.py`

## 2. 环境说明

本机 `.venv` 中的 `flash_attn` Python 包当前不能加载：

```text
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
```

因此本轮 attention kernel 对照使用的是 PyTorch：

```text
torch.nn.functional.scaled_dot_product_attention
forced FLASH_ATTENTION backend
```

这仍然是 fused attention kernel 对照，但不是 `flash_attn.flash_attn_func` 包的直接调用。后续如果要在论文图里写成 FlashAttention-2，需要换到能正常 import `flash_attn` 的环境重新确认一次。

另一个注意点：默认 CUDA 枚举顺序下，`CUDA_VISIBLE_DEVICES=1` 对应的是 A6000，不是 `nvidia-smi index 1`。必须加：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
```

才能锁定物理 3090。

## 3. 初步结果

配置：

- `batch_size = 1`
- `query_heads = 32`
- `kv_heads = 8`
- `head_dim = 128`
- `dtype = float16`
- `observe window W = 64`
- `chunk_length = 20`
- `compression_ratio = 0.5`
- `warmup = 10`
- `repeat = 50`

| L | prefill FA ms | decode FA ms | window FA ms | SnapKV score ms | ChunkKV total ms | Snap / prefill | Snap / decode | Chunk / decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.671 | 0.074 | 0.106 | 0.400 | 2.432 | 0.596 | 5.43 | 32.99 |
| 4096 | 2.694 | 0.086 | 0.148 | 0.649 | 4.456 | 0.241 | 7.56 | 51.88 |
| 8192 | 10.356 | 0.114 | 0.229 | 1.224 | 8.595 | 0.118 | 10.78 | 75.69 |
| 16384 | 47.045 | 0.168 | 0.395 | 2.812 | 15.232 | 0.060 | 16.76 | 90.79 |

## 4. 当前能支持的结论

### 4.1 SnapKV scoring 对 decode-like attention 已经非常贵

`SnapKV score / decode FA` 从 `5.43x` 增长到 `16.76x`。

这说明如果在 decode 或在线 refresh 路径里频繁做 SnapKV 风格 recent-window token-level scoring，打分本身会比单步 attention kernel 贵很多。这个结论正好支持我们要强调的系统问题：

> 现有 KV 重要性估计不是免费元数据计算，而可能成为在线压缩/选择的主瓶颈。

### 4.2 对 full prefill attention，打分占比随长度下降

`SnapKV score / prefill FA` 从 `59.6%` 降到 `6.0%`。

这也符合复杂度预期：full prefill attention 近似随 `L^2` 增长，而固定 window 的 SnapKV scoring 近似随 `W * L` 增长。因此不能简单说 “SnapKV 一定比 FlashAttention prefill 更慢”。

更严谨的说法是：

> 在 full prefill attention 中，SnapKV scoring 对短中长度仍不可忽略；但在长上下文下，它相对 full prefill kernel 的比例会下降。真正更尖锐的问题出现在 decode-like refresh、分块热度更新、offload/prefetch 在线决策等场景。

### 4.3 ChunkKV 当前路径的额外开销很大

本地 `ChunkKVPress` 是 `SnapKV score -> chunk aggregation -> top-k chunks -> token index construction -> gather K/V`。

本轮 synthetic 实现里，`ChunkKV total / decode FA` 从 `32.99x` 到 `90.79x`。这说明如果把 ChunkKV 的选择逻辑放到在线路径，当前实现形态的系统开销非常高。

需要注意：当前脚本把 chunk 聚合、top-k/index、gather 合成了 `chunk_extra_ms`，尚未进一步拆成三列。因此下一步应该把它拆开，判断主要开销到底来自：

- token score 到 chunk score 的聚合；
- top-k 和 token index 构造；
- K/V gather；
- 还是大量小 kernel / Python loop 造成的调度开销。

## 5. 下一步建议

### Step 1：补 window sweep

固定 `L = 8192, 16384`，扫：

```text
W = 16, 32, 64, 128, 256
```

目标：证明 `SnapKV score` 基本随 observe window 线性增长。

### Step 2：拆 ChunkKV breakdown

把当前 `chunk_extra_ms` 拆成：

```text
chunk_score_agg_ms
chunk_topk_index_ms
chunk_gather_ms
```

目标：明确 ChunkKV 在线路径最贵的是打分继承、索引构造，还是 gather。

### Step 3：补真实 press hook 计时

在 `SnapKVPress.score()` 和 `ChunkKVPress.compress()` 上插临时 profiling wrapper，用 synthetic prompt 跑一次真实模型 prefill，确认 microbenchmark 的趋势能在实际 kvpress 路径中复现。

### Step 4：如果需要论文级图，换可用 FlashAttention-2 环境复跑

当前 PyTorch SDPA forced flash 可以作为快速证据，但最终若要严谨写 `FlashAttention-2`，应修复 `flash_attn` 的 GLIBC 依赖或换容器后复跑。

