# SnapKV / ChunkKV 打分开销实验修订方案

## 1. 修订原因

上一版实验把 `ChunkKV` 的 K/V `gather` 也计入总开销，这会把两个问题混在一起：

1. 重要性打分和索引构建是否昂贵；
2. 按索引搬运 K/V tensor 是否昂贵。

这次实验只回答第一个问题：

> `SnapKV` / `ChunkKV` 为了得到保留索引，需要额外花多少时间做 attention score、score 聚合和 top-k/index 构建？

因此新实验不再把 K/V gather 计入 `SnapKV` 或 `ChunkKV` 的核心开销。

## 2. 新计时口径

### 2.1 SnapKV

`SnapKV` 只计：

```text
snap_total_no_gather =
  snap_attention_score
  + snap_token_topk_index
```

其中：

- `snap_attention_score`：recent-window query 与所有 key 计算 attention scores，再做 softmax、window mean、pooling、GQA group reduce、recent window pad。
- `snap_token_topk_index`：对 token score 做 top-k，构造后续 gather 会使用的 index tensor。
- 不计：`keys.gather()`、`values.gather()`。

### 2.2 ChunkKV

`ChunkKV` 只计：

```text
chunkkv_total_no_gather =
  snap_attention_score
  + chunk_score_aggregation
  + chunk_topk
  + token_index_construction
```

其中：

- `chunk_score_aggregation`：把 token score 聚合成 chunk score。
- `chunk_topk`：选择 top chunks。
- `token_index_construction`：把 chunk index 展开成 token index，并排序。
- 不计：`keys.gather()`、`values.gather()`。

当前脚本中先合并记录为：

```text
chunk_index_ms = chunk_score_aggregation + chunk_topk + token_index_construction
chunkkv_total_no_gather = snap_attention_score + chunk_index_ms
```

如果后续需要更细 breakdown，再把 `chunk_index_ms` 继续拆成三列。

## 3. Attention 对照的公平性

`SnapKV` 打分的核心 attention 形状是：

```text
q_len = W
kv_len = L
attention_pairs = W * L
```

因此 attention kernel 对照不应该只用完整 `L x L` prefill。完整 prefill 的 causal attention pair 数约为：

```text
L * (L + 1) / 2
```

它远大于 `W * L`，会让 SnapKV 打分显得过于便宜。

新实验采用三组 attention 对照。

### 3.1 Fair prefill attention

选择一个 `S_prefill`，让 causal prefill 的 attention pair 数尽量接近 `W * L`：

```text
S_prefill * (S_prefill + 1) / 2 ~= W * L
S_prefill = floor((sqrt(1 + 8 * W * L) - 1) / 2)
```

记录：

```text
fair_prefill_fa_ms
fair_prefill_attention_pairs
```

这个指标用于回答：

> 如果 prefill attention 的计算量与 SnapKV scoring 接近，FlashAttention kernel 本身需要多长时间？

### 3.2 Fair batched decode attention

用 `W` 个 decode query 同时 attend 到长度为 `L` 的 KV cache：

```text
q_len = W
kv_len = L
attention_pairs = W * L
```

记录：

```text
decode_fair_batched_fa_ms
decode_attention_pairs
```

这个对照的计算量与 SnapKV scoring 完全一致，且只比较 fused attention kernel 本身。

### 3.3 Sequential decode attention reference

真实 decode 常是 `q_len = 1`。因此额外记录：

```text
decode_single_fa_ms
decode_single_times_window_ms = decode_single_fa_ms * W
```

它不是主公平对照，因为重复 `W` 次会引入 kernel launch 开销；但它更贴近在线 decode step 的实际调度形态。

### 3.4 Full prefill attention 只作为参考

仍可记录：

```text
full_prefill_fa_ms
```

但它不是主要公平比较对象，只用于说明完整 prefill 的 `L^2` 计算量增长趋势。

## 4. 主实验配置

建议先跑 3090 上的主实验：

```text
L = 2048, 4096, 8192, 16384
W = 64
chunk_length = 20
compression_ratio = 0.5
batch_size = 1
query_heads = 32
kv_heads = 8
head_dim = 128
dtype = float16
warmup = 10
repeat = 50
```

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

## 5. 输出字段

关键字段：

| 字段 | 含义 |
|---|---|
| `score_attention_pairs` | `W * L` |
| `fair_prefill_len` | 与 `W * L` 近似匹配的 prefill 长度 |
| `fair_prefill_attention_pairs` | `S_prefill * (S_prefill + 1) / 2` |
| `decode_attention_pairs` | `W * L` |
| `fair_prefill_fa_ms_median` | 计算量匹配后的 prefill FlashAttention 时间 |
| `decode_fair_batched_fa_ms_median` | 计算量匹配后的 batched decode FlashAttention 时间 |
| `decode_single_times_window_ms` | 单步 decode 时间乘以 `W` |
| `snap_score_ms_median` | SnapKV attention score 计算时间 |
| `snap_topk_index_ms_median` | SnapKV token top-k/index 时间 |
| `snap_total_no_gather_ms_median` | SnapKV 不含 gather 的总时间 |
| `chunk_index_ms_median` | ChunkKV chunk 聚合/top-k/index 时间 |
| `chunkkv_total_no_gather_ms_median` | ChunkKV 不含 gather 的总时间 |

主要比例：

```text
snap_vs_fair_prefill_fa
snap_vs_decode_fair_batched_fa
snap_vs_score_shape_fa
chunk_vs_fair_prefill_fa
chunk_vs_decode_fair_batched_fa
chunk_vs_score_shape_fa
snap_vs_decode_single_times_window
```

## 6. FlashAttention 环境处理

当前 `.venv` 里的 `flash_attn 2.8.3` wheel 不能用，原因是：

```text
flash_attn_2_cuda...so requires GLIBC_2.32
host glibc = 2.31
```

修复策略：

1. 不再使用这个预编译 wheel；
2. 优先尝试使用本机 `nvcc 12.4` 从源码编译 `flash-attn`，使生成的 `.so` 链接本机 glibc；
3. 如果源码编译耗时过长、失败或磁盘不足，则卸载坏 wheel，避免上层误判 `flash_attn` 可用；
4. 退回 PyTorch SDPA forced `FLASH_ATTENTION` backend，并在结果 metadata 中明确标注。

注意：本仓库 `.venv` 没有 `pip` 模块，包管理应使用 `uv pip`。

## 7. 预期结论口径

修订后的结论应该写成：

> 在排除 K/V gather 后，SnapKV / ChunkKV 的核心在线开销仍来自 recent-window attention score 和 top-k/index 构建。与计算量匹配的 FlashAttention prefill/decode kernel 相比，如果 `snap_total_no_gather` 或 `chunkkv_total_no_gather` 仍显著更高，就能证明当前重要性打分路径本身存在系统瓶颈，而不是 K/V 搬运造成的假象。

避免写成：

> ChunkKV 很慢主要是 gather 慢。

这不是本轮实验要证明的问题。
