# SnapKV / ChunkKV / BlockWisePress 稀疏索引开销实验方案

## 1. 问题定义

本实验只比较三类方法在“得到稀疏索引”上的时间开销，不比较 K/V gather、真实 attention forward、模型端到端生成速度或质量指标。

目标问题：

> 给定一层 KV cache 和当前请求上下文，SnapKV、ChunkKV、BlockWisePress 为了计算重要性分数并得到保留 token/block 索引，各自需要多少时间？

这里的“稀疏索引开销”定义为：

- 计算重要性分数；
- 对分数做聚合、保护区处理、top-k；
- 构造最终保留 token index 或 block index；
- 不包含 `keys.gather()` / `values.gather()`；
- 不包含真实模型 attention kernel；
- 不包含 tokenizer、数据集、调度器等外部开销。

## 2. 方法计时边界

### 2.1 SnapKV

计入：

1. 最近 `window_size` 个 query 与全量 K 做 attention score；
2. causal mask / softmax；
3. 对 recent-window 维度求均值；
4. `avg_pool1d` 平滑；
5. GQA group 聚合到 KV heads；
6. recent window padding；
7. token-level `topk` 得到保留 token index。

不计入：

- 根据 index gather K/V；
- RoPE 和 hidden_states 到 Q 的真实模型投影开销。benchmark 中直接构造等价 `q_window`，避免把模型算子混入稀疏索引开销。

核心复杂度近似：

```text
O(B * Hq * W * L * D) + O(B * Hkv * L log K)
```

其中 `W=window_size`，`L=request length`。

### 2.2 ChunkKV

本地 `ChunkKVPress` 的实际边界是 `underlying ScorerPress.score()` 加 chunk-level selection。为了和代码一致，默认底层 scorer 使用 `SnapKVPress`。

计入：

1. 完整 SnapKV score 计算；
2. token score 到 chunk score 的聚合；
3. chunk-level `topk`；
4. chunk index 展开成 token index；
5. sort / stack 等索引构造开销。

不计入：

- `keys.gather()` / `values.gather()`；
- 因真实 Python for-loop 实现带来的 gather 后处理之外的任何 K/V 访问。

核心复杂度近似：

```text
SnapKV score cost + O(B * Hkv * L) aggregation + O(B * Nchunk log Kchunk) topk/index
```

### 2.3 BlockWisePress

BlockWisePress 的开销需要拆成两类，因为 block summary 可以缓存复用。

**一次性 summary 构建成本**

计入：

- 每个 block 的 `mean_keys`、`mean_values`；
- `representative_mode=key_norm` 下的 block 内代表 key 选择；
- `topk_key_means` / `multi_rep_keys` 的构造；
- 写入 `last_block_summary`。

这部分只在 KV 内容变化或强制 refresh 时发生。用户指出“块摘要生成是一次性的，后面可直接使用之前结果”，因此主对比不应把它完整算到每次迭代。

**每次迭代在线索引成本**

计入：

1. 使用缓存好的 block summary；
2. 当前 tail query window 与 block summary anchor 做 coarse score：
   - `mean_keys` score；
   - `topk_key_means` score；
   - 若使用 `multi_rep_max`，再计入 multi-representative score；
3. query 维度聚合；
4. head 维度聚合；
5. prefix sink / recent protected blocks / partial tail block 保护逻辑；
6. block-level `topk`；
7. block index 展开为 token index。

不计入：

- `gather_by_token_indices()`；
- 重新 summarize compressed keys 的后处理；
- 真实模型 Q projection。

最终报告两个 BlockWise 数字：

```text
blockwise_online_ms
blockwise_amortized_ms = blockwise_online_ms + summary_build_ms / reuse_steps
```

`reuse_steps` 表示同一份 block summary 被复用多少次。建议 sweep：

```text
reuse_steps in {1, 4, 16, 64, 256}
```

主文结论使用 `reuse_steps=64` 或 `256`，同时保留 `reuse_steps=1` 作为最保守上界。

## 3. 实验维度

### 3.1 默认参数

建议使用与当前仓库和 Llama-family 模型一致的 synthetic tensor：

| 参数 | 默认值 |
|---|---:|
| GPU | physical device 1, RTX 3090 |
| dtype | `float16` |
| batch size | `1, 2, 4, 8` |
| request length `L` | `2048, 4096, 8192, 16384, 32768` |
| query heads `Hq` | `32` |
| KV heads `Hkv` | `8` |
| head dim `D` | `128` |
| SnapKV window `W` | `64` |
| SnapKV kernel size | `5` |
| ChunkKV chunk length | `20` |
| BlockWise block size | `16` |
| BlockWise q window | `32` |
| compression ratio | `0.5, 0.7` |
| warmup / repeat | `10 / 50`，OOM 或长长度可降到 `5 / 20` |

### 3.2 Sweep 设计

至少做两组 sweep。

**A. 长度 sweep**

固定：

```text
B=1
compression_ratio=0.5
L in {2k, 4k, 8k, 16k, 32k}
```

目的：展示分数计算和索引构造随请求长度增长的趋势。SnapKV/ChunkKV 近似随 `W * L` 增长；BlockWise online 近似随 `q_window * (L / block_size)` 增长。

**B. batch sweep**

固定：

```text
L=8192
B in {1, 2, 4, 8}
compression_ratio=0.5
```

目的：展示在正常 batch 场景下，三种方法的索引开销是否随 batch 线性放大，以及 Python 索引构造是否成为 ChunkKV/BlockWise 的额外瓶颈。

**C. compression ratio sweep**

固定：

```text
B=1
L=8192
compression_ratio in {0.3, 0.5, 0.7, 0.9}
```

目的：区分“score 计算成本”和“top-k/index 构造成本”。score 计算通常与压缩率弱相关；top-k 和最终 index 长度与压缩率相关。

**D. BlockWise amortization sweep**

固定：

```text
B=1
L=8192
compression_ratio=0.5
reuse_steps in {1, 4, 16, 64, 256}
```

目的：明确 BlockWise summary 一次性成本被摊薄后，在线索引成本是否显著低于 SnapKV/ChunkKV。

## 4. 计时实现

### 4.1 Synthetic module

为了不引入真实模型 forward，benchmark 里构造一个最小 `FakeLlamaAttentionModule`：

- `config.num_attention_heads = 32`
- `config.num_key_value_heads = 8`
- `head_dim = 128`
- `layer_idx = 0`

对于 SnapKV/ChunkKV，不直接调用 `get_prerope_query_states()`，而是实现等价的 synthetic score kernel，输入已经是 `q_window` 和 `keys`。

对于 BlockWise，也建议先实现 synthetic 等价路径：

- 直接构造 `kv_query_states: [B, Hkv, q_window, D]`；
- 直接调用 `_compute_summary_scores_per_head()`、`aggregate_head_scores()`、`_select_top_block_indices()` 和 `expand_blocks_to_token_indices()`；
- summary build 单独调用 `_summarize_blocks()` 或等价函数。

这样能保证计时对象就是“稀疏索引逻辑”，不会混入真实模型 Q projection。

### 4.2 CUDA event 计时

每个 case：

1. 预生成 tensor，避免把随机初始化计入；
2. warmup；
3. 使用 `torch.cuda.Event` 记录 GPU elapsed time；
4. 每次 timed op 后 `torch.cuda.synchronize()`；
5. 报告 median / p10 / p90；
6. 记录 peak memory。

对于 Python-heavy index 构造，例如 ChunkKV 的 loop 展开 token index，同时记录：

- CUDA event time；
- wall-clock time。

如果二者差距明显，说明瓶颈在 CPU/Python 调度或同步。

## 5. 输出指标

CSV 建议字段：

| 字段 | 含义 |
|---|---|
| `method` | `snapkv` / `chunkkv` / `blockwise` |
| `phase` | `score` / `topk_index` / `summary_build` / `online_index` / `amortized_total` |
| `batch_size` | batch size |
| `length` | request length |
| `compression_ratio` | 压缩率 |
| `score_ms_median` | 分数计算 median |
| `topk_index_ms_median` | top-k/index median |
| `total_index_ms_median` | score + topk/index |
| `summary_build_ms_median` | BlockWise 一次性 summary 成本 |
| `reuse_steps` | BlockWise summary 复用步数 |
| `amortized_total_ms_median` | BlockWise 摊销后成本 |
| `cuda_event_ms_*` | GPU event 统计 |
| `wall_ms_*` | wall-clock 统计 |
| `peak_memory_mb` | 峰值显存 |

## 6. 图表

建议至少生成 4 张图：

1. `sparse_index_length_sweep.png`
   - x: request length
   - y: total sparse-index time, log scale
   - lines: SnapKV, ChunkKV, BlockWise online, BlockWise amortized

2. `sparse_index_batch_sweep.png`
   - x: batch size
   - y: total sparse-index time
   - lines: 三种方法

3. `sparse_index_breakdown.png`
   - stacked bar
   - SnapKV: score / topk
   - ChunkKV: SnapKV score / chunk aggregation+index
   - BlockWise: summary_build/reuse_steps / online score / topk+expand

4. `blockwise_amortization.png`
   - x: reuse steps
   - y: BlockWise amortized total time
   - reference horizontal lines: SnapKV total、ChunkKV total

## 7. 预期结论与风险

### 预期结论

1. SnapKV 和 ChunkKV 的主要在线成本来自 recent-window attention score，随 `W * L` 增长。
2. ChunkKV 相比 SnapKV 的额外成本主要来自 chunk 聚合、chunk top-k 和 Python token index 展开；如果不计 gather，增量应小于旧实验。
3. BlockWise 的 summary build 可能不便宜，但这是一次性成本；当 `reuse_steps` 足够大时，摊销后在线索引开销应该显著低于 SnapKV/ChunkKV。
4. 如果 `reuse_steps=1` 时 BlockWise 不占优，但 `reuse_steps>=64` 后占优，这正好支持“块摘要可复用”的系统设计论点。

### 风险

1. Synthetic query 与真实模型 query 分布不同：本实验只用于时间开销，不用于质量结论。
2. Python loop 会放大 ChunkKV 和 BlockWise 的 index 展开成本：需要同时报告 CUDA event 和 wall-clock。
3. BlockWise 当前 `compress()` 会强制 refresh summary 并 gather；benchmark 必须绕开 `compress()`，直接测 `build_block_plan()` 的子步骤。
4. 如果 batch 内不同样本的尾块长度不同，BlockWise token index 展开可能要求相同 kept length；synthetic sweep 先使用同长 batch。

## 8. 最小验证计划

1. 在 RTX 3090 上先跑 smoke case：

```text
B=1, L=4096, ratio=0.5
```

确认三种方法输出 index shape 正确，且不触发 K/V gather。

2. 跑完整 length sweep 和 batch sweep。
3. 生成 CSV、metadata、README 和 4 张图。
4. 在结果分析中只下时间开销结论，不讨论质量优劣。

