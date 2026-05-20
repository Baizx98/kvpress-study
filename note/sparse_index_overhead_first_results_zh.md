# SnapKV / ChunkKV / BlockWisePress 稀疏索引开销全层结果

## 1. 实验范围

本轮覆盖掉之前的单层结果，改为在物理 Device 0 的 L40S 上，对 `/Tan/model/Llama-3.1-8B-Instruct` 的全部 32 个 attention 层逐层测试，并对层维度取平均。

计时范围：

- 重要性分数计算；
- top-k / block selection / index construction；
- 不包含 K/V gather；
- 不包含真实模型 attention forward；
- 不把 embedding / q_proj / k_proj 计入 timed region。

真实权重使用方式：

- 直接从 safetensors index 加载真实权重张量；
- 加载 `model.embed_tokens.weight`；
- 加载 `model.layers.{0..31}.self_attn.q_proj.weight`；
- 加载 `model.layers.{0..31}.self_attn.k_proj.weight`；
- 对每个 layer，用真实 embedding 和该层 Q/K projection 在计时前生成 `q_window`、`keys`、`block_q_window`；
- 每个 op 使用 `warmup=5, repeat=10`，主结果使用 repeat mean；
- 主 CSV 再对 32 层的 mean 结果取平均。

BlockWisePress 已按要求删除 `mean_values` 和 `multi_rep_keys` block summary 构建逻辑。BlockWise summary 只构造低开销 key 相关摘要：

- `mean_keys`
- `topk_key_means`
- `token_counts`

实现侧同步做了两点优化：

- `representative_mode=key_norm` 的 summary 构建改为按 block 维度 reshape/vectorize，避免 Python 逐块循环；
- `summary_mode=mean_plus_norm_topk_mean` 且 `query_agg_mode=mean` 时，把 `mean_keys` 与 `topk_key_means` 先线性融合为一个 weighted anchor，只做一次 query-anchor 点积。

## 2. 运行配置

| 项目 | 值 |
|---|---:|
| GPU | physical device 0, NVIDIA L40S |
| `CUDA_VISIBLE_DEVICES` | `0` |
| 模型 | `/Tan/model/Llama-3.1-8B-Instruct` |
| attention layers | `32` |
| `hidden_size` | `4096` |
| `num_attention_heads` | `32` |
| `num_key_value_heads` | `8` |
| `head_dim` | `128` |
| dtype | `float16` |
| SnapKV window | `64` |
| ChunkKV chunk length | `20` |
| BlockWise block size | `16` |
| BlockWise q window | `32` |
| warmup / repeat | `5 / 10` |

产物：

- 全层平均 CSV: `evaluation/results/experiments/sparse_index_overhead_snapkv_chunkkv_blockwise/artifacts/sparse_index_overhead_summary.csv`
- 每层明细 CSV: `evaluation/results/experiments/sparse_index_overhead_snapkv_chunkkv_blockwise/artifacts/sparse_index_overhead_layers.csv`
- Metadata: `evaluation/results/experiments/sparse_index_overhead_snapkv_chunkkv_blockwise/artifacts/metadata.json`
- 脚本: `evaluation/bench_sparse_index_overhead.py`

## 3. 核心结果

### 3.1 长度 sweep

设置：`B=1, ratio=0.5, BlockWise reuse_steps=64`。下表为 32 层平均的 repeat mean。

| L | SnapKV online ms | ChunkKV online ms | BlockWise online ms | BlockWise amortized ms | BlockWise summary ms |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.301 | 1.356 | 0.518 | 0.524 | 0.388 |
| 4096 | 0.340 | 2.279 | 0.618 | 0.624 | 0.384 |
| 8192 | 0.695 | 4.144 | 0.756 | 0.761 | 0.332 |
| 16384 | 2.101 | 8.673 | 1.125 | 1.131 | 0.338 |
| 32768 | 4.596 | 19.243 | 2.099 | 2.108 | 0.598 |

观察：

- ChunkKV 仍然最慢，因为它继承 SnapKV token score，并额外做 chunk aggregation 与 token index 展开。
- 去掉 `multi_rep_keys` 并向量化 summary 后，BlockWise 一次性 summary 构建成本从上一版几十到数百 ms 降到 `0.33-0.60ms`。
- BlockWise online 随长度增长更慢；在 `L>=16384` 时 online 和 amortized 均明显低于 SnapKV。
- `reuse=64` 时，BlockWise 摊销后全长度均低于 ChunkKV；在长请求区间也低于 SnapKV。

### 3.2 Batch sweep

设置：`L=8192, ratio=0.5, BlockWise reuse_steps=64`。下表为 32 层平均的 repeat mean。

| B | SnapKV online ms | ChunkKV online ms | BlockWise online ms | BlockWise amortized ms | BlockWise summary ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.704 | 4.423 | 0.841 | 0.846 | 0.368 |
| 2 | 2.063 | 9.632 | 1.325 | 1.331 | 0.398 |
| 4 | 4.517 | 17.465 | 1.731 | 1.740 | 0.581 |
| 8 | 8.953 | 36.648 | 3.216 | 3.235 | 1.193 |

观察：

- batch 增大后，SnapKV 和 ChunkKV 放大更明显。
- BlockWise online 和 amortized 在 `B>=2` 后均低于 SnapKV。
- `B=8,L=8192` 下，BlockWise amortized `3.235ms`，低于 SnapKV `8.953ms` 和 ChunkKV `36.648ms`。

### 3.3 Compression ratio sweep

设置：`B=1, L=8192, BlockWise reuse_steps=64`。下表为 32 层平均的 repeat mean。

| ratio | SnapKV online ms | ChunkKV online ms | BlockWise online ms | BlockWise amortized ms | BlockWise summary ms |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.755 | 5.550 | 0.924 | 0.929 | 0.345 |
| 0.5 | 0.705 | 4.491 | 0.875 | 0.881 | 0.383 |
| 0.7 | 0.702 | 3.035 | 0.713 | 0.719 | 0.386 |
| 0.9 | 0.699 | 1.534 | 0.511 | 0.517 | 0.371 |

观察：

- SnapKV 主要由 score 计算决定，对 compression ratio 不敏感。
- ChunkKV 随保留 chunk 数减少而下降明显。
- BlockWise online 随压缩率提高而下降；在高压缩率 `ratio=0.9` 下，BlockWise amortized `0.517ms`，低于 SnapKV `0.699ms`。

### 3.4 BlockWise summary 摊销

设置：`B=1, L=8192, ratio=0.5`。下表为 32 层平均的 repeat mean。

| reuse steps | SnapKV online ms | ChunkKV online ms | BlockWise online ms | BlockWise summary ms | BlockWise amortized ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.705 | 4.435 | 0.800 | 0.372 | 1.173 |
| 4 | 0.705 | 4.467 | 0.811 | 0.373 | 0.904 |
| 16 | 0.709 | 4.557 | 0.843 | 0.399 | 0.868 |
| 64 | 0.716 | 4.530 | 0.835 | 0.386 | 0.841 |
| 256 | 0.704 | 4.475 | 0.816 | 0.387 | 0.817 |

观察：

- BlockWise summary build 一次性成本约 `0.37-0.40ms`。
- 即使 `reuse=1`，BlockWise amortized total 也只有 `1.173ms`，约为 ChunkKV 的 `26.4%`。
- 当 summary 被复用到 `reuse>=4`，BlockWise amortized total 降到 `0.90ms` 左右；`reuse=64` 时为 `0.841ms`。
- 在线路径本身约 `0.80-0.84ms`，远低于 ChunkKV；与 SnapKV 相比，在 `B=1,L=8192,ratio=0.5` 下略高，但在长请求、大 batch、高压缩率下更低。

## 4. 当前结论

confirmed:

1. 结果已覆盖上一版，并确认运行在 L40S：metadata 中 `gpu_name = NVIDIA L40S`、`cuda_visible_devices = 0`。
2. 本轮使用 Llama-3.1-8B-Instruct 全部 32 个 attention 层的真实 Q/K projection 权重。
3. 每个 op 都进行 `10` 次 timed repeat，主结果使用 repeat mean；summary CSV 是 32 层平均。
4. 删除 `mean_values` 和 `multi_rep_keys` 后，BlockWise summary 构建成本已经降到亚毫秒级；online index path 相对轻。
5. ChunkKV 不计 gather 后仍显著慢于 SnapKV/BlockWise，主要因为它同时承担 SnapKV token score 和 chunk/token index 构造。
6. BlockWise 的论点应表述为：它把昂贵的 token-level attention score 转换为 block-level summary score；一次性 summary 构建可复用，摊销后在长请求、大 batch 或高压缩率下明显低于 SnapKV，并在全部设置下显著低于 ChunkKV。

limits:

1. 本轮没有运行完整 transformer forward；只用真实 embedding 与各 attention 层 Q/K projection 权重在计时前生成 Q/K。
2. timed region 不含 Q/K 生成，因此结果回答的是“已有 Q/K 后构造稀疏索引”的开销。
3. BlockWise benchmark 使用最小等价索引逻辑，避免导入整个 `kvpress` 包引入额外依赖和退出卡住问题；它对齐当前 key-summary / score / top-k / token-index 逻辑，但不调用完整 `compress()`。
4. 结果只支持“稀疏索引构造开销”结论，不支持质量优劣结论。
