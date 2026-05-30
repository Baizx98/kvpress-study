# ATC26 BlockWise 时间步索引相似性实验方案

## 1. 问题定义

这组实验回答一个面向系统优化的问题：

> 对同一模型、同一层、同一条样本，BlockWisePress 在连续 decode step 上选择的 KV block index 是否高度相似？如果相邻 step 相似，而远距离 step 逐渐下降，是否可以每隔固定步长才重新计算一次稀疏索引？

已有 `ATC26_blockwise_attention_similarity_hotpotqa_3samples` 实验主要比较 layer 间和 KV head 间的 kept-block overlap。它不能直接支持“固定步长复用稀疏索引”，因为它没有记录同一层在 decode 时间维度上的选择变化。

## 2. 假设

核心假设：

1. 同一层相邻 decode step 的 block score 分布高度相似。
2. 同一层相邻 decode step 的 top-k kept block index 集合高度重叠。
3. 随着 step distance 变大，相似度下降，但在一个中等窗口内仍保持较高水平。
4. 因此，BlockWisePress 可以把在线稀疏索引刷新频率从 every step 降到 every `R` steps，例如 `R=4/8/16`，从而减少 scoring 和 top-k/index 构建开销。

需要避免过度表述：这个实验只能证明 index 选择的时间稳定性，不直接证明最终质量不下降。若要在论文中主张“可以固定步长刷新”，还需要后续做复用索引的质量和延迟实验。

## 3. 核心设计

### 3.1 Trace 对象

在 decode 过程中，对每个生成 step `t`、每一层 `l` 记录：

```text
score[l, t, b] = BlockWise block score for block b
S[l, t] = top-k selected historical block index set
```

主指标只比较 stable historical block universe：

```text
U0 = answer decode 开始前已经存在的 context + question KV blocks
```

原因是 decode 过程中 KV cache 会继续增长。如果把新生成 token 所在 block 也纳入比较，相似度会受到 block universe 变化和 recent-block 保护策略影响，不能纯粹反映“历史块重要性是否稳定”。论文主图应使用 `U0`；附录可以补充 dynamic full-cache 口径。

### 3.2 选择集合定义

对每个 layer 和 step，先在固定 block universe `U0` 上计算 BlockWise score，再用固定预算选择 top-k：

```text
K0 = ceil(|U0| * (1 - compression_ratio))
S[l, t] = TopK(score[l, t, b], b in U0, K0)
```

主口径建议排除以下总是被保护、且不代表 score 决策的块：

- prefix sink blocks
- protected recent blocks
- partial tail block

因此保存两组集合：

1. `selected_blocks_full`: 包含保护块，贴近真实 BlockWise kept set。
2. `selected_blocks_scored_only`: 只包含通过 score 决策选出的历史块，作为论文主指标。

### 3.3 单步 query 与窗口 query

同时记录两种 score：

1. `single_query`: 每个 decode step 只用当前 token 的 hidden state 做 block score。
2. `window_query`: 使用最近 `W` 个 decode hidden states 聚合后做 block score，默认 `W=16`。

论文主结论应优先看 `single_query`，因为它最严格地验证相邻 token 的注意力分布是否稳定。`window_query` 更接近工程实现里的 fixed-interval refresh，但相邻窗口天然共享大量 query，会让相似度偏高，适合作为系统实现口径。

## 4. 指标

### 4.1 相邻 step 相似性

同一层内比较 `t` 和 `t+1`：

```text
Jaccard[l, t, 1] = |S[l,t] intersection S[l,t+1]| / |S[l,t] union S[l,t+1]|
Overlap[l, t, 1] = |S[l,t] intersection S[l,t+1]| / K0
Cosine[l, t, 1] = cosine(score[l,t], score[l,t+1])
```

其中 `Overlap` 更直接对应“复用旧索引时能覆盖多少新 oracle block”。

### 4.2 不同 step distance 曲线

对 lag 做分组：

```text
lag in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
```

计算：

```text
Jaccard(lag) = mean_{l,t} Jaccard(S[l,t], S[l,t+lag])
Overlap(lag) = mean_{l,t} |S[l,t] intersection S[l,t+lag]| / K0
Cosine(lag) = mean_{l,t} cosine(score[l,t], score[l,t+lag])
```

主图建议画 `layer x lag` heatmap，以及全层均值的 lag 曲线。论文文字重点比较：

- adjacent: `lag=1`
- practical refresh interval: `lag=4/8/16`
- far steps: `lag=64/128/256/512`

### 4.3 固定步长复用收益指标

定义 oracle 为每个 step 都重新算索引：

```text
S_oracle[l,t] = S[l,t]
```

定义固定步长刷新：

```text
S_reuse_R[l,t] = S[l, floor(t/R) * R]
```

计算：

```text
ReuseRecall_R = |S_reuse_R[l,t] intersection S_oracle[l,t]| / |S_oracle[l,t]|
ReuseJaccard_R = Jaccard(S_reuse_R[l,t], S_oracle[l,t])
RefreshReduction_R = 1 - 1/R
```

建议报告 `R in {2, 4, 8, 16, 32, 64, 128, 256, 512}`。如果 `R=8` 或 `R=16` 仍能保持高 `ReuseRecall`，就可以作为“fixed-step sparse index refresh”的主要证据。

## 5. 实验矩阵

### 5.1 主实验

主实验优先用一个模型和一个稳定长输出数据源，保证 A6000 上快速出结论。

```text
model: /Tan/model/Llama-3.1-8B-Instruct
dataset: PG19 continuation
context_lengths: 8192, 16384
samples_per_length: 4
decode_steps: 256
compression_ratio: 0.5
block_size: 16
window_query_size: 16
dtype: auto / fp16
seed: 42
device: physical GPU 2, A6000
```

选择 PG19 的原因：它天然提供长文本 continuation，适合 teacher-forced decode trace，不会受到 QA 任务答案过短或 EOS 过早停止的影响。

### 5.2 论文稳健性补充

如果主实验支持假设，再扩展到 ATC26 已使用的模型和 LongBench 样本：

```text
models:
  - /Tan/model/Llama-3.1-8B-Instruct
  - /Tan/model/Mistral-7B-Instruct-v0.3
  - /Tan/model/Qwen3-8B

datasets:
  - PG19 continuation, context_length=8192, samples=4
  - LongBench/hotpotqa, samples=3, forced greedy decode_steps=128

compression_ratios:
  - 0.3
  - 0.5
  - 0.7
```

LongBench 只作为跨任务补充，不作为主口径，因为生成 QA 答案容易很短。若使用 LongBench，脚本应固定 decode steps 并忽略 EOS，仅用于 trace attention/index，不用于质量评测。

## 6. 采集脚本设计

建议新增脚本：

```text
evaluation/ATC26_collect_blockwise_temporal_index_similarity.py
```

关键实现点：

1. 使用 `DynamicCache` 手写 prefill + decode loop，不走会提前遇到 EOS 停止的 pipeline 默认 `generate_answer`。
2. prefill 后冻结 `U0`，只比较 answer decode 开始前的 historical blocks。
3. 对每个 decode step，拿每层当前 hidden state 和 cache 里的 K/V，调用 BlockWisePress 的 `build_block_plan()` 或复用其 `analyze_blocks()` 逻辑，只记录分数和索引，不真的 gather/压缩 K/V。
4. 保存每层、每 step 的 score summary 和 top-k index。完整 score tensor 较大，默认保存 fp16 或 top-k score；debug 模式再保存完整 score。
5. teacher-forced 模式优先：PG19 continuation tokens 作为 decode 输入，保证每次运行可复现。
6. greedy forced 模式作为 LongBench 补充：固定生成 `decode_steps`，即使生成 EOS 也继续喂入 greedy token。

推荐输出：

```text
evaluation/results/experiments/ATC26_blockwise_temporal_index_similarity/
  README.md
  artifacts/
    ATC26_temporal_similarity_config.json
    ATC26_temporal_similarity_manifest.json
    ATC26_temporal_similarity_raw.jsonl
    ATC26_temporal_similarity_aggregate.csv
    ATC26_temporal_similarity_aggregate.json
    scores/*.npz
    logs/
```

推荐绘图脚本：

```text
figure/ATC26_plot_blockwise_temporal_index_similarity.py
```

推荐图目录：

```text
figure/experiments/ATC26_blockwise_temporal_index_similarity/
```

## 7. Watchdog 设计

建议新增：

```text
evaluation/ATC26_watch_blockwise_temporal_index_similarity.sh
```

运行设备明确绑定到 physical GPU 2：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
PYTHONUNBUFFERED=1 \
.venv/bin/python evaluation/ATC26_collect_blockwise_temporal_index_similarity.py \
  --device cuda:0 \
  --model-key llama31_8b_instruct \
  --dataset pg19 \
  --context-lengths 8192 16384 \
  --samples-per-length 4 \
  --decode-steps 512 \
  --compression-ratio 0.5 \
  --block-size 16 \
  --window-query-size 16 \
  --seed 42 \
  --resume
```

这里用 `CUDA_VISIBLE_DEVICES=2` 后，脚本内部应使用 `cuda:0`。这样比直接传 `cuda:2` 更不容易误用其它 GPU。

Watchdog 不应只看进程退出码，还要看 heartbeat：

```text
artifacts/ATC26_temporal_similarity_heartbeat.json
```

heartbeat 至少包含：

```json
{
  "updated_at": "ISO timestamp",
  "current_model": "llama31_8b_instruct",
  "current_dataset": "pg19",
  "current_context_length": 8192,
  "current_sample": 2,
  "current_step": 137,
  "completed_jobs": 5,
  "total_jobs": 8
}
```

重启规则：

1. 子进程正常退出且 aggregate 完整：watchdog 退出。
2. 子进程异常退出：等待 60 秒后 `--resume` 重启。
3. 子进程仍在但 heartbeat 超过 20 分钟未更新：记录 `ps`、`nvidia-smi`、最近日志；先向日志写入 stale 状态，再 kill 当前进程树并 `--resume` 重启。
4. 若连续失败超过 5 次：watchdog 退出并留下失败摘要，避免无限重启掩盖真实 bug。

## 8. 论文图表建议

主文建议放 2 张图：

1. `layer x lag` heatmap：展示同一层内不同 step distance 的 kept-block Jaccard 或 Overlap。
2. `reuse interval curve`：横轴为 refresh interval `R`，左轴为 `ReuseRecall_R`，右轴标注 `RefreshReduction_R`。

主文文字可以这样组织：

> Within the same layer, BlockWise block selection is highly stable across adjacent decoding steps. The overlap gradually decreases as the step distance grows, but remains high within practical refresh intervals. This suggests that sparse block indices can be refreshed every few decoding steps instead of being recomputed at every step.

如果结果不支持假设，应改写为：

> Block score distributions remain smooth across adjacent steps, but hard top-k indices fluctuate near the decision boundary. In that case, a soft score reuse or candidate-union policy may be more robust than directly reusing a single hard index set.

## 9. 最小验证计划

1. Smoke：`Llama-3.1-8B / PG19 / 1 sample / 8192 context / 32 decode steps / ratio=0.5`。
2. 主实验：`Llama-3.1-8B / PG19 / 4 samples x 2 lengths / 256 decode steps / ratio=0.5`。
3. 稳健性：加入 `ratio=0.3/0.7`，确认高压缩率下 top-k 边界是否更不稳定。
4. 跨模型：加入 Mistral 和 Qwen，只跑 `context_length=8192`。
5. LongBench 补充：`hotpotqa` 3 samples，固定 greedy decode 128 steps，作为与现有 ATC26 LongBench 实验的连接。

## 10. 风险与限制

1. Teacher-forced PG19 更适合稳定 trace，但和真实自由生成存在分布差异。
2. LongBench QA 输出较短，若不强制继续 decode，会导致 step 数不足。
3. 相邻 window-query 相似度可能被窗口重叠放大，不能单独作为主证据。
4. 只看 index similarity 不能替代质量实验；后续仍需跑 fixed-refresh BlockWise 的任务分数。
5. A6000 上 16k context x 256 decode steps 可能较慢，必须有 `--resume`、heartbeat 和 watchdog。
