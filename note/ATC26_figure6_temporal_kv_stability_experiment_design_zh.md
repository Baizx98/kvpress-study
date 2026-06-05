# ATC26 Figure 6：重要 KV Block 集合时间稳定性实验方案

## 1. 问题定义

论文 Observation 2/3 需要证明同一条生成轨迹中，step `i` 与 step `i + Delta` 的重要 KV block 集合存在两个同时成立的现象：

1. 当 `Delta` 很小时，重要 KV block 集合高度重叠，说明当前 step 的 sparse score 对近未来访问有预测性。
2. 当 `Delta` 变大时，重叠率逐渐下降，尤其在更严格的 sparsity budget 下更明显，说明一次性静态压缩不能覆盖整个 generation。

因此 Figure 6 的核心不是证明 BlockWisePress 的最终任务质量，而是证明 sparse block selection 的 temporal predictability 和 long-horizon drift。

## 2. 核心假设

### Hypothesis A：短期局部稳定

在相邻或近邻 decode steps 中，query 语义变化较小，attention 仍倾向访问相似的历史 KV blocks。因此：

```text
Overlap(S_i, S_{i+Delta}) 高，Delta in {1, 2, 4, 8, 16}
```

这支撑 Observation 2：runtime 可以用当前 sparse scores 近似 near-future access，并据此保留 resident blocks 或触发短期 prefetch。

### Hypothesis B：长期模式漂移

随着生成推进，局部上下文、新生成 token 和语义焦点都会改变，重要 KV block 集合会逐渐漂移。因此：

```text
Overlap(S_i, S_{i+Delta}) 随 Delta 增大下降，Delta in {64, 128, 256, 512}
```

这支撑 Observation 3：static compression 不可靠，sparse scores 必须在线刷新。

### Hypothesis C：高压缩率下漂移更明显

当保留预算更小，top-k 边界更敏感，少量 score 排序变化就会改变 selected set。因此：

```text
compression_ratio=0.7 的远距离 overlap 低于 compression_ratio=0.5/0.3
```

注意这里沿用仓库内 BlockWisePress 语义：`compression_ratio=0.7` 表示压缩掉 70%，保留约 30% blocks。

## 3. 方法

### 3.1 重要 KV 集合定义

对每个 decode step `i`、layer `l`、compression ratio `c`，在完整 KV cache 上计算 block importance score，不真正压缩 KV：

```text
score[l, i, b] = BlockWise score of KV block b at decode step i
K(c, i) = ceil(num_blocks(i) * (1 - c))
S[l, i, c] = top-K(c, i) blocks ranked by score[l, i, :]
```

Figure 6 主指标使用 hard top-k set overlap，因为它最直接对应“如果 runtime 沿用当前 sparse set，能覆盖多少未来 oracle-important blocks”。

### 3.2 对比对象

比较同一 layer 内不同 step distance：

```text
S[l, i, c] vs. S[l, i + Delta, c]
```

`Delta` sweep：

```text
1, 2, 4, 8, 16, 32, 64, 128, 256, 512
```

主图建议聚合所有 layer 和样本，画均值曲线；附录再给 layer-by-lag heatmap。

### 3.3 主要指标

主指标：

```text
Overlap(Delta, c) =
  |S[l, i, c] intersection S[l, i + Delta, c]| / |S[l, i + Delta, c]|
```

解释：如果当前 step 的 sparse set 用来近似 `Delta` steps 后的 oracle set，Overlap 就是 future oracle block recall。

辅助指标：

```text
Jaccard(Delta, c) =
  |S[l, i, c] intersection S[l, i + Delta, c]|
  / |S[l, i, c] union S[l, i + Delta, c]|
```

```text
DecodeNewEntryRatio(Delta, c) =
  newly-entered blocks from S[l, i + Delta, c] \ S[l, i, c]
  that originate from decode-stage KV blocks
```

`DecodeNewEntryRatio` 用来解释 drift 的来源：如果远距离新增重要 blocks 很多来自 decode 阶段，说明生成过程中产生的新 KV 逐渐变得重要。

### 3.4 复用刷新指标

为了把 Observation 转成系统 implication，额外报告固定刷新间隔 `R`：

```text
S_reuse[l, t, c, R] = S[l, floor(t / R) * R, c]
ReuseRecall(R, c) =
  |S_reuse[l, t, c, R] intersection S[l, t, c]| / |S[l, t, c]|
RefreshReduction(R) = 1 - 1 / R
```

这不是 Figure 6 必需项，但适合放在补充图或同一 figure 的右侧小图，用来连接“必须在线刷新”和“可以不必每步刷新”。

## 4. 实验设置

### 主实验

直接复用当前仓库已完成的 ranked top-k temporal trace：

```text
script: evaluation/ATC26_collect_blockwise_ranked_topk_temporal_similarity.py
result: evaluation/results/experiments/ATC26_blockwise_ranked_topk_temporal_similarity/artifacts/decode1024/
model: /Tan/model/Llama-3.1-8B-Instruct
dataset: PG19 test, /Tan/dataset/pg19-test
context_lengths: 8192, 16384
samples_per_length: 4
decode_steps: 1024
block_size: 16
window_query_size: 16
compression_ratios: 0.7, 0.5, 0.3
lags: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
reuse_intervals: 2, 4, 8, 16, 32, 64, 128, 256, 512
seed: 42
```

选择 PG19 的原因：它提供长 continuation，可稳定获得 1024 个 decode steps；LongBench QA 容易答案过短，不适合作为 Figure 6 主证据。

### 主图口径

建议 Figure 6 使用：

```text
mode: window
metric: overlap
context_length: 8192 and 16384 both aggregate, or use 8192 as main and 16384 as appendix
compression_ratios: 0.3, 0.5, 0.7
aggregation: mean over samples and layers, with std/CI band if space allows
```

`window` 模式更接近 runtime 刷新 sparse score 的工程实现。若审稿人可能质疑 window overlap 因共享 query window 被放大，可以在附录补 `single` 模式曲线，证明趋势一致。

## 5. Figure 6 设计

### 推荐主图：一张双面板图

Panel A：重要 KV set overlap 随 step distance 变化。

```text
x-axis: Delta, log2 scale
y-axis: Overlap(S_i, S_{i+Delta})
lines: compression_ratio = 0.3, 0.5, 0.7
markers: Delta in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
```

这张图同时支撑两句话：

1. 小 `Delta` 下 overlap 高，说明 local temporal stability。
2. 大 `Delta` 下 overlap 下降，说明 long-horizon drift。

Panel B：固定刷新间隔与 every-step oracle 的接近程度。

```text
x-axis: refresh interval R, log2 scale
left y-axis: ReuseRecall(R)
right y-axis: RefreshReduction(R)
lines: compression_ratio = 0.3, 0.5, 0.7
```

Panel B 用来把 observation 转成 system implication：短期可以复用，但 interval 不能无限变大。

### 可选附录图

1. Layer-by-lag heatmap：展示每层趋势一致性，避免均值掩盖异常层。
2. `single` vs `window` 对比曲线：说明 window 不是唯一证据。
3. Decode-new block ratio 曲线：解释 drift 的原因。
4. Context length 8192 vs 16384 对比：说明长上下文会稀释 decode blocks 的占比。

## 6. 预期结果与论文解释

仓库中已完成的 `decode1024` 结果已经支持主趋势。可作为写作时的定量 anchor：

```text
window, lag=1:
  overlap about 0.98-0.99 across context lengths and compression ratios

window, lag=512:
  compression_ratio=0.3: overlap about 0.83-0.86
  compression_ratio=0.5: overlap about 0.76-0.80
  compression_ratio=0.7: overlap about 0.68-0.73

window, R=32:
  reuse recall about 0.89-0.95

window, R=512:
  high compression ratio reuse recall drops to about 0.73-0.76
```

可写成：

```text
Important KV block sets are highly stable across adjacent decode steps,
but the overlap decays with step distance. The decay is stronger under
tighter sparsity budgets, which indicates that one-shot sparse decisions
cannot reliably approximate future access throughout a long generation.
```

限制必须写清楚：

1. 该实验证明的是 sparse index temporal behavior，不直接证明任务质量。
2. PG19 teacher-forced/continuation trace 与真实自由生成存在分布差异。
3. `window` 模式更接近系统实现，但可能高估相邻 step 相似性；需要附录补 `single`。

## 7. 最小验证计划

### Step 1：确认现有 artifact 完整性

检查以下文件：

```text
evaluation/results/experiments/ATC26_blockwise_ranked_topk_temporal_similarity/artifacts/decode1024/ATC26_ranked_topk_temporal_similarity_aggregate.csv
evaluation/results/experiments/ATC26_blockwise_ranked_topk_temporal_similarity/artifacts/decode1024/ATC26_ranked_topk_temporal_similarity_aggregate.json
evaluation/results/experiments/ATC26_blockwise_ranked_topk_temporal_similarity/artifacts/decode1024/raw/ATC26_ranked_topk_temporal_similarity_raw.jsonl
```

通过条件：

```text
8 / 8 jobs complete
aggregate rows cover 2 context lengths x 4 samples x 32 layers x 2 modes x 3 ratios x all lags/reuse intervals
```

### Step 2：生成 Figure 6 专用数据表

从 aggregate CSV 生成一个 paper-facing CSV：

```text
figure/experiments/ATC26_figure6_temporal_kv_stability/figure6_overlap_curve.csv
```

列建议：

```text
mode, context_length, compression_ratio, lag, overlap_mean, overlap_std, jaccard_mean, sample_count
```

Reuse panel 另存：

```text
figure/experiments/ATC26_figure6_temporal_kv_stability/figure6_reuse_curve.csv
```

列建议：

```text
mode, context_length, compression_ratio, reuse_interval, reuse_recall_mean, reuse_recall_std, refresh_reduction
```

### Step 3：绘制论文图

建议新增或改造绘图脚本：

```text
figure/ATC26_plot_figure6_temporal_kv_stability.py
```

输出：

```text
figure/experiments/ATC26_figure6_temporal_kv_stability/figure6_temporal_kv_stability.pdf
figure/experiments/ATC26_figure6_temporal_kv_stability/figure6_temporal_kv_stability.png
figure/experiments/ATC26_figure6_temporal_kv_stability/README.md
figure/experiments/ATC26_figure6_temporal_kv_stability/summary.json
```

### Step 4：补充 single 模式附录图

若主文用 `window`，附录必须放 `single` 曲线，证明 Observation 2/3 不是 window overlap artifact。

### Step 5：必要时补质量实验

如果论文从 observation 推到具体机制，例如 fixed-refresh sparse index，应补：

```text
R in {16, 32, 64, 128}
compression_ratio in {0.5, 0.7}
datasets: PG19 PPL + LongBench/Needle quality
metrics: quality, latency, memory footprint
```

这一步不是 Figure 6 的前置条件，但它是把 observation 转成系统优化 claim 的必要支撑。

## 8. 风险与应对

1. 风险：高 overlap 被认为只是相邻 query window 重叠导致。
   应对：主文或附录报告 `single` 模式，强调趋势一致。

2. 风险：只用 PG19 被认为任务单一。
   应对：Figure 6 主证据保留 PG19，附录补少量 LongBench/Needle trace；不要把 LongBench 短答案作为主图。

3. 风险：Overlap 高但质量仍下降。
   应对：在 observation 段只声称 temporal predictability，不直接声称质量无损；质量放到后续机制实验。

4. 风险：不同 context length 趋势不同。
   应对：主图聚合前先分别画 8192/16384；若差异明显，主图分两个子面板或只用 8192，16384 放附录。

## 9. 结论

这组实验的最小闭环是：

```text
problem: sparse future-access oracle 无法直接获得
hypothesis: 当前重要 KV set 对近未来有预测性，但长期会漂移
method: 比较 S_i 和 S_{i+Delta} 的 top-k block overlap
experiment: Llama-3.1-8B + PG19 + 1024 decode steps + ratio sweep + lag sweep
result: 小 Delta overlap 高，大 Delta overlap 下降，高压缩率下降更明显
conclusion: sparse scores 可以近似短期未来访问，但必须在线刷新
```

Figure 6 应优先画 `Overlap(S_i, S_{i+Delta})` 随 `Delta` 的下降曲线。这个图直接对应 Observation 2/3 的文字，并且能自然导出 runtime 需要 periodic refresh 的设计动机。
