# ATC26 注意力分布相似性实验方案

## 目标

为 ATC26 论文补一组解释性实验，说明 `BlockWisePress` 在不同 layer 和不同 KV head 上选择的 KV block 是否相似。

这组实验不比较最终任务分数，只分析压缩决策本身：

- layer 间相似性：同一条样本、同一模型、同一压缩率下，比较每一层最终保留的 block index 集合。
- head 间相似性：在每一层内部，基于 BlockWise 的 per-head block score，计算每个 KV head 如果独立 top-k 会保留哪些 block，再比较这些集合。

## 实验矩阵

模型：

- `/Tan/model/Llama-3.1-8B-Instruct`
- `/Tan/model/Mistral-7B-Instruct-v0.3`
- `/Tan/model/Qwen3-8B`

方法：

- `BlockWisePress`
- 主配置与现有 ATC26 sweep 对齐：
  - `block_size=16`
  - `q_window_size=64`
  - `summary_topk_keys=4`
  - `mean_key_weight=0.75`
  - `representative_k=4`
  - `multi_rep_k=4`
  - `query_topr=16`
  - `head_topk=1`
  - `summary_mode=mean_plus_norm_topk_mean`
  - `representative_mode=key_norm`
  - `query_agg_mode=max`
  - `head_agg_mode=uniform_mean`

压缩率：

- `0.3`
- `0.5`
- `0.7`

数据集：

- `Xnhyacinth/LongBench`
- config: `hotpotqa`
- 随机选择 3 条样本，默认 `seed=42`

总配置数：

- `3 models x 3 ratios = 9`

## layer 相似性定义

对每个样本、模型、压缩率，BlockWise 在每层会得到一个最终保留 block 集合：

```text
S_l = {kept block indices at layer l}
```

任意两层 `i, j` 的相似性使用 Jaccard：

```text
J(i, j) = |S_i ∩ S_j| / |S_i ∪ S_j|
```

每个配置保存：

- 每条样本的 `n_layers x n_layers` 原始矩阵
- 3 条样本平均后的 `n_layers x n_layers` 矩阵
- 上三角均值，作为一个 summary number

论文图优先使用平均矩阵。后续如果只想展示一个模型或一个压缩率，可以直接从同一个 aggregate JSON 里筛选。

## head 相似性方案

不建议直接拿最终 `kept_block_indices` 比 head，因为 `BlockWisePress` 当前会先计算 per-head block score，再通过 `head_agg_mode=uniform_mean` 聚合成单个最终 block 排序。最终 kept set 已经没有 head 维度。

更合理的做法是：

1. 保留每层的 `summary_scores_per_head`。
2. 对每个 KV head 独立执行与 BlockWise 相同的预算规则：
   - 同样的 compression ratio
   - 同样的 `prefix_sink_blocks`
   - 同样的 `protected_recent_blocks`
   - 同样保护 partial tail block
3. 得到每层每个 KV head 的假想独立保留集合：

```text
H_{l,h} = {kept block indices if KV head h selects independently at layer l}
```

4. 在同一层内计算 KV head 两两 Jaccard：

```text
J_l(h_a, h_b) = |H_{l,h_a} ∩ H_{l,h_b}| / |H_{l,h_a} ∪ H_{l,h_b}|
```

5. 对所有 layer 和 3 条样本取平均，得到每个模型/压缩率的一张 `n_kv_heads x n_kv_heads` 热力图。

这个定义和 layer 相似度保持一致，优点是直接对应“最终会保留哪些 KV block”。同时脚本额外保存 per-head score vector，并输出 score-vector cosine 热力图，方便后续如果觉得 Jaccard 太离散，可以改用 score 分布相似性而不重跑模型。

注意：Llama/Mistral/Qwen 这类 8B 模型通常使用 GQA。这里的 head 默认指 `KV head / KV head group`，不是展开后的所有 query attention head。原因是 KV cache 的 block 选择实际发生在 KV head 粒度上。论文文字里建议写成 `KV head groups`，避免和 query head 混淆。

## 原始数据与产物

实验名：

```text
ATC26_blockwise_attention_similarity_hotpotqa_3samples
```

运行脚本：

```text
evaluation/ATC26_collect_attention_similarity.py
```

绘图脚本：

```text
figure/ATC26_plot_attention_similarity.py
```

结果目录：

```text
evaluation/results/experiments/ATC26_blockwise_attention_similarity_hotpotqa_3samples/
```

图像目录：

```text
figure/experiments/ATC26_blockwise_attention_similarity_hotpotqa_3samples/
```

关键原始文件：

- `artifacts/ATC26_attention_similarity_aggregate.json`
- `artifacts/raw/ATC26_attention_similarity_raw.jsonl`
- `artifacts/scores/*.npz`
- `artifacts/ATC26_hotpotqa_sample_manifest.json`

`raw.jsonl` 保存每个样本的 kept block indices 和 Jaccard 矩阵；`scores/*.npz` 保存每层 block scores、per-head scores、最终 layer kept blocks、per-head kept blocks。后续换 colormap、只画一个模型、只画一个压缩率、或改成 score cosine / rank correlation 时，不需要重新跑模型。

## 执行命令

采集数据：

```bash
.venv/bin/python evaluation/ATC26_collect_attention_similarity.py --device cuda:0
```

只跑单模型 smoke：

```bash
.venv/bin/python evaluation/ATC26_collect_attention_similarity.py --models llama31_8b_instruct --device cuda:0
```

绘图：

```bash
.venv/bin/python figure/ATC26_plot_attention_similarity.py
```

## 预期论文呈现

建议主图不要一次塞满 9 张 layer 图和 9 张 head 图。

更清晰的呈现方式：

1. 主文放一个代表性模型和压缩率，例如 `Llama-3.1-8B-Instruct, ratio=0.5`：
   - layer-to-layer kept-block Jaccard heatmap
   - KV-head-to-KV-head kept-block Jaccard heatmap
2. 主文用一句话报告 9 组配置的均值范围。
3. appendix 放完整 `3 x 3` grid。

如果结果显示 layer 相似性明显高而 head 相似性也高，可以支持：

- 不同 layer/head 会重复关注一批相似的 hot KV blocks。
- 这为 block-level metadata reuse、跨层/跨头共享候选块、以及更低开销的块级调度提供动机。

如果 head 相似性低但 score cosine 高，则应改写为：

- 各 head 的 score distribution 相似，但 top-k 边界附近排序存在差异。
- 论文里更适合展示 score cosine 或 top-k overlap at multiple k，而不是只展示单一压缩率下的 Jaccard。
