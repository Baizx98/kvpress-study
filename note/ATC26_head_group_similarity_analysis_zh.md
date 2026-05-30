# ATC26 Head Group Similarity 分析

## 结论先行

原始 `ATC26_head_similarity_grid` 的 head-head Jaccard 低，不能直接说明不同 head 的注意力分布完全不同。它更说明：当每个 head 单独做 hard top-k block selection 时，top-k 边界附近的小排序差异会被放大成不同的 kept block set。

`ATC26_head_score_cosine_grid` 很高，说明不同 KV head 的 block score 向量方向高度一致。也就是说，大部分 head 对“哪些 block 大体重要”判断相近，但单个 head 的 top-k 离散选择不稳定。

因此更适合论文叙述的实验不是“单 head 是否能独立决定 KVCache”，而是“把若干 KV head 合并成 group 后，group-level selection 是否接近全头平均 selection”。这和最终方法更一致。

## 新实验定义

输入来自已有原始分数，不重新跑模型：

- `per_head_scores`: 每层每个 KV head 的 block score
- `kept_layer_blocks`: 当前 `head_agg_mode=uniform_mean` 的全头平均 block selection
- `kept_head_blocks`: 单 head 独立选择的 block set

对 KV heads 做固定连续分组，group size 为 `1/2/4/8`。每个 group 内对 score 取平均，再按相同 keep budget 选择 block。

主要指标：

- `Jaccard vs. all-head selection`: 每个 head group 的选择与当前全头平均选择的 Jaccard。
- `Jaccard among head groups`: 不同 head group 之间的 Jaccard。
- `union recall vs. all-head`: 多个 group 的 union 对全头平均选择的覆盖率。

## 关键数值

| model | ratio | g=1 vs all | g=2 vs all | g=4 vs all | g=8 vs all |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | 0.3 | 0.760 | 0.823 | 0.884 | 0.995 |
| Llama-3.1-8B | 0.5 | 0.645 | 0.734 | 0.823 | 0.992 |
| Llama-3.1-8B | 0.7 | 0.535 | 0.643 | 0.765 | 0.991 |
| Mistral-7B-v0.3 | 0.3 | 0.759 | 0.820 | 0.885 | 0.995 |
| Mistral-7B-v0.3 | 0.5 | 0.637 | 0.723 | 0.816 | 0.993 |
| Mistral-7B-v0.3 | 0.7 | 0.521 | 0.629 | 0.748 | 0.989 |
| Qwen3-8B | 0.3 | 0.786 | 0.842 | 0.894 | 0.995 |
| Qwen3-8B | 0.5 | 0.685 | 0.766 | 0.841 | 0.989 |
| Qwen3-8B | 0.7 | 0.585 | 0.687 | 0.785 | 0.984 |

## 解释

1. `g=1` 对应每个 KV head 独立选择，和原来的 head Jaccard 一样容易受到 top-k 边界扰动影响。
2. `g=2/4` 会明显稳定选择结果；它衡量的是“合并部分 head 后是否仍接近全头平均”。
3. `g=8` 是所有 KV head 合成一个 group，本质上退化为当前 `uniform_mean`，因此和 all-head selection 一致。
4. 这支持一个更合理的系统设计方向：不要选择单个 head，也不要让每个 head 单独维护完全独立的 KV block set；应该将相似 score distribution 的 head 合并成少量 group，再做 group-level block selection。

## 图像

- `figure/experiments/ATC26_blockwise_head_group_similarity_hotpotqa_3samples/ATC26_head_group_vs_all_selection.png`
- `figure/experiments/ATC26_blockwise_head_group_similarity_hotpotqa_3samples/ATC26_head_group_pairwise_similarity.png`
- `figure/experiments/ATC26_blockwise_head_group_similarity_hotpotqa_3samples/<model>__per_layer_group_vs_all.png`
- `figure/experiments/ATC26_blockwise_head_group_similarity_hotpotqa_3samples/per_layer_head_similarity/*all_layers_head_similarity.png`

## 论文建议

主文不要展示原始 `head_similarity_grid` 作为主要证据，因为它回答的是“单 head hard top-k 是否一致”，而不是最终方法需要的“head group 是否可共享选择”。更推荐展示：

- 一张 `head_score_cosine` 或文字说明 score distribution 高相似。
- 一张 `head_group_vs_all_selection`，说明合并 2/4 个 KV heads 后选择已经接近全头平均。
- appendix 放 per-layer head Jaccard，说明单 head top-k 的不稳定性来自 hard selection 边界。
