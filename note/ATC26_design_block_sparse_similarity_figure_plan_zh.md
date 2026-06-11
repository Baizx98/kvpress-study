# ATC26 Design 块摘要稀疏 Head/Layer 相似度组图规划

## 目标

为论文 Design 中“块摘要稀疏方法”补一组解释性热力图，说明 BlockWise 的稀疏 block index 在不同 KV head group 和不同 layer 之间存在可复用或可共享的相似性。

## 复用数据

优先复用已完成的 `ATC26_blockwise_attention_similarity_hotpotqa_3samples`，不重新跑模型。

- 结果目录：`evaluation/results/experiments/ATC26_blockwise_attention_similarity_hotpotqa_3samples/`
- 聚合文件：`artifacts/ATC26_attention_similarity_aggregate.json`
- 原始记录：`artifacts/raw/ATC26_attention_similarity_raw.jsonl`
- per-head score 和 kept block 数组：`artifacts/scores/*.npz`
- 旧绘图脚本：`figure/ATC26_plot_attention_similarity.py`
- 旧说明：`note/ATC26_attention_similarity_experiment_plan_zh.md`

数据覆盖：

- 模型：Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen3-8B
- 数据集：LongBench HotpotQA
- 样本数：每个配置 3 条样本
- 压缩率：0.3、0.5、0.7
- 指标：layer-to-layer kept-block Jaccard、KV-head-to-KV-head kept-block Jaccard、KV-head score cosine

术语上，论文中建议写 `KV head groups`，而不是泛称 `heads`，因为这些本地 8B 模型是 GQA，KV cache 相关维度是 `num_key_value_heads=8`。

## 主图方案

推荐新建图目录：

`figure/experiments/ATC26_design_block_sparse_head_layer_similarity/`

推荐新建绘图脚本：

`figure/ATC26_plot_design_block_sparse_head_layer_similarity.py`

主文组图建议做成 1 行 2 列：

1. Panel A：KV-head score cosine heatmap
   - 数据：`artifacts/scores/*.npz` 中的 `per_head_scores`
   - 计算：对每层、每样本的 KV-head block score vector 做 head-head cosine，再对 layer 和样本平均
   - 代表配置：Llama-3.1-8B-Instruct，compression ratio=0.5
   - 目的：说明不同 KV head group 对 block 重要性的连续分布判断相似

2. Panel B：Layer-to-layer kept-block Jaccard heatmap
   - 数据：`ATC26_attention_similarity_aggregate.json` 中的 `layer_similarity_matrix_mean`
   - 计算：每两层最终 kept block index 集合的 Jaccard，相同模型/压缩率下对 3 条样本平均
   - 代表配置：Llama-3.1-8B-Instruct，compression ratio=0.5
   - 目的：说明不同 layer 的 block index 选择存在明显重复，为跨层候选块复用提供设计动机

这两个 panel 对应一条 Design 叙事：

- KV-head score cosine 高：head 维度上可以用 group-level summary，而不是每个 KV head 完全独立维护选择逻辑。
- Layer Jaccard 有结构性相似：layer 维度上可以复用候选 block metadata 或减少重复 block selection 开销。

## 备选与附录

如果主文空间允许，可以追加 2x2：

3. Panel C：KV-head kept-block Jaccard heatmap
   - 作为 hard top-k 选择的一致性补充
   - 风险：Jaccard 可能低于 score cosine，容易被误读为 head 不相似；正文需要解释 top-k 边界放大了排序扰动

4. Panel D：3 models x 3 ratios summary grid 或均值条形图
   - 目的：说明代表配置不是 cherry-pick
   - 如果主文空间不够，放 appendix

## 输出文件

计划生成：

- `ATC26_design_head_layer_similarity_main.pdf`
- `ATC26_design_head_layer_similarity_main.png`
- `ATC26_design_head_layer_similarity_appendix_grid.pdf`
- `ATC26_design_head_layer_similarity_appendix_grid.png`
- `summary.json`
- `README.md`

## 图形风格

- 论文主图优先 PDF 矢量输出，PNG 用于快速查看
- colormap 使用同一套 sequential map，范围固定到 `[0, 1]`
- 两个 panel 共用 colorbar 或保持一致 colorbar 范围
- 标题尽量短：`KV-head score cosine`、`Layer block-index Jaccard`
- 坐标轴：`KV head group`、`Layer`
- 默认不在主图里写大段解释文字

## 限制

这组图只证明 block selection / block score 的相似性，不直接证明端到端质量或 latency 收益。论文文字中应把它作为 Design motivation，而不是最终性能结论。
