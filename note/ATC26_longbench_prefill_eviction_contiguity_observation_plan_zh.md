# ATC26 LongBench Prefill 驱逐连续性观察图实验方案

## 1. 问题陈述

论文观察小节要说明：对长请求做 prefill 阶段 KV 压缩时，如果根据 attention-derived token importance 选择保留 KV，被驱逐的 KV token 往往不是均匀散落，而是形成连续片段。这个现象可以作为使用较大粒度管理和驱逐 KV 的动机，因为现有推理系统通常已经以 block/page 为单位管理显存 KV cache。

这组实验的定位是 observation / motivation figure，不直接声明质量收益。核心问题是：

> token-level attention 压缩决策本身是否已经具有明显的空间连续性，从而能被 block-level eviction 近似？

## 2. 核心假设

### H1: token-level eviction mask 存在长连续 run

对 LongBench 长请求，按照 prefill attention score 保留 top-K KV token 后，evicted token 的连续 run 长度显著大于随机同保留率 mask。

### H2: block-level 近似的额外误差较小

如果把 token-level mask 投影到固定大小 block，例如 16 或 32 token，一个 block 内多数 token 被驱逐时整块驱逐，多数 token 被保留时整块保留，那么产生的 token-level decision mismatch 较低。

### H3: 连续性不是单个样本偶然现象

该现象在多个 LongBench 子任务、多个随机样本、不同压缩率下都存在，但强弱可能随任务类型和 query 形式变化。

## 3. 方法设计

### 3.1 主证据使用 token-level attention scorer，而不是 BlockWise scorer

为了避免循环论证，主图不直接画 `BlockWisePress` 的 block-level 选择结果。主证据先用 token-level attention score 得到保留/驱逐 mask：

1. 对 LongBench 请求执行一次 prefill。
2. 收集每层最后 `q_window_size=64` 个 query 对所有 key positions 的 attention。
3. 对 query window、KV head group 和 layer 做聚合，得到每个 token position 的全局重要性分数。
4. 按压缩率选择 top-K token 为 kept，其余 token 为 evicted。
5. 画 token position 上的 kept/evicted binary mask。

这样得到的连续性来自 attention score 本身，而不是来自 blockwise 方法的结构先验。

### 3.2 默认 scorer

默认 scorer 与当前仓库的 prefill 设定保持接近：

- query window: `q_window_size=64`
- query aggregation: `max`
- head aggregation: `mean` 或 `uniform_mean`
- layer aggregation: 主图使用 `mean over layers`，附图可画 per-layer mask
- protected region:
  - sink token 保留，建议前 4 个 block，即 `4 * 16 = 64` tokens
  - recent token 保留，建议最后 4 个 block，即 `64` tokens
  - 这与常见 KV 压缩策略一致，避免把系统通常会强制保留的区域误算为驱逐连续性

注意：如果实现时直接复用 `SnapKVPress` 或当前 `BlockWisePress` 的 attention trace 接口，需要确认 score 聚合语义与这里一致。主图 caption 应写清楚 scorer 是 token-level attention score，而不是 block-level score。

### 3.3 压缩率

主图建议固定：

- `compression_ratio=0.5`

附加统计覆盖：

- `compression_ratio=0.3`
- `compression_ratio=0.5`
- `compression_ratio=0.7`

原因：`0.5` 直观且容易解释；`0.3/0.7` 用来说明连续性不是某个阈值偶然造成。

### 3.4 block size

统计中比较：

- `block_size=16`：与当前 `BlockWisePress` 默认配置一致。
- `block_size=32`：代表更粗粒度管理，观察是否仍可近似。
- 可选 `block_size=64`：只放 appendix 或 text number，不建议主图塞太多。

## 4. 实验矩阵

### 4.1 模型

主图建议只用一个模型：

- `/Tan/model/Llama-3.1-8B-Instruct`

鲁棒性统计可选三模型：

- `/Tan/model/Llama-3.1-8B-Instruct`
- `/Tan/model/Mistral-7B-Instruct-v0.3`
- `/Tan/model/Qwen3-8B`

主图只放一个模型的原因是观察图要清晰，不宜把多个模型混成复杂网格。多模型结果更适合用一个小表或 appendix 图报告均值范围。

### 4.2 数据集与样本

数据集使用 `Xnhyacinth/LongBench`，优先覆盖不同长请求形态：

- `hotpotqa`：多跳 QA，已有相近 attention trace 脚本，便于复用。
- `multifieldqa_en`：长文档问答，证据位置更分散。
- `qasper`：论文 QA，结构化长上下文。
- `gov_report`：长摘要任务，输入更长，适合展示大段连续冷区。

抽样规则：

- 固定 `seed=42`。
- 每个子任务随机抽 `2` 条请求，总计 `8` 条。
- 过滤 tokenized input length 小于 `4096` 的样本；如果某个子任务不足，继续按随机顺序补抽。
- 记录 `dataset_row_index`、`sample_id`、`input_length`、`context_sha1`，保证图可复现。

如果只做最小版图，可先用：

- `hotpotqa=3` 条
- `gov_report=3` 条

这样成本更低，但覆盖面弱一些。

## 5. 指标设计

### 5.1 主可视化：kept/evicted mask heatmap

图形设计：

- x 轴：token position，按 block 边界加浅灰竖线；长请求可以 bucket 到每 8 或 16 token 一个像素。
- y 轴：不同请求，或者同一请求的不同 layer。
- 颜色：
  - kept: 深色
  - evicted: 浅色
  - protected sink/recent: 单独颜色或 hatch 标注
- 每行右侧标注：dataset、input length、compression ratio。

建议主图布局：

1. 左侧大图：8 条 LongBench 请求的 token-level kept/evicted mask。
2. 右侧小图 A：evicted run length 分布，比较 attention score vs random mask。
3. 右侧小图 B：block projection mismatch 随 block size 变化的柱状图。

这比单张 heatmap 更强，因为 heatmap 给直觉，run length 和 mismatch 给量化证据。

### 5.2 连续 run 指标

对每个样本的 evicted binary mask 计算：

- `mean_evicted_run_length`
- `median_evicted_run_length`
- `p90_evicted_run_length`
- `max_evicted_run_length`
- `num_evicted_runs`
- `evicted_tokens_per_run = evicted_count / num_evicted_runs`

随机 baseline：

- 保持相同 input length、compression ratio、protected sink/recent 区域。
- 在非保护区域随机选择相同数量 kept tokens。
- 每条样本重复 `100` 次，报告均值和 95% interval。

预期表达：

> attention-based eviction has substantially longer evicted runs than random masks under the same eviction budget.

### 5.3 block projection 指标

给定 token-level mask 和 block size `B`：

1. 将非保护区域划分为 block。
2. 对每个 block 计算 `evicted_fraction`。
3. 若 `evicted_fraction >= 0.5`，则整块视为 evicted，否则整块视为 kept。
4. 与原 token-level mask 比较 mismatch。

指标：

- `token_decision_mismatch_rate`
- `false_eviction_rate`: token-level kept 但 block projection evicted
- `false_keep_rate`: token-level evicted 但 block projection kept
- `pure_block_ratio`: block 内 token 决策全一致的比例
- `majority_pure_block_ratio`: block 内多数决策比例大于等于 0.8 的比例

这个指标直接回答“用较大粒度管理和驱逐 KV 是否可行”。

### 5.4 可选佐证：score smoothness

如果时间允许，额外保存 attention score 序列并画一条轻量曲线：

- x 轴 token position
- y 轴 normalized attention importance
- 在曲线下方叠加 kept/evicted mask

它能解释为什么 mask 连续：attention importance 在位置轴上通常有低频趋势，而不是独立随机噪声。

## 6. 产物路径

实验名：

```text
ATC26_longbench_prefill_eviction_contiguity
```

建议新增采集脚本：

```text
evaluation/ATC26_collect_longbench_prefill_eviction_contiguity.py
```

建议新增绘图脚本：

```text
figure/ATC26_plot_longbench_prefill_eviction_contiguity.py
```

结果目录：

```text
evaluation/results/experiments/ATC26_longbench_prefill_eviction_contiguity/
```

图像目录：

```text
figure/experiments/ATC26_longbench_prefill_eviction_contiguity/
```

计划输出：

- `artifacts/ATC26_eviction_contiguity_manifest.json`
- `artifacts/raw/ATC26_eviction_contiguity_raw.jsonl`
- `artifacts/scores/*.npz`
- `artifacts/ATC26_eviction_contiguity_summary.csv`
- `artifacts/ATC26_eviction_contiguity_summary.json`
- `figure/experiments/ATC26_longbench_prefill_eviction_contiguity/ATC26_eviction_mask_heatmap_main.png`
- `figure/experiments/ATC26_longbench_prefill_eviction_contiguity/ATC26_evicted_run_length_vs_random.png`
- `figure/experiments/ATC26_longbench_prefill_eviction_contiguity/ATC26_block_projection_mismatch.png`
- `note/ATC26_longbench_prefill_eviction_contiguity_results_zh.md`

## 7. 最小执行计划

### Stage 0: smoke

目的：确认 attention trace、样本过滤、输出格式都正常。

配置：

- model: Llama-3.1-8B-Instruct
- dataset: `hotpotqa`
- sample count: `1`
- ratio: `0.5`
- max context length: 可先限制到 `8192`

成功条件：

- 产出 raw jsonl、summary csv、npz score 文件。
- heatmap 能正常显示 kept/evicted mask。
- token budget 和 compression ratio 语义正确。

### Stage 1: 主图数据

配置：

- model: Llama-3.1-8B-Instruct
- datasets: `hotpotqa`, `multifieldqa_en`, `qasper`, `gov_report`
- samples per dataset: `2`
- ratio: `0.5`
- max context length: 建议 `16384`；如果显存压力大，先用 `8192`

成功条件：

- 8 条样本都有有效 token-level mask。
- attention mask 的 kept 数量与设定保留预算一致。
- run length 指标显著高于 random baseline。
- block size 16/32 的 mismatch 可接受。

### Stage 2: 鲁棒性统计

配置：

- ratios: `0.3,0.5,0.7`
- block sizes: `16,32,64`
- models: 先 Llama；如果主图现象明确，再补 Mistral/Qwen

成功条件：

- 能在结果文档中报告不同 ratio 和 block size 下的均值范围。
- 如果某些任务连续性弱，需要明确说明边界条件，而不是只挑好看的样本。

## 8. 论文图建议

主文建议放一个三联图：

```text
(a) Token-level kept/evicted mask over LongBench requests
(b) Evicted run length vs. random mask
(c) Block projection mismatch for B=16/32
```

caption 重点：

- `compression_ratio=0.5`
- token-level attention scorer
- protected sink/recent regions are excluded from run statistics
- block projection is only used for analysis, not for generating the token-level mask

建议论文文字：

> We first compute token-level KV importance from prefill attention scores and evict the lowest-scored tokens under a fixed budget. The resulting eviction masks are highly clustered along the sequence dimension, producing long contiguous cold regions. This suggests that a block/page-level eviction interface can approximate fine-grained attention-based decisions while matching the granularity used by existing KV-cache memory managers.

## 9. 风险与应对

### 风险 1: output attentions 显存开销过高

应对：

- smoke 阶段限制 `max_context_length=8192`。
- 只保留最后 `q_window_size=64` query 对 key 的 attention。
- 分样本逐条运行并立即写盘，避免多个样本常驻内存。

### 风险 2: 不同 layer 的 mask 差异较大

应对：

- 主图使用 layer-aggregated score。
- 附图或 raw artifact 保存 per-layer mask。
- 如果差异大，结论改为“aggregated token-level decision is clustered”，不扩展到每层都完全一致。

### 风险 3: 随机样本不够长或图不好看

应对：

- 固定随机 seed 后先按 input length 过滤。
- 不能手工挑样本；如果必须替换样本，记录替换规则。
- 结果文档同时列出所有候选样本指标，防止 selective visualization。

### 风险 4: 连续性主要来自 protected recent/sink

应对：

- run length 和 block projection 统计默认排除 protected region。
- heatmap 中单独标注 protected region。
- 论文文字只讨论非保护区域的 cold KV 分布。

## 10. 确认后要实现的代码改动

确认方案后再开始执行，预计改动包括：

1. 新增 `evaluation/ATC26_collect_longbench_prefill_eviction_contiguity.py`。
2. 新增 `figure/ATC26_plot_longbench_prefill_eviction_contiguity.py`。
3. smoke 后生成 `note/ATC26_longbench_prefill_eviction_contiguity_results_zh.md`。
4. 按 `evaluation/results/experiments/ATC26_longbench_prefill_eviction_contiguity/` 组织 raw artifacts。

