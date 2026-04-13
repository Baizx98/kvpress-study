# Blockwise Prefill Compression Stage2 设计报告

## 背景

本报告基于以下两份 stage1 消融实验报告整理：

- [blockwise_ablation_ratio70_stage1_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/blockwise_ablation_ratio70_stage1_analysis_zh.md)
- [blockwise_ablation_ratio70_longbench_stage1_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/blockwise_ablation_ratio70_longbench_stage1_analysis_zh.md)

目标是基于已有证据，给出：

- 跨 `RULER` 与 `LongBench` 的稳健结论
- `stage2` 的精简实验矩阵
- 下一步方法改进方向
- 时间有限时的优先级建议

## 一、跨 Benchmark 的共同结论

### 1. 已验证的稳定结论

- `query_agg=max` 是目前最强、最稳定的单点改动。
  - RULER 上 baseline 从 `33.00 -> 40.00`
  - LongBench 上：
    - `hotpotqa`: `53.20 -> 56.27`
    - `multifieldqa_en`: `43.62 -> 54.30`
- `key_norm` 仍然是最稳的 representative selection 基线。
  - RULER 上 `tail_query_relevance` 未超过它
  - LongBench 上也没有形成全局替代优势
- `Quest-prefill` 当前更适合作为对照组，而不是主线方法。
  - RULER 上平均仅 `20.76`
  - LongBench 上整体仍弱于 summary-based blockwise
- `random_topk`、`strength_weighted`、`top_head_only` 都没有形成跨 benchmark 的稳定收益。

### 2. benchmark-specific 结论

- RULER 更偏好 `norm_topk_mean_only`
  - 说明检索型任务更依赖块内少数高强度 token
- LongBench 更偏好 `mean_plus_norm_topk_mean`
  - 至少在 `hotpotqa` 与 `multifieldqa_en` 上，最优配置都是：
    - `mean_plus_norm_topk_mean + key_norm + max + uniform_mean`
- `triviaqa` 是显著例外
  - 最优配置是：
    - `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean = 98.00`
  - 说明它更依赖 query relevance，而不是固定偏向 `max`

### 3. 当前总判断

- 跨 benchmark 最稳的部分是：
  - `query_agg=max`
  - `representative_mode=key_norm`
- 当前最值得继续挖的主变量是：
  - `summary_mode`
- `tail_query_relevance` 更像 LongBench 某些任务的特化分支，而不是全局默认项

## 二、Stage2 消融实验建议

## 1. 总体原则

`stage2` 不建议继续做大规模全组合，而应转向“小矩阵验证”：

- 固定最稳部分
  - `representative_mode=key_norm`
  - `head_agg_mode=uniform_mean`
- 把主要搜索预算放在：
  - `summary_mode`
  - `query_agg_mode`
- 只保留极少数有明确证据支持的特化分支

## 2. 推荐主矩阵

### 配置 1：主线锚点

- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

目的：

- 作为统一主线锚点
- 验证 stage1 的跨 benchmark 最优信号是否稳定复现

### 配置 2：RULER 风格 summary 候选

- `summary_mode=norm_topk_mean_only`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

目的：

- 验证 RULER 上“更简洁 summary”在引入 `max` 后是否仍保持优势

### 配置 3：多 representative 候选

- `summary_mode=multi_rep_max`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

目的：

- 验证 multi-key / single-key 检索增强在更强 query aggregation 下是否仍有价值

### 配置 4：LongBench 特化分支

- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=tail_query_relevance`
- `query_agg_mode=mean`
- `head_agg_mode=uniform_mean`

目的：

- 仅在 LongBench 上验证 `triviaqa` 的特化收益是否稳定

建议：

- 不要把它直接并入全 benchmark 主线
- 只作为 LongBench 特化验证项

## 3. 建议淘汰的方向

以下方向不建议进入 stage2 主线：

- `random_topk`
- `strength_weighted`
- `top_head_only`
- `Quest-prefill`

原因：

- 没有形成跨 benchmark 的稳定收益
- 复杂化成本高于当前证据支持的收益

## 4. 最小可执行矩阵

如果只做最小 stage2，我建议先跑 3 组：

1. 主线锚点
2. `norm_topk_mean_only + key_norm + max + uniform_mean`
3. `multi_rep_max + key_norm + max + uniform_mean`

然后把：

- `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean`

作为 LongBench 附加验证项。

## 三、下一步方法改进方向

## 1. 自适应 query aggregation

动机：

- stage1 已经证明 `max` 很强
- 但 `triviaqa` 表明 `mean` 仍可能在特定任务上更优

预期收益：

- 同时兼顾检索型任务与 QA 型任务
- 降低固定 `mean` 或固定 `max` 的偏置

主要风险：

- 引入门控后，可能变成新的超参问题

最小验证方案：

- 先做 `mean/max` 二选一规则门控
- 不训练参数
- 只用 query window 的 top-gap、峰值占比或熵做决策

## 2. 自适应 summary fusion

动机：

- RULER 更偏 `norm_topk_mean_only`
- LongBench 更偏 `mean_plus_norm_topk_mean`

预期收益：

- 降低固定 `mean_key_weight` 的脆弱性
- 提高跨 benchmark 的适应性

主要风险：

- 可能退化成复杂但无效的启发式

最小验证方案：

- 让 `mean` 与 `norm-topk-mean` 的融合权重依赖块内统计量
- 例如：
  - token norm 方差
  - top-k 集中度
  - top-k / mean 比值

## 3. 结构化 multi-representative

动机：

- `multi_rep_max` 已经显示出潜力
- 但目前稳定性还不足

预期收益：

- 更好覆盖多峰块
- 对 multi-key retrieval 更友好

主要风险：

- representative 过多会引入噪声

最小验证方案：

- 在 `multi_rep_max` 上加入简单多样性约束
- 比如：
  - 位置最小间隔
  - 相似度阈值
  - norm 排名去冗余

## 4. 轻量 head calibration

动机：

- `uniform_mean` 当前最稳
- 但这不代表 head 维度没有进一步利用空间

预期收益：

- 在不牺牲鲁棒性的前提下，提高 head 维度辨别能力

主要风险：

- head 权重设计一旦不稳定，会破坏跨 benchmark 泛化

最小验证方案：

- 只做轻量 calibration
- 例如：
  - per-layer head temperature scaling
  - 基于 entropy 的固定归一化

## 5. Quest 做粗筛而不是最终 scorer

动机：

- Quest 直接做最终 scorer 的效果较差
- 但它的 envelope 思路可能仍适合低成本预过滤

预期收益：

- 在超长上下文下减少后续精排成本

主要风险：

- 如果粗筛过严，会误杀有效块

最小验证方案：

- 只在块数特别多的长上下文样本上试
- 采用：
  - Quest 粗筛
  - blockwise summary 精排

## 四、风险与优先级

如果研究时间有限，我建议优先顺序如下：

1. 固化主线配置
2. 只扩两个 `summary_mode` 候选
3. 把 `tail_query_relevance` 放到 LongBench 特化验证
4. 暂停投入 `random/head/Quest` 这些低 ROI 分支

原因：

- `query_agg=max` 是当前最强跨 benchmark 信号
- `key_norm` 是最稳默认项
- `summary_mode` 是当前最值得继续投入研究预算的核心变量
- 其他方向要么收益弱，要么只有局部 benchmark 上成立

## 五、Stage2 推荐配置

### 主线配置

- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

### 备选配置 1

- `summary_mode=norm_topk_mean_only`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

### 备选配置 2

- `summary_mode=multi_rep_max`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

### LongBench 特化配置

- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=tail_query_relevance`
- `query_agg_mode=mean`
- `head_agg_mode=uniform_mean`

## 六、结论

当前证据表明，`stage2` 最合理的推进方式不是继续扩大全组合，而是围绕以下问题做精细验证：

- `query_agg=max` 作为主线是否在更广设置下保持最稳
- `summary_mode` 是否可以进一步统一为一个跨 benchmark 更稳的形式
- `tail_query_relevance` 是否只是一种 LongBench 特化技巧

因此，最推荐的策略是：

- 先用主线锚点配置稳定推进
- 再把主要实验预算投入到 `summary_mode` 的两到三个候选上
- 将方法创新集中到“自适应 query aggregation / 自适应 summary fusion / 结构化 multi-representative”三条主线上
