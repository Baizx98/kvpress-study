# Blockwise Stage2 结论与 Stage3 计划

## 文档目的

本报告基于以下材料整理：

- [blockwise_stage2_design_report_zh.md](/home10T/bzx/workspace/kvpress-study/note/blockwise_stage2_design_report_zh.md)
- [blockwise_stage2_ratio70_fraction20_multidataset_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/blockwise_stage2_ratio70_fraction20_multidataset_analysis_zh.md)

目标是回答两个问题：

1. `stage2` 到底验证出了什么
2. 下一步 `stage3` 应该如何聚焦，而不是继续盲目扩展组合

## 一、Stage2 的核心结论

## 1. stage2 没有支持“单一 blockwise 配置全局最优”

这是最重要的结论。

在 `ratio=0.7`、`fraction=0.2`、多数据集设置下，不同任务的最优配置明显分化：

- `RULER / 4096`
  - 最优：`chunkkv_prefill_per_layer = 83.61`
- `LongBench / qasper`
  - 最优：`blockwise_norm_topk = 41.51`
- `LongBench / multifieldqa_en`
  - 最优：`blockwise_multi_rep = 56.57`
- `LongBench / hotpotqa`
  - 最优：`blockwise_main = 56.27`
- `LongBench / 2wikimqa`
  - 最优：`chunkkv_prefill_per_layer = 45.13`
- `LongBench / musique`
  - 最优：`chunkkv_prefill_per_layer = 35.63`
- `LongBench / triviaqa`
  - 最优：`blockwise_tail_query_special = 98.00`
- `Needle in a Haystack / 16384`
  - 最优：`blockwise_multi_rep = 73.50`

这说明：

- 当前 blockwise family 已经具备竞争力
- 但还没有收敛到一个跨 benchmark 的统一最优配置
- `stage3` 的重点不应再是“多跑几个组合碰碰运气”，而应转向“解释这种分化，并把它转化成结构化方法”

## 2. `query_agg=max` 依然是 blockwise 主线上最稳的组成部分

从 `stage1` 到 `stage2`，`query_agg=max` 的信号是最稳定的：

- 它支撑了 `blockwise_main`
- 也支撑了 `blockwise_norm_topk`
- 还支撑了 `blockwise_multi_rep`

而 `stage2` 中获胜的 blockwise 方案里，除了 `triviaqa` 特化分支外，其余主力候选都建立在 `max` 上。

因此可以把下面这条判断视为 **已验证结论**：

- 对 blockwise 主线而言，`query_agg=max` 应继续作为默认配置

## 3. `summary_mode` 才是当前最关键的主变量

`stage2` 的结果实际上把 `summary_mode` 的角色进一步放大了：

- `qasper`：`norm_topk_mean_only` 最好
- `multifieldqa_en`：`multi_rep_max` 最好
- `hotpotqa`：`mean_plus_norm_topk_mean` 最好
- `needle`：`multi_rep_max` 最好

这说明：

- 当前 blockwise 的性能差异，更多来自“块摘要如何表达”
- 而不是来自 `head aggregation` 或随机 representative 这些外围因素

所以 `stage3` 最值得投入的部分，不是继续扩展 `head` 或 `Quest`，而是：

- 如何把不同 `summary_mode` 的优点统一起来

## 4. `tail_query_relevance` 不适合作为全局默认，但值得保留为任务特化分支

`stage2` 中：

- `triviaqa` 上最优是 `blockwise_tail_query_special = 98.00`
- 但在其它 LongBench 任务上，它没有形成系统性优势
  - `qasper`: `33.24`
  - `multifieldqa_en`: `44.41`
  - `hotpotqa`: `53.20`
  - `2wikimqa`: `42.87`
  - `musique`: `26.24`

因此这里的合理结论不是：

- “`tail_query_relevance` 更强”

而是：

- “`tail_query_relevance` 对某些任务存在特化上限，但不应升级为全局默认策略”

## 5. `chunkkv` 是真实强 baseline，不能只当陪跑

这是 `stage2` 里另一个非常重要的发现。

`chunkkv_prefill_per_layer` 并没有只是“顺手测一下”，而是在多个数据集上直接赢了：

- `RULER / 4096`: `83.61`
- `2wikimqa`: `45.13`
- `musique`: `35.63`

并且在一些数据集上与 blockwise 主线非常接近：

- `multifieldqa_en`
  - `chunkkv = 53.85`
  - `blockwise_main = 53.74`
- `qasper`
  - `chunkkv = 39.77`
  - `blockwise_main = 40.31`

因此可以把下面这条判断视为 **已验证结论**：

- `chunkkv` 现在是必须严肃对待的主基线，而不是附带 baseline

## 6. 当前最有希望的 blockwise 配置不是一个，而是三个

如果只看 blockwise family，本轮最值得保留的候选有三个：

1. `blockwise_main`
   - `mean_plus_norm_topk_mean + key_norm + max + uniform_mean`
   - 优点：整体最稳，`hotpotqa` 最优

2. `blockwise_norm_topk`
   - `norm_topk_mean_only + key_norm + max + uniform_mean`
   - 优点：在 `qasper` 上最优，延续了 RULER 的 `norm-topk` 信号

3. `blockwise_multi_rep`
   - `multi_rep_max + key_norm + max + uniform_mean`
   - 优点：在 `multifieldqa_en` 和 `needle` 上最优，说明多峰块建模是有效方向

而 `blockwise_tail_query_special` 更适合作为：

- `triviaqa` 特化分支

## 二、Stage2 的研究含义

## 1. blockwise 的问题已经不是“有没有效果”，而是“为什么不同任务偏好不同摘要”

到 `stage2` 为止，这个问题已经很明确：

- blockwise 是有效的
- 但任务之间偏好的 summary 不同

这意味着 `stage3` 的重点不应再停留在：

- “继续枚举更多 summary 组合”

而应转向：

- “解释不同任务需要什么类型的块表达”
- “构造一个能自适应选择摘要方式的 blockwise”

## 2. 目前最值得推进的是“结构改进”，而不是继续加小技巧

`stage1 + stage2` 共同说明：

- `random_topk` 没有持续价值
- `strength_weighted` 没有形成稳定收益
- `top_head_only` 只在局部数据集有正信号
- `Quest-prefill` 没有成为主线

这些都意味着：

- 下一步最值得投入的方法创新，不是继续在外围 aggregation 技巧上微调
- 而是围绕 `summary representation` 本身做结构改进

## 三、Stage3 的总体目标

我建议把 `stage3` 的目标明确定义为：

> 在保留 `query_agg=max` 和 `key_norm` 稳定优势的前提下，设计一个比 `blockwise_main / norm_topk / multi_rep` 更统一、更具任务适应性的块摘要机制，并与 `chunkkv` 做正面对比。

对应地，`stage3` 应该分成两条并行主线：

## 主线 A：Blockwise 内部统一化

目标：

- 把 `main / norm_topk / multi_rep` 三条分化路线统一成一个更强的 block summary

## 主线 B：Blockwise vs ChunkKV 正面对抗

目标：

- 明确 blockwise 该在哪些任务上争胜
- 以及如果不占优，问题到底出在哪里

## 四、Stage3 具体计划

## 1. Stage3-A：统一 blockwise family

### 方向 A1：自适应 summary fusion

动机：

- `hotpotqa` 偏 `mean_plus_norm_topk_mean`
- `qasper` 偏 `norm_topk_mean_only`
- `multifieldqa_en/needle` 偏 `multi_rep_max`

预期收益：

- 用一个统一结构替代三套手工模式
- 提高跨任务稳健性

建议方法：

- 保留三个基础 summary 分量：
  - `mean summary`
  - `norm-topk summary`
  - `multi-representative summary`
- 根据块内统计量做轻量 gating 或插值

最小验证方案：

- 不训练参数
- 先做规则型权重：
  - 如果块内 norm 峰值集中，则偏向 `norm-topk`
  - 如果块内峰值分散，则偏向 `multi_rep`
  - 其它情况保留 `mean+norm-topk`

### 方向 A2：结构化 multi-representative

动机：

- `multi_rep_max` 在 `multifieldqa_en` 与 `needle` 上最好
- 说明“多峰块建模”是有效的

预期收益：

- 改善 multi-key retrieval / 多证据 QA

主要风险：

- representative 之间高度冗余时会放大噪声

最小验证方案：

- 在 `multi_rep_max` 中加入多样性约束：
  - 位置间隔
  - 相似度去重
  - top-k 中心分散

### 方向 A3：任务无监督自适应 query aggregation

动机：

- 大多数任务偏 `max`
- 但 `triviaqa` 明显偏 `tail_query_relevance + mean`

预期收益：

- 减少“一个固定 query aggregation 打全部任务”的偏差

主要风险：

- 容易引入新的 heuristic 不稳定性

最小验证方案：

- 只做 `mean/max` 二选一门控
- 门控依据：
  - query window 峰值比
  - top-1 与均值差距
  - query entropy

## 2. Stage3-B：与 ChunkKV 的正面对比

### 方向 B1：明确 blockwise 的优势区间

根据当前结果：

- blockwise 更强的任务：
  - `hotpotqa`
  - `multifieldqa_en`
  - `triviaqa`
  - `needle`
- chunkkv 更强的任务：
  - `RULER`
  - `2wikimqa`
  - `musique`

这意味着 `stage3` 需要回答：

- blockwise 为何在 retrieval-heavy 任务上不如 chunkkv？
- chunkkv 为何在多文档 QA 某些任务上更稳？

### 方向 B2：构造 blockwise-chunk hybrid baseline

动机：

- `chunkkv` 保留局部语义一致性强
- `blockwise` 在 query-aware coarse selection 上更灵活

预期收益：

- 有机会融合两者优点

建议方法：

- 先做 block-level 选择
- 再在保留块内使用 chunk-wise token selection

最小验证方案：

- 仅实现：
  - `blockwise coarse select`
  - `chunkkv intra-block keep`

如果这条 hybrid 有提升，它会比继续微调单一 block summary 更有论文价值。

## 五、Stage3 实验设计建议

## 1. 不建议的做法

不要在 `stage3` 再做：

- 大规模全组合
- 继续测试 `random_topk`
- 继续测试 `strength_weighted`
- 继续测试 `top_head_only`
- 继续推进当前这版 `Quest-prefill`

这些方向目前证据不足，ROI 太低。

## 2. 建议的最小 stage3 实验矩阵

### 第一组：统一 blockwise family

需要比较：

1. `blockwise_main`
2. `blockwise_norm_topk`
3. `blockwise_multi_rep`
4. `adaptive_summary_fusion`（新方法）

推荐数据集：

- `RULER / 4096`
- `hotpotqa`
- `multifieldqa_en`
- `triviaqa`
- `needle_in_haystack`

理由：

- 这 5 组已经足够覆盖当前分化现象

### 第二组：blockwise vs chunkkv

需要比较：

1. `best_blockwise_from_group1`
2. `chunkkv_prefill_per_layer`
3. `blockwise_chunk_hybrid`（如果实现）

推荐数据集：

- `RULER / 4096`
- `2wikimqa`
- `musique`
- `hotpotqa`
- `needle_in_haystack`

理由：

- 这组能最清楚地拉开 blockwise 与 chunkkv 的优劣区间

### 第三组：比率 sweep

在最终候选上做：

- `ratio=0.3`
- `ratio=0.5`
- `ratio=0.7`

目的：

- 避免当前结论只成立于高压缩率 `0.7`

## 六、当前最推荐的 Stage3 优先级

如果时间有限，我建议优先顺序如下：

1. `adaptive_summary_fusion`
2. `structured_multi_rep`
3. `blockwise vs chunkkv` 正面对比
4. `blockwise-chunk hybrid`

不建议优先做：

- 更复杂的 head aggregation
- 更复杂的 tail-query 变体
- Quest 相关迁移

## 七、结论

`stage2` 的真正价值，不是证明了某一个固定 blockwise 配置已经全局最优，而是把问题收缩到了一个更清晰的层面：

- `query_agg=max` 和 `key_norm` 可以先固定
- 关键矛盾集中在 `summary representation`
- `chunkkv` 是必须认真应对的强基线

因此，`stage3` 的主线不应再是继续枚举组合，而应当是：

- 设计一个能统一 `main / norm_topk / multi_rep` 优点的 block summary
- 并与 `chunkkv` 做正面对抗或融合

如果只给一句最简洁的路线建议，那就是：

> 先做 `adaptive_summary_fusion`，再用它去和 `chunkkv` 在 `RULER + hotpotqa + multifieldqa_en + 2wikimqa + needle` 上打正面对比。


## 附注

- 当前 `failed_jobs_final.jsonl` 中关于 `needle` 的失败记录与最终落盘结果不一致，说明控制器的失败判定逻辑还有瑕疵。
- 但这不影响本轮核心实验结果已经生成并被分析文档读取。

