# RULER 失败案例块分布分析

## 1. 分析范围

本次分析针对 `RULER` 上 `ratio=0.7` 的失败案例，筛选规则为：

- `BlockWisePress` 回答错误
- `ChunkKV` 回答正确
- 任务仅保留：
  - `niah_multikey_3`
  - `niah_single_3`

共选出 6 个代表性样本：

- `niah_multikey_3`：样本 `49`、`64`、`67`
- `niah_single_3`：样本 `27`、`117`、`172`

图像位置：

- [case_0027_niah_single_3.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_failure_block_analysis/case_0027_niah_single_3.png)
- [case_0049_niah_multikey_3.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_failure_block_analysis/case_0049_niah_multikey_3.png)
- [case_0064_niah_multikey_3.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_failure_block_analysis/case_0064_niah_multikey_3.png)
- [case_0067_niah_multikey_3.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_failure_block_analysis/case_0067_niah_multikey_3.png)
- [case_0117_niah_single_3.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_failure_block_analysis/case_0117_niah_single_3.png)
- [case_0172_niah_single_3.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_failure_block_analysis/case_0172_niah_single_3.png)

原始数值导出：

- `evaluation/results/experiments/ruler_failure_block_analysis/artifacts/case_*.json`

## 2. 关键观察

### 2.1 失败并不都来自“答案块完全被删掉”

6 个样本里，只有 1 个最极端的案例表现出“答案块在所有层都未被 BlockWise 保留”：

- 样本 `67`，`niah_multikey_3`
  - 答案块：`80`
  - 问题起始块：`188`
  - `BlockWise`：`32/32` 层都未保留答案块
  - `ChunkKV`：后期层对答案块仍有部分覆盖

这是最纯粹的“关键块被完全错删”。

但另外 5 个失败案例并不是这样：

- 样本 `27`：答案块在 `25/32` 层被 `BlockWise` 保留
- 样本 `49`：答案块在 `24/32` 层被保留
- 样本 `64`：答案块在 `27/32` 层被保留
- 样本 `117`：答案块在 `29/32` 层被保留
- 样本 `172`：答案块在 `28/32` 层被保留

这说明当前问题不能简单概括成“BlockWise 总是删掉答案块”。

### 2.2 更核心的问题是：BlockWise 对答案块的保留不稳定，而 ChunkKV 更连续

在这 6 个 case 中，`ChunkKV` 对答案块的保留呈现出更强的连续性：

- 有些样本从较早层开始就几乎稳定保留
- 有些样本虽然早层覆盖不高，但中后层很快提升到 `0.75` 或 `1.0`

相反，`BlockWise` 的答案块保留更像“跳变型”：

- 某些层保留
- 某些层突然掉掉
- 有时最后一层又丢掉

这种不稳定性在高压缩率下会很危险，因为：

- `RULER` 这类任务对少数关键 token 极其敏感
- 即使答案块在很多层都保留，只要关键层发生断裂，也可能导致最终生成失败

### 2.3 BlockWise 的块分数对“远距离关键块”仍然不够稳

从答案块位置与问题块位置的距离看：

- 样本 `27`：距离 `212` 块
- 样本 `64`：距离 `110` 块
- 样本 `67`：距离 `108` 块
- 样本 `117`：距离 `102` 块

也就是说，不少失败样本中的答案块都离问题尾部很远。

当前 `BlockWisePress` 使用的是：

- 较短的 `question-aware q window`
- 块级摘要 `mean_key + topk_key_mean`
- 最近块保护

这套机制虽然在平均表现上更稳，但对这种“远距离、极少数关键 token 主导”的块，打分仍然会偏弱或波动较大。

### 2.4 `niah_multikey_3` 比 `niah_single_3` 更难

从 6 个 case 看：

- `niah_single_3` 中有些样本，答案块虽然被保留很多层，但仍会答错
- `niah_multikey_3` 中则更容易出现答案块完全错删或最后层分数极低

这说明多 key 检索任务对当前块摘要机制更不友好：

- 单 key 任务中，答案块只要被保留，模型有时仍能恢复
- 多 key 任务中，不仅要保留目标块，还要保证相关支持信息的块分布也足够完整

因此，`BlockWisePress` 当前不只是“答案块打分不够”，还可能存在：

- 与目标块相关的支持块召回不足
- 块间语义关系被高压缩破坏

## 3. 方法层面的判断

### 3.1 当前短板更像“块级分数不稳定 + 块内关键 token 被稀释”

这轮失败 case 支持下面这个判断：

1. `BlockWise` 的确有一部分样本会直接错删答案块。
2. 但更多样本不是简单的“删没了”，而是：
   - 答案块保留不连续
   - 最后一层得分不够稳
   - 相关支持块分布可能不足

所以当前最核心的问题不是 recent 保护不够，而是：

- 块摘要仍然太粗
- 远距离关键块分数不稳定
- 多 key 检索场景下，单块级摘要不足以稳定表达关键 token

### 3.2 这进一步支持引入轻量 token correction

这轮分析更加强化了此前的判断：

- 不应该回到全 token 粒度重方法
- 但只靠当前纯块摘要，也不足以支撑 `RULER 0.7`

因此最合理的下一步仍然是：

- 以块摘要为主
- 在块内增加 very-cheap 的 token correction

例如：

- 仅对块内极少量代表 token 计算 query 相似度
- 用它对块分数做一个轻量校正项

这样可以同时保持：

- 块级元数据常驻显存
- 未来卸载友好
- 比纯摘要更敏锐地识别块内少数关键 token

## 4. 改进建议

### 建议 1：加入 lightweight token correction

优先级最高。

建议形式：

- 每块保留 `top-1` 或 `top-2` 个代表 token 的真实 key
- 块分数由：
  - `summary_score`
  - `token_correction_score`
  做线性组合

这样不会把系统改重，但能明显增强对块内稀有关键 token 的感知能力。

### 建议 2：保留块选择时加入更稳定的 hot block 机制

这轮分析说明 recent protection 不是主要矛盾。

更值得尝试的是：

- 在 recent blocks 之外
- 再显式保留少量高热度远距离块

尤其适合：

- `niah_multikey_3`
- `qa` 类远距离检索任务

### 建议 3：优先针对 RULER 0.7 做小规模验证

不需要先跑全套多数据集。

下一轮建议只测：

- `RULER`
- `ratio=0.7`
- 关注：
  - `niah_multikey_2`
  - `niah_multikey_3`
  - `niah_single_3`

如果这些关键子任务显著回升，再扩展到 LongBench / Needle。

## 5. 一句话结论

这轮失败 case 分析说明，`BlockWisePress` 的问题并不只是“把答案块删掉”，更常见的是：

- 远距离关键块的保留不稳定
- 多 key 任务下支持块召回不足
- 纯块摘要对块内少数关键 token 的表达仍然不够敏锐

因此，下一步最值得做的不是继续堆复杂保护机制，而是引入一个保持低开销的 `lightweight token correction`。
