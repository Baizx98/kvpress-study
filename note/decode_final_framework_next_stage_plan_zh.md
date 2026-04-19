# 最终推理框架下一阶段实验规划

## 1. 目标

当前 `decode_long_output_longbench_stage1` 已经说明：

- 只做 decode fixed-budget 策略时
- `permanent_fixed_budget` 和 `compute_cold_fixed_budget`
- 在长输出任务上都能维持接近 baseline 的质量

因此，下一阶段不应再停留在“decode 单点可行性”上，而应转向最终推理框架视角：

> 最终要部署或写进论文的方法，究竟应当采用怎样的 prefill + decode 组合框架？

这意味着下一阶段的核心问题不再是：

- 单独某个 decode 技巧是否存在

而是：

1. decode 策略在不同 budget 下是否稳定
2. decode 策略与 prefill 压缩结合后是否仍然成立
3. 最终框架是否需要“单一统一策略”，还是“prefill 与 decode 分别负责不同目标”

---

## 2. 实验原则

本阶段明确不把重点放在：

- `TPOT`
- `peak memory`
- `live block`
- `active block`

这些系统指标上。

当前阶段优先看：

- 最终任务质量
- 不同 budget 下的质量稳定性
- 不同推理框架组合下的质量一致性

一句话说，就是：

> 先把“最终框架该长什么样”定下来，再决定是否值得围绕它去做更重的系统优化。

---

## 3. 最终框架视角下的候选路线

下一阶段建议只保留 4 条候选路线，不再扩太多组合。

## 3.1 路线 A：No-Prefill-Compression + Permanent Decode

记作：

- `dense_prefill + permanent_decode`

含义：

- prefill 不压缩
- decode 用 `permanent_fixed_budget`

这条路线的意义是：

- 检查如果把主要预算控制完全交给 decode 阶段，质量能否稳定

## 3.2 路线 B：No-Prefill-Compression + Compute-Cold Decode

记作：

- `dense_prefill + compute_cold_decode`

含义：

- prefill 不压缩
- decode 用 `compute_cold_fixed_budget`

这条路线的意义是：

- 检查如果 decode 阶段采用可逆稀疏，是否比永久驱逐更适合作为最终策略

## 3.3 路线 C：Blockwise Prefill + Permanent Decode

记作：

- `blockwise_prefill + permanent_decode`

含义：

- prefill 用 `blockwise_main`
- decode 用 `permanent_fixed_budget`

这条路线更像：

- prefill 先做一次静态结构化缩减
- decode 再做严格预算控制

## 3.4 路线 D：Blockwise Prefill + Compute-Cold Decode

记作：

- `blockwise_prefill + compute_cold_decode`

含义：

- prefill 用 `blockwise_main`
- decode 用 `compute_cold_fixed_budget`

这条路线更像：

- prefill 负责静态筛减
- decode 负责动态 query-aware 调度

---

## 4. 数据集设计

## 4.1 主验证集：LongBench 长输出

主验证仍建议保持：

- `gov_report`
- `qmsum`
- `multi_news`

原因：

- 当前 decode 机制确实需要长输出才能充分展开
- 这 3 个任务已经证明能稳定提供长生成

采样策略继续固定：

- `min_answer_tokens=64`
- `min_context_tokens=4000`
- `max_filtered_samples=20`

这样下一阶段能和上一轮直接对齐。

## 4.2 补充验证集：RULER

这次可以把 `RULER` 加进来，但只作为补充验证，不作为主搜索集。

建议使用：

- `niah_single_3`
- `niah_multikey_2`
- `niah_multikey_3`
- `qa_2`

理由：

- 这些任务在仓库里已有稳定使用历史
- 能很好地检验：
  - 永久驱逐是否会误删稀疏关键块
  - `compute_cold` 对多 key / 多支持证据是否更稳

但要明确：

- `RULER` 输出通常较短
- 所以它不用于证明“decode 长输出收益”
- 它只用于补充回答：
  - 最终框架在检索型、合成型任务上有没有明显副作用

---

## 5. 预算设计

这次的核心 sweep 维度仍然是 budget，而不是 compression ratio。  
但这里的 budget 必须是：

- 固定的绝对块数

而不是：

- 根据请求长度变化的百分比

因为你现在关心的是最终推理框架在固定容量约束下的稳定性，而不是“不同长度样本各自按比例缩放后”的相对表现。

### 建议固定 budget

统一使用 3 档固定历史块 budget：

- `tight = 96 blocks`
- `medium = 128 blocks`
- `loose = 160 blocks`

在当前 `block_size=16` 下，对应的大致历史 token 容量为：

- `96 blocks ~= 1536 tokens`
- `128 blocks ~= 2048 tokens`
- `160 blocks ~= 2560 tokens`

此外继续固定：

- `protected_recent_blocks = 2`

因此 decode 时实际 live/active 上界是：

- `budget + 2 recent blocks`

### 为什么用这三档

这组预算的优点是：

1. 对 `LongBench` 长输出任务有足够约束
   - 当前筛出的上下文最短也在 `4000` token 左右
2. 对 `RULER 4096` 也有区分度
   - `4096 / 16 = 256 blocks`
3. 能同时适用于：
   - `dense_prefill + decode`
   - `blockwise_prefill + decode`

也就是说，这次的 budget 语义就是：

> 不管请求多长，也不管 prefill 是否先压缩，decode 阶段都只能在同一组固定绝对容量内工作。

---

## 6. 实验矩阵

## 6.1 第一阶段：只做 LongBench 主矩阵

先在主验证集上跑：

- 路线 A
- 路线 B
- 路线 C
- 路线 D

每条路线 3 档固定 budget：

- `96 blocks`
- `128 blocks`
- `160 blocks`
- `192 blocks`

总计：

- `4 routes x 3 budgets x 3 datasets = 36 runs`

这是下一阶段最关键的一组实验。

## 6.2 第二阶段：RULER 补充矩阵

只保留最有代表性的 2 条路线：

- `dense_prefill + permanent_decode`
- `dense_prefill + compute_cold_decode`

budget 只跑：

- `160 blocks`
- `192 blocks`

任务：

- `niah_single_3`
- `niah_multikey_2`
- `niah_multikey_3`
- `qa_2`

总计：

- `2 routes x 2 budgets x 4 tasks = 16 runs`

这样 `RULER` 不会反客为主，但足够给出补充判断。

---

## 7. 我建议的分析重点

这次不要再把主要精力放在：

- 谁更快
- 谁更省显存

而是看下面 4 个判断。

## 7.1 最终框架到底需不需要 prefill 压缩

这个问题可以通过比较：

- 路线 A/B
- 路线 C/D

来回答。

如果 `dense_prefill + decode` 已经很好，那说明：

- prefill 压缩未必是最终框架必须项

如果 `blockwise_prefill + decode` 明显更稳，那说明：

- prefill 静态筛减仍然是最终框架的重要组成部分

## 7.2 最终框架到底该选 permanent 还是 compute-cold

这是通过比较：

- A vs B
- C vs D

来回答的。

如果 `compute_cold` 在 LongBench 长输出上持续更稳，且在 RULER 多 key 任务上也更稳，那么：

- 最终框架应优先围绕 `compute_cold`

如果 `permanent` 一直不输，甚至在部分任务略好，那么：

- 更简单的永久驱逐可能已经足够

## 7.3 结论是否依赖 budget

如果某条路线只在 `160 blocks` 下成立，而一旦收紧到 `96 blocks` 就崩，那么：

- 它不适合作为“稳健最终框架”

我们更应该关注：

- 哪条路线在 `96/128/160 blocks` 三档固定 budget 下结论最一致

## 7.4 RULER 与 LongBench 是否给出冲突信号

如果：

- LongBench 更偏 `compute_cold`
- RULER 更偏 `permanent`

那说明：

- 最终框架不能只靠一个 benchmark 定结论

这时论文里更好的说法会是：

- 长输出真实任务更适合可逆 decode 稀疏
- 高稀疏检索型任务更依赖稳定保留关键块

---

## 8. 我对下一阶段结果的预期

当前我更倾向的工作假设是：

1. `compute_cold` 会在 `LongBench / multi_news` 上继续优于 `permanent`
2. `gov_report / qmsum` 上两者差异仍然很小
3. `RULER / niah_multikey_2/3` 上，`compute_cold` 可能更稳
4. `blockwise_prefill + decode` 大概率会比 `dense_prefill + decode` 更稳一些

但我不认为这些结论已经被证明，所以仍应以实验为准。

---

## 9. 推荐执行顺序

为了控制实验量，我建议这样推进：

1. 先跑 LongBench 主矩阵中的 `128 blocks`
   - 先粗看路线优劣
2. 再补 `96 / 160 blocks`
   - 看稳定性
3. 最后跑 RULER 补充矩阵
   - 看是否出现冲突结论

如果第一步就出现明显趋势：

- `permanent` 明显输
- 或 `dense_prefill + decode` 明显输

那么后续矩阵可以直接裁掉一半。

---

## 10. 最终建议

如果从“最终推理框架”出发，而不是继续做单点 decode 技巧，我建议下一阶段把实验问题改写成：

> 在长输出任务和检索型补充任务上，最终推理框架究竟应采用：
> 1) 是否需要 prefill 压缩
> 2) decode 阶段是永久驱逐还是可逆计算冷块
> 3) 这一结论在不同 budget 下是否稳定

对应的实验最小闭环就是：

- `4` 条框架路线
- `3` 档 budget
- `LongBench` 长输出主验证
- `RULER` 补充验证

这会比继续围绕单一 decode 方法做局部优化，更接近你真正要落地的最终推理框架。*** End Patch
