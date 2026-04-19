# 最终推理框架下一步建议

## 1. 先回答核心问题

> 下一步还要不要继续在算法上探索？

结论是：

- **要，但只能做非常收敛的探索**
- **不要再做大范围算法搜索**

更具体地说：

- 不值得继续的，是：
  - `blockwise` 内部 summary / representative / aggregation 的继续排列组合
  - prefill 阶段再扩很多新启发式
  - decode 阶段再同时发散出很多不同机制
- 仍然值得继续的，是：
  - 围绕**最终推理框架**做 1 条到 2 条非常聚焦的算法路线验证

也就是说，下一步不是“继续算法海选”，而是“框架定型前的最后一轮算法判别”。

---

## 2. 为什么不该继续大范围算法搜索

结合你前面的结果，已经有 3 个比较稳定的事实。

### 2.1 prefill 方向已经接近阶段性上限

前面的 stage1 / stage2 / stage3 已经说明：

- `blockwise` family 内部不同 summary 组合确实会影响结果
- 但这种影响越来越像任务特化差异，而不是结构性突破

因此继续在：

- `mean / topk / rep`
- `query agg`
- `head agg`
- representative 去冗余

这些点上深挖，收益已经明显下降。

### 2.2 最终框架结果已经给出更强信号

fixed-budget stage1 的结果说明：

- 当前 `LongBench` 长输出上，最优整体路线不是 `blockwise_prefill + decode`
- 而是 `dense_prefill + decode`

这说明：

- 当前阶段 prefill 压缩并不是最终框架的默认赢家

所以如果这时候还继续在 prefill 里大规模探索算法，方向就会偏。

### 2.3 decode 机制差异已经缩到少数关键矛盾

现在真正还没回答清楚的不是：

- decode 有没有很多花样可试

而是非常具体的一个问题：

- 最终框架里，decode 应该偏 `permanent`，还是偏 `compute-cold`

这已经不是“大范围算法搜索”的问题了，而是“最后一轮框架路线选择”的问题。

---

## 3. 我建议停止的探索

下面这些方向，我建议现在就停，不要再投入主力：

### 3.1 停止继续做 blockwise 内部细碎消融

包括：

- 新的 summary mixing
- 新的 representative 打分
- 新的 query/head 聚合小变体

原因：

- 已有结果足够说明这条线的上限快到了
- 很难再产出“改变最终框架选择”的结论

### 3.2 暂停把 prefill 压缩当作主线继续扩展

不是说 prefill 永远没价值，而是：

- 你当前结果已经显示 `dense_prefill + decode` 更强

所以这时候继续把主要精力放在 prefill 算法上，不是最优投入。

### 3.3 暂停继续扩展更多 decode 机制分支

例如：

- 再加第 4、第 5 种 decode 策略
- 再加很多复杂状态机

原因：

- 你现在最需要的是缩小选择集，而不是扩大选择集

---

## 4. 我建议继续的算法探索

如果还要继续做算法，我只建议保留 1 条主线：

## 4.1 做一个 hybrid decode 策略

这是我认为现在最值得继续的唯一算法方向。

### 动机

当前结果已经给出非常明确的偏好分裂：

- `gov_report / qmsum` 更偏 `permanent`
- `multi_news` 更偏 `compute-cold`

这意味着：

- `permanent` 和 `compute-cold` 各自抓住了不同任务的需求
- 但没有哪一个单独统治所有长输出任务

所以最自然的下一步不是再多试几个单独策略，而是：

- 做一个两者结合的 hybrid

### 建议形式

一个很自然的 hybrid 是：

- `permanent core + cold fringe`

也就是：

1. 保留一部分 **核心块** 为永久保留
2. 额外留一部分 **边缘块** 为 cold blocks
3. decode 时：
   - 核心块始终在
   - fringe 块可热可冷，可重新激活

### 为什么这个方向合理

它和当前结果的关系非常直接：

- `permanent` 的优势：
  - 稳定
  - 不容易丢核心信息
- `compute-cold` 的优势：
  - 对后续 query 演化更宽容

hybrid 的本质就是同时保这两个优点。

### 最小实现建议

先不要做复杂动态配比，只做固定分配：

- `budget = 128`
  - `core = 96`
  - `fringe = 32`
- `budget = 160`
  - `core = 128`
  - `fringe = 32`

只要先验证：

- 这种两段式预算是否比纯 `permanent` 或纯 `compute-cold` 更稳

就够了。

---

## 5. 下一步实验应该怎么安排

如果从“最终推理框架”考虑，我建议下一轮实验分三步。

## 5.1 第一步：先收缩到 dense_prefill 主线

只保留：

- `dense_prefill + permanent_decode`
- `dense_prefill + compute_cold_decode`

原因：

- 当前它们已经整体强于 `blockwise_prefill + decode`
- 先把最终框架主线缩到最强候选，避免再被 prefill 干扰

数据集：

- `LongBench / gov_report`
- `LongBench / qmsum`
- `LongBench / multi_news`

budget：

- `128`
- `160`

这一步的目标是：

- 先确定最强 decode 主路线

## 5.2 第二步：只加 1 个 hybrid 算法

加入：

- `dense_prefill + hybrid_decode`

与前两条路线对比：

- `dense + permanent`
- `dense + compute-cold`
- `dense + hybrid`

如果 hybrid 没有明显增益，就说明：

- 当前阶段不值得继续做算法探索

如果 hybrid 有明确增益，就说明：

- 这就是下一阶段该继续推进的唯一路线

## 5.3 第三步：用 RULER 只做补充验证

不要拿 `RULER` 做主搜索，只做补充确认。

任务：

- `niah_single_3`
- `niah_multikey_2`
- `niah_multikey_3`
- `qa_2`

目的只有一个：

- 看最强 LongBench 路线在合成检索任务上有没有明显副作用

---

## 6. 如果你不想再继续算法探索

其实也完全说得通。

因为如果你现在的目标是：

- 尽快把最终框架定下来
- 开始做更大规模主实验
- 开始写故事和论文结构

那你也可以选择：

- 停止算法探索
- 直接把当前最优路线定成：
  - `dense_prefill + permanent_decode @ 160`
  - 或者把 `multi_news` 单独列出来说明 `compute-cold` 更好

这条路线的优点是：

- 节奏更快
- 结论更稳定

缺点是：

- 你可能会错过一次很有希望的 `hybrid` 增益机会

所以我的建议是：

- **最多只再给算法探索一次机会**
- 这次机会只留给 `hybrid decode`

如果它不成，就彻底停。

---

## 7. 最终建议

最简洁的判断是：

1. **没必要继续做大范围算法探索**
2. **还有必要做一次非常聚焦的算法探索**
3. **这次探索只应该是 hybrid decode**
4. **如果 hybrid 没明显赢，就停止算法探索，转入框架定型和主实验**

所以如果你问我一句最直接的话：

> 还有必要继续在算法上探索吗？

我的回答是：

> 还有必要，但只值得再做最后一轮，而且只能做 `hybrid decode` 这一条。除此之外，不建议再继续扩算法树。
