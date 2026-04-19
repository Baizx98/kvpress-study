# Decode Final Framework Fixed-Budget Stage1 分析

## 1. 实验目的

本轮实验从“最终推理框架”视角比较 4 条路线：

- `dense_prefill + permanent_decode`
- `dense_prefill + compute_cold_decode`
- `blockwise_prefill + permanent_decode`
- `blockwise_prefill + compute_cold_decode`

并且所有 decode 路线都使用固定绝对 budget，而不是按请求长度百分比：

- `96 blocks`
- `128 blocks`
- `160 blocks`

本轮重点只看任务质量，不把时间、显存作为主判断标准。

---

## 2. 数据与去重说明

主验证集：

- `LongBench / gov_report`
- `LongBench / qmsum`
- `LongBench / multi_news`

补充验证集：

- `RULER / 4096`
  - `niah_single_3`
  - `niah_multikey_2`
  - `niah_multikey_3`
  - `qa_2`

原始结果里共有：

- `48` 个 `metrics.json`

但逻辑配置只有：

- `40` 组

原因是：

- `RULER` 的 4 组 `dense_prefill` 配置出现了控制器“假失败重跑”
- `return_code=0` 的成功任务仍被记进了 `failed_jobs`
- 因此产生了 `/1`、`/2` 重跑目录

本分析采用的规则是：

- 按逻辑配置去重
- 只保留每组配置最新的一份有效 `metrics.json`

所以以下结论基于：

- `40` 组去重后的有效结果

---

## 3. LongBench 主结果

## 3.1 每个数据集的最优配置

- `gov_report`
  - 最优：`dense_prefill + permanent_decode @ 160 blocks = 32.65`
- `qmsum`
  - 最优：`dense_prefill + permanent_decode @ 160 blocks = 24.87`
- `multi_news`
  - 最优：`dense_prefill + compute_cold_decode @ 160 blocks = 25.38`

LongBench 宏平均最优：

- `dense_prefill + permanent_decode @ 160 blocks = 27.16`

这说明：

- 当前阶段最优框架并不是 `blockwise_prefill + decode`
- 而是 `dense_prefill + decode`

这和之前“prefill 先压缩再 decode”直觉并不完全一致，是这轮最值得重视的结果。

## 3.2 预算趋势

整体上，LongBench 三个长输出任务都有一个很稳定的现象：

- `160 blocks` 普遍优于 `128 blocks`
- `128 blocks` 普遍优于 `96 blocks`

也就是说，在当前这组任务和模型上：

- 预算越宽，质量越高

这本身并不意外，但它说明：

- 当前三档 fixed budget 都还处在“有效约束区间”
- 没有出现某一档已经宽到近似无约束的情况

因此，这组 fixed budget 是有分析价值的。

## 3.3 路线比较

### `dense_prefill + permanent_decode`

这是当前 LongBench 上最强的总体路线。

优点：

- `gov_report` 最优
- `qmsum` 最优
- 宏平均最优

含义：

- 在长输出摘要/会议问答类任务上，当前阶段 prefill 不压缩、decode 再做固定预算永久驱逐，反而是最稳的路线

### `dense_prefill + compute_cold_decode`

优点：

- `multi_news` 最优
- 在 `96/128/160` 三档 budget 下整体都不差

含义：

- 对多文档摘要这类需要后续重新访问不同证据块的任务，可逆的 `compute-cold` 更合适

### `blockwise_prefill + permanent_decode`

这是当前 LongBench 上最弱的一条路线之一，尤其在：

- `multi_news`
- `qmsum`

上表现一般。

说明：

- prefill 已经先做一次静态压缩后，再叠加永久驱逐，可能过早损失了后续 decode 仍然需要的信息

### `blockwise_prefill + compute_cold_decode`

它整体略好于 `blockwise_prefill + permanent_decode`，但仍未超过 `dense_prefill` 两条路线。

说明：

- 当前 `blockwise_prefill` 这一步本身已经带来了不小的质量负担
- decode 再怎么选，短期内都很难完全补回来

---

## 4. RULER 补充结果

去重后，RULER 的结果非常整齐：

### `128 blocks`

- `dense_prefill + permanent_decode = 80.0`
- `dense_prefill + compute_cold_decode = 80.0`

### `160 blocks`

- `dense_prefill + permanent_decode = 87.5`
- `dense_prefill + compute_cold_decode = 87.5`

更细看任务：

- `niah_single_3`
  - 两者都稳定 `100`
- `niah_multikey_2`
  - `128 -> 95`
  - `160 -> 100`
- `niah_multikey_3`
  - `128 -> 65`
  - `160 -> 90`
- `qa_2`
  - 始终 `60`

结论：

- 在这组 `RULER` 补充验证里，`permanent` 与 `compute-cold` 没有拉开差距
- 真正有影响的是 budget 大小
- 从 `128` 放宽到 `160` 时，`niah_multikey_3` 改善最明显

这说明：

- RULER 这轮主要在回答“预算够不够”
- 而不是“decode 机制选哪条路线”

---

## 5. 核心结论

这轮 fixed-budget stage1 的关键信号可以收敛成 4 条。

### 5.1 最终框架暂时不需要 prefill 压缩

至少在当前这组长输出任务上：

- `dense_prefill + decode`
  明显强于
- `blockwise_prefill + decode`

这说明当前最终框架不应默认把 prefill 压缩作为必须组件。

### 5.2 LongBench 更偏向 decode-only 框架

更具体地说：

- `gov_report / qmsum` 更偏 `dense + permanent`
- `multi_news` 更偏 `dense + compute-cold`

因此目前更像是：

- decode-only 框架已经足够强
- 关键问题变成 decode 阶段到底该选哪条路线

### 5.3 permanent 与 compute-cold 的胜负依赖任务

当前没有证据支持：

- `compute-cold` 全面优于 `permanent`

也没有证据支持：

- `permanent` 全面优于 `compute-cold`

更准确的说法是：

- `permanent` 对 `gov_report / qmsum` 更稳
- `compute-cold` 对 `multi_news` 更稳

这说明最终框架可能需要：

- 更细的任务感知选择
- 或者更鲁棒的统一策略

### 5.4 RULER 这轮主要说明“预算重要”，而不是“机制不同”

在当前 4 个 RULER 子任务上：

- 机制差异几乎看不出来
- budget 差异很明显

所以 RULER 在下一轮仍然适合做：

- 固定 budget 合理性验证

但不适合单独决定：

- `permanent` 还是 `compute-cold`

---

## 6. 对下一轮实验的建议

如果从最终推理框架角度继续推进，我建议下一轮收缩成下面两步。

### 第一步：只保留 dense_prefill 两条 decode 路线

保留：

- `dense_prefill + permanent_decode`
- `dense_prefill + compute_cold_decode`

原因：

- 这两条路线已经全面优于 `blockwise_prefill + decode`
- 继续在当前阶段把 blockwise_prefill 带进来，收益不高

### 第二步：继续做 fixed budget，但集中看 LongBench

建议只保留：

- `128`
- `160`

必要时再加：

- `192`

因为：

- `96` 已经能看到明显退化
- `128/160` 更接近真正可用的工作区间

### RULER 的角色

RULER 下一轮只作为补充验证即可：

- 验证 budget 放宽是否继续改善 `niah_multikey_3`
- 看 `dense + permanent` 与 `dense + cold` 是否开始出现差异

---

## 7. 一句话总结

当前 fixed-budget stage1 最重要的结论是：

> 在当前长输出任务上，最终推理框架暂时更应优先考虑 `dense_prefill + decode`，而不是 `blockwise_prefill + decode`；其中 `gov_report/qmsum` 更偏 `permanent`，`multi_news` 更偏 `compute-cold`，而 `RULER` 主要说明 budget 大小比 decode 机制差异更重要。
