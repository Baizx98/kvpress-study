# Decode 阶段长输出实验方案（修订版）

## 1. 本轮实验的固定前提

根据当前研究推进节奏，这轮 decode 实验不再继续扩展 prefill 内部消融，而是固定前半段，只研究 decode 阶段策略。

本轮固定约束如下：

- 数据集只使用 `LongBench` 中长输出任务
- prefill 固定为之前最稳的 `blockwise_main`
- prefill 压缩率固定为 `0.3`
  - 按当前项目实现，`compression_ratio=0.3` 表示压掉 `30%` 块，保留约 `70%`
- decode 阶段每隔 `block_size` 个 step 才重新压缩/重评分一次
- decode 打分统一使用：
  - 固定 query window
  - window 内 `q-max`
  - 与所有块摘要打分

这意味着本轮实验的目标已经非常明确：

> 在固定 prefill backbone 的前提下，判断 decode 阶段“永久驱逐”与“计算稀疏”谁更值得继续做。

---

## 2. 数据集选择

这轮只保留 `LongBench` 中真正长输出的 3 个任务：

- `gov_report`
- `qmsum`
- `multi_news`

理由：

- 这 3 个任务在项目里的 `max_new_tokens` 都是 `512 + 20`
- 它们比 `hotpotqa / triviaqa / musique / 2wikimqa` 更能展开 decode 动态
- 生成足够长，才可能真实观察到：
  - 后续 query 演化
  - 冷块重新变热
  - 永久驱逐的不可逆错误
  - 稀疏 decode 的长期收益

因此，本轮不再混入：

- `needle_in_haystack`
- `PG19`
- 其他短输出 LongBench 子任务

不是因为这些数据集没价值，而是因为它们会把这轮 decode 机制验证的信号冲淡。

---

## 3. 样本筛选

即使在长输出任务里，也不是每条样本都会真的产生长 decode。

因此建议先离线筛样本，再固定样本集合。

### 3.1 输出长度筛选

基于参考答案 token 长度筛选：

- 首选：`answer_length >= 160`
- 如果样本不足：退到 `>= 128`
- 如果仍不足以覆盖 `multi_news`：退到 `>= 64`

### 3.2 上下文长度筛选

只保留足够长的输入：

- 首选：`context_length >= 8000`
- 如果样本不足：退到 `>= 6000`
- 如果仍不足以保证三任务各 `20` 条：退到 `>= 4000`

### 3.3 固定样本数量

建议第一轮：

- `gov_report`: `20` 条
- `qmsum`: `20` 条
- `multi_news`: `20` 条

总计约 `60` 条样本。

一旦筛好，这 `60` 条后续全部固定，不再变化。

按当前 `LongBench` 缓存数据的实际分布，首轮正式执行建议直接采用：

- `min_answer_tokens=64`
- `min_context_tokens=4000`
- `max_filtered_samples=20`

原因是：

- `160/8000` 对 `qmsum` 和 `multi_news` 过严
- `128/6000` 仍无法给 `multi_news` 凑够 `20` 条
- `64/4000` 能保证 3 个任务都取到稳定的 `20` 条长输出样本

---

## 4. 固定 backbone

## 4.1 Prefill 方案

prefill 全部固定为：

- `blockwise_main`
  - `summary_mode=mean_plus_norm_topk_mean`
  - `representative_mode=key_norm`
  - `query_agg_mode=max`
  - `head_agg_mode=uniform_mean`

并固定：

- `compression_ratio=0.3`
- `block_size=16`
  - 若你当前主线不是 `16`，则沿用仓库当前 blockwise 默认实验设置

### 4.2 Prefill 后的基准预算

记：

- `N_prefill_blocks`
  - prefill 前上下文块总数
- `B_prefill_keep`
  - prefill 后保留下来的块数

按当前实现：

- `B_prefill_keep = ceil((1 - 0.3) * N_prefill_blocks)`

后续 decode 阶段所有预算设计，都以 `B_prefill_keep` 为锚点。

这样做的好处是：

- decode 策略的预算有统一参照
- 不会因为不同样本原始长度不同而完全不可比

---

## 5. Decode 打分机制

这轮 decode 阶段统一使用一套固定 scorer，不再随方法变化。

## 5.1 Query window

每次重评分时，只看最近固定窗口内的 decode queries。

建议：

- `decode_q_window = block_size`

也就是：

- 若 `block_size=16`
- 则每次重评分只使用最近 `16` 个 decode step 的 query

这样设计的原因是：

- 你本来就计划每隔 `block_size` 步刷新一次
- 所以用最近一个 block 的 query 作为当前局部意图，是最自然且一致的设计

## 5.2 Query aggregation

固定为：

- `q-max`

具体做法：

1. 收集最近 `decode_q_window` 个 query
2. 每个 query 和所有块摘要分别打分
3. 在 query 维度上取 `max`
4. 得到每个块的当前 decode score

即：

- `score(block) = max_{q in recent window} sim(q, block_summary)`

### 为什么这里固定用 `q-max`

原因是：

- 你之前 prefill 阶段已经看到 `max` 比 `mean` 更稳
- decode 阶段我们更关心“最近窗口里是否有某一步强烈需要该块”
- 这比平均需求更符合永久驱逐/冷块调度的语义

因此这轮不要再扫：

- `q-mean`
- `top-r mean`
- `adaptive mean-max`

全部固定成 `q-max`

---

## 6. Decode 刷新节奏

decode 阶段不需要每一步都重压缩，否则开销太大，也会把策略实现复杂度提前拉高。

本轮统一固定：

- 每隔 `block_size` 个 decode step 重评分一次

也就是：

- refresh interval = `block_size`

如果 `block_size=16`，则：

- 每生成 `16` 个 token
- 做一次块打分与状态更新

在两个 refresh 之间：

- `permanent eviction` 直接沿用上一次的幸存块
- `compute cold` 直接沿用上一次的 active block 集合

这样做的意义是：

- 计算开销可控
- 和块粒度设计对齐
- 更接近未来系统实现，而不是理想化每步 oracle

---

## 7. 永久驱逐到底该用 ratio 还是 fixed budget

这是这轮实验里最关键的设计选择之一。

我的明确建议是：

- 主线实验使用 `fixed block budget`
- 不用 `ratio`

## 7.1 为什么不建议把 ratio 作为主线

如果 decode 永久驱逐用比例：

- 每次保留固定比例的块

那么随着 decode 持续生成，新块数会不断增长，于是：

- 幸存块绝对数量也会不断增长

这会带来两个问题：

### 问题 1：显存上界变得模糊

decode 实验研究的是：

- 能不能真正控制 decode 阶段的 memory growth

但 ratio 会让 live KV 总量随着生成长度继续增长，只是增长变慢。

这会导致：

- 永久驱逐看起来“在压缩”
- 但其实没有给出一个稳定的 decode memory cap

### 问题 2：和计算稀疏不公平

如果 `compute cold` 用的是固定 active budget，而 `permanent eviction` 用的是比例增长预算，那么两者比较会不公平：

- 一个在固定资源上做选择
- 一个在资源池随时间增大

这样最后很难解释：

- 质量差异到底来自策略本身
- 还是来自 budget 越跑越宽

## 7.2 为什么 fixed budget 更适合作为主线

decode 永久驱逐更适合表达为：

- 在一个固定大小的历史块预算里，持续做不可逆保留

这才真正对应系统问题：

- 显存预算固定
- 需要不断决定谁永久留下，谁永久删除

因此推荐：

- 历史 context block 使用固定 budget
- 最近 decode block 作为 protected region 单独保留

也就是把 live cache 拆成两部分：

1. `protected recent decode blocks`
2. `fixed historical block budget`

## 7.3 推荐的具体 budget 形式

记：

- `B_prefill_keep`
  - prefill 后保留的历史块数

则 decode 永久驱逐的主线预算定义为：

- `B_hist = B_prefill_keep`

并始终额外保护：

- 最近 `R` 个 decode blocks

建议：

- `R = 2`

因此任一 refresh 时刻，永久驱逐的 live block 总数为：

- `B_live = B_hist + R`

其中：

- 历史块只能保留 `B_hist`
- 新生成的最近 `R` 个 decode blocks 永远不参与驱逐

这个设计的优点是：

- 预算上界稳定
- 与 decode 长度解耦
- 易于解释
- 更符合“永久驱逐”的真实系统目标

## 7.4 Ratio 该怎么处理

不是完全不能用 ratio，而是不建议它做主线。

更合理的位置是：

- 作为后续一组补充对照

即：

- `permanent_eviction_fixed_budget`
- `permanent_eviction_ratio_budget`

只做一个小规模补充实验，用来验证：

- ratio 增长预算是否真的会让结果更好
- 以及它是不是只是靠“越跑预算越大”换来的

但第一轮主实验不要把它放进来。

---

## 8. 方法矩阵

这轮 decode 实验建议只保留 3 个方法。

## 8.1 `prefill_only_no_decode_pruning`

- prefill：固定 `blockwise_main @ compression_ratio=0.3`
- decode：不做驱逐，不做稀疏
- 作用：质量上界参考

## 8.2 `decode_permanent_eviction_fixed_budget`

- prefill：固定 `blockwise_main @ compression_ratio=0.3`
- decode：
  - 每隔 `block_size` 步刷新
  - 用最近固定 query window 的 `q-max` 打分
  - 历史块只保留固定 `B_hist`
  - 最近 `R=2` 个 decode blocks 永远保护
  - 被删块物理上永久移除

这是这轮永久驱逐主线方案。

## 8.3 `decode_compute_cold_fixed_active_budget`

- prefill：固定 `blockwise_main @ compression_ratio=0.3`
- decode：
  - 每隔 `block_size` 步刷新
  - 用最近固定 query window 的 `q-max` 打分
  - 所有 KV 物理保留
  - 只有固定数量 active blocks 参与计算

这里也建议不要直接用 ratio，而是先用固定 active budget：

- `B_active = B_prefill_keep`

这样它和永久驱逐是公平的：

- 一个是固定历史存活预算
- 一个是固定历史计算预算

这两者的差异才真正代表：

- “物理删除”与“仅停止计算”之间的区别

---

## 9. 指标设计

## 9.1 任务质量

沿用 LongBench 现有 scorer：

- `gov_report`
- `qmsum`
- `multi_news`

同时新增：

- `generated_length`

因为 decode 策略有可能让模型输出提前收缩，必须单独监控。

## 9.2 系统指标

至少记录：

- `TTFT`
- `TPOT`
- `peak_gpu_memory`
- `decode_avg_live_blocks`
- `decode_avg_active_blocks`

其中：

- 永久驱逐更关注 `live_blocks`
- 计算稀疏更关注 `active_blocks`

## 9.3 过程诊断指标

建议增加：

- `reactivation_rate`
  - 仅对 `compute_cold` 记录
- `eviction_regret_rate`
  - 对永久驱逐记录
- `late_step_quality_drop`
  - 比较前半段与后半段生成质量

这几个指标会直接帮助解释：

- 为何长输出下 `compute_cold` 可能更稳
- 为何永久驱逐可能在后半段 summary 中累积错误

---

## 10. 第一轮最小实验矩阵

数据集：

- `gov_report`
- `qmsum`
- `multi_news`

样本：

- 每个任务 `20` 条

方法：

- `prefill_only_no_decode_pruning`
- `decode_permanent_eviction_fixed_budget`
- `decode_compute_cold_fixed_active_budget`

固定超参：

- `prefill_compression_ratio=0.3`
- `block_size=16`
- `decode_refresh_interval=16`
- `decode_q_window=16`
- `decode_query_agg=max`
- `R=2`

这样总共是：

- `3 datasets x 3 methods = 9 runs`

这是一个很干净的第一轮最小闭环。

---

## 11. 第二轮补充实验

若第一轮看到了明显差异，再补两件事：

### 11.1 Budget 形式补充

只在一个代表性任务上加：

- `permanent_eviction_ratio_budget`

作用：

- 验证 ratio 是否只是靠预算膨胀获益

### 11.2 Budget 大小补充

只扫少量 budget：

- `0.75 * B_prefill_keep`
- `1.00 * B_prefill_keep`
- `1.25 * B_prefill_keep`

这能帮助判断：

- 当前差异是策略问题
- 还是只是 budget 太紧/太松

---

## 12. 结论

按你现在的约束，这轮 decode 实验应该被改写成一个更严格、也更容易解释的版本：

1. 只用 `LongBench` 长输出任务：
   - `gov_report`
   - `qmsum`
   - `multi_news`
2. prefill 完全固定为：
   - `blockwise_main @ compression_ratio=0.3`
3. decode 完全固定为：
   - 每 `block_size` 步刷新一次
   - 固定 query window
   - window 内 `q-max`
4. 永久驱逐主线采用：
   - `fixed historical block budget`
   - 而不是 ratio
5. 计算稀疏主线采用：
   - `fixed active block budget`

最关键的判断是：

> 对 decode 永久驱逐来说，第一轮主线应该优先使用固定大小的 block budget，而不是比例预算。因为只有 fixed budget 才能真正检验：在长输出、固定资源约束下，不可逆删除是否优于可逆的计算稀疏。
