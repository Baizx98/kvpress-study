# 面向低开销块摘要的 BlockWisePress 简化方案

## 1. 背景与目标

你当前对 `BlockWisePress` 的定位已经非常清楚：

- 它不是要变成 `SnapKV` 或 `ChunkKV` 那样的 token-level 压缩器；
- 它的核心目标是：
  1. 以块为单位压缩；
  2. 用低开销的块摘要替代 token 级 KV；
  3. 让块摘要将来可以常驻显存，作为卸载系统中的热度元数据；
  4. 在 prefill / decode 中，只需要用 query 和每个块的摘要交互即可评分。

因此，当前最重要的不是继续给 `BlockWisePress` 加越来越复杂的打分函数，而是回到最核心的四个问题：

1. 如何构造块摘要
2. prefill 阶段 query window 多大合适
3. 块大小应该是多少
4. 哪些块应当无条件保留

我这里会结合以下三个来源来给出一版更简洁可行的方案：

- `SnapKV` 的代码和论文思想
- `ChunkKV` 的代码和论文思想
- 你当前 `BlockWisePress` 在 10% / 50% 实验中的实际表现

相关实现：

- [snapkv_press.py](/home10T/bzx/workspace/kvpress-study/kvpress/presses/snapkv_press.py)
- [chunkkv_press.py](/home10T/bzx/workspace/kvpress-study/kvpress/presses/chunkkv_press.py)
- [block_wise_press.py](/home10T/bzx/workspace/kvpress-study/kvpress/presses/block_wise_press.py)

相关论文：

- SnapKV: https://arxiv.org/abs/2404.14469
- ChunkKV: https://arxiv.org/abs/2502.00299

## 2. 先看 SnapKV 和 ChunkKV 到底借鉴什么

### 2.1 SnapKV 的核心思想

从代码和论文来看，SnapKV 的核心链路非常直接：

1. 取 prompt 末尾的 observation window 中的 query；
2. 计算这些 query 对前面 token 的 attention；
3. 在 query 维度平均；
4. 再做局部平滑；
5. 按头分组平均；
6. 得到 token 级重要性分数；
7. 强制保留 observation window，本质上不压最后窗口。

它的优点是：

- query-aware 非常强；
- 末尾窗口保护天然稳定；
- 对检索类任务特别有效。

但它的缺点也很明确：

- 需要和 token 级 key 做交互；
- score 是 token-level 的；
- 这条路线不适合你后续“块摘要常驻显存”的目标。

所以对你来说，SnapKV 最值得借的是两点：

1. 只看 prompt 尾部的一段 query 来做评分是合理的
2. 最近窗口/最近区域必须无条件保留

### 2.2 ChunkKV 的核心思想

ChunkKV 的本地实现和论文的共同点是：

- 它先有 token-level 分数；
- 再把 token 分数聚合为 chunk 分数；
- 然后按 chunk 选择。

本地 `ChunkKVPress` 的具体实现方式是：

1. 先调用底层 `ScorerPress` 得到 token 分数；
2. 按 chunk 切分；
3. chunk 内对 token 分数求平均；
4. 选 top-k chunks。

所以你理解得基本是对的：

- 它确实是先 token 打分；
- 再按 chunk 聚合；
- 当前本地实现里，聚合时先在 head 上 `sum(dim=1)`，再在 chunk token 维度 `mean(dim=-1)`。

论文里更强调的是：

- chunk 保持局部语义连续性；
- 比离散 token selection 更稳；
- 并且跨层块索引更相似，可做 reuse。

对你来说，ChunkKV 最值得借的是：

1. 块压缩不能破坏语义片段
2. 最近局部区域和局部连续结构要稳定保住

但它不适合直接成为你的方案，因为它还是建立在 token-level 打分之上。

## 3. 当前 BlockWisePress 的问题本质

从我们前面的实验结果看，当前 `BlockWisePress` 的问题不是“完全做不通”，而是：

1. 块摘要还不够稳定
2. 高压缩率下缺少简单而可靠的保留规则
3. 头维度相关机制已经开始复杂化，但实际收益不明显

更具体地说：

- `needle` 上，新版比旧版好，说明“块内多代表 token”是对的；
- 但 `ruler` 和 `longbench` 仍明显不如 `snapkv/chunkkv`；
- 说明问题不是 query-aware 本身，而是：
  - 块摘要还不够好
  - 无条件保留规则还不够合理
  - 当前实现开始混入过多次要细节，例如 head redundancy penalty

所以我同意你的判断：

- 现在 `BlockWisePress` 有点过于复杂了
- 应该收敛回“低开销块摘要 + 简洁筛选规则”的主线

## 4. 四个关键问题的收敛答案

## 4.1 问题一：如何构造块摘要

这是最核心的问题。

你未来的使用场景决定了块摘要必须满足：

1. 小
2. 稳
3. query-aware 评分时有区分度
4. 能保住块内少量关键 token

### 4.1.1 token 维度如何聚合

我建议不要再继续沿着复杂的非线性函数走，而是回到一个非常简洁的设计：

每个块只保留两类 key 摘要：

1. `mean_key`
   - 表示块整体语义中心
2. `topk_key_mean`
   - 取块内范数最大的前 `k` 个 token 的 key，再求均值

注意：

- 不是保留一个 `peak_key`
- 也不是保留所有 `top-k key`
- 而是保留“前 k 个关键 token 的均值”

我建议这么做的原因是：

#### 原因一：比单个 peak 稳

单个 `peak_key` 容易被偶然极值干扰，不稳定。

#### 原因二：比保留全部 top-k keys 更便宜

如果把所有 `top-k` 代表 key 都留在摘要里，块摘要尺寸会随 `k` 增长。
你将来要让摘要常驻显存，这部分不能太重。

#### 原因三：比纯 mean 更能保住少量关键 token

`topk_key_mean` 本质上是一个“关键 token 子空间中心”，它比整块 mean 更能代表少数关键信息。

### 4.1.2 head 维度如何聚合

我建议现阶段彻底简化：

- 不做 head redundancy penalty
- 不做复杂 head clustering
- 不做 per-head 稀疏选择

直接采用：

1. 先在每个 head 上独立计算块分数
2. 再在 head 维度做简单平均

也就是：

`block_score = mean_over_heads(block_score_per_head)`

为什么我建议这样做：

1. 你当前核心目标是低开销，而不是榨干 head-level 最优性
2. 现有实验没有显示 head 冗余建模带来了明确收益
3. Head 相关复杂逻辑会大幅增加方法描述和调参复杂度
4. 将来如果真有需要，再加一个 very-light head weight 就行，不必现在就上

### 4.1.3 块摘要的最终建议

我建议把块摘要收敛成：

- `mean_key`
- `topk_key_mean`
- `token_count`

可选：

- `mean_value`

但如果只是用于重要性评分，`mean_value` 不是必须项。
如果未来 offload 系统要把它当块元数据一起维护，可以保留。

### 4.1.4 当前 `k` 取多少

你前面说希望先用固定值，不要非线性函数。

我同意。

在当前阶段，建议先固定：

- `block_size=16` 时，`k=2`
- `block_size=32` 时，`k=4`

如果你只先跑一种块大小，我建议先统一用：

- `block_size=16`
- `k=2`

这是一个非常干净的起点：

- 一个块的整体语义：`mean_key`
- 一个块的关键子集语义：`top2_key_mean`

然后块分数可以写成：

`score = alpha * sim(q, mean_key) + (1 - alpha) * sim(q, topk_key_mean)`

建议固定：

- `alpha = 0.5`

也就是先不要三项混合了，直接两项混合即可。

## 4.2 问题二：prefill 阶段参与计算的 q 的窗口大小应该是多少

这部分我建议直接借 SnapKV 的思路，简单固定，不用函数。

### 4.2.1 固定窗口比自适应函数更适合当前阶段

理由：

1. 现在你最需要的是稳定对比，而不是继续增加一个动态来源
2. SnapKV 已经说明“末尾 observation window”是有效的
3. 你的 query-aware 是 question-aware prefill 场景，末尾窗口天然就是最相关区域

### 4.2.2 推荐值

我建议当前阶段直接固定：

- `q_window = 64`

如果 prompt 很短，就取：

- `min(seq_len, 64)`

为什么先选 `64`：

1. SnapKV 默认就是 `64`
2. 这个值已经在长上下文压缩里被大量使用
3. 对你来说，`64` 个 query 和块摘要交互，代价已经很低

如果后面觉得 `64` 还是太贵，可以再试：

- `32`
- `64`

但第一轮我建议就固定 `64`，不要再引入自适应函数。

## 4.3 问题三：块大小应该是多少

这部分要分“当前研究原型”和“未来系统目标”来看。

### 4.3.1 从系统目标看

你最终是为了：

- 块管理
- 块卸载
- 低元数据开销

因此块不能太小。

如果块太小：

- 管理粒度太细
- 块数太多
- 元数据和调度成本上升

### 4.3.2 从精度看

如果块太大：

- 一个块里的语义混合更严重
- 块摘要更容易平均掉关键 token

### 4.3.3 当前建议

我建议当前阶段只认真比较两个值：

- `block_size = 16`
- `block_size = 32`

不建议一上来扫太多值。

从目前你的结果和目标看，我更推荐优先把默认值设成：

- `block_size = 16`

原因：

1. 对块摘要方法来说，`16` 比 `32` 更容易保真
2. 未来即使做卸载，`16` 也还是一个合理的管理单元
3. `ChunkKV` 的强势恰恰说明：语义连续单元不宜过大

因此我的建议是：

- 当前主线方法先固定 `block_size=16`
- 如果后面要补系统实验，再额外比较 `32`

## 4.4 问题四：哪些块应该无条件保留

这里我建议只保留两类，而且都尽量简单。

### 4.4.1 第一类：最近块

这部分几乎必须保留，原因直接来自 SnapKV：

- 最近窗口里的 query 对后续生成最关键
- 最近上下文也最可能被立刻复用

我建议当前阶段固定：

- `recent_blocks = 4`

如果 `block_size=16`，那就是无条件保留最近 `64` 个 token。

这和 SnapKV 的 `window_size=64` 正好对齐，语义非常清楚。

### 4.4.2 第二类：尾部不完整块

如果最后一个块是不完整块，建议无条件保留。

原因：

- 这部分通常就是最新加入的 token
- 同时可以避免不同层 cache 长度不一致

### 4.4.3 暂时不要无条件保留 `top hot blocks`

你之前问过 `top hot blocks` 是什么。

在目前这个更简化的阶段，我反而建议：

- 先不要上这个机制

原因：

1. 它虽然有道理，但会引入另一层筛选逻辑
2. 你现在的首要问题不是“块类型不够多”，而是“块摘要是否足够好”
3. 最近块 + 正常 top-k 筛选，已经够构成一个清楚的第一版

也就是说，当前无条件保留就两类：

1. 最近 `4` 个块
2. 尾部不完整块

其余块全部通过分数竞争。

如果压缩率极端高，预算不足以保住这些 recent blocks，那么就像你之前要求的：

- recent blocks 也参与压缩
- 打一条日志

## 5. 一版更简洁的 BlockWisePress 方案

基于上面的收敛，我建议把下一版 `BlockWisePress` 简化成下面这套：

### 5.1 固定超参数

- `block_size = 16`
- `q_window = 64`
- `topk = 2`
- `recent_blocks = 4`
- `alpha = 0.5`

### 5.2 块摘要

每个块缓存：

- `mean_key`
- `top2_key_mean`

可选：

- `mean_value`

### 5.3 块分数

先对每个 head 分别算：

- `sim(q, mean_key)`
- `sim(q, top2_key_mean)`

在 query 维度做平均，然后 head 维度做平均：

`score(block) = 0.5 * avg_qh sim(q, mean_key) + 0.5 * avg_qh sim(q, top2_key_mean)`

这里的 `avg_qh` 表示：

- 先对 query window 求平均
- 再对 heads 求平均

### 5.4 块筛选

1. 先保护最近 `4` 个块
2. 先保护尾部不完整块
3. 剩余预算从其余块里按分数 top-k 选
4. 如果预算太紧，保护块也参与筛选，并输出日志

### 5.5 明确去掉的复杂项

我建议这版先去掉：

- head redundancy penalty
- 复杂 head weighting
- 多种 head scoring method
- 非线性 `k(block_size)` 函数
- 非线性 `recent(num_blocks)` 函数
- `top hot blocks`
- 层间 reuse

原因很简单：

- 这些东西现在都不是主矛盾
- 会让方法描述复杂化
- 不利于你后面把它放进卸载系统里讲成一个简洁、可信的故事

## 6. 下一步最值得做的实验

我建议下一轮不要再大扫全量组合，而是先做一个小而准的 ablation：

### 6.1 固定主方案

- `block_size=16`
- `q_window=64`
- `topk=2`
- `recent_blocks=4`

### 6.2 只测这几个对照

1. `block_wise_simple`
   - 只用 `mean_key`
2. `block_wise_simple + top2_key_mean`
3. `block_wise_simple + top2_key_mean + recent_blocks`

然后和：

- `snapkv`
- `chunkkv`

对比。

### 6.3 重点观察

只看最有信息量的点：

- `ruler`: `0.5 / 0.7`
- `longbench(triviaqa)`: `0.5 / 0.7`
- `needle_in_haystack`: `0.7`

这样你能快速回答两个问题：

1. 块摘要改良本身有没有用
2. recent block 保留有没有用

## 7. 总结

一句话总结：

你现在不该继续给 `BlockWisePress` 叠加复杂机制，而应该把它收敛成一个真正面向“块摘要常驻显存”的低开销设计。

我建议的收敛版是：

- 固定 `block_size=16`
- 固定 `q_window=64`
- 每块只保留 `mean_key + top2_key_mean`
- query 和块摘要交互后，在 query/head 上直接平均
- 只无条件保留最近 `4` 个块和尾部不完整块
- 去掉 head redundancy、非线性函数和其它复杂筛选

这样既保留了你的核心系统目标，也借到了 SnapKV / ChunkKV 最值得借的地方，而且方法会比现在更容易调、也更容易讲清楚。

## 参考

- SnapKV 论文：https://arxiv.org/abs/2404.14469
- ChunkKV 论文：https://arxiv.org/abs/2502.00299
- [snapkv_press.py](/home10T/bzx/workspace/kvpress-study/kvpress/presses/snapkv_press.py)
- [chunkkv_press.py](/home10T/bzx/workspace/kvpress-study/kvpress/presses/chunkkv_press.py)
