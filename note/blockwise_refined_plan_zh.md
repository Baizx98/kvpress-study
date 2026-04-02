# BlockWisePress 优化计划（结合 ChunkKV 借鉴与当前约束）

## 1. 你的新要求总结

基于你刚刚的反馈，后续优化要满足下面这些约束：

1. 类名保持为 `BlockWisePress`，不改命名。
2. 当前单一 `peak_key` 要扩展成少量代表性 token 摘要。
3. `top-k` 里的 `k` 不能写死，要和 `block_size` 相关，而且最好是非线性关系。
4. 块分数继续改成“整体 + 峰值 + 多峰”混合。
5. `recent blocks` 机制要继续保留，但要明确：
   - `N_recent` 可以是固定值，也可以和长度相关；
   - 这些块默认不受压缩率影响；
   - 但如果压缩率极端高，预算不够覆盖非 recent 块，那么 recent 块也必须参与压缩；
   - 发生这种情况时，要输出日志。
6. `top hot blocks` 的定义和筛选方式需要解释清楚。
7. `layer-wise reuse` 先不做，现阶段仍然每层单独重新算。
8. ChunkKV 借鉴的是其“语义感知的 chunk 压缩思想”，但它的 chunk 大小本质上仍然是固定块大小，这一点与你的 block-size 设定并不冲突。

这份文档的目的，就是在这些条件下给出一版更可执行的优化计划。

## 2. 对你几个问题的直接回答

### 2.1 为什么保留 `BlockWisePress` 命名是合理的

我同意不改名。

原因有三点：

1. 你当前方法的核心身份仍然是“块粒度压缩”，不是在做一个新的独立算法家族。
2. 后续你还要和 `DualPhasePerLayerPress` 配套使用，名字上保持连续性更清楚。
3. 论文里也更容易表达为：
   - `BlockWisePress` 是基础块压缩器
   - `DualPhasePerLayerPress` 是其分层分阶段调度器

所以后续应该做的是增强 `BlockWisePress` 的内部机制，而不是重命名。

### 2.2 `top-k representative keys` 中的 k 应该怎么选

这个问题很关键。

我不建议：

- `k=1`，太脆弱；
- `k` 与 `block_size` 线性增长，比如 `block_size=32` 就直接 `k=8`，会让摘要变重得太快。

更合理的是：

- `k` 随 `block_size` 增长，但增长速度慢于线性；
- 也就是用一个次线性的、非线性的映射。

我建议用下面这个默认规则：

`k = clamp(round(sqrt(block_size) / 2), min=1, max=4)`

这个规则下：

- `block_size=16` 时，`sqrt(16)/2 = 2`，所以 `k=2`
- `block_size=32` 时，`sqrt(32)/2 ≈ 2.83`，所以 `k=3`
- `block_size=64` 时，`sqrt(64)/2 = 4`，所以 `k=4`

我更推荐这个映射，而不是简单的 `block_size/8` 之类线性规则，原因是：

1. 代表 token 的数量不需要和块大小同比例增长；
2. 一个块里真正关键的 token 数量通常远少于块大小；
3. 这个规则可以在 `block_size=16/32` 这两个你最关心的点上给出自然区别：
   - `16 -> 2`
   - `32 -> 3`
4. 对将来的 GPU 常驻块摘要缓存也更友好。

如果你担心 `k=4` 对某些大块还是偏大，也可以把上限先设成 `3`，但我个人建议：

- 第一版先用 `max=4`
- 因为我们真正要解决的是高压缩率下块内关键 token 被淹没的问题

### 2.3 “核心改动二”的具体含义

你同意的“核心改动二”，我这里再用更实现导向的话说一遍：

当前块分数不要只由：

- 块均值摘要和 query 的交互
- 单个 peak key 和 query 的交互

来决定，而是改成三部分混合：

1. `mean_score`
   - 块整体语义中心与 query 的相似度
2. `peak_score`
   - 最强代表 token 与 query 的相似度
3. `topk_peak_mean_score`
   - 少量代表 token 与 query 的相似度均值

推荐默认形式：

`block_score = 0.4 * mean_score + 0.3 * peak_score + 0.3 * topk_peak_mean_score`

这个设计的意义是：

- `mean_score` 保整体语义
- `peak_score` 保最尖锐热点
- `topk_peak_mean_score` 保“块里有多个关键 token”的情况

这对 `ruler` 的 multikey / multivalue 和 `needle` 都更友好。

### 2.4 `recent blocks` 中 N 应该固定还是随长度变化

这是个需要平衡的问题。

#### 方案 A：固定值

例如：

- 永远保留最近 `2` 个块
- 或最近 `4` 个块

优点：

- 简单、稳定、容易解释
- 不会因为超长序列导致 protected 区域无限变大
- 对未来 offload 系统也更像一个固定 GPU 热区

缺点：

- 对非常长的上下文，固定 `2` 或 `4` 个块可能不够

#### 方案 B：长度相关的非线性值

例如：

`N_recent = clamp(round(log2(num_blocks)), min=2, max=8)`

举例：

- `num_blocks=16`，`log2=4`，保留 `4`
- `num_blocks=64`，`log2=6`，保留 `6`
- `num_blocks=256`，`log2=8`，保留 `8`

优点：

- 随序列变长，recent 区域也会适度增长
- 但增长速度很慢，不会失控

缺点：

- 比固定值多一层动态性
- 在实现和调参上略复杂一点

#### 我的建议

我建议第一版做成：

- 支持两种模式
- 默认用长度相关的非线性模式

即：

- 若用户显式设置 `protected_recent_blocks`，就用固定值
- 否则默认：
  `N_recent = clamp(round(log2(num_blocks)), min=2, max=8)`

原因是：

1. 这和你“希望是固定值或长度相关非线性变量”的想法一致
2. 非线性 `log` 增长比线性更合理
3. 它能适配请求长度分布差异很大的情况
4. 不会把 protected 区域撑得太大

### 2.5 `recent blocks` 如何与压缩率共同工作

这部分需要非常明确。

我建议的规则是：

1. 先根据压缩率算出本层总共能保留多少块 `n_keep`
2. 再划出 `recent blocks` 作为优先保留集合
3. 如果 `recent_count < n_keep`：
   - recent 块全部保留
   - 剩余预算再由非 recent 块去竞争
4. 如果 `recent_count >= n_keep`：
   - 说明压缩率太高，预算已经不足以完整保 recent
   - 此时 recent 块内部也必须再筛选
   - 同时输出一条 warning / info 日志

这恰好符合你说的语义：

- 正常情况下 recent 块不受压缩率影响
- 但在极端压缩率下，预算约束优先

建议日志类似：

`Requested compression is too aggressive: protected recent blocks exceed keep budget, falling back to scoring within recent blocks.`

### 2.6 `top hot blocks` 到底是什么

你问得很对，这个概念如果不讲清楚，很容易变得空泛。

我这里把它重新定义得更具体：

`top hot blocks` 指的是：

- 在去掉 recent blocks 之后，
- 剩余块里按当前块热度分数排序，
- 选出分数最高的一小批块，
- 将其视作“强制优先保留块”。

也就是说，它不是另一个复杂评分器，而是：

- 还是用当前已经有的块分数
- 只是把排序最靠前的那一小批块额外标成 must-keep

#### 一个清晰的筛选流程

可以这样做：

1. 计算所有块的 `block_score`
2. 先划出 recent blocks
3. 对非 recent 块按 `block_score` 排序
4. 取前 `N_hot` 个，作为 `top hot blocks`
5. 剩余块再按预算正常截断

这样它的作用是：

- recent blocks 负责保护最近局部上下文
- top hot blocks 负责保护全局最热块
- 普通预算筛选负责控制整体压缩率

#### 为什么需要它

因为只保留 recent blocks 还不够。

例如：

- 问题相关的一些关键证据块可能在很早的位置
- 它们不是 recent
- 但热度非常高

如果不额外保护，它们在高压缩率时仍然容易被删掉。

#### 第一版是否必须做

我认为：

- 这个机制是有价值的
- 但优先级低于 `top-k representative keys` 和 `recent blocks`

所以更稳妥的实现顺序是：

1. 先做 `top-k representative keys`
2. 再做 `recent block protection`
3. 最后再加 `top hot block protection`

## 3. 结合 ChunkKV 和当前实验结果后的优化方向

### 3.1 ChunkKV 最值得借鉴的点

我现在更明确地认为，ChunkKV 对你最有价值的不是“固定 chunk 大小”本身，而是以下三点：

1. chunk / block 是语义连续性的载体
2. recent window 需要稳定保留
3. 高压缩率下不能只依赖纯比例截断

你补充说得对：

- ChunkKV 虽然讲的是语义感知 chunk 稀疏
- 但 chunk 仍然是通过参数敏感性分析后确定的固定块大小

这和你的 block-wise 设定并不冲突。

也就是说：

- “固定块大小”并不妨碍做语义感知块压缩
- 真正决定效果的是块内表示和块间筛选机制

### 3.2 当前 BlockWisePress 需要补的两类能力

从前面的实验结果来看，当前 `BlockWisePress` 主要缺两类能力：

#### 能力一：块内关键 token 不能被均值淹没

这就需要：

- `single peak` 扩展为 `top-k representative keys`
- 块分数做多项混合，而不是只看均值或单峰

#### 能力二：高压缩率下需要安全边界

这就需要：

- `recent block protection`
- 后续可选地加入 `top hot block protection`

## 4. 一版新的实现计划

下面是我建议的实际实现顺序。

### 第一阶段：增强块摘要

目标：

- 不改变 `BlockWisePress` 命名
- 不改变整体“块摘要直接评分”的路线
- 只增强块内关键信息表达

具体改动：

1. 新增 `summary_topk_keys`
   - 支持用户显式指定
   - 若未指定，则由 `block_size` 自动推导
2. 默认自动推导规则：
   - `k = clamp(round(sqrt(block_size) / 2), min=1, max=4)`
3. 当前块摘要从：
   - `mean_keys`
   - `peak_keys`
   扩展到：
   - `mean_keys`
   - `peak_keys`
   - `topk_peak_keys`

### 第二阶段：增强块打分

目标：

- 让块内多个关键 token 都能影响块热度

具体改动：

1. 加入三个可调分量：
   - `mean_score`
   - `peak_score`
   - `topk_peak_mean_score`
2. 默认混合权重：
   - `mean=0.4`
   - `peak=0.3`
   - `topk_peak_mean=0.3`

### 第三阶段：加入 recent blocks 保底

目标：

- 减少高压缩率下误删最近局部上下文

具体改动：

1. 支持固定 recent：
   - `protected_recent_blocks = int`
2. 若未显式给定，则默认自动推导：
   - `N_recent = clamp(round(log2(num_blocks)), min=2, max=8)`
3. 正常情况下：
   - recent blocks 在压缩预算之外保留
4. 极端压缩时：
   - 若 `recent_count > n_keep`
   - 则 recent 内部也要重新筛
   - 并打印日志

### 第四阶段：可选加入 top hot blocks

目标：

- 保护“远处但非常重要”的少量块

建议先设计接口，但不一定立刻作为默认开启：

1. 新增：
   - `protected_hot_blocks`
2. 筛选方式：
   - 先去掉 recent
   - 再从剩余块中选 top-`N_hot`

我建议：

- 第一轮代码里可以先把接口留好
- 但默认关闭
- 等我们看完 recent 机制的收益后再决定是否默认打开

### 第五阶段：现阶段不做层间复用

你已经明确了：

- 暂时每层都单独重新算

我同意这一步先不做。

原因是：

1. 当前最核心的瓶颈还在块内摘要和高压缩率鲁棒性
2. 层间复用是效率优化，不是当前准确率短板的第一来源
3. 先把精度问题解决，再引入 reuse 更稳

## 5. 最终建议

如果只选最值得先做的两项，我建议就是：

1. `top-k representative keys`
2. `recent block protection`

这是最有希望直接提升：

- `ruler` 高压缩率
- `needle_in_haystack`
- `longbench` 在 `0.7` 时的断崖退化

而且它们都不需要把方法变得太复杂，也不会破坏你后面和 KV block offload 结合时最关键的低开销特性。

## 6. 一句话总结

下一版 `BlockWisePress` 不需要改名，也不需要改成 ChunkKV 式 token-level 压缩；更合理的路线是：

- 保持块摘要直接评分
- 用与 `block_size` 非线性相关的 `top-k representative keys` 增强块内表示
- 用 `recent block protection` 给高压缩率提供稳定兜底
- `top hot blocks` 作为第二阶段的可选增强

这条路线最符合你当前的系统目标和实验反馈。
