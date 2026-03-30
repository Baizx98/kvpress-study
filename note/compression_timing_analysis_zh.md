# KV Cache 压缩时机分析：先计算后压缩 vs 先压缩后计算

## 1. 你当前理解的问题本质上是什么

你现在关心的核心问题不是“块怎么打分”，而是：

**压缩动作到底发生在当前层 / 当前步 attention 计算之前，还是之后？**

这会直接决定两件事：

1. 压缩影响的是“当前层 / 当前步的计算”，还是“下一层 / 下一步的计算”
2. 压缩带来的收益主要体现为：
   - 当前计算量减少
   - 当前显存峰值降低
   - 还是只是为下一步 / 下一层节省 KV cache

这两类机制的系统含义差异很大。

---

## 2. 你现在这版 BlockWisePress / DualPhasePerLayerPress 属于哪一类

在当前 `kvpress` 框架下，`BasePress` 的入口是 `forward_hook`，
也就是 **attention forward 执行完成之后** 才会被调用。

对应文件：

- `kvpress/presses/base_press.py`
- `kvpress/presses/block_wise_press.py`
- `kvpress/presses/dual_phase_per_layer_press.py`

所以你现在这版方法，本质上属于：

### 2.1 Prefill 阶段

- 当前层 attention 先算完
- 当前层的新 KV cache 已经生成
- 再根据 prompt 末尾窗口的 query 和当前层现有块做重要性评估
- 删除一部分块，保留一部分块
- 压缩后的 KV cache 供 **后续层** 和 **decode 阶段** 使用

也就是说，当前实现的 prefill 更接近：

**先计算当前层 attention，再为下一层 / 下一阶段准备更小的 KV cache**

### 2.2 Decode 阶段

- 当前步 attention 先算完
- 当前 token 的新 KV 被追加到 cache 中
- 再根据当前缓冲的尾部 query 和现有块做重要性评估
- 决定哪些块永久删除，哪些块下一步不参与计算

也就是说，当前实现的 decode 更接近：

**用当前步的信息指导下一步的块选择**

因此你刚才的判断是对的：

> 当前方案更偏“先计算后压缩”，而不是“先压缩后计算”。

---

## 3. 不同方法的压缩时机可以怎么分类

可以把主流方法大致分成三类。

### 3.1 天然后压缩：必须先看到当前层 attention 才能打分

典型代表：

- `ObservedAttentionPress`，可视作 H2O 风格的“观测到的注意力”路线
- `TOVAPress`
- 很多直接依赖真实 attention map 的方法

这类方法的特点是：

- 它们依赖当前层已经算出来的 attention weights
- 因此天然适合 **attention 之后再压缩**
- 当前层算完之后，才能知道哪些历史 token / 块更重要

优点：

- 评分更贴近“真实被关注了多少”
- 重要性估计通常更直接

缺点：

- 当前层的 attention 计算量已经付出了
- 当前层无法享受到压缩带来的算力 / 显存收益
- 更像是“为后续步骤省钱”，不是“为当前步骤省钱”

### 3.2 天然前压缩可行：不依赖当前层真实 attention map

典型代表：

- `StreamingLLM`
- `KVzap`
- 以及你想做的“块摘要 + query-aware” block-wise 方法

这类方法的特点是：

- 不要求先拿到当前层真实 attention weights
- 只要在 attention 之前拿到足够的信息，就可以先做判断

例如：

#### StreamingLLM

它本质上是启发式窗口保留：

- 保留前若干 sink token
- 保留最近窗口
- 删中间

它几乎不需要额外评分，所以在时机上最自由：

- 放在 attention 前可以
- 放在 attention 后也可以

只是在当前 `kvpress` 里，为了统一框架，仍然走的是 post-hook 路径。

#### KVzap

KVzap 用的是轻量 surrogate model 直接从 hidden states 预测分数。

从信息依赖关系看，它不一定需要当前层真实 attention map，
因此是 **有希望做成 attention 前压缩** 的。

优点：

- 评分快
- 更容易放进在线 serving 路径

缺点：

- 依赖 surrogate 预测质量
- 如果放到 attention 前，当前层输入 hidden state 到底多可靠，取决于训练好的代理模型是否泛化得足够好

### 3.3 中间地带：通常后压缩，但也可以改成近似前压缩

典型代表：

- `SnapKV`
- 你想做的 block-wise query-aware 方法

这类方法的共同点是：

- 它们依赖 query-aware 信号
- 但不一定非要等当前层真实 attention 完整算出来

例如 `SnapKV`：

- 标准实现里通常使用最后窗口 token 的注意力模式来打分
- 如果拿到了真实 attention，当然最自然
- 但如果在 attention 前就拿到当前层 Q/K，也可以自己算一个局部近似 attention，用它近似 SnapKV 风格的打分

所以这类方法不是“绝对只能后压缩”，而是：

**后压缩最自然，前压缩需要额外工程改造和精度-开销折中**

---

## 4. 你提出的“先压缩后计算” block-wise 方案是什么

你现在想要的是：

### Prefill 阶段

在当前层 attention 真正计算之前：

1. 先拿到当前层的 Q/K/V 投影
2. 先维护块级聚合结果
3. 用 prompt 末尾窗口的 Q 和块摘要交互，算块重要性
4. 决定：
   - 哪些块永久删除
   - 哪些块本次不参与 attention
   - 哪些块本次参与 attention
5. 只让保留下来的 active blocks 进入当前层真正的 attention 计算

这就是典型的：

**先压缩后计算**

### Decode 阶段

在当前步 attention 之前：

1. 当前 token 的 Q/K/V 已经可以投影出来
2. 当前缓存块摘要已经在 GPU 上
3. 用当前 token 或最近几个 token 的 Q 去和块摘要交互
4. 先决定当前步 active blocks
5. 再让这些 active blocks 参与当前步 attention

这也是：

**先决定当前步参与集合，再执行当前步 attention**

---

## 5. 这种“先压缩后计算”方案的可行性

### 5.1 从信息依赖上看，可行

你的 block-wise 方案如果改成：

- 不依赖真实 attention map
- 只依赖当前层预投影出来的 Q/K/V
- 并且块重要性依赖的是块摘要而不是 token 级全量扫描

那它在理论上是完全可以放到 attention 前面的。

因为在 attention 前，你已经能拿到：

- 当前层输入 hidden states
- 当前层线性投影后的 Q/K/V
- 已有历史 KV cache
- 已有历史块摘要

这些信息足以支持一次块级筛选。

### 5.2 从系统工程上看，也可行，但要改框架

当前 `kvpress` 框架是 post-hook 驱动的。

如果你要真正实现“先压缩后计算”，你需要至少改一项：

1. 使用 `forward_pre_hook`
2. 或者直接 patch attention module，把块筛选逻辑塞进 attention forward 内部
3. 或者自定义 attention wrapper，在 Q/K/V 投影之后、attention 核函数之前插入块筛选

否则仅靠现在的 `forward_hook` 是做不到“当前层先压缩再计算”的。

### 5.3 最关键的工程挑战：不能重复做一遍昂贵投影

如果你只是用 pre-hook，但为了打分又单独重新算一遍 Q/K/V，
那你会遇到一个大问题：

- attention 还没开始算
- 你已经额外做了一遍投影和块筛选
- 然后 attention forward 里面又会再投影一遍

这样容易出现：

- 额外重复计算
- latency 上升
- 破坏 fused attention 路径

所以真正合理的实现方式，通常是：

**把块筛选嵌入 attention forward 内部，在同一份 Q/K/V 上完成筛选与注意力计算。**

---

## 6. 先压缩后计算 vs 先计算后压缩：利害分析

### 6.1 先计算后压缩

也就是你当前这版更接近的方式。

优点：

- 框架实现简单
- 不改原 attention 主路径
- 容易和现有 KV cache hook 兼容
- 不容易破坏模型数值稳定性

缺点：

- 当前层 / 当前步的 attention 开销已经付了
- 压缩收益主要延迟到下一层 / 下一步
- 对缓解“当前步峰值算力 / 当前步峰值显存”帮助有限

### 6.2 先压缩后计算

也就是你想要推进的方式。

优点：

- 当前层 / 当前步就能减少 attention 参与的 KV 数量
- 更直接降低当前计算开销
- 更接近真正的在线内存管理 / 在线卸载系统
- 更容易把“active / resident / offloaded / prefetch”这些状态直接接入当前步调度

缺点：

- 工程复杂度明显更高
- 很可能要改 attention forward 主路径
- 如果筛错，当前层输出立刻受损，比后压缩更敏感
- 对近似评分精度要求更高
- 更需要严格控制筛选开销，否则容易“压缩节省 < 筛选成本”

---

## 7. 对不同方法而言，哪种时机更自然

### 7.1 StreamingLLM

最适合做前压缩。

因为它本质上是静态规则：

- 不需要真实 attention
- 不需要复杂打分
- 最容易嵌进当前步计算前

### 7.2 KVzap

比较适合做前压缩。

因为它依赖轻量 surrogate score，
核心问题不是时机，而是 surrogate 精度和部署开销。

### 7.3 H2O / ObservedAttention / TOVA

更自然地属于后压缩。

因为它们依赖当前层已经算出来的 attention。
如果强行前压缩，就必须用“近似 attention”替代“真实 attention”，
这时方法本身就变味了。

### 7.4 SnapKV

两边都可以，但标准路线更偏后压缩或“准后压缩”。

如果你愿意自己在 attention 前用当前层 Q/K 近似计算最后窗口 attention，
那它可以改成前压缩。
但工程上会更复杂。

### 7.5 你的 block-wise summary 方法

这是最值得推进前压缩的一类。

因为它具备几个非常好的条件：

1. 评分对象已经从 token 变成块摘要
2. 块摘要体积小，适合常驻 GPU
3. query-aware，但不要求真实完整 attention map
4. 与未来 offload / prefetch 的块级调度目标天然一致

也就是说：

**在所有这些方法里，你的方法最适合被做成“真正的先压缩后计算”块调度器。**

---

## 8. Prefill 和 Decode 上的区别

### 8.1 Prefill

在 prefill 阶段，先压缩后计算的主要收益是：

- 当前层 attention 直接减负
- 当前层输出就建立在压缩后的 active blocks 上

但风险也最大：

- 如果当前层误删关键块，这个损失会立即传给后续层

因此 prefill 更适合：

- 用较保守的压缩率
- 更偏 recall 的块评分
- 保留尾部窗口和若干安全块

### 8.2 Decode

在 decode 阶段，先压缩后计算的收益更直接：

- 每步只对 active blocks 做注意力
- 更贴近在线调度目标

但 decode 也有自己的难点：

- 每步 query 太短，块热度估计更容易抖动
- 如果每步都全量重算块分数，开销可能反而不划算

因此 decode 更适合：

- 块摘要 + 历史热度 EMA
- 间隔刷新
- 冷块 / resident / prefetch 的分层状态机制

这也正是你现在在 `DualPhasePerLayerPress` 里想做的近似更新路线。

---

## 9. 对你当前研究方向的建议

如果你的最终目标是：

1. 缓解 batch 场景下 KV cache 动态增长导致的抢占
2. 将稀疏压缩和 GPU-CPU 两级卸载统一起来

那么从论文和系统设计上，最值得强调的路线是：

### 9.1 在论文中明确区分两种压缩时机

- 后压缩：用当前信息优化下一步
- 前压缩：用当前信息直接缩减当前步参与集合

你的方法应该主打第二类。

### 9.2 将 block summary 作为核心系统抽象

块摘要应该成为：

- 低开销块评分的输入
- GPU 常驻元数据
- offload / prefetch 的热度依据
- decode 近似更新的缓存对象

### 9.3 将当前工作分成两个阶段描述

#### 当前已完成 / 易完成

- 后压缩版本
- 块状态分类
- 历史热度与近似刷新

#### 下一阶段关键实现

- attention 前块筛选
- active block set 驱动当前步 attention
- 与 GPU-CPU offload manager 联合调度

这样叙事会更清楚，也更符合系统论文的推进路径。

---

## 10. 结论

一句话总结：

- `H2O / ObservedAttention / TOVA` 这类方法天然更偏后压缩
- `StreamingLLM / KVzap` 更适合前压缩
- `SnapKV` 处于中间地带
- **你的 block-wise summary 方法，是最有潜力做成“先压缩后计算”的一类**

而且它的系统价值不只是压缩本身，而是：

**它可以把“当前步参与计算的块”和“未来卸载 / 预取的块”统一到同一个块级热度框架下。**

这正是你后面要讲的系统故事的核心。
