# BlockWisePress 研究总结：结合 ChunkKV 与 Quest 的下一步方向

## 1. 先把问题说清楚

当前 `BlockWisePress` 的目标不是把 KV 压缩做成 token 级别的重型选择器，而是保持它作为一个**块粒度、低开销、batch 推理友好、未来可接 GPU-CPU 两级卸载**的压缩单元。

这意味着它的核心约束不是“极致精细”，而是：

- 块摘要必须足够轻
- query-aware 必须尽量保留
- 选择逻辑必须可复用、可缓存、可作为 offload 热度信号
- 不要把系统做成 token-level 复杂控制器

结合当前仓库里的实现，`BlockWisePress` 已经收敛到了一个比较清晰的主线：

- 块摘要
  - `mean_keys`
  - `topk_key_means`
- query-aware 打分
  - 用最后一段 `q_window`
  - 对块摘要做相似度计算
- 简单保底
  - recent blocks
  - 尾部不完整块

当前更值得做的不是继续把方法加复杂，而是把这条主线往“更有语义保真、更稳定、更能服务 batch / offload”的方向推。

## 2. ChunkKV 最值得借鉴什么

参考论文：

- ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference
- OpenReview/论文入口：<https://openreview.net/forum?id=i0cjbEuezL>

本地实现：

- [chunkkv_press.py](/home10T/bzx/workspace/kvpress-study/kvpress/presses/chunkkv_press.py)

### 2.1 核心思想不是“chunk 这个名字”，而是“语义连续性”

ChunkKV 最有价值的地方，不是简单地把 token 改成 chunk，而是它明确指出：

- 连续 token 往往共同表达一个语义片段
- 只按单 token 独立裁剪，容易把语义结构切碎
- 因此压缩单元应该尽量跟语义连续边界对齐

对我们来说，这个启发非常直接：

- `BlockWisePress` 的块不是纯管理单元
- 块本身也应该带语义保真约束

### 2.2 recent-window 保留很关键，而且便宜

ChunkKV 里的 observe window 思想非常实用：

- 最近上下文通常更有用
- 强制保留 recent window 成本低
- 能显著减少“压缩太狠把局部上下文删掉”的问题

这点比复杂的 token 级补丁更值得借鉴。

### 2.3 layer-wise index reuse 很适合你的系统目标

ChunkKV 论文里还有一个很重要的效率观察：

- 相邻层保留的索引往往相似
- 不必每层都完全重新计算
- 可以复用一部分索引 / 选择结果

这对你的后续方向尤其重要，因为你最终要走向：

- 分层压缩率
- batch 推理
- 未来的块热度作为 offload/prefetch 信号

所以，ChunkKV 真正值得借鉴的，是：

- 语义连续性
- recent-window 兜底
- 层间复用/稳定性

### 2.4 ChunkKV 对我们当前实现的启发

当前仓库里的 `ChunkKVPress` 其实更像一个简化 wrapper：

- 先有 token scorer
- 再按 chunk 聚合
- 再挑选 chunk

它和论文原意有差异，但仍然提供了一个重要信号：

- 相比 token 级孤立打分，chunk / block 级保留更容易维持语义结构

这说明 `BlockWisePress` 不能只盯着“块均值是否高”，还要照顾：

- 块内少数关键 token
- 块的完整语义片段

## 3. Quest 最值得借鉴什么

参考论文：

- QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- OpenReview PDF：<https://openreview.net/pdf/33999165c53a8778cce8830b43d58381eea3b389.pdf>

### 3.1 核心不是“稀疏”，而是“query-aware criticality”

Quest 的关键思想非常明确：

- KV 的重要性不是静态的
- 它强烈依赖当前 query
- 因此 criticality 必须随着 query 动态估计

这比“历史热度”或“全局固定重要性”更接近真实推理过程。

### 3.2 两阶段估计是 Quest 最值得借鉴的设计

Quest 的方法很适合我们这类系统，因为它不是简单地全量算注意力，而是：

1. 先用很便宜的方式估计 page/block 的 criticality
2. 再只对少量候选做真正的 attention / sparse attention

论文里用的是 page granularity，并且在 stage 1 里借助 query 和每页的 min/max key 做快速评估，再进入 stage 2 的稀疏注意力。

这件事对我们非常有启发：

- 低开销块摘要本质上就应该是一个 coarse criticality estimator
- 真正的“精确选择”应该只留给少量候选块

### 3.3 Quest 的 page budget / sparse load 视角很适合 batch 推理

Quest 还有一个很实际的点：

- 它把 KV 选择和“内存加载预算”联系起来
- 不是只看准确率
- 而是看能加载多少、能省多少 memory movement

这和你后续要做的 batch 推理、GPU-CPU 两级卸载直接对齐。

换句话说，Quest 提供的是一种很强的系统叙事：

- query-aware 不是为了论文里的漂亮名字
- 而是为了把“热块”变成更低的 memory traffic

### 3.4 Quest 对我们当前实现的启发

最值得借的是这三点：

- query-aware criticality 必须是当前 query 驱动的
- 评分过程应当是两阶段的 coarse-to-fine
- 选择结果要服务于 memory movement / block loading，而不是只服务于 accuracy

## 4. 我认为最值得尝试的 3 个优化点

下面这 3 个点是我结合 ChunkKV + Quest + 当前 `BlockWisePress` 约束后，认为最值得尝试的方向。

### 4.1 方向一：Query-aware 双锚点块摘要

#### 问题

当前 `BlockWisePress` 的块摘要虽然轻，但主要还是：

- mean
- top-k mean

这类表示容易把块内少数非常关键的 token 稀释掉，尤其在高压缩的检索型任务里。

#### 动机

Quest 说明了重要性高度依赖当前 query；ChunkKV 则说明语义连续性不能被简单打散。  
因此我们需要一个既轻、又 query-aware、又能保住块内尖峰信息的摘要。

#### 方法

每个块只保留极少量 summary anchor，例如：

- 块均值
- 一个“正锚点”代表块内最强响应 token
- 一个“反锚点”或第二代表 token

然后用当前 query 对这些少量锚点做 coarse scoring。  
如果 coarse score 足够高，再对少量候选块做更精确的二次筛选。

这本质上是：

- ChunkKV 的语义连续性意识
- Quest 的 query-aware coarse-to-fine
- 但仍然保持 block 粒度和低元数据

#### 预期收益

- 提升多 key / 少数关键 token 任务的召回
- 比纯 mean 更能捕捉块内峰值
- 比 token-level correction 更低开销

#### 风险

- 锚点选得不好会和 mean 信息重复
- 如果锚点过多，块摘要会膨胀
- 如果 coarse-to-fine 设计太复杂，会丢失系统简洁性

#### 最小验证方案

- 先只在 `RULER` 的 `niah_multikey_2/3` 上验证
- 对比：
  - 当前 `BlockWisePress`
  - 双锚点版本
- 只看 `compression_ratio=0.7`

---

### 4.2 方向二：Recent + Hot 双预算保留

#### 问题

当前纯按比例删块的方式，在高压缩率下很容易：

- 误删最近上下文
- 误删远距离但很重要的支持块
- 导致块选择太“硬”，缺少安全边界

#### 动机

ChunkKV 的 recent-window 说明最近上下文要兜底；  
Quest 的 top-K page selection 说明只要 query-aware 足够强，就应该把预算留给最热的候选块。

#### 方法

把最终保留预算拆成两部分：

- recent budget：固定保留最近若干块
- hot budget：从全局候选中保留少量高热块

最终剩余的块再按普通分数补齐。

这个策略的重点不是复杂，而是把“稳定性”和“检索性”分开。

#### 预期收益

- 降低高压缩时的灾难性误删
- 对 batch 推理更友好，因为 recent 部分可以近似看作稳定缓存
- 对 offload 更友好，因为 hot blocks 可以直接作为 prefetch 热度信号

#### 风险

- budget 切分不好会浪费保留名额
- hot block 机制如果过重，可能重新变复杂

#### 最小验证方案

- 只测 `RULER 0.7`
- 重点看：
  - `niah_single_3`
  - `qa_1`
  - `niah_multikey_3`
- 先验证“recent 保底”是否足够，再决定要不要加 hot budget

---

### 4.3 方向三：层间/步间 score reuse，而不是残差

#### 问题

每层、每步都重新做完整精细评分，成本高，而且在 batch 推理里容易引入抖动。

#### 动机

ChunkKV 论文的 layer-wise index reuse 给了一个很好的启发；  
Quest 的两阶段 criticality 也说明，很多选择不需要每次都全量重算。

#### 方法

不是做“跨层残差”这种强耦合，而是做更轻的：

- score cache
- lazy refresh
- layer-wise / step-wise reuse

例如：

- 每隔若干层刷新一次完整评分
- 其余层复用最近一次的 top block set
- 或只对候选集做局部 refresh

这更像“稳定控制”和“计算节流”，而不是把错误强行传递到下一层。

#### 预期收益

- 明显减少评分开销
- 对 batch 推理更稳定
- 更适合把块分数变成卸载热度指标

#### 风险

- 如果 refresh 间隔过长，会造成 stale ranking
- 如果 reuse 规则不清晰，会影响评测公平性

#### 最小验证方案

- 在 `LongBench` 或 `InfiniteBench` 上做小样本对比
- 测：
  - 全量每层重算
  - 间隔刷新
  - 复用 top blocks

## 5. 哪些看起来合理，但现在不值得做

我建议当前阶段不要优先做下面这些：

### 5.1 token-level correction

已经验证过，收益很有限，且容易和摘要主路径重复。

### 5.2 复杂 head clustering / head redundancy penalty

看起来合理，但会把方法迅速带复杂，而且当前主要瓶颈不在 head 冗余。

### 5.3 跨层残差作为主线

残差确实有一点稳定作用，但它更像辅助项，不像主解法。  
如果把它变成主线，会让方法状态耦合变重。

### 5.4 动态可变 block size

这会损害你后续做卸载和内存管理的简洁性。  
当前阶段固定 block size 更适合系统化叙事。

### 5.5 过早引入学习型 gating / policy network

这会让整个方法从“轻量启发式”变成“训练型控制器”，不适合当前定位。

## 6. 更适合 batch 推理的测试数据集建议

你说得对，后面不必死磕 `RULER` 和 `Needle`。  
如果最终目标是 batch 推理和 KV cache 卸载，我建议把数据集分成三层：

### 6.1 主实验层：更贴近 batch inference 的长文本任务

优先推荐：

- `LongBench`
- `LongBench-v2`
- `InfiniteBench`
- `LooGLE`

其中最值得优先测的子任务是：

- `LongBench`
  - `qasper`
  - `hotpotqa`
  - `multifieldqa_en`
  - `triviaqa`
  - `narrativeqa`
  - `govreport`
- `LongBench-v2`
  - 作为更新的长文本补充
- `InfiniteBench`
  - `passkey`
  - `kv_retrieval`
  - `number_string`
  - `longbook_qa_eng`
  - `longdialogue_qa_eng`
  - `code_run`
  - `code_debug`
- `LooGLE`
  - `shortdep_qa`
  - `longdep_qa`
  - `shortdep_cloze`
  - `longdep_summarization`

这些任务的共同点是：

- 请求长度分布更复杂
- 有长短依赖混合
- 对 batch 推理更有代表性
- 更能体现“压缩 + 内存 + 选择”的综合效果

### 6.2 Stress test 层：保留少量极限任务

建议继续保留：

- `RULER`
- `Needle in a Haystack`

但不要把它们当唯一主战场。  
它们更适合作为：

- 极限检索能力测试
- 失败案例分析
- 压缩策略鲁棒性验证

### 6.3 系统补充层：多任务混合、输出长度差异大的任务

如果你后面要讲 batch 推理故事，以下任务也值得考虑：

- `Zero Scrolls`
- `AIME25` 作为非长上下文控制任务

其中：

- `Zero Scrolls` 适合补充 summarization / QA / reordering 这类复杂输出
- `AIME25` 不一定是 KV 压缩主力，但适合做“非长上下文条件下的性能保持”对照

### 6.4 我建议的评测组合

如果要形成一个适合论文的、比较均衡的组合，我建议：

1. 主线：
   - `LongBench`
   - `InfiniteBench`
   - `LooGLE`
2. 压力测试：
   - `RULER`
   - `Needle`
3. 系统补充：
   - `Zero Scrolls`
   - `AIME25`

这样你就不会被单一的检索 benchmark 绑死，也更符合 batch 推理与卸载系统的叙事。

## 7. 最终建议

如果把问题压缩成一句话，我的判断是：

- `ChunkKV` 值得借鉴的是语义连续性、recent 保底和层间复用
- `Quest` 值得借鉴的是 query-aware 两阶段 criticality estimation
- 对 `BlockWisePress` 最值得做的三件事，是：
  - Query-aware 双锚点块摘要
  - Recent + hot 双预算保留
  - score reuse / lazy refresh

而对于你的最终目标：

- 不必死磕 `RULER` 和 `Needle`
- 也不必追求所有任务都超过 `ChunkKV`
- 更重要的是选择一组真正能体现 batch 推理和卸载价值的 benchmark

如果你愿意，下一步我可以继续把这份总结再压成一版“可直接写进论文 introduction / motivation”的故事线。
