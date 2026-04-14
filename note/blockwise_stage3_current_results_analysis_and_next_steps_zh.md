# Blockwise Stage3 当前结果分析与下一步方向

## 1. 文档目的

本文档基于当前已经完成的 `stage3` 第一批实验结果，回答三个问题：

1. 现有 `stage3` 结果说明了什么
2. 为什么不建议继续把主要精力投入到 blockwise 内部细碎消融
3. 下一步更值得做什么

本分析使用的有效结果范围为：

- `LongBench`
  - `qasper`
  - `multifieldqa_en`
  - `hotpotqa`
  - `2wikimqa`
  - `musique`
  - `triviaqa`
- `needle_in_haystack / 16384`

说明：

- `PG19` 因官方源下载不稳定，本轮不纳入结论
- `stage3` 当前只分析已完成的有效结果，不把历史重试中的假失败记录当作真实结论

---

## 2. 当前实验设置

本轮比较的方法有：

- `blockwise_main`
  - `mean_plus_norm_topk_mean + key_norm + max + uniform_mean`
- `blockwise_multi_rep`
  - `multi_rep_max + key_norm + max + uniform_mean`
- `blockwise_adaptive_fusion_v1`
  - 基于 `mean / norm-topk / multi-rep` 的规则融合
- `blockwise_multi_rep_diverse_v1`
  - 在 `multi_rep_max` 上加入 representative 去冗余
- `chunkkv_prefill_per_layer`

统一设置：

- `compression_ratio=0.7`
- `fraction=0.2`

---

## 3. 当前结果

## 3.1 各数据集最优方法

- `LongBench / qasper`
  - 最优：`blockwise_multi_rep = 40.61`
- `LongBench / multifieldqa_en`
  - 最优：`blockwise_adaptive_fusion_v1 = 57.65`
- `LongBench / hotpotqa`
  - 最优：`blockwise_main = 56.27`
- `LongBench / 2wikimqa`
  - 最优：`chunkkv_prefill = 45.13`
- `LongBench / musique`
  - 最优：`chunkkv_prefill = 35.63`
- `LongBench / triviaqa`
  - 最优：`blockwise_main = 96.00`
  - `blockwise_multi_rep` 与 `adaptive_fusion_v1` 同分
- `needle_in_haystack / 16384`
  - 最优：`blockwise_multi_rep = 73.50`

## 3.2 关键对比

### 跨数据集的整体观感

如果只看当前有效的 `7` 个任务：

- `blockwise_multi_rep` 拿到 `3` 个数据集最优
- `blockwise_main` 拿到 `2` 个数据集最优
- `blockwise_adaptive_fusion_v1` 拿到 `2` 个数据集最优
- `chunkkv_prefill` 拿到 `2` 个数据集最优

但这个“最优次数”并不能直接说明谁更值得继续深挖，因为不同任务的优势来源并不一致。更有意义的是看它们相对 `chunkkv` 的整体差值：

- `blockwise_main`
  - 相对 `chunkkv` 的平均差值约为 `-1.04`
- `blockwise_multi_rep`
  - 相对 `chunkkv` 的平均差值约为 `+0.42`
- `blockwise_adaptive_fusion_v1`
  - 相对 `chunkkv` 的平均差值约为 `-0.77`
- `blockwise_multi_rep_diverse_v1`
  - 相对 `chunkkv` 的平均差值约为 `-1.62`

这组数字说明：

- `blockwise_multi_rep` 是当前最有竞争力的 retrieval-oriented 分支
- `blockwise_main` 仍然是最稳的通用配置，但还不能稳定压过 `chunkkv`
- `adaptive_fusion_v1` 虽然能在局部任务上夺冠，但并没有形成更强的总体竞争力
- `multi_rep_diverse_v1` 没有形成值得继续主攻的信号

### `adaptive_fusion_v1`

结果：

- `qasper`: `39.48`
- `multifieldqa_en`: `57.65`
- `hotpotqa`: `54.37`
- `2wikimqa`: `38.76`
- `musique`: `31.90`
- `triviaqa`: `96.00`
- `needle`: `69.85`

结论：

- 它只在 `multifieldqa_en` 上真正形成优势
- 在 `triviaqa` 上只做到与已有强配置持平
- 在 `2wikimqa` 上明显退化
- 在 `needle` 上也未超过 `multi_rep`

这说明：

- 当前这版规则融合并没有形成“统一 summary”能力
- 它更像一个局部有效的启发式，而不是新的主线

### `multi_rep_diverse_v1`

结果：

- `qasper`: `39.80`
- `multifieldqa_en`: `53.35`
- `hotpotqa`: `54.17`
- `2wikimqa`: `39.81`
- `musique`: `29.36`
- `triviaqa`: `93.00`
- `needle`: `72.58`

结论：

- 它在大多数任务上都不如原始 `multi_rep`
- 只在 `needle` 上略接近 `multi_rep`
- 整体上说明“简单代表去冗余”没有解决主要矛盾

### `chunkkv`

结果：

- `2wikimqa`: `45.13`
- `musique`: `35.63`
- `hotpotqa`: `54.61`
- `multifieldqa_en`: `53.85`
- `qasper`: `39.70`
- `triviaqa`: `93.50`
- `needle`: `70.97`

结论：

- `chunkkv` 仍然是非常强的基线
- 尤其在 `2wikimqa` 与 `musique` 上依然稳稳领先
- 这说明 blockwise 的问题并不是“有没有效果”，而是“为什么在某类任务上仍然打不过 chunkkv”

---

## 4. 这轮结果的核心含义

## 4.1 继续做 blockwise 内部消融的收益已经明显下降

这是这轮最重要的判断。

当前已有证据是：

- `blockwise_main` 在 `hotpotqa / triviaqa` 这类任务上已经很强
- `blockwise_multi_rep` 在 `qasper / needle` 这类任务上已经很强
- `adaptive_fusion_v1` 虽然在 `multifieldqa_en` 上有提升，但没有形成统一优势
- `multi_rep_diverse_v1` 基本没有带来普遍收益

也就是说，现阶段 blockwise family 的主要现象已经很清楚：

- 任务间偏好不同 summary form
- 简单增加组合或局部启发式，已经很难形成跨任务的一致提升

因此，如果继续主要投入在：

- 再试几个 summary mixing 权重
- 再换几种 representative 去冗余
- 再加一些 query aggregation 小技巧

那么很可能只会得到：

- 个别任务小涨
- 另一些任务小跌
- 总体没有形成新的结构性突破

这说明 blockwise 内部排列组合已经比较接近它当前范式下的上限。

## 4.2 当前最值得保留的 blockwise 结论已经足够稳定

到目前为止，可以认为已经比较稳定的结论有：

- `query_agg=max` 仍然应该保留为 blockwise 主线默认项
- `blockwise_main` 是最稳的通用主线
- `blockwise_multi_rep` 是最强的检索/多峰块候选
- `chunkkv` 是必须正视的强基线

所以从研究推进角度看，现在不缺“更多消融”，而是缺：

- 一个更高层次的新变量

---

## 5. 下一步最值得做的方向

## 5.1 方向一：按层设置不同压缩率或 budget

### 动机

当前所有比较基本都还停留在：

- 一个全局 `compression_ratio`
- 各层共享同一个 budget 分配原则

但从长上下文压缩的经验来看，不同层的重要性通常并不相同：

- 底层更偏局部 lexical / surface pattern
- 中层更偏实体、关系、片段组织
- 高层更偏任务相关聚合与最终决策

如果所有层都用同一个 ratio，很可能会：

- 在不重要层保留过多 budget
- 在关键层压得过狠

### 预期收益

- 比继续调 block summary 更可能带来真实增益
- 能直接回答“预算该分配给哪些层”这个更本质的问题
- 也更容易和论文中的系统设计主张结合

### 风险

- 搜索空间会扩大
- 如果直接全层自由搜索，实验成本会很高

### 最小验证方案

建议不要一上来做全层自由优化，而是先做 3 类简单分配：

1. `front-light / middle-heavy / back-heavy`
2. `front-heavy / middle-heavy / back-light`
3. `U-shape / inverted-U`

更具体一点，可以先做：

- 例如 32 层模型分成三段
  - `0-9`
  - `10-21`
  - `22-31`
- 给三段分别设不同压缩率

推荐先试：

- `0.8 / 0.6 / 0.8`
- `0.8 / 0.7 / 0.6`
- `0.7 / 0.6 / 0.8`

这里的重点不是立刻找到最优值，而是先验证：

- “分层 budget” 是否比继续做 summary 消融更有价值

## 5.2 方向二：decode 阶段的永久驱逐

### 动机

当前 prefill 压缩已经把主要现象摸清了，但 decode 阶段仍然有独立价值：

- 生成越长，decode KV 会持续增长
- 仅靠 prefill 压缩不能解决长生成中的持续显存压力
- 某些 token 在 decode 中应当被永久驱逐，而不是一轮轮重复参与候选

### 预期收益

- 显存收益会更直接
- 能把“prefill 压缩”和“decode 内存管理”真正连起来
- 也更适合作为系统论文里的第二条主线

### 风险

- 永久驱逐若做错，质量会出现不可逆损失
- 需要明确什么信息应该永久保留，什么可以一旦驱逐就不再回来

### 最小验证方案

建议先从最简单的 permanent eviction policy 开始：

1. `age-based`
   - 极老 token 在满足若干条件后永久移除
2. `low-score-and-stable`
   - 多轮都低分的 token 才永久移除
3. `block-level permanent eviction`
   - 不是单 token 驱逐，而是整块永久移除

其中第三种尤其值得做，因为它和你现在的 blockwise 架构更一致。

## 5.3 方向三：decode 阶段的计算稀疏

### 动机

永久驱逐解决的是“存不存”，但还没有解决“算不算”。

即使某些 token 还留在 cache 里，也未必要每一步都参与完整 attention 计算。

所以 decode 阶段的下一层问题是：

- 是否可以让一部分 cache 常驻但不参与每步全量计算

这会把问题从“memory pruning”推进到“memory + compute co-design”。

### 预期收益

- 比单纯做永久驱逐更有系统价值
- 更容易拿到 latency / throughput 改善
- 与论文中的“计算稀疏”叙事天然契合

### 风险

- 实现复杂度明显高于纯 eviction
- 如果稀疏策略不稳定，可能对生成质量造成不可预测影响

### 最小验证方案

建议先做最轻量版本：

1. `two-tier compute`
   - 热块每步参与计算
   - 冷块隔若干步才刷新
2. `periodic refresh`
   - decode 中不是每步全量关注冷块，而是按固定 interval 激活
3. `summary-assisted sparse decode`
   - 冷块平时只通过 summary 参与，必要时再回到原始 KV

这第三种方向尤其重要，因为它可能把你前面积累的 block summary 研究真正转化成 decode 价值，而不是停留在 prefill ranking。

---

## 6. 我对下一阶段的建议

如果按优先级排序，我建议下一阶段这样推进：

### 第一优先级

- 分层压缩率 / 分层 budget

原因：

- 这是新的高价值变量
- 与当前 blockwise / chunkkv 都兼容
- 成本明显低于立刻做复杂 decode 稀疏实现

### 第二优先级

- decode 阶段的 block-level permanent eviction

原因：

- 和当前 blockwise 架构天然兼容
- 可以自然接到“不同块长期价值不同”的叙事上

### 第三优先级

- decode 阶段的计算稀疏

原因：

- 研究价值很高
- 但实现和验证成本也最高
- 更适合在 permanent eviction 有初步结论后再展开

---

## 7. 结论

当前 `stage3` 的结果已经足够支持下面这条判断：

> 不建议再把主要精力放在 blockwise 内部 summary / representative / aggregation 的继续消融上，因为这类组合已经比较接近当前范式下的收益上限，进一步扩展更可能只带来局部涨跌，而不是结构性突破。

下一步更值得做的是：

1. 分层压缩率 / 分层 budget
2. decode 阶段的永久驱逐
3. decode 阶段的计算稀疏

换句话说，研究问题应该从：

- “哪个 summary 组合更好”

转向：

- “预算如何跨层分配”
- “哪些信息该在 decode 中长期保留”
- “哪些信息可以不参与每步计算”

这会比继续做细碎消融更有研究价值，也更接近真正能形成系统贡献的方向。
