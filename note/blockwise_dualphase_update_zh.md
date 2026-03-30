# BlockWisePress / DualPhasePerLayerPress 修改说明

## 1. 修改目标

这次修改主要围绕以下几个目标展开：

1. 将 `query_aware` 明确收敛为这里的 `question-aware`：
   在评测阶段将问题拼接到上下文中，再进行压缩。
2. 将 `BlockWisePress` 从“先算 token 分数，再聚合块分数”的路径，
   改成“先缓存块摘要，再直接对块打分”的低开销路径。
3. 让 `DualPhasePerLayerPress` 成为真正和 `BlockWisePress` 配套的
   分层、分阶段块压缩器，并在 decode 阶段加入近似刷新机制。
4. 为未来 KV cache 块卸载系统预留块状态分类机制，
   即使当前还没有真实接入 GPU-CPU 两级卸载框架，也能先输出这些逻辑状态。

## 2. BlockWisePress 的主要修改

文件：
`kvpress/presses/block_wise_press.py`

### 2.1 从 token 评分改成块摘要评分

旧思路是：

- 对 query window 和所有 token 逐个计算重要性；
- 再把 token 分数聚合成 block 分数。

新思路是：

- 先为每个块构造 compact summary；
- query window 直接和 block summary 交互；
- 直接得到 block 粒度的重要性分数。

### 2.2 当前使用的块摘要

每个块当前缓存以下摘要：

- `mean_keys`：块内 key 的均值，表示块的整体语义中心
- `peak_keys`：块内 key 范数最大的代表 key，近似少数关键 token
- `mean_values`：块内 value 的均值，作为未来卸载/预取侧的附加摘要
- `token_counts`：块内真实 token 数

这样的摘要尺寸远小于原始 token 级 KV cache，
更适合作为未来 offload 系统中常驻 GPU 的“块级元数据”。

### 2.3 块分数计算方式

当前块打分是：

1. 取最后若干个 question-aware queries
2. 分别与 `mean_keys` 和 `peak_keys` 做交互
3. query 维度上做 `mean + max` 混合聚合
4. 在头维度上做可筛选的加权聚合

因此它仍然同时考虑：

- 块整体的重要性
- 块内少数关键 token 的存在

但不再需要逐 token 打分。

### 2.4 新增缓存

新增的内部缓存包括：

- `last_block_summary`
- `last_block_heat`
- `last_block_heat_ema`

其中 `last_block_summary` 可以直接看作未来块卸载框架的一个可复用接口。

## 3. DualPhasePerLayerPress 的主要修改

文件：
`kvpress/presses/dual_phase_per_layer_press.py`

### 3.1 分层、分阶段

当前仍保留：

- `layer_phase_ratios`
- `default_phase_ratios`
- `layer_phase_cold_ratios`
- `default_phase_cold_ratios`

因此可以按 attention layer、按 prefill / decode 阶段分别配置。

### 3.2 永久删除 + 暂时冷块

当前 decode 中的块分为两层处理：

1. 永久删除块：
   直接从当前 active KV 中移除
2. 暂时冷块：
   物理上保留，但通过 `masked_key_indices` 在后续迭代中不参与当前注意力计算

这样就能表达：

- 真正死掉的块
- 当前轮次不活跃、但后续有可能重新变热的块

### 3.3 decode 近似刷新

为了避免 decode 每步都重算完整块打分，增加了两种节流机制：

- `compression_interval`
- `score_refresh_interval`

其中：

- `compression_interval` 控制永久删除 / 物理压缩的刷新频率
- `score_refresh_interval` 控制块热度和块状态分类的刷新频率

如果没有到刷新步，则沿用上一次的块状态和 mask，
从而降低 decode 阶段的块评分开销。

### 3.4 历史热度复用

新增 `layer_heat_ema` 作为历史热度缓存，
使用 EMA 来稳定块热度，降低单步波动带来的抖动。

## 4. 块状态分类机制

当前在 `DualPhasePerLayerPress` 中，已经能为每层维护以下逻辑状态：

- `active`
- `resident_gpu`
- `permanently_deleted`
- `offloaded_cpu`
- `prefetch_to_gpu`

说明：

- 这些状态目前是逻辑分类，不会真的搬运数据
- 但已经具备未来接入 GPU-CPU 卸载调度器所需要的接口形态

当前分类逻辑大致是：

1. 先选出当前参与计算的 `active`
2. 永久删掉的是 `permanently_deleted`
3. 不活跃块中，一部分保留为 `resident_gpu`
4. 其余块中，一部分标为 `prefetch_to_gpu`
5. 剩余块标为 `offloaded_cpu`

目前这些类别主要由当前块热度和历史热度 EMA 决定。

## 5. 评测逻辑修改

文件：

- `evaluation/evaluate.py`
- `evaluation/evaluate_config.yaml`
- `evaluation/README.md`
- `evaluation/leaderboard.sh`

主要改动：

1. 在 `EvaluationConfig` 中把字符串形式的 `query_aware`
   统一转换成布尔值
2. 在 `evaluate.py` 中加入 `_press_requires_question_aware`
   自动判断某个 press 是否应强制启用 question-aware
3. 对 `BlockWisePress`、`DualPhasePerLayerPress` 等方法，
   若未显式打开 `query_aware`，则自动强制为 `True`
4. README 和配置注释中明确：
   这里的 `query_aware` 指 question-aware，而不是 Quest 风格的 decode query-aware

## 6. 测试修改

文件：
`tests/test_dual_phase_per_layer_press.py`

新增 / 保留的验证点包括：

- 分层分阶段压缩率是否生效
- prefill / decode 切换是否正常
- 暂时冷块是否能在不物理删除的前提下被 mask
- 永久删除和暂时冷块是否能混合使用
- 尾部残块是否被稳定保留，从而避免层间 cache 长度不一致
- `BlockWisePress` 是否真的构建了块摘要缓存
- `DualPhasePerLayerPress` 是否真的记录了块状态分类

## 7. 当前局限

1. 块摘要目前仍然是“从当前完整块重新构建”的实现，
   已经避免了 token 级打分，但还没有接成真正的在线增量更新器。
2. 块状态分类已经存在，但尚未真正驱动 GPU-CPU 数据迁移。
3. decode 近似刷新机制已经接入，但参数仍需要更多 benchmark 去调优。
4. 多数据集 benchmark 仍需要继续扩展，尤其是更慢的自然任务数据集。

## 8. 建议的下一步

1. 将块摘要改成真正的增量更新形式：
   每个块填满时固化 summary，只维护最后一个未满块的临时 summary
2. 将块状态分类接入真实 offload 管理器
3. 基于多数据集结果调 `score_refresh_interval`、`resident_gpu_ratio`、`prefetch_ratio`
4. 在论文中将“块摘要常驻 GPU、原始块可冷存”的设计写成核心系统卖点
