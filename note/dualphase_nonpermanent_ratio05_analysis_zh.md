# dualphase_nonpermanent_ratio05 结果分析

## 1. 实验目标

这轮实验比较三种设置在 `compression_ratio=0.5` 下的表现：

- `BlockWisePress`
  - 永久压缩，物理删除部分 KV
- `DualPhasePerLayerPress`
  - 非永久驱逐，物理保留全部 KV
  - 仅让约 `50%` 的块参与计算
- `ChunkKVPress`
  - 作为参考基线

实验数据集只使用前一轮已经稳定完成的 4 个任务：

- `LongBench / hotpotqa`
- `LongBench / multifieldqa_en`
- `LongBench / triviaqa`
- `LongBench-v2 / 0shot`

其中 `LongBench-v2 / 0shot` 使用 `max_context_length=32768`。

## 2. 结果

### LongBench

- `hotpotqa`
  - `BlockWisePress`: `58.30`
  - `DualPhasePerLayerPress`: `58.10`
  - `ChunkKVPress`: `59.16`
- `multifieldqa_en`
  - `BlockWisePress`: `53.75`
  - `DualPhasePerLayerPress`: `53.55`
  - `ChunkKVPress`: `54.81`
- `triviaqa`
  - `BlockWisePress`: `91.04`
  - `DualPhasePerLayerPress`: `90.47`
  - `ChunkKVPress`: `91.43`

### LongBench-v2 / 0shot

- `average`
  - `BlockWisePress`: `0.0855`
  - `DualPhasePerLayerPress`: `0.0815`
  - `ChunkKVPress`: `0.0835`
- `easy`
  - 三者均为 `0.0833`
- `hard`
  - `BlockWisePress`: `0.0868`
  - `DualPhasePerLayerPress`: `0.0804`
  - `ChunkKVPress`: `0.0836`
- `short`
  - `BlockWisePress`: `0.2389`
  - `DualPhasePerLayerPress`: `0.2278`
  - `ChunkKVPress`: `0.2333`
- `medium`
  - 三者均为 `0.0`
- `long`
  - 三者均为 `0.0`

## 3. 结论

### 3.1 非永久驱逐没有带来精度增益

这轮最直接的观察是：

- `DualPhasePerLayerPress` 没有超过永久压缩版 `BlockWisePress`
- 在 4 个已完成数据集上，它都略低一点

这种差距不大，但方向很一致。  
这说明在当前实现里，“物理保留全部 KV，只稀疏参与计算”并没有自动转化成更高的精度。

### 3.2 当前差距更像实现策略带来的损失，而不是非永久驱逐思想本身无效

这里要特别区分两件事：

1. **思想层面**
   - 非永久驱逐本来是合理的
   - 它对未来块级卸载、热度维护、GPU-CPU 两级调度都更自然

2. **当前实现层面**
   - `DualPhasePerLayerPress` 现在更像是在 `BlockWisePress` 外包了一层 active-mask 逻辑
   - 它不是为“只保留 active blocks 参与计算”专门优化过的推理路径
   - 因此精度和实现开销都还不是最终形态

所以这轮结果更适合解读为：

- 当前这版 `dual_phase_per_layer` 还没有把“非永久驱逐”的潜力发挥出来
- 但它已经说明：在不做真实删除的情况下，精度也没有出现灾难性退化

### 3.3 从研究路线看，永久压缩仍然是当前更稳的验证基线

如果当前目标是继续验证块重要性评分算法本身，那么：

- 永久压缩版 `BlockWisePress` 更简单
- 结果也略好
- 更适合继续做算法精度迭代

而 `DualPhasePerLayerPress` 更适合后续承担：

- “物理保留、逻辑不参与”的块状态管理
- 步间 / 层间分数重用
- 未来与 GPU-CPU 卸载系统结合

也就是说：

- `BlockWisePress` 更适合当前做“评分算法”的主验证
- `DualPhasePerLayerPress` 更适合后续做“系统机制”的主验证

## 4. 对下一步工作的启发

基于这轮结果，我的建议是：

1. 继续把 `BlockWisePress` 作为永久压缩主线基线。
2. `DualPhasePerLayerPress` 暂时不要继续拿精度和 `BlockWisePress` 硬比。
3. 后续如果继续做 `dual_phase_per_layer`，更值得关注：
   - 计算参与稀疏是否能降低时延
   - 是否便于维护块热度
   - 是否更适合和卸载 / 预取策略结合

换句话说，这轮结果支持一个更清晰的角色分工：

- `BlockWisePress`：验证块评分与永久压缩精度
- `DualPhasePerLayerPress`：验证非永久驱逐与块状态管理机制

## 5. 配套图

- [dualphase_nonpermanent_ratio05_compare.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/dualphase_nonpermanent_ratio05/dualphase_nonpermanent_ratio05_compare.png)
