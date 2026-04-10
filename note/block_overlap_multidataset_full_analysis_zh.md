# 多数据集块保留相似性分析

## 1. 实验目的

这组实验不是比较最终准确率，而是比较 `BlockWisePress` 和 `ChunkKVPress` 在不同层、不同压缩率下，**到底保留了哪些 KV 块**。

我们关心两类相似性：

1. 跨方法、同层相似性  
   同一条样本、同一层，比较 `BlockWise` 和 `ChunkKV` 保留块 ID 集合的 Jaccard 相似度。

2. 同方法、跨层相似性  
   对单个方法，计算 32 层两两之间保留块 ID 集合的 Jaccard 相似度，形成 `32 x 32` 矩阵。

这个分析的价值在于回答：

- 两种方法是否其实保留了相似的块，只是评分实现不同？
- 哪些层两者更一致，哪些层差异更大？
- `BlockWise` 是否比 `ChunkKV` 更容易在层间发生块集合漂移？


## 2. 当前已完成的数据集

本轮脚本已经按新的绘图方式重跑，并保留了原始 JSON：

- 原始结果：
  [block_overlap_multidataset_results.json](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/block_overlap_multidataset_full/artifacts/block_overlap_multidataset_results.json)
- 日志：
  [run.log](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/block_overlap_multidataset_full/artifacts/run.log)

当前**完整落盘**的数据集有 3 个：

- `LongBench / triviaqa`
- `LongBench / hotpotqa`
- `LongBench / multifieldqa_en`

`LongBench-v2 / 0shot`、`RULER` 和 `Needle in a Haystack` 没有完整跑完，原因是后续阶段触发了 `CUDA OOM`。  
因此下面的结论只针对上述 3 个已完成数据集。


## 3. 总体观察

### 3.1 跨方法同层相似性随压缩率升高明显下降

把所有样本、所有层的同层 Jaccard 相似度取平均，可以得到：

### `LongBench / triviaqa`

- `ratio=0.3`: `0.6948`
- `ratio=0.5`: `0.5700`
- `ratio=0.7`: `0.4538`

### `LongBench / hotpotqa`

- `ratio=0.3`: `0.7026`
- `ratio=0.5`: `0.5568`
- `ratio=0.7`: `0.4247`

### `LongBench / multifieldqa_en`

- `ratio=0.3`: `0.7158`
- `ratio=0.5`: `0.5669`
- `ratio=0.7`: `0.4267`

结论：

- 在 `0.3` 时，两种方法保留块集合已经有比较明显的一致性，Jaccard 大约在 `0.70` 左右。
- 到 `0.5` 时，一致性下降到 `0.56~0.57`。
- 到 `0.7` 时，一致性进一步下降到 `0.42~0.45`。

这说明：

- `BlockWise` 和 `ChunkKV` 在**中低压缩率**下倾向于保留相近的块。
- 但压缩率越高，它们的块选择路径越分化。


### 3.2 `ChunkKV` 的层间块集合更稳定

把每个方法自己的 `32 x 32` 层间相似矩阵的上三角均值取出来，可以看到：

### `LongBench / triviaqa`

- `ratio=0.3`:
  - `BlockWise`: `0.7563`
  - `ChunkKV`: `0.7921`
- `ratio=0.5`:
  - `BlockWise`: `0.6394`
  - `ChunkKV`: `0.6960`
- `ratio=0.7`:
  - `BlockWise`: `0.5282`
  - `ChunkKV`: `0.6443`

### `LongBench / hotpotqa`

- `ratio=0.3`:
  - `BlockWise`: `0.7537`
  - `ChunkKV`: `0.7853`
- `ratio=0.5`:
  - `BlockWise`: `0.6342`
  - `ChunkKV`: `0.6702`
- `ratio=0.7`:
  - `BlockWise`: `0.5189`
  - `ChunkKV`: `0.5667`

### `LongBench / multifieldqa_en`

- `ratio=0.3`:
  - `BlockWise`: `0.7496`
  - `ChunkKV`: `0.8050`
- `ratio=0.5`:
  - `BlockWise`: `0.6390`
  - `ChunkKV`: `0.6840`
- `ratio=0.7`:
  - `BlockWise`: `0.5310`
  - `ChunkKV`: `0.5856`

结论：

- 三个数据集、三个压缩率上，`ChunkKV` 的层间相似性都高于 `BlockWise`。
- 这说明 `ChunkKV` 更倾向于在不同层保留相似的块集合，层间选择更稳定。
- `BlockWise` 的块选择层间漂移更明显，尤其在高压缩率下更容易发生变化。


## 4. 分层现象

### 4.1 早层的跨方法一致性通常更差

按层统计同层 Jaccard 的均值后，可以看到一个很稳定的模式：

- 在三个数据集上，`layer 0` 和 `layer 1` 基本都是最不一致的层。
- 到中后层，两者的保留块集合更接近。

例如：

### `triviaqa, ratio=0.7`

- 最低层：
  - `layer 0: 0.246`
  - `layer 1: 0.302`
- 较高层：
  - `layer 7: 0.541`
  - `layer 28: 0.512`
  - `layer 26: 0.508`

### `hotpotqa, ratio=0.7`

- 最低层：
  - `layer 1: 0.268`
  - `layer 0: 0.272`
- 较高层：
  - `layer 7: 0.498`
  - `layer 6: 0.476`
  - `layer 18: 0.473`

这说明：

- 早层的块选择更依赖方法本身的归纳偏置。
- 到中后层，两种方法都更容易收敛到“任务相关”的区域。


### 4.2 `BlockWise` 和 `ChunkKV` 的主要差异不是“完全不同”，而是“稳定性不同”

从 `0.3` 压缩率下约 `0.70` 的同层相似度可以看出：

- `BlockWise` 并不是在保留完全不同的块。
- 它在很多层已经和 `ChunkKV` 有较大重合。

但从层间矩阵看：

- `BlockWise` 的保留块集合更容易随着层数变化而漂移。
- `ChunkKV` 更像是在较多层重复保留一组相对稳定的关键块。

这意味着当前 `BlockWise` 的主要短板更可能是：

- 保留块排序的层间稳定性不够
- 而不是块级摘要路线完全错误


## 5. 对方法设计的启发

### 5.1 当前 `BlockWise` 的方向仍然成立

这组分析支持一个很重要的判断：

- `BlockWise` 在低开销块摘要下，已经能和 `ChunkKV` 保留出相当一部分相似块。
- 尤其在 `ratio=0.3` 时，这种相似性已经不低。

所以：

- 不需要回退到完全 token-level 的 ChunkKV 风格实现。
- 块摘要路线依然是合理的主线。


### 5.2 更值得优化的是“块选择稳定性”，不是继续堆复杂摘要

因为现象更像：

- 方法之间并非完全保留不同块
- 而是 `BlockWise` 更容易在不同层改变保留块集合

所以后续更值得考虑的优化方向是：

1. `sink + recent + hot` 的预算再细化  
   目标：降低高压缩率下层间块选择漂移。

2. 在 `DualPhasePerLayerPress` 中实现层间或步间的分数重用  
   但默认关闭，只在系统实验中开启。  
   目标：让块状态管理更平滑，而不是把逻辑直接堆回 `BlockWise`。

3. 控制高压缩率下的块集合剧烈变化  
   例如：
   - 对上一层已保留块给予轻微优先级
   - 或在 hot block 排序中加入很弱的历史热度项

这里要注意：

- 这类“稳定性增强”更适合放在 `DualPhasePerLayerPress`
- 而不是重新把 `BlockWisePress` 变复杂


### 5.3 目前不建议优先回到重型 token correction

这组分析更像是在说：

- `BlockWise` 和 `ChunkKV` 的差距主要体现在层间稳定性
- 不完全是块内关键 token 信息不够

因此下一步不建议优先回到：

- 重型 token correction
- 复杂 head redundancy
- 过于激进的二次筛选

因为这些会增加实现复杂度，但不一定直击当前观察到的主要问题。


## 6. 对你后续系统工作的意义

从你的最终目标看，这组结果其实是正面的。

你要做的是：

- 面向 batch 推理
- 面向块级 KV 管理
- 未来接 GPU-CPU 两级卸载

在这个背景下，`BlockWise` 的优势不是一定要在所有层完全复现 `ChunkKV` 的选择，而是：

- 用更低的元数据和计算开销
- 保留一组“足够接近有效块”的块集合
- 并且天然适合块级存储、调度和卸载

这组 overlap 结果说明：

- `BlockWise` 并没有偏离 `ChunkKV` 太远
- 只是更缺少层间稳定性

这反而很适合作为论文里的叙述：

- `ChunkKV` 提供了精细 token-level 的强基线
- `BlockWise` 用更系统友好的块摘要近似它
- 二者在低中压缩率下保留块集合已经高度重叠
- 差距主要来自高压缩和层间稳定性，而不是方法方向错误


## 7. 建议的下一步

我建议后面按这个顺序继续：

1. 暂时不再把 `BlockWisePress` 做复杂  
   保持 `mean_key + topk_key_mean + sink/recent/hot` 的简洁结构。

2. 如果继续做 overlap 研究，优先单独补 `RULER` 和 `Needle`  
   但应采用更小批、更短上下文、更强显存清理的方式，避免这次这种多数据集串行 OOM。

3. 后续真正想改精度时，优先把“稳定性机制”放到 `DualPhasePerLayerPress`  
   而不是重新把 `BlockWise` 改成复杂算法。


## 8. 相关产物

图像目录：

- [block_overlap_multidataset_full](/home10T/bzx/workspace/kvpress-study/figure/experiments/block_overlap_multidataset_full)

原始结果：

- [block_overlap_multidataset_results.json](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/block_overlap_multidataset_full/artifacts/block_overlap_multidataset_results.json)

运行日志：

- [run.log](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/block_overlap_multidataset_full/artifacts/run.log)
