# batch_main_compare_ratio05 部分结果解读

## 1. 背景

本轮实验原计划按照 `note/blockwise_chunkkv_quest_research_zh.md` 中“更适合 batch 推理的测试数据集建议”，在 `compression_ratio=0.5` 下比较：

- `BlockWisePress`
- `ChunkKVPress`

但在全量数据、单卡 `cuda:0` 条件下，后续的 `InfiniteBench` 与 `LooGLE` 任务持续触发显存不足，因此本次分析只基于已经完整完成的 4 个数据集：

- `LongBench / hotpotqa`
- `LongBench / multifieldqa_en`
- `LongBench / triviaqa`
- `LongBench-v2 / 0shot`

其中 `LongBench-v2 / 0shot` 为了适配单卡环境，使用了 `max_context_length=32768`。

## 2. 结果概览

### LongBench

- `hotpotqa`
  - `BlockWise`: `58.30`
  - `ChunkKV`: `59.16`
- `multifieldqa_en`
  - `BlockWise`: `53.75`
  - `ChunkKV`: `54.81`
- `triviaqa`
  - `BlockWise`: `91.04`
  - `ChunkKV`: `91.43`

### LongBench-v2 / 0shot

- `average`
  - `BlockWise`: `0.0855`
  - `ChunkKV`: `0.0835`
- `easy`
  - 两者均为 `0.0833`
- `hard`
  - `BlockWise`: `0.0868`
  - `ChunkKV`: `0.0836`
- `short`
  - `BlockWise`: `0.2389`
  - `ChunkKV`: `0.2333`
- `medium`
  - 两者均为 `0.0`
- `long`
  - 两者均为 `0.0`

## 3. 结果解读

### 3.1 在已完成的 LongBench 任务上，BlockWise 与 ChunkKV 非常接近

三项 `LongBench` 任务中，`BlockWise` 都略低于 `ChunkKV`，但差距很小：

- `hotpotqa` 差约 `0.86`
- `multifieldqa_en` 差约 `1.06`
- `triviaqa` 差约 `0.39`

这说明当前的 `BlockWisePress` 在真实长文 QA / 多字段信息提取任务上，已经具备相当强的竞争力。  
至少在 `ratio=0.5` 这个压缩率下，它并没有表现出像在某些极限检索基准中那样明显的劣势。

### 3.2 LongBench-v2 上，BlockWise 反而略优于 ChunkKV

在 `LongBench-v2 / 0shot` 上：

- `average`
- `hard`
- `short`

这三个维度上，`BlockWise` 都略优于 `ChunkKV`。  
这说明块摘要 + question-aware 打分这条路线，在更接近真实开放式长上下文任务时，未必天然弱于 token/chunk 粒度更细的方法。

### 3.3 当前实验更支持“BlockWise 适合作为 batch 推理系统中的实用方案”

这轮结果和前面在 `RULER` 上观察到的现象形成了一个互补结论：

- 在极限检索、多 key、超长距离精确定位任务上，`BlockWise` 仍明显弱于 `ChunkKV`
- 但在更接近真实 batch inference workload 的任务上，`BlockWise` 和 `ChunkKV` 的精度差距显著缩小

这对你的研究主线其实是有利的。  
因为你的目标并不是“在所有检索型 benchmark 上都打赢 ChunkKV”，而是：

- 设计一个更适合块级管理、块级卸载、低开销热度评估的 KV 压缩机制
- 并在更贴近 batch 推理负载的任务上保持足够好的效果

从这个角度看，这轮结果是正面的。

## 4. 限制

这轮结果不能过度外推，原因有三点：

1. `InfiniteBench` 和 `LooGLE` 由于显存限制未能完成，因此本次结论仍然主要来自 `LongBench` 系列。
2. `LongBench-v2` 使用了 `max_context_length=32768`，它不是完全原始长度下的评测。
3. 当前只测试了 `compression_ratio=0.5`，还没有覆盖更激进压缩率下的表现。

## 5. 当前可得出的阶段性结论

可以较有把握地说：

- `BlockWisePress` 在更贴近 batch 推理主场景的任务上，已经接近 `ChunkKVPress`
- 它的优势不在于极限检索精度，而在于更自然地支持块摘要、块热度、块级管理和未来块级卸载
- 因此后续工作应继续围绕“系统友好性 + 足够好的真实任务精度”推进，而不是一味追逐 `RULER` 类基准上的最优分数

## 6. 配套图

- [batch_main_compare_ratio05_partial.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/batch_main_compare_ratio05/batch_main_compare_ratio05_partial.png)
