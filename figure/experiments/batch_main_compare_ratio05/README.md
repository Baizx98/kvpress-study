# batch_main_compare_ratio05

## 图像说明

本目录保存 `BlockWisePress vs ChunkKVPress @ ratio=0.5` 在 batch 推理主实验数据集组合上的图像结果。

当前已完成并已纳入图像分析的数据集：

- `LongBench / hotpotqa`
- `LongBench / multifieldqa_en`
- `LongBench / triviaqa`
- `LongBench-v2 / 0shot`

当前未纳入图像分析的数据集：

- `InfiniteBench`
- `LooGLE`

原因是这些任务在单卡 `cuda:0`、全量 `fraction=1` 条件下持续触发显存不足。

## 当前图像

- [batch_main_compare_ratio05_partial.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/batch_main_compare_ratio05/batch_main_compare_ratio05_partial.png)

## 配套解读

- [batch_main_compare_ratio05_partial_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/batch_main_compare_ratio05_partial_analysis_zh.md)
