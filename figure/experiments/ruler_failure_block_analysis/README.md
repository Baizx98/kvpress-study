# RULER Failure Block Analysis

## 目的

针对 `RULER ratio=0.7` 中 `BlockWisePress` 与 `ChunkKV` 差距最大的失败样本，分析两种方法在各层保留/删除 KV 块的分布特点。

## 分析范围

- 任务：
  - `niah_multikey_3`
  - `niah_single_3`
- 样本：
  - `BlockWise` 失败且 `ChunkKV` 成功的代表性 case
- 方法：
  - `BlockWisePress`
  - `ChunkKVPress(SnapKVPress)`

## 相关结果

- 原始分析产物：
  `evaluation/results/experiments/ruler_failure_block_analysis/artifacts/`
- 代表性图：
  本目录下的 `case_*.png`

## 说明

本组图主要用于诊断：

- 是否错删了答案相关块
- 是否过度偏向 recent/question 尾部块
- `BlockWise` 的块摘要分数与 `ChunkKV` 的 token->chunk 分数在关键失败 case 上有何区别
