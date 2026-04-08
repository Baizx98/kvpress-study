# dualphase_nonpermanent_ratio05

## 图像说明

本目录保存 `compression_ratio=0.5` 下三种方法的对比图：

- `BlockWisePress`
  - 永久压缩
- `DualPhasePerLayerPress`
  - 非永久驱逐，物理保留全部 KV，仅选择约 `50%` 块参与计算
- `ChunkKVPress`

## 已完成数据集

- `LongBench / hotpotqa`
- `LongBench / multifieldqa_en`
- `LongBench / triviaqa`
- `LongBench-v2 / 0shot`

## 关键图

- [dualphase_nonpermanent_ratio05_compare.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/dualphase_nonpermanent_ratio05/dualphase_nonpermanent_ratio05_compare.png)

## 配套解读

- [dualphase_nonpermanent_ratio05_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/dualphase_nonpermanent_ratio05_analysis_zh.md)
