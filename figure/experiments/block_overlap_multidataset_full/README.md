# block_overlap_multidataset_full

## 图像说明

本目录保存 `BlockWisePress` 与 `ChunkKVPress` 在多个代表性数据集上的块保留相似性可视化图。

图中包含两类相似性：

- 跨方法、同层相似性
  - 同一层 `BlockWise` 和 `ChunkKV` 保留块集合的 Jaccard 相似度曲线
- 同方法、跨层相似性
  - 每个方法内部 `32 x 32` 的层间块集合 Jaccard 热图

## 当前已完成的数据集

- `LongBench / triviaqa`
- `LongBench / hotpotqa`
- `LongBench / multifieldqa_en`

## 当前未完整完成的数据集

由于后续阶段触发 `CUDA OOM`，下面这些数据集没有完整产出图像：

- `LongBench-v2 / 0shot`
- `RULER / 4096`
- `Needle in a Haystack`

## 关键图

建议优先查看每个数据集的 aggregate 图：

- `longbench_triviaqa/longbench_triviaqa_aggregate.png`
- `longbench_hotpotqa/longbench_hotpotqa_aggregate.png`
- `longbench_multifieldqa_en/longbench_multifieldqa_en_aggregate.png`

然后再结合各自的 `sample_*.png` 看具体样本。

## 绘图脚本

- [analyze_block_overlap_multidataset.py](/home10T/bzx/workspace/kvpress-study/figure/analyze_block_overlap_multidataset.py)

## 配套解读

- [block_overlap_multidataset_full_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/block_overlap_multidataset_full_analysis_zh.md)

## 备注

本仓库当前通过 `.gitignore` 默认忽略 `figure/` 下的图片文件，避免将可复现图像直接同步到远端仓库。
