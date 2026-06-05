# ATC26 Decode Prompt KVCache Importance Heatmap LongBench Figures

## 实验目的

展示 LongBench 长输出样本在 decode 阶段不同 step 下，prompt token positions 的 token-level 保留/丢弃状态。

## 图像说明

- `heatmap_*`: 主图。x 轴为 prompt token position，y 轴为 decode step，粉色为保留，蓝色为丢弃。
- `keep_frequency_*`: 每个 prompt token 在所有 decode steps 中被保留的频率。
- `jaccard_adjacent_summary.png`: 不同压缩率下相邻 step retained set Jaccard。

## 对应结果目录

- `evaluation/results/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/`

## 推荐阅读

- `note/decode_prompt_kvcache_importance_heatmap_longbench_results_zh.md`

