# ruler_token_correction_50pct

## 图像说明

本目录用于保存 `BlockWisePress + lightweight token correction` 在 `RULER` 上的详细对比图。

## 实验配置

- 数据集：`RULER`
- 压缩率：`0.7`
- 采样比例：`0.5`
- 方法：
  - 当前 `block_wise`
  - 上一轮已完成的 `chunkkv` 对照

## 推荐查看方式

重点关注以下子任务：

- `niah_multikey_2`
- `niah_multikey_3`
- `niah_single_3`
- `qa_1`
- `qa_2`

## 配套结果

- [results README](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/ruler_token_correction_50pct/README.md)
