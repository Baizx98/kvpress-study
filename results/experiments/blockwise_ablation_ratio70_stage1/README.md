# blockwise_ablation_ratio70_stage1

## 实验目的

在 `compression_ratio=0.7` 的高压缩设置下，对当前 `block_wise` baseline 做第一阶段逐轴消融，回答四个问题：

- A. 哪种 `block summary form` 更好
- B. `representative selection` 是否值得从 `key_norm` 换成更复杂方案
- C. `query window aggregation` 应该用 `mean / max / top-r mean` 中哪一个
- D. `head aggregation` 是否值得复杂化

同时加入 `Quest-style prefill block scorer` 作为并列对照组。

## 运行脚本

- [run_blockwise_ablation_ratio70_stage1.sh](/home10T/bzx/workspace/kvpress-study/evaluation/run_blockwise_ablation_ratio70_stage1.sh)

## 数据集

- `RULER (4096)`
- 任务过滤：
  - `niah_single_3`
  - `niah_multikey_3`
  - `qa_2`

## 方法

- `block_wise_prefill_per_layer`
- `quest_blockwise_prefill_per_layer`

## 关键 sweep 维度

- `compression_ratio=0.7`
- `fraction=0.2`
- `block_size=16`
- `q_window_size=64`
- `prefill_skip_first_layers=1`

逐轴消融：

- `summary_mode`
- `representative_mode`
- `query_agg_mode`
- `head_agg_mode`
- `Quest minmax scorer`

## 产物位置

- 原始运行日志：
  - [run.log](/home10T/bzx/workspace/kvpress-study/results/experiments/blockwise_ablation_ratio70_stage1/artifacts/run.log)
- 所有配置的原始结果：
  - [artifacts](/home10T/bzx/workspace/kvpress-study/results/experiments/blockwise_ablation_ratio70_stage1/artifacts)

每个配置目录下保留：

- `config.yaml`
- `predictions.csv`
- `metrics.json`

## 推荐优先查看

- 中文分析：
  - [blockwise_ablation_ratio70_stage1_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/blockwise_ablation_ratio70_stage1_analysis_zh.md)

## 当前结论摘要

- 这轮最强信号来自 `query_agg_mode=max`
- `norm-topk-mean only` 优于当前 `mean + norm-topk-mean` baseline
- `tail_query_relevance` 没有打赢 `key_norm`
- `uniform_mean` 仍然是最稳的 head aggregation
- `Quest-prefill` 在这组 prefill RULER 上明显弱于 summary-based blockwise
