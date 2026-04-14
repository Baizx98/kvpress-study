# Evaluation Result Index

本目录按“正式实验分组”和“零散历史结果”两类组织。

## 正式实验分组

- `experiments/prefill_sweep_10pct_blockwise_snapkv`
- `experiments/prefill_compare_15pct_blockwise_chunkkv`
- `experiments/prefill_compare_50pct_four_methods`
- `experiments/prefill_compare_50pct_blockwise_chunkkv`
- `experiments/ruler_ablation_10pct`
- `experiments/ruler_failure_block_analysis`
- `experiments/ruler_token_correction_50pct`
- `experiments/ruler_cross_layer_residual_50pct`
- `experiments/ruler_residual_ablation_fast`
- `experiments/batch_main_compare_ratio05`
- `experiments/blockwise_ablation_ratio70_longbench_stage1`

- `experiments/blockwise_stage2_ratio70_fraction20_multidataset`
每组实验目录下统一包含：

- `artifacts/`
  存放原始 `config.yaml`、`predictions.csv`、`metrics.json`、`run.log`
- `README.md`
  说明实验目的、运行脚本、数据集与关键配置

## 历史零散结果

- `ad_hoc_baselines/`

这里保留尚未归并成正式实验组的早期结果，避免信息丢失。
- [blockwise_stage3_ratio70_fraction20_primarybench](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_stage3_ratio70_fraction20_primarybench/README.md)
