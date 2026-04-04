# Figure Experiment Index

本目录只保留绘图脚本与按实验分组后的图像结果。

## 绘图脚本

- `plot_prefill_detailed.py`
- `plot_prefill_sweep.py`
- `plot_ruler_ablation.py`

## 实验分组

- `experiments/prefill_sweep_10pct_blockwise_snapkv`
- `experiments/prefill_compare_15pct_blockwise_chunkkv`
- `experiments/prefill_compare_50pct_four_methods`
- `experiments/prefill_compare_50pct_blockwise_chunkkv`
- `experiments/ruler_ablation_10pct`
- `experiments/ruler_failure_block_analysis`
- `experiments/ruler_token_correction_50pct`
- `experiments/ruler_cross_layer_residual_50pct`
- `experiments/ruler_residual_ablation_fast`

每个实验子目录包含：

- 本组图像文件
- 一个 `README.md`，说明实验设置、配套结果目录、推荐阅读顺序
