# ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19

## 说明

本目录保存 ATC26 prefill-only 压缩实验的聚合图。图像由 `figure/ATC26_plot_prefill_sweep.py` 优先从 `ATC26_metrics_full_long.csv` 生成。

配套结果目录：

`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/`

完整性分析：

`note/ATC26_prefill_sweep_completeness_zh.md`

## 图像

- `ATC26_longbench_quality_vs_compression_by_model.png`
- `ATC26_needle_accuracy_vs_compression_by_model.png`
- `ATC26_pg19_ppl_vs_compression_by_model.png`
- `ATC26_longbench_subdataset_quality_grid.png`
- `ATC26_needle_depth_quality_grid.png`

## 推荐查看顺序

1. 先看 `ATC26_longbench_quality_vs_compression_by_model.png`，确认 6 个 LongBench 子任务的整体趋势。
2. 再看 `ATC26_longbench_subdataset_quality_grid.png`，确认 LongBench 子数据集之间是否趋势一致。
3. 看 `ATC26_needle_accuracy_vs_compression_by_model.png` 和 `ATC26_needle_depth_quality_grid.png`，检查长上下文检索是否存在明显断点。
4. 最后看 `ATC26_pg19_ppl_vs_compression_by_model.png`，注意 PG19 perplexity 是 lower-is-better。
