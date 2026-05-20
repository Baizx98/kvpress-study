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
- `experiments/batch_main_compare_ratio05`
- `experiments/blockwise_ablation_ratio70_longbench_stage1`

- `experiments/blockwise_stage2_ratio70_fraction20_multidataset`
每个实验子目录包含：

- 本组图像文件
- 一个 `README.md`，说明实验设置、配套结果目录、推荐阅读顺序
- [blockwise_stage3_ratio70_fraction20_primarybench](/home10T/bzx/workspace/kvpress-study/figure/experiments/blockwise_stage3_ratio70_fraction20_primarybench/README.md)
- [decode_long_output_longbench_stage1](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_long_output_longbench_stage1/README.md)
- [decode_final_framework_fixed_budget_stage1](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_final_framework_fixed_budget_stage1/README.md)
- [decode_hybrid_final_stage](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/README.md)
- `ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19`: ATC26 prefill-only sweep figures for BlockWise, SnapKV, and ChunkKV.
- `scoring_overhead_snapkv_chunkkv`: SnapKV and ChunkKV scoring overhead figures against fused attention kernels.
- `sparse_index_overhead_snapkv_chunkkv_blockwise`: paper-style sparse-index overhead figure for SnapKV, ChunkKV, and KVCore.
