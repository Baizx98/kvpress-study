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
- [decode_long_output_longbench_stage1](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_long_output_longbench_stage1/README.md)
- [decode_final_framework_fixed_budget_stage1](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_final_framework_fixed_budget_stage1/README.md)
- [decode_hybrid_final_stage](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_hybrid_final_stage/README.md)
- `ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19`: ATC26 prefill-only sweep for BlockWise, SnapKV, and ChunkKV.
- `ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv`: ATC26 prefill-only sweep for BlockWise, SnapKV, and ChunkKV.
- `ATC26_blockwise_attention_similarity_hotpotqa_3samples`: ATC26 BlockWise layer/KV-head kept-block similarity on 3 LongBench hotpotqa samples.
- `ATC26_blockwise_head_group_similarity_hotpotqa_3samples`: ATC26 BlockWise KV-head group selection similarity using saved per-head scores.
- [ATC26_blockwise_temporal_index_similarity](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/ATC26_blockwise_temporal_index_similarity/README.md)
- [ATC26_decode_prompt_kvcache_importance_heatmap_longbench](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/README.md): token-level prompt KVCache keep/discard heatmaps across LongBench decode steps.
- [ATC26_token_level_temporal_similarity](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/ATC26_token_level_temporal_similarity/README.md): token-level important KV set temporal overlap and fixed-refresh recall on PG19 decode traces.
- `scoring_overhead_snapkv_chunkkv`: SnapKV and ChunkKV scoring overhead microbenchmark against fused attention kernels.
- `sparse_index_overhead_snapkv_chunkkv_blockwise`: sparse-index score/top-k overhead for SnapKV, ChunkKV, and BlockWisePress.
- [end2end_serving_kvcore_vllm_infinigen_longreq](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/README.md): unified request manifests for KVCore, vLLM, and InfiniGen end-to-end serving throughput/latency experiments.
