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
- `ATC26_blockwise_attention_similarity_hotpotqa_3samples`: ATC26 BlockWise layer/KV-head kept-block similarity heatmaps.
- `ATC26_blockwise_head_group_similarity_hotpotqa_3samples`: ATC26 BlockWise KV-head group selection similarity figures.
- `ATC26_design_block_sparse_head_layer_similarity`: paper-style independent head/layer similarity heatmaps for ATC26 Design.
- [ATC26_blockwise_temporal_index_similarity](/home10T/bzx/workspace/kvpress-study/figure/experiments/ATC26_blockwise_temporal_index_similarity/README.md)
- [ATC26_decode_prompt_kvcache_importance_heatmap_longbench](/home10T/bzx/workspace/kvpress-study/figure/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/README.md): token-level prompt KVCache keep/discard heatmaps across LongBench decode steps.
- [ATC26_token_level_temporal_similarity](/home10T/bzx/workspace/kvpress-study/figure/experiments/ATC26_token_level_temporal_similarity/preview_delta1024/README.md): token-level important KV set overlap and fixed-refresh recall across decode-step deltas.
- `scoring_overhead_snapkv_chunkkv`: SnapKV and ChunkKV scoring overhead figures against fused attention kernels.
- `sparse_index_overhead_snapkv_chunkkv_blockwise`: paper-style sparse-index overhead figure for SnapKV, ChunkKV, and KVCore.
- `end2end_serving_placeholder`: placeholder end-to-end serving figures using measured vLLM and simulated KVCore/InfiniGen data.
- `vllm_formal_pg19_in6k`: measured vLLM PG19 in6k figures split by model, batch size, and output length.
- `end2end_serving_llama31_real_infinigen_kvcore_sim_20260610`: Llama-3.1-8B single-column serving figures with measured vLLM/InfiniGen and simulated KVCore.
- `end2end_serving_llama31_real_infinigen_kvcore_sim_mergedbs_20260610`: Llama-3.1-8B merged-BS wide serving figures with measured vLLM/InfiniGen and simulated KVCore.
- `end2end_serving_paper_draft_predicted_20260610`: paper-draft serving figures with measured vLLM/Llama InfiniGen and predicted missing systems.
- `end2end_serving_paper_draft_model_comparison_20260610`: diagnostic model-comparison figures for paper-draft serving predictions.
- `end2end_serving_paper_draft_modelaware_predicted_20260610`: model-aware paper-draft serving figures with measured vLLM/Llama InfiniGen and predicted missing systems.
- `end2end_serving_paper_draft_modelaware_comparison_20260610`: diagnostic model-comparison figures for model-aware paper-draft serving predictions.
- `end2end_serving_paper_draft_modelaware_comparison_singlecol_20260610`: single-column model-aware serving comparison figures.
- `end2end_serving_paper_draft_modelaware_by_model_horizontal_20260610`: horizontal by-model model-aware serving comparison figures.
- `kvcore_preemption_sim_from_vllm_test_20260610`: vLLM-measured and KVCore-simulated request preemption figures.
- `kvcore_preemption_sim_a6000_large_bs_long_out_20260610`: A6000 vLLM-measured and KVCore-simulated request preemption figures.
- [kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2](/home10T/bzx/workspace/kvpress-study/figure/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/README.md): task-group average LongBench score bars.
