# ATC26 BlockWise Temporal Index Similarity Figures

Main source: `evaluation/results/experiments/ATC26_blockwise_temporal_index_similarity/artifacts/decode1024/ATC26_temporal_similarity_aggregate.csv`

Analysis note: `note/ATC26_blockwise_temporal_index_similarity_results_analysis_zh.md`

The root directory contains figures from the earlier 512-step run. Use `decode1024/` for the complete paper-facing run because it includes lag 512.

Main views:
- `*_lag_overlap_heatmap.*`: layer-by-lag top-k overlap.
- `*_lag_curve.*`: layer-averaged temporal similarity over lag.
- `*_reuse_curve.*`: fixed refresh interval versus every-step oracle.
