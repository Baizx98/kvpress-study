# kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2

## Purpose

Single-column grouped bar chart showing average absolute LongBench score by task category.

## Source Data

- `evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/summary.csv`
- `note/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2_results_zh.md`

## Methods

- `Full KV`
- `KVCore`: decode query-aware BlockWise active set, top-p `p=0.95`, first 2 layers skipped, LongBench fraction `0.02`, seed `43`.

## Aggregation

Scores are averaged within each LongBench task category. These are absolute LongBench scores, not deltas.

## Outputs

- `figure/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/longbench_task_group_absolute_score_singlecol.pdf`
- `figure/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/longbench_task_group_absolute_score_singlecol.png`
- `figure/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/longbench_task_group_absolute_score_singlecol.csv`
- `figure/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/summary.json`
