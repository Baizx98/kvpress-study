# sparse_index_overhead_snapkv_chunkkv_blockwise

## Purpose

Paper-style single-column figure for sparse-index construction overhead.
The figure uses the 32-layer averaged Llama-3.1-8B-Instruct results on L40S.

## Source Data

- `evaluation/results/experiments/sparse_index_overhead_snapkv_chunkkv_blockwise/artifacts/sparse_index_overhead_summary.csv`
- `evaluation/results/experiments/sparse_index_overhead_snapkv_chunkkv_blockwise/artifacts/sparse_index_overhead_layers.csv`
- `evaluation/results/experiments/sparse_index_overhead_snapkv_chunkkv_blockwise/artifacts/metadata.json`

## Figures

- `sparse_index_overhead_paper_acm_wide.pdf`
- `sparse_index_overhead_paper_acm_wide.png`

## Notes

Panel (a) shows request-length scaling, panel (b) shows batch-size scaling, and panel (c) shows BlockWise summary amortization for `B=1,L=4096,ratio=0.5`.
BlockWisePress does not build `mean_values` or `multi_rep_keys` summaries in this benchmark.
All plotted values are CUDA-time repeat means averaged over 32 attention layers.
