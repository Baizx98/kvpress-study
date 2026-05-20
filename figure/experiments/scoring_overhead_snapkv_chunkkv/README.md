# scoring_overhead_snapkv_chunkkv

## Purpose

Figures for the SnapKV / ChunkKV scoring-overhead microbenchmark on the RTX 3090.
The current figures use the revised no-gather timing definition: score computation plus top-k/index construction only.

## Source Data

- `evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/scoring_overhead_summary.csv`
- `evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/metadata.json`

## Figures

- `scoring_overhead_absolute_time.png`
- `scoring_overhead_absolute_time.pdf`
- `scoring_overhead_ratios.png`
- `scoring_overhead_ratios.pdf`
- `chunkkv_overhead_breakdown.png`
- `chunkkv_overhead_breakdown.pdf`
- `scoring_overhead_presentation_summary.png`
- `scoring_overhead_presentation_summary.pdf`

## Notes

The attention baseline uses `flash_attn.flash_attn_func` when available. On the current host it falls back to PyTorch SDPA forced `FLASH_ATTENTION` backend because the local `flash_attn` package cannot be imported due to a GLIBC version mismatch.
