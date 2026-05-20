# sparse_index_overhead_snapkv_chunkkv_blockwise

## Purpose

Compare sparse-index construction overhead for SnapKV, ChunkKV, and BlockWisePress.
The measurement includes score computation and top-k/index construction only; it excludes K/V gather.

## Model Shape

The benchmark loads `/Tan/model/Llama-3.1-8B-Instruct` weights by default.
It uses the real embedding table and every tested layer's Q/K projection weights to generate Q/K tensors before timing sparse-index logic.

## Artifacts

- `artifacts/sparse_index_overhead_summary.csv`
- `artifacts/sparse_index_overhead_layers.csv`
- `artifacts/metadata.json`
- Plan: `note/sparse_index_overhead_snapkv_chunkkv_blockwise_plan_zh.md`
