# end2end_serving_paper_draft_modelaware_comparison_singlecol_20260610

## Purpose

Single-column model-aware comparison figures for paper-draft end-to-end serving results.

## Source

- Metrics table: `figure/experiments/end2end_serving_paper_draft_modelaware_predicted_20260610/paper_draft_end2end_modelaware_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_modelaware_comparison_singlecol.py`

## Figures

- `modelaware_singlecol_throughput.pdf`
- `modelaware_singlecol_throughput.png`
- `modelaware_singlecol_ttft.pdf`
- `modelaware_singlecol_ttft.png`
- `modelaware_singlecol_p99_e2e.pdf`
- `modelaware_singlecol_p99_e2e.png`

## Layout

Each metric is one single-column figure. The three vertical panels are vLLM, InfiniGen, and KVCore. Within each panel, bars compare Llama-3.1, Mistral, and Qwen3 at the same batch size and output length. Batch-size regions use the merged-BS shaded layout.
