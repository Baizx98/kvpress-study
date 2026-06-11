# end2end_serving_paper_draft_model_comparison_20260610

## Purpose

Diagnostic three-model comparison figures for the paper-draft end-to-end serving table.
These figures are meant to make model-level differences visible, because the previous per-model plots used the same relative prediction rule for KVCore and for missing InfiniGen rows.

## Source

- Metrics table: `figure/experiments/end2end_serving_paper_draft_predicted_20260610/paper_draft_end2end_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_model_comparison.py`

## Figures

- `model_compare_throughput.pdf`
- `model_compare_throughput.png`
- `model_compare_ttft.pdf`
- `model_compare_ttft.png`
- `model_compare_p99_e2e.pdf`
- `model_compare_p99_e2e.png`

## Layout

Each metric is one figure with three panels: vLLM, InfiniGen, and KVCore. Within each panel, bars compare the three models at the same batch size and output length. Batch-size regions use the same merged-BS shaded layout as the previous figures.
