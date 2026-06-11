# end2end_serving_paper_draft_modelaware_comparison_20260610

## Purpose

Diagnostic three-model comparison figures for the model-aware paper-draft prediction table.

## Source

- Metrics table: `figure/experiments/end2end_serving_paper_draft_modelaware_predicted_20260610/paper_draft_end2end_modelaware_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_modelaware_comparison.py`

## Figures

- `modelaware_model_compare_throughput.pdf`
- `modelaware_model_compare_throughput.png`
- `modelaware_model_compare_ttft.pdf`
- `modelaware_model_compare_ttft.png`
- `modelaware_model_compare_p99_e2e.pdf`
- `modelaware_model_compare_p99_e2e.png`

## Layout

Each metric is one figure with three panels: vLLM, InfiniGen, and KVCore. Within each panel, bars compare Llama-3.1, Mistral, and Qwen3 at the same batch size and output length.
