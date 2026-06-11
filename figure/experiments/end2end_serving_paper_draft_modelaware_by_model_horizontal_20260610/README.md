# end2end_serving_paper_draft_modelaware_by_model_horizontal_20260610

## Purpose

Double-column horizontal model-comparison figures for the model-aware paper-draft serving data.

## Source

- Metrics table: `figure/experiments/end2end_serving_paper_draft_modelaware_predicted_20260610/paper_draft_end2end_modelaware_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_modelaware_by_model_horizontal.py`

## Figures

- `modelaware_by_model_throughput.pdf`
- `modelaware_by_model_throughput.png`
- `modelaware_by_model_ttft.pdf`
- `modelaware_by_model_ttft.png`
- `modelaware_by_model_p99_e2e.pdf`
- `modelaware_by_model_p99_e2e.png`

## Layout

Each metric is one figure with three horizontal model panels: Llama-3.1-8B, Mistral-7B, and Qwen3-8B. Within each model panel, batch sizes `1`, `8`, and `16` are shaded regions; each region compares vLLM, InfiniGen, and KVCore at output lengths `1k`, `2k`, and `6k`.
