# vllm_formal_pg19_in6k

## Purpose

Measured vLLM-only figures for the formal PG19 in6k serving sweep.
The three models are plotted separately. Within each model figure, batch size is the x-axis and output length is encoded by color and marker.

## Source Data

- Raw JSONL: `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/`
- Metrics CSV: `figure/experiments/vllm_formal_pg19_in6k/vllm_formal_pg19_in6k_metrics.csv`
- Plotting script: `figure/plot_vllm_formal_pg19_in6k.py`

## Figures

- `vllm_pg19_in6k_llama31_8b_instruct_by_bs_output.pdf`
- `vllm_pg19_in6k_llama31_8b_instruct_by_bs_output.png`
- `vllm_pg19_in6k_qwen3_8b_by_bs_output.pdf`
- `vllm_pg19_in6k_qwen3_8b_by_bs_output.png`
- `vllm_pg19_in6k_mistral_7b_instruct_v03_by_bs_output.pdf`
- `vllm_pg19_in6k_mistral_7b_instruct_v03_by_bs_output.png`
- `vllm_pg19_in6k_all_models_tpot_heatmap.pdf`
- `vllm_pg19_in6k_all_models_tpot_heatmap.png`

## Metric Definitions

- Median TTFT: median `ttft_s` over completed requests.
- Median E2E latency: median `e2e_latency_s` over completed requests.
- Median TPOT: median `tpot_ms` over completed requests.
- Decode throughput: `sum(actual_output_len) / (max(finish_time_s) - min(submit_time_s))` for completed requests in the point.

## Notes

- Warmup requests are excluded by the runner and therefore excluded from these plots.
- Failed points are shown as `OOM / failed` markers instead of being interpolated.
- The Llama out2k bs8 point uses the successful rerun raw file from 2026-06-09.
