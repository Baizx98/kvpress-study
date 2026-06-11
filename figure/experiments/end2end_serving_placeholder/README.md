# end2end_serving_placeholder

## Purpose

Placeholder figures for the end-to-end serving comparison among vLLM, InfiniGen, and KVCore.
vLLM points are measured from the formal PG19 in6k raw JSONL outputs.
InfiniGen and KVCore points are simulated placeholders for figure layout and narrative planning only.

## Source Data

- Measured vLLM raw: `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/`
- Plotting intermediate: `figure/experiments/end2end_serving_placeholder/placeholder_metrics.csv`
- Plotting script: `figure/plot_end2end_serving_placeholder.py`

## Figures

- `throughput_vs_batch_by_model.pdf`
- `throughput_vs_batch_by_model.png`
- `median_e2e_vs_batch_by_model.pdf`
- `median_e2e_vs_batch_by_model.png`
- `p99_e2e_vs_batch_by_model.pdf`
- `p99_e2e_vs_batch_by_model.png`
- `tpot_vs_batch_by_model.pdf`
- `tpot_vs_batch_by_model.png`
- `throughput_vs_batch_llama31.pdf`
- `throughput_vs_batch_llama31.png`
- `p99_latency_llama31.pdf`
- `p99_latency_llama31.png`
- `ttft_tpot_llama31.pdf`
- `ttft_tpot_llama31.png`
- `feasibility_matrix_all_models.pdf`
- `feasibility_matrix_all_models.png`

## Metric Definitions

- Decode throughput: `sum(actual_output_len) / (max(finish_time_s) - min(submit_time_s))`, measured-run only, warmup excluded.
- Request throughput: `num_completed / (max(finish_time_s) - min(submit_time_s))`, measured-run only, warmup excluded.
- P99 latency: empirical P99 over 32 or 48 requests per point.
- The `*_by_model` figures use rows for models, columns for output lengths, and batch size on the x-axis.

## Notes

- `KVCore (sim.)` and `InfiniGen (sim.)` are not real end-to-end results.
- vLLM failed points remain marked as OOM rather than interpolated.
- The Llama out2k bs8 point uses the successful rerun raw file from 2026-06-09.
