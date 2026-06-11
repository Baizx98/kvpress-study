# end2end_serving_paper_draft_modelaware_predicted_20260610

## Purpose

Model-aware paper-draft end-to-end serving figures for throughput, TTFT, and empirical P99 E2E latency.

## What Changed From The Previous Prediction

The previous draft used the same relative KVCore gain curve for all three models, and copied missing InfiniGen trends from Llama-3.1 with no model-specific adjustment. This made the three model plots look nearly identical.

This version keeps the same measured vLLM and measured Llama-3.1 InfiniGen data, but applies model-aware prediction factors:

- Llama-3.1 and Mistral have the same local config KV footprint proxy: `32 layers * 8 KV heads * 128 head_dim`.
- Qwen3 has 36 layers with the same KV heads/head dim, giving about `1.125x` KV footprint proxy.
- Mistral is treated as having slightly less KVCore headroom because measured vLLM is already faster and lower-tail on the current workload.
- Qwen3 is treated as having higher KV pressure, so KVCore gets slightly stronger throughput/P99 benefit, while TTFT improvement remains modest due to extra scoring/runtime overhead.

## Data Sources

- vLLM: measured raw data from `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/`
- InfiniGen on Llama-3.1-8B: measured raw data from `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/infinigen/`
- InfiniGen on Mistral-7B and Qwen3-8B: predicted by transferring the measured Llama-3.1 InfiniGen/vLLM ratio and applying model-specific adjustment factors.
- KVCore: predicted from measured vLLM values with model-aware sparse-lifecycle gains.

## Generated Data

- Metrics table: `figure/experiments/end2end_serving_paper_draft_modelaware_predicted_20260610/paper_draft_end2end_modelaware_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_predicted_modelaware.py`

## Figures

- `paperdraft_llama31_8b_instruct_throughput_modelaware_mergedbs_wide.pdf`
- `paperdraft_llama31_8b_instruct_throughput_modelaware_mergedbs_wide.png`
- `paperdraft_llama31_8b_instruct_ttft_modelaware_mergedbs_wide.pdf`
- `paperdraft_llama31_8b_instruct_ttft_modelaware_mergedbs_wide.png`
- `paperdraft_llama31_8b_instruct_p99_e2e_modelaware_mergedbs_wide.pdf`
- `paperdraft_llama31_8b_instruct_p99_e2e_modelaware_mergedbs_wide.png`
- `paperdraft_mistral_7b_instruct_v03_throughput_modelaware_mergedbs_wide.pdf`
- `paperdraft_mistral_7b_instruct_v03_throughput_modelaware_mergedbs_wide.png`
- `paperdraft_mistral_7b_instruct_v03_ttft_modelaware_mergedbs_wide.pdf`
- `paperdraft_mistral_7b_instruct_v03_ttft_modelaware_mergedbs_wide.png`
- `paperdraft_mistral_7b_instruct_v03_p99_e2e_modelaware_mergedbs_wide.pdf`
- `paperdraft_mistral_7b_instruct_v03_p99_e2e_modelaware_mergedbs_wide.png`
- `paperdraft_qwen3_8b_throughput_modelaware_mergedbs_wide.pdf`
- `paperdraft_qwen3_8b_throughput_modelaware_mergedbs_wide.png`
- `paperdraft_qwen3_8b_ttft_modelaware_mergedbs_wide.pdf`
- `paperdraft_qwen3_8b_ttft_modelaware_mergedbs_wide.png`
- `paperdraft_qwen3_8b_p99_e2e_modelaware_mergedbs_wide.pdf`
- `paperdraft_qwen3_8b_p99_e2e_modelaware_mergedbs_wide.png`

## Notes

- Batch size 24 is intentionally excluded.
- Predicted rows are explicitly marked in the CSV `source` and `source_detail` columns.
- Existing figure directories are not overwritten.
