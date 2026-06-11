# end2end_serving_paper_draft_predicted_20260610

## Purpose

Paper-draft end-to-end serving figures for throughput, TTFT, and empirical P99 E2E latency across Llama-3.1-8B, Mistral-7B, and Qwen3-8B.

## Data Sources

- vLLM: measured raw data from `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/`
- InfiniGen on Llama-3.1-8B: measured raw data from `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/infinigen/`
- InfiniGen on Mistral-7B and Qwen3-8B: predicted by transferring the measured Llama-3.1 InfiniGen/vLLM ratio to each model's measured vLLM value.
- KVCore: predicted from measured vLLM values with mechanism-based gains. The prediction assumes sparse block lifecycle management reduces most but not all request preemptions, with larger gains under larger batch sizes and longer outputs.

## Generated Data

- Metrics table: `figure/experiments/end2end_serving_paper_draft_predicted_20260610/paper_draft_end2end_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_predicted.py`

## Figures

- `paperdraft_llama31_8b_instruct_throughput_mergedbs_wide.pdf`
- `paperdraft_llama31_8b_instruct_throughput_mergedbs_wide.png`
- `paperdraft_llama31_8b_instruct_ttft_mergedbs_wide.pdf`
- `paperdraft_llama31_8b_instruct_ttft_mergedbs_wide.png`
- `paperdraft_llama31_8b_instruct_p99_e2e_mergedbs_wide.pdf`
- `paperdraft_llama31_8b_instruct_p99_e2e_mergedbs_wide.png`
- `paperdraft_mistral_7b_instruct_v03_throughput_mergedbs_wide.pdf`
- `paperdraft_mistral_7b_instruct_v03_throughput_mergedbs_wide.png`
- `paperdraft_mistral_7b_instruct_v03_ttft_mergedbs_wide.pdf`
- `paperdraft_mistral_7b_instruct_v03_ttft_mergedbs_wide.png`
- `paperdraft_mistral_7b_instruct_v03_p99_e2e_mergedbs_wide.pdf`
- `paperdraft_mistral_7b_instruct_v03_p99_e2e_mergedbs_wide.png`
- `paperdraft_qwen3_8b_throughput_mergedbs_wide.pdf`
- `paperdraft_qwen3_8b_throughput_mergedbs_wide.png`
- `paperdraft_qwen3_8b_ttft_mergedbs_wide.pdf`
- `paperdraft_qwen3_8b_ttft_mergedbs_wide.png`
- `paperdraft_qwen3_8b_p99_e2e_mergedbs_wide.pdf`
- `paperdraft_qwen3_8b_p99_e2e_mergedbs_wide.png`

## Plot Layout

Each figure is one long boxed plotting area. Batch sizes `1`, `8`, and `16` are separated by lightly shaded background regions. Within each batch-size region, bars compare systems at output lengths `1k`, `2k`, and `6k`.

## Notes

- Batch size 24 is intentionally excluded to match the current paper-draft figure layout.
- The figures are for draft visualization only; predicted rows are explicitly marked in the CSV `source` and `source_detail` columns.
- Existing figure directories are not overwritten.
