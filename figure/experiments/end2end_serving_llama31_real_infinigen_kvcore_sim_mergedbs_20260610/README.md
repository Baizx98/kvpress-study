# end2end_serving_llama31_real_infinigen_kvcore_sim_mergedbs_20260610

## Purpose

Wide single-axis Llama-3.1-8B end-to-end serving figures with measured vLLM and InfiniGen data plus simulated KVCore placeholder data.

## Source Data

- vLLM raw: `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/`
- InfiniGen raw: `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/infinigen/`
- Metrics CSV: `figure/experiments/end2end_serving_llama31_real_infinigen_kvcore_sim_mergedbs_20260610/llama31_real_infinigen_kvcore_sim_mergedbs_metrics.csv`
- Plotting script: `figure/plot_end2end_llama31_infinigen_kvcore_sim_mergedbs.py`

## Figures

- `llama31_decode_throughput_mergedbs_wide.pdf`
- `llama31_decode_throughput_mergedbs_wide.png`
- `llama31_median_e2e_mergedbs_wide.pdf`
- `llama31_median_e2e_mergedbs_wide.png`
- `llama31_p99_e2e_mergedbs_wide.pdf`
- `llama31_p99_e2e_mergedbs_wide.png`
- `llama31_median_ttft_mergedbs_wide.pdf`
- `llama31_median_ttft_mergedbs_wide.png`
- `llama31_median_tpot_mergedbs_wide.pdf`
- `llama31_median_tpot_mergedbs_wide.png`
- `llama31_gpu_peak_memory_mergedbs_wide.pdf`
- `llama31_gpu_peak_memory_mergedbs_wide.png`

## Plot Layout

Each metric uses one long boxed plotting area instead of three batch-size subplots. Batch sizes `1`, `8`, and `16` are separated by lightly shaded background regions, and each region repeats output lengths `1k`, `2k`, and `6k`. The x-axis label appears once per figure.

## Notes

- vLLM and InfiniGen are measured raw results.
- `KVCore (sim.)` is simulated placeholder data. The simulation is anchored to measured vLLM and gives larger improvements as batch size and output length increase.
- Batch size 24 is intentionally excluded.
- Existing figure directories are not overwritten.
