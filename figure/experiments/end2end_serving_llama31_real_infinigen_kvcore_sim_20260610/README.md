# end2end_serving_llama31_real_infinigen_kvcore_sim_20260610

## Purpose

Single-column Llama-3.1-8B end-to-end serving figures with measured vLLM and InfiniGen data plus simulated KVCore placeholder data.

## Source Data

- vLLM raw: `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/`
- InfiniGen raw: `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/infinigen/`
- Metrics CSV: `figure/experiments/end2end_serving_llama31_real_infinigen_kvcore_sim_20260610/llama31_real_infinigen_kvcore_sim_metrics.csv`
- Plotting script: `figure/plot_end2end_llama31_infinigen_kvcore_sim.py`

## Figures

- `llama31_decode_throughput_singlecol.pdf`
- `llama31_decode_throughput_singlecol.png`
- `llama31_median_e2e_singlecol.pdf`
- `llama31_median_e2e_singlecol.png`
- `llama31_p99_e2e_singlecol.pdf`
- `llama31_p99_e2e_singlecol.png`
- `llama31_median_ttft_singlecol.pdf`
- `llama31_median_ttft_singlecol.png`
- `llama31_median_tpot_singlecol.pdf`
- `llama31_median_tpot_singlecol.png`
- `llama31_gpu_peak_memory_singlecol.pdf`
- `llama31_gpu_peak_memory_singlecol.png`

## Plot Layout

Each figure is one single-column group plot. The three subplots are batch sizes `1`, `8`, and `16`.
Within each subplot, the x-axis is output length `1k`, `2k`, and `6k`, and bars compare vLLM, InfiniGen, and KVCore.

## Notes

- vLLM and InfiniGen are measured raw results.
- `KVCore (sim.)` is simulated placeholder data. The simulation is anchored to measured vLLM and gives larger improvements as batch size and output length increase.
- Batch size 24 is intentionally excluded.
- Existing figure directories are not overwritten.
