# kvcore_preemption_sim_a6000_large_bs_long_out_20260610

## Purpose

Request preemption figures using the A6000 large-batch long-output vLLM motivation data and simulated KVCore residual preemption.

## Source Data

- Source root: `/home10T/bzx/workspace/vllm-test/experiment_results/a6000_motivation_20260530_215742_large_bs_long_out`
- vLLM summary: `/home10T/bzx/workspace/vllm-test/experiment_results/a6000_motivation_20260530_215742_large_bs_long_out/analysis/a6000_preemption_summary.csv`
- Model: `/Tan/model/Llama-3.1-8B-Instruct`
- Workload: RULER 8192 prompts truncated to 3072 input tokens.
- Sweep: batch sizes `12`, `16`, `20`, `24`; output lengths `1K`, `2K`, `4K`, `6K`; KV budget `10 GB`.
- Metrics table: `figure/experiments/kvcore_preemption_sim_a6000_large_bs_long_out_20260610/kvcore_preemption_sim_a6000_metrics.csv`
- Plotting script: `figure/plot_kvcore_preemption_sim_a6000.py`

## Figures

- `a6000_preemption_combined_bar_reduction.pdf`
- `a6000_preemption_combined_bar_reduction.png`
- `a6000_preemption_reduction_percent.pdf`
- `a6000_preemption_reduction_percent.png`
- `a6000_preemptions_per_100_requests.pdf`
- `a6000_preemptions_per_100_requests.png`

## Simulation Assumption

KVCore avoids most but not all vLLM request preemptions by reducing dynamic GPU KV pressure. Residual preemption increases slightly under higher vLLM pressure, larger batch size, and longer output length. Points where vLLM has zero preemptions have undefined reduction and are omitted from the reduction-percent line.
