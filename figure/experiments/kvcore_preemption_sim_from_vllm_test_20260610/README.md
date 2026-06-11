# kvcore_preemption_sim_from_vllm_test_20260610

## Purpose

Paper-style request preemption figures using measured vLLM preemption data and simulated KVCore residual preemption.

## Source Data

- vLLM summary: `/home10T/bzx/workspace/vllm-test/experiment_results/preemption_motivation_long_output_20260510_170956/analysis/preemption_summary.csv`
- Model: `Llama-3.1-8B-Instruct`
- Workload: 64 offline requests, input length 512, forced output length 1536.
- Metrics table: `figure/experiments/kvcore_preemption_sim_from_vllm_test_20260610/kvcore_preemption_sim_metrics.csv`
- Plotting script: `figure/plot_kvcore_preemption_sim.py`

## Figures

- `preemption_reduction_percent.pdf`
- `preemption_reduction_percent.png`
- `preemptions_per_100_requests.pdf`
- `preemptions_per_100_requests.png`

## Simulation Assumption

KVCore avoids most request preemptions by reducing dynamic GPU KV pressure before vLLM-style preemption becomes necessary. The residual preemption fraction is pressure-dependent, so heavier vLLM preemption points retain slightly more residual KVCore preemption. The simulated KVCore rows are explicitly marked as `source=simulated` in the CSV.
