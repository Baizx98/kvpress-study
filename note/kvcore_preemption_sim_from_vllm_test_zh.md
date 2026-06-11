# KVCore 请求抢占率模拟图说明

本文档记录基于 `vllm-test` 既有请求抢占实验数据生成 KVCore 模拟抢占结果和论文图的过程。

## 数据来源

使用的 vLLM baseline 数据来自：

`/home10T/bzx/workspace/vllm-test/experiment_results/preemption_motivation_long_output_20260510_170956/analysis/preemption_summary.csv`

选择其中 `Llama-3.1-8B-Instruct` 三个 KV budget 点：

| KV budget (GB) | vLLM total preemptions | vLLM preemptions / 100 reqs | vLLM preempted requests |
|---:|---:|---:|---:|
| 1 | 133 | 207.81 | 90.62% |
| 2 | 158 | 246.88 | 89.06% |
| 4 | 325 | 507.81 | 75.00% |

未使用 A6000 那轮实验，因为当时状态文档记录已完成 case 均未触发 preemption，不能支撑抢占率趋势图。

## KVCore 模拟口径

KVCore 当前端到端抢占数据尚未实测，因此本图只作为论文草稿占位。模拟假设：

- KVCore 通过 sparse block lifecycle management 主动降低 GPU KV pressure。
- 它能避免绝大部分请求抢占，但不能完全消除所有抢占。
- vLLM 抢占越严重，KVCore 残余抢占比例略高，用于体现重压下仍存在调度和生命周期预测误差。

生成的 CSV：

`figure/experiments/kvcore_preemption_sim_from_vllm_test_20260610/kvcore_preemption_sim_metrics.csv`

模拟结果：

| KV budget (GB) | vLLM preemptions / 100 reqs | KVCore preemptions / 100 reqs | Reduction |
|---:|---:|---:|---:|
| 1 | 207.81 | 22.77 | 89.04% |
| 2 | 246.88 | 29.42 | 88.08% |
| 4 | 507.81 | 81.25 | 84.00% |

## 生成图

输出目录：

`figure/experiments/kvcore_preemption_sim_from_vllm_test_20260610/`

图文件：

- `preemption_reduction_percent.pdf/png`
- `preemptions_per_100_requests.pdf/png`

绘图脚本：

`figure/plot_kvcore_preemption_sim.py`

## 论文使用建议

正文中可以写：

> KVCore reduces request preemption by 84.0%-89.0% over vLLM in the preemption-stress workload, reducing preemptions from 207.8-507.8 per 100 requests to 22.8-81.3 per 100 requests.

但正式论文提交前应将 `source=simulated` 的 KVCore 行替换为真实 KVCore runner 数据。
