# 端到端 Serving 草稿图 Model-aware 预测更新

## 背景

上一版预测图中，三个模型的曲线/柱形结构看起来过于相似。检查后原因是：

- KVCore 对 Llama-3.1、Mistral、Qwen3 使用了完全相同的相对收益曲线。
- Mistral/Qwen3 的 InfiniGen 缺失数据使用 Llama-3.1 的 `InfiniGen / vLLM` 比值直接迁移，没有模型特异修正。
- 每个模型单独成图时，纵轴范围相近，视觉上进一步弱化了三模型的绝对差异。

## 新版修正

新增 model-aware 预测版本：

- 数据表：`figure/experiments/end2end_serving_paper_draft_modelaware_predicted_20260610/paper_draft_end2end_modelaware_metrics_table.csv`
- 九张系统对比图：`figure/experiments/end2end_serving_paper_draft_modelaware_predicted_20260610/`
- 三张模型诊断对比图：`figure/experiments/end2end_serving_paper_draft_modelaware_comparison_20260610/`
- 预测脚本：`figure/plot_end2end_paper_draft_predicted_modelaware.py`
- 诊断图脚本：`figure/plot_end2end_paper_draft_modelaware_comparison.py`

新版仍然保留以下实测数据：

- vLLM：三个模型全部使用实测 raw JSONL。
- InfiniGen：Llama-3.1 使用实测 raw JSONL。

新版对缺失数据做如下修正：

- InfiniGen Mistral/Qwen3：仍以 Llama-3.1 的 `InfiniGen / vLLM` 实测比值为基础，但加入模型修正因子。
- KVCore：以 vLLM 实测值为基础，加入模型结构和 KV pressure 修正。

## 模型依据

本地模型 config 显示：

| 模型 | Layers | KV heads | Head dim | KV footprint proxy |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 32 | 8 | 128 | 1.000x |
| Mistral-7B | 32 | 8 | 128 | 1.000x |
| Qwen3-8B | 36 | 8 | 128 | 1.125x |

因此新版假设：

- Mistral 的 vLLM 实测 throughput 更高、P99 更低，说明当前 workload 下 baseline 已经较稳，所以 KVCore 的相对 headroom 略小。
- Qwen3 的 KV footprint proxy 更高，长输出和大 batch 下 KV pressure 更强，因此 KVCore 对 throughput/P99 的收益略强，但 TTFT 改善仍保持温和。

## 新版相对收益范围

KVCore 相对 vLLM：

| 模型 | Throughput speedup median | TTFT improvement median | P99 E2E improvement median |
|---|---:|---:|---:|
| Llama-3.1-8B | 1.780x | 1.124x | 1.493x |
| Mistral-7B | 1.673x | 1.105x | 1.439x |
| Qwen3-8B | 1.894x | 1.105x | 1.629x |

InfiniGen 相对 vLLM：

| 模型 | Throughput ratio median | TTFT improvement median | P99 E2E improvement median |
|---|---:|---:|---:|
| Llama-3.1-8B | 0.299x | 0.552x | 0.229x |
| Mistral-7B | 0.323x | 0.575x | 0.244x |
| Qwen3-8B | 0.275x | 0.521x | 0.212x |

对 latency 指标，`improvement` 定义为 `vLLM / system`，因此小于 1 表示比 vLLM 更慢。

## 使用限制

这仍然是论文草稿占位数据，不是 KVCore 的端到端实测结果。新版的价值是让草稿图的模型间趋势更合理，避免三个模型共享完全相同的相对曲线。正式论文应以后续 KVCore runner 的 raw JSONL 替换预测行。
