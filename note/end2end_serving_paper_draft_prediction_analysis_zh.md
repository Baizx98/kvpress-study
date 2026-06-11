# 端到端 Serving 草稿图预测数据分析

本文档记录论文草稿中 `vLLM / InfiniGen / KVCore` 端到端 serving 图的数据来源、推断规则和指标影响判断。

## 1. 论文依据

用户提到最新论文位于 `note/main.tex`，但当前 `kvpress-study/note/` 下未找到该文件；当前实际可读的论文主文为 `note/ATC26.tex`。本次分析依据 `note/ATC26.tex` 中的 KVCore 叙事：

- vLLM-style runtime 在长输出和大 batch 下会因为 KV cache 持续增长触发 request preemption。
- KVCore 将 block-level sparse score 提升为 runtime lifecycle hint，用于 GPU residency、CPU offload、prefetch 和 permanent eviction。
- KVCore 的主要收益来自提前降低 GPU KV pressure、减少大部分但不是全部 preemption，并降低 decode 阶段冗余 KV 访问。

因此，对三个目标指标的合理影响应为：

1. **Decode throughput**：应有最明显提升。压力越大，即 batch size 越大、output length 越长，KVCore 越能通过减少 KV resident footprint 和 preemption 提升有效 token serving capacity。
2. **TTFT**：应只有温和改善。TTFT 主要由长 prompt prefill、batch 形成和首轮调度决定；KVCore 的 block sparse lifecycle 主要作用在 decode 阶段和后续 KV pressure，因此不能假设 TTFT 大幅下降。
3. **P99 E2E latency**：应明显改善，且改善幅度大于 median/TTFT。P99 对 preemption、swap/recompute、排队抖动最敏感；KVCore 避免绝大部分 preemption 后，tail latency 应显著下降。但由于不能保证完全避免所有 preemption，预测不应把 P99 画成接近理想线性下界。

## 2. 数据来源与推断规则

生成的数据总表：

`figure/experiments/end2end_serving_paper_draft_predicted_20260610/paper_draft_end2end_metrics_table.csv`

绘图脚本：

`figure/plot_end2end_paper_draft_predicted.py`

数据表共 81 行，覆盖：

- 3 个模型：Llama-3.1-8B、Mistral-7B、Qwen3-8B
- 3 个 batch size：1、8、16
- 3 个 output length：1k、2k、6k
- 3 个系统：vLLM、InfiniGen、KVCore

数据来源标记如下：

| 系统 | 行数 | 来源 |
|---|---:|---|
| vLLM | 27 | 实测 raw JSONL |
| InfiniGen / Llama-3.1 | 9 | 实测 raw JSONL |
| InfiniGen / Mistral+Qwen | 18 | 用同配置下 Llama 的 `InfiniGen / vLLM` 比值迁移到对应模型的 vLLM 实测值 |
| KVCore | 27 | 基于 vLLM 实测值和论文机制的预测数据 |

### 2.1 InfiniGen 缺失数据

InfiniGen 在 Llama-3.1 上已有完整实测数据。由于 vLLM 的三个模型整体变化趋势接近，本次对 Mistral 和 Qwen 的 InfiniGen 数据采用比值迁移：

```text
InfiniGen(model, config, metric)
  = vLLM(model, config, metric)
    * InfiniGen(Llama, config, metric) / vLLM(Llama, config, metric)
```

这样保留了每个模型自身的 vLLM 绝对性能水平，同时继承 Llama 上实际测到的 InfiniGen 相对趋势。

### 2.2 KVCore 预测数据

KVCore 当前尚未完成端到端实测。本次预测遵循三条约束：

1. 吞吐提升随 batch size 和 output length 增加而增强。
2. TTFT 只做温和改善，不夸大 prefill 侧收益。
3. P99 E2E 改善比 TTFT 更明显，但保留残余 preemption / transfer / scheduling 抖动。

当前脚本采用的 KVCore 相对 vLLM 范围：

| 指标 | 预测范围 | 中位数 | 解释 |
|---|---:|---:|---|
| Decode throughput speedup | 1.28x - 2.28x | 1.78x | 对齐论文中 1.5x-2.3x throughput 叙事，低压力点收益较小 |
| TTFT improvement | 1.04x - 1.22x | 1.12x | 只体现较少排队/首轮压力，不声称大幅降低 prefill |
| P99 E2E improvement | 1.19x - 2.00x | 1.49x | 体现 preemption 大幅减少，但不是全部消除 |

## 3. 当前数据趋势

vLLM 实测值范围：

| 指标 | min | median | max |
|---|---:|---:|---:|
| Decode throughput (tok/s) | 32.777 | 149.248 | 224.912 |
| Median TTFT (s) | 1.733 | 13.602 | 28.016 |
| P99 E2E latency (s) | 27.898 | 97.357 | 790.056 |

InfiniGen 当前 Llama 实测值迁移后，在这批长请求上相对 vLLM 较慢：

| 指标 | min | median | max |
|---|---:|---:|---:|
| Throughput vs. vLLM | 0.160x | 0.299x | 0.357x |
| TTFT improvement vs. vLLM | 0.300x | 0.552x | 0.767x |
| P99 E2E improvement vs. vLLM | 0.128x | 0.229x | 0.335x |

这里的 `improvement vs. vLLM` 对 latency 指标定义为 `vLLM / system`，因此小于 1 表示比 vLLM 更慢。

KVCore 预测值相对 vLLM：

| 指标 | min | median | max |
|---|---:|---:|---:|
| Throughput vs. vLLM | 1.280x | 1.780x | 2.280x |
| TTFT improvement vs. vLLM | 1.042x | 1.124x | 1.220x |
| P99 E2E improvement vs. vLLM | 1.190x | 1.493x | 2.000x |

## 4. 生成的九张组图

输出目录：

`figure/experiments/end2end_serving_paper_draft_predicted_20260610/`

九张组图均同时生成 PDF 和 PNG：

| 模型 | 指标 | 文件前缀 |
|---|---|---|
| Llama-3.1-8B | Decode throughput | `paperdraft_llama31_8b_instruct_throughput_mergedbs_wide` |
| Llama-3.1-8B | Median TTFT | `paperdraft_llama31_8b_instruct_ttft_mergedbs_wide` |
| Llama-3.1-8B | P99 E2E latency | `paperdraft_llama31_8b_instruct_p99_e2e_mergedbs_wide` |
| Mistral-7B | Decode throughput | `paperdraft_mistral_7b_instruct_v03_throughput_mergedbs_wide` |
| Mistral-7B | Median TTFT | `paperdraft_mistral_7b_instruct_v03_ttft_mergedbs_wide` |
| Mistral-7B | P99 E2E latency | `paperdraft_mistral_7b_instruct_v03_p99_e2e_mergedbs_wide` |
| Qwen3-8B | Decode throughput | `paperdraft_qwen3_8b_throughput_mergedbs_wide` |
| Qwen3-8B | Median TTFT | `paperdraft_qwen3_8b_ttft_mergedbs_wide` |
| Qwen3-8B | P99 E2E latency | `paperdraft_qwen3_8b_p99_e2e_mergedbs_wide` |

## 5. 使用限制

- 这些图是论文草稿占位图，不应在正式论文中表述为 KVCore 端到端实测结果。
- InfiniGen 的 Mistral/Qwen 数据是推断值，即使趋势合理，也需要后续实测替换。
- KVCore 的预测收益已尽量保持保守：吞吐和 P99 是主要改善，TTFT 只温和改善；但最终数值必须以后续完整 runner 的 raw JSONL 为准。
- 当前 P99 是 empirical P99，每个配置请求数有限，不等同于生产级大样本 tail latency。
