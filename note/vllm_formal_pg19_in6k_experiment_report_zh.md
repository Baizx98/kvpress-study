# vLLM formal PG19 in6k 端到端实验结果汇报

生成时间：2026-06-09

## 实验范围

本次汇报基于已经完成的 vLLM formal sweep，不包含早期 smoke 数据。2026-06-09 对 `Llama-3.1-8B-Instruct / out2k / bs8` 做了单点重跑，替换了原先因启动时显存不足导致的 invalid failure。

- 系统：vLLM
- 硬件目标：单卡 NVIDIA RTX A6000
- Workload：PG19
- 输入长度：in6k
- 输出长度：out1k / out2k / out6k
- Batch size：1 / 8 / 16 / 24
- 模型：Llama-3.1-8B-Instruct、Qwen3-8B、Mistral-7B-Instruct-v0.3
- 随机种子：2026
- warmup：每个点 warmup 1 次
- 运行时间：2026-06-08T12:59:31Z 到 2026-06-09T01:38:08Z

关键配置从日志确认：

- `enforce_eager=True`，日志显示 CUDAGraph disabled under eager mode。
- `tensor_parallel_size=1`，单 GPU。
- `quantization=None`。
- `speculative_config=None`。
- `enable_prefix_caching=False`。
- `enable_chunked_prefill=False`。
- attention backend 为 `TRITON_ATTN`，不是 FlashAttention / FlashInfer backend。

## 完成情况

总计 36 个 formal 点，其中 33 个完成，3 个失败。

- 总请求数：1296
- 成功请求数：1152
- 失败请求数：144
- 成功点数：33 / 36
- 失败点数：3 / 36

失败点如下：

| 模型 | 输出长度 | BS | 请求数 | 成功数 | 失败原因 |
|---|---:|---:|---:|---:|---|
| Llama-3.1-8B-Instruct | 6k | 24 | 48 | 0 | CUDA OOM，profile / KV cache 初始化阶段申请 15.75 GiB 失败 |
| Qwen3-8B | 6k | 24 | 48 | 0 | CUDA OOM，profile / KV cache 初始化阶段申请 13.50 GiB 失败 |
| Mistral-7B-Instruct-v0.3 | 6k | 24 | 48 | 0 | CUDA OOM，profile / KV cache 初始化阶段申请 15.75 GiB 失败 |

原始 sweep 中的 Llama out2k bs8 不是模型本身的必然 OOM；日志显示启动时 GPU free memory 只有 39.68 GiB，低于 `gpu_memory_utilization=0.9` 对应的 42.59 GiB 需求。该点已在清空 A6000 后重跑成功，正式汇总中按重跑结果计入成功点。

out6k bs24 的三个模型失败更像真实容量边界：失败发生在 vLLM profile / KV cache 初始化阶段，日志中有明确 CUDA OOM。

## 中位数结果

指标为成功请求的 median。失败点记为 NA。

### Llama-3.1-8B-Instruct

| 输出长度 | BS | 成功/总数 | TTFT (s) | E2E latency (s) | TPOT (ms) | GPU peak memory (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 1 | 32/32 | 1.761 | 29.917 | 27.547 | 43.274 |
| 1k | 8 | 32/32 | 18.259 | 75.629 | 56.118 | 45.711 |
| 1k | 16 | 32/32 | 26.967 | 82.938 | 55.005 | 43.604 |
| 1k | 24 | 48/48 | 46.116 | 168.344 | 119.597 | 34.901 |
| 2k | 1 | 32/32 | 1.764 | 58.150 | 27.559 | 43.157 |
| 2k | 8 | 32/32 | 13.414 | 97.169 | 40.920 | 41.907 |
| 2k | 16 | 32/32 | 26.974 | 180.374 | 77.001 | 46.803 |
| 2k | 24 | 48/48 | 113.057 | 333.854 | 107.804 | 31.463 |
| 6k | 1 | 32/32 | 1.758 | 173.391 | 27.944 | 42.751 |
| 6k | 8 | 32/32 | 13.602 | 286.882 | 44.501 | 38.786 |
| 6k | 16 | 32/32 | 15.389 | 783.319 | 123.933 | 30.138 |
| 6k | 24 | 0/48 | NA | NA | NA | NA |

### Qwen3-8B

| 输出长度 | BS | 成功/总数 | TTFT (s) | E2E latency (s) | TPOT (ms) | GPU peak memory (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 1 | 32/32 | 1.743 | 30.991 | 28.591 | 43.278 |
| 1k | 8 | 32/32 | 13.835 | 55.234 | 40.576 | 43.194 |
| 1k | 16 | 32/32 | 27.770 | 85.876 | 56.913 | 44.069 |
| 1k | 24 | 48/48 | 47.125 | 178.299 | 127.237 | 35.925 |
| 2k | 1 | 32/32 | 1.759 | 59.539 | 28.230 | 43.146 |
| 2k | 8 | 32/32 | 14.005 | 100.849 | 42.419 | 42.413 |
| 2k | 16 | 32/32 | 28.016 | 203.349 | 85.726 | 42.423 |
| 2k | 24 | 48/48 | 113.444 | 365.325 | 122.417 | 31.323 |
| 6k | 1 | 32/32 | 1.751 | 177.513 | 28.610 | 42.755 |
| 6k | 8 | 32/32 | 13.942 | 293.089 | 45.677 | 39.717 |
| 6k | 16 | 32/32 | 15.854 | 772.605 | 123.119 | 34.215 |
| 6k | 24 | 0/48 | NA | NA | NA | NA |

### Mistral-7B-Instruct-v0.3

| 输出长度 | BS | 成功/总数 | TTFT (s) | E2E latency (s) | TPOT (ms) | GPU peak memory (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 1 | 32/32 | 1.748 | 27.607 | 25.305 | 43.288 |
| 1k | 8 | 32/32 | 13.005 | 51.648 | 37.529 | 43.352 |
| 1k | 16 | 32/32 | 25.956 | 79.474 | 51.837 | 44.149 |
| 1k | 24 | 48/48 | 23.198 | 135.317 | 109.455 | 36.585 |
| 2k | 1 | 32/32 | 1.749 | 53.686 | 25.384 | 43.171 |
| 2k | 8 | 32/32 | 13.449 | 91.901 | 38.462 | 42.539 |
| 2k | 16 | 32/32 | 26.561 | 145.618 | 58.152 | 43.691 |
| 2k | 24 | 48/48 | 117.894 | 296.247 | 87.171 | 33.346 |
| 6k | 1 | 32/32 | 1.733 | 159.489 | 25.681 | 42.765 |
| 6k | 8 | 32/32 | 13.424 | 271.312 | 42.130 | 38.784 |
| 6k | 16 | 32/32 | 18.142 | 677.916 | 106.924 | 31.940 |
| 6k | 24 | 0/48 | NA | NA | NA | NA |

## 结果是否合理

整体趋势基本合理：

1. 对固定 batch size，输出长度从 1k 增加到 2k / 6k 时，E2E latency 近似随 decode token 数增长。bs1 下 TPOT 基本稳定在 25-29 ms，说明单请求 decode 阶段较稳定。
2. batch size 增大后，TTFT 明显上升，这是长 prompt prefill 和批处理排队共同造成的。bs24 的 TTFT 可到 100 s 量级，符合 in6k 长输入下单卡处理大 batch 的现象。
3. bs8 通常比 bs1 的 TPOT 更低或相近，说明 batch 提高了 decode 吞吐利用率；但 bs16/bs24 在 out6k 或长输出下 TPOT 明显升高，说明已经接近显存和调度容量边界。
4. out6k bs24 三个模型都 OOM，和 48 GiB A6000 上 in6k + out6k + bs24 的 KV cache 需求一致，不能把这个点作为成功性能数据。

需要谨慎解释的地方：

- `gpu_peak_memory_gb` 在 bs24 某些点反而低于 bs8/bs16，这不代表 bs24 更省显存。更可能是 vLLM 在不同 profile / cache block 分配 / failed-capacity 边界下的可用 KV cache 与峰值统计口径变化。正式论文图里建议不要只用这个字段推导显存效率。
- Llama out2k bs8 的原失败是启动时已有显存占用导致的 invalid failure；已重跑成功，正式分析应使用重跑后的 raw/metadata。
- 当前只是 vLLM 单系统结果，还不能推出 KVCore / InfiniGen 的相对优势。

## 结论

vLLM formal PG19 in6k sweep 已经给出一个可用 baseline：33/36 个配置完成，主要容量边界出现在 out6k bs24。对论文或系统比较而言，建议把 33 个成功点作为 vLLM baseline，并把 out6k bs24 标为 A6000 单卡容量不可行点。

## 证据路径

- 聚合结果：`evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/logs/vllm/vllm_formal_pg19_in6k_aggregate_for_report.json`
- sweep summary：`evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/logs/vllm/vllm_formal_pg19_in6k_summary.json`
- sweep status：`evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/logs/vllm/vllm_formal_pg19_in6k_status.jsonl`
- raw JSONL：`evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/`
- vLLM logs：`evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/logs/vllm/`
