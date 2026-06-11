# 端到端系统性能实验结果记录

更新时间：2026-06-08

本文记录 `KVCore / vLLM / InfiniGen` 端到端 serving 对比实验的当前结果。当前完成了带 warmup 的 vLLM smoke/full-manifest run，不包含 KVCore、InfiniGen，也不应作为三系统性能结论。

## 1. Problem statement

需要验证三套系统后续都能读取同一份 manifest，并把 per-request raw metrics 和 run-level metadata 写回 `kvpress-study` 的统一结果目录。当前阶段只验证 vLLM 路径是否打通。

## 2. Hypothesis

如果 runner 正确实现，vLLM 应能在单卡 NVIDIA RTX A6000 上完成 32 条 PG19 长请求 smoke workload，并输出符合统一 schema 的 raw JSONL：

- 每条请求固定 `prompt_token_len=4096`；
- 每条请求目标 `max_new_tokens=1024`；
- closed-loop batch size 为 1；
- raw JSONL 中所有必需字段齐全；
- metadata 中明确记录公平性相关配置和实际未关闭的组件。

## 3. Method

### Runner

vLLM runner 位于：

```text
/home10T/bzx/workspace/vllm-test/experiments/end2end_serving/run_vllm_manifest.py
```

runner 输入 manifest：

```text
/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/manifests/llama31_8b_instruct__pg19__in4k_out1k__bs1__seed2026.jsonl
```

输出 raw JSONL：

```text
/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/vllm__llama31_8b_instruct__pg19__in4k_out1k__bs1__seed2026.jsonl
```

输出 metadata：

```text
/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/environment/vllm__llama31_8b_instruct__pg19__in4k_out1k__bs1__seed2026.json
```

运行监控日志：

```text
/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/logs/vllm/vllm__llama31_8b_instruct__pg19__in4k_out1k__bs1__seed2026__warmup1.log
```

### Fairness config

本次 smoke 的 vLLM 配置：

| item | value |
|---|---|
| GPU | NVIDIA RTX A6000 |
| CUDA_VISIBLE_DEVICES | 2 |
| attention backend | TRITON_ATTN |
| CUDA graph | disabled via `enforce_eager=True` |
| tensor parallel | 1 |
| speculative decoding | disabled; no `speculative_config` |
| quantization | None |
| LoRA | disabled |
| prefix caching | disabled |
| chunked prefill | disabled |
| dtype | auto, runtime resolved to bf16 |
| max model len | 5120 |
| max num batched tokens | 5120 |
| max num seqs | 1 |
| warmup requests | 1 |

实际限制：

- vLLM 日志显示 attention backend 为 `AttentionBackendEnum.TRITON_ATTN`，没有使用 FlashAttention/FlashInfer attention backend。
- vLLM 日志显示 top-k/top-p sampling 使用了 FlashInfer sampler。该组件不是 attention backend，但属于无法完全关闭的实际启用状态，已写入 metadata 的 `config.sampling_backend_observed`。
- 本次 run 在 measured raw 之前执行了 1 条 warmup request。日志显示 Triton JIT 编译发生在 warmup 阶段，measured raw 不包含该 warmup 请求。
- vLLM 返回的 `actual_output_len` 为 1023，而不是 manifest 中的 `max_new_tokens=1024`；raw 中按协议记录实际输出长度。

## 4. Experiment

运行命令：

```bash
cd /home10T/bzx/workspace/vllm-test
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  .venv/bin/python experiments/end2end_serving/run_vllm_manifest.py --warmup-count 1
```

环境：

| item | value |
|---|---|
| vLLM git commit | c29f214405a6c057dbc451bc7aacca523777df15 |
| kvpress-study git commit | 3f08efcd3bbe0227c3d76a455a74184a6e28cd06 |
| torch | 2.11.0+cu129 |
| vLLM | 0.20.2rc1.dev133+gf9b9bf3bb.d20260509 |
| run start UTC | 2026-06-08T11:05:52.576252+00:00 |
| warmup start UTC | 2026-06-08T11:07:41.390398+00:00 |
| warmup end UTC | 2026-06-08T11:08:08.804740+00:00 |
| run end UTC | 2026-06-08T11:23:28.462233+00:00 |
| warmup duration | 27.414 s |
| warmup GPU peak memory | 43.196 GB |

## 5. Result

### Completion

| metric | value |
|---|---:|
| manifest requests | 32 |
| warmup requests | 1 |
| measured requests | 32 |
| raw rows | 32 |
| completed | 32 |
| failed | 0 |
| OOM | 0 |
| prompt token length | 4096 |
| actual output token length | 1023 |
| total generated tokens | 32736 |

### Latency and throughput

吞吐率按 raw 中 per-request 时间窗口聚合计算：

- `measured_wall_time_s = max(finish_time_s) - min(submit_time_s)`
- `output_throughput_toks_per_s = sum(actual_output_len) / measured_wall_time_s`
- `total_throughput_toks_per_s = sum(prompt_token_len + actual_output_len) / measured_wall_time_s`

| metric | value |
|---|---:|
| measured wall time | 919.645 s |
| output throughput | 35.596 tok/s |
| total throughput | 178.121 tok/s |
| e2e latency mean | 28.738 s |
| e2e latency P50 | 28.790 s |
| e2e latency P90 | 28.999 s |
| e2e latency P99 | 29.097 s |
| TTFT mean | 1.114 s |
| TTFT P50 | 1.115 s |
| TTFT P90 | 1.127 s |
| TTFT P99 | 1.131 s |
| TPOT mean | 27.030 ms |
| TPOT P50 | 27.075 ms |
| TPOT P90 | 27.288 ms |
| TPOT P99 | 27.366 ms |
| GPU peak memory mean | 43.196 GB |
| GPU peak memory max | 43.196 GB |

## 6. Conclusion

Confirmed findings:

1. vLLM smoke runner 已经打通：能读取统一 manifest，完成 32 条请求，并写回统一 raw JSONL 和 metadata。
2. raw JSONL schema 检查通过：要求的关键字段均存在，32 条均为 `status=completed`。
3. 单卡 A6000 上该 smoke 点没有 OOM，最大显存约 43.20 GB。
4. vLLM 已按要求关闭 CUDA graph、tensor parallel、quantization、LoRA、prefix caching 和 chunked prefill；attention backend 使用 `TRITON_ATTN`。
5. warmup 已经生效：Triton JIT 警告出现在 warmup 阶段，measured raw 从 warmup 之后开始写入。

## 7. Sanity check

整体判断：本次 smoke 指标是正常且合理的，但只能说明 vLLM runner 与测量链路打通，不能作为最终系统对比结论。

Reasoning:

1. Latency/TPOT 数值自洽。单条请求 `actual_output_len=1023`，TTFT 约 1.11 s，TPOT 约 27.03 ms。按 `TTFT + (1023 - 1) * TPOT` 估算，端到端延迟约 28.7 s，与 raw 中 `e2e_latency_s` mean 28.738 s 一致。
2. 吞吐率和 batch size 语义一致。当前是 closed-loop `bs=1`，因此 output throughput 约等于单请求 decode throughput，即 35.6 tok/s。它和 vLLM 日志中 36-37 tok/s 的 generation throughput 同量级。
3. TTFT 约 1.1 s 是合理的。prompt 长度 4096，attention backend 使用 `TRITON_ATTN`，并且 `enforce_eager=True` 关闭 CUDA graph/compile 优化；在 A6000 上该 TTFT 没有异常偏高。
4. 显存约 43.2 GB 不代表 4K+1K 单请求真实需要这么多 KV，而主要反映 vLLM 按 `gpu_memory_utilization=0.9` 预留 KV cache。metadata 中 vLLM 日志显示 GPU KV cache size 为 220,912 tokens，远大于单请求 5120 token 上限，因此 smoke 没有触及 KV cache 容量压力。
5. `actual_output_len=1023` 需要注意，但不是本次 smoke 的失败。vLLM 返回 token_ids 长度比 `max_new_tokens=1024` 少 1，raw 已按实际输出长度记录。正式聚合必须使用 `actual_output_len`，不要用目标输出长度代替。

Limits:

1. 这只是 vLLM smoke，不包含 KVCore/InfiniGen 对比，不能主张系统优劣。
2. vLLM 仍使用 FlashInfer top-k/top-p sampler；metadata 已记录，后续正式公平性设置需要决定是否接受该采样路径或进一步绕开。
3. `actual_output_len=1023`，后续聚合和画图必须用实际输出长度，不要假设等于 `max_new_tokens=1024`。
4. 这次只覆盖 Llama-3.1-8B-Instruct / PG19 / in4k_out1k / bs1 一个点，尚未覆盖 stage 1/2 的多模型、多长度、多 batch sweep。

## 8. Next steps

1. 实现 KVCore 和 InfiniGen runner，复用同一 manifest 和 raw schema。
2. 写聚合脚本生成 `request_metrics.csv` 和 `aggregate_metrics.csv`，避免手工计算指标。
3. 扩展 vLLM 到 stage 1 的多模型、多长度、多 batch sweep。
4. 对比三系统前，先确认三者对 EOS、max tokens、sampling backend、prefix/cache reuse 的实际语义是否一致。
