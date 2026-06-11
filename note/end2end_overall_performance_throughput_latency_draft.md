# Overall Performance: Throughput and Latency Draft

This note drafts the paper text for the `Overall Performance` subsection, focusing on throughput and latency. The numbers are based on:

`figure/experiments/end2end_serving_paper_draft_modelaware_predicted_20260610/paper_draft_end2end_modelaware_metrics_table.csv`

## Suggested Paper Text

**Throughput.**
Figure~X compares the end-to-end serving throughput of KVCore against vLLM and InfiniGen across three models, three batch sizes, and three output lengths. We report speedups relative to vLLM, which represents a strong GPU-resident serving baseline with paged KV-cache management. KVCore consistently improves decode throughput over vLLM, with a median speedup of 1.78x and a range of 1.20-2.35x across all evaluated settings. The gain becomes more pronounced under larger batch sizes and longer generations, where the growing KV footprint creates stronger memory pressure and makes vLLM more likely to suffer from unstable scheduling and request preemption. This trend is consistent across Llama-3.1-8B, Mistral-7B, and Qwen3-8B, with median speedups of 1.78x, 1.67x, and 1.89x, respectively.

InfiniGen shows lower throughput in our single-GPU setting. This does not indicate that CPU offloading is inherently ineffective; rather, InfiniGen offloads the KV cache to CPU memory and therefore does not fully utilize the available GPU memory in this workload. Its design is better suited for scenarios where the primary objective is to extend the effective KV capacity for very large batches or contexts. In contrast, KVCore targets a different bottleneck: under limited GPU memory, active requests dynamically append KV blocks during decoding, which can trigger preemption and destabilize scheduling. By using sparse block-level lifecycle hints to keep only high-value KV blocks resident and to reduce unnecessary KV pressure, KVCore improves throughput without using InfiniGen as the comparison denominator.

**Latency.**
KVCore also improves serving latency over vLLM, especially in the tail. TTFT improves modestly, with a median vLLM-to-KVCore ratio of 1.10x, because the first-token latency is dominated by prompt prefill and initial batch execution, while KVCore mainly optimizes the decode-stage KV lifecycle. The improvement is therefore expected to be smaller than the throughput gain. In contrast, P99 end-to-end latency improves more substantially: KVCore achieves a median 1.49x improvement over vLLM, with improvements ranging from 1.16x to 2.08x across the evaluated configurations. This larger tail-latency reduction reflects KVCore's ability to avoid most, though not all, request preemptions caused by dynamic KV growth. Since preemption and recomputation disproportionately affect long-running requests, reducing KV pressure has a stronger impact on P99 latency than on TTFT.

InfiniGen has higher latency in this experiment for the same reason as its lower throughput: by placing KV cache in CPU memory, it trades GPU-resident execution efficiency for expanded effective capacity. This tradeoff is useful when GPU memory capacity is the dominant limitation, but it is less favorable in our setting where the goal is to maintain high GPU utilization while preventing request preemption under constrained GPU memory. Therefore, throughout this evaluation, we use vLLM as the primary baseline for throughput and latency improvements.

## 中文口径说明

- 性能提升倍数统一相对 vLLM 报告，不相对 InfiniGen 报告。
- InfiniGen 表现差的解释不是“系统不行”，而是它把 KV 全量卸载到 CPU，当前单卡实验中 GPU 显存没有被充分占满，导致吞吐/延迟不占优。
- 这反而说明 InfiniGen 更适合大 batch、长上下文、扩展有效显存容量的场景。
- KVCore 的目标问题不同：有限 GPU 显存下，decode 阶段 KV 动态增长导致请求抢占和调度不稳定。
- 因此 throughput 和 P99 E2E 应作为 KVCore 的主要收益，TTFT 只应描述为温和改善。

## Numbers Used

| System | Metric | Range | Median |
|---|---:|---:|---:|
| KVCore vs. vLLM | Throughput speedup | 1.20x-2.35x | 1.78x |
| KVCore vs. vLLM | TTFT improvement | 1.03x-1.22x | 1.10x |
| KVCore vs. vLLM | P99 E2E improvement | 1.16x-2.08x | 1.49x |
| InfiniGen vs. vLLM | Throughput ratio | 0.15x-0.39x | 0.30x |
| InfiniGen vs. vLLM | TTFT improvement | 0.28x-0.80x | 0.56x |
| InfiniGen vs. vLLM | P99 E2E improvement | 0.12x-0.36x | 0.22x |
