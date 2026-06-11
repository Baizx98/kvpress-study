# End-to-End Serving 占位图规划

生成时间：2026-06-09

## 当前真实数据可用性

vLLM formal PG19 in6k raw JSONL 已经包含 per-request 指标：

- `ttft_s`
- `e2e_latency_s`
- `tpot_ms`
- `actual_output_len`
- `prompt_token_len`
- `gpu_peak_memory_gb`
- `batch_size`
- `model_key`
- `input_len_bucket`
- `output_len_bucket`
- `status`

因此当前可以从 vLLM 真实数据计算：

- Median / P95 / empirical P99 TTFT
- Median / P95 / empirical P99 E2E latency
- Median / P95 / empirical P99 TPOT
- 完成率 / OOM failure matrix
- GPU peak memory
- 近似吞吐量

注意：当前每个点只有 32 或 48 条请求，P99 只是 empirical tail，不是严格稳定的生产级 P99。论文里正式使用前最好增加 repeat 或请求数；占位图中可以先用 `P99 (empirical)` 标注。

## 吞吐量口径

建议先用两个口径，图中明确标注：

1. Decode token throughput：
   `sum(actual_output_len) / (max(finish_time_s) - min(submit_time_s))`

   这个口径包含 measured run 内的 prefill + decode 时间，不包含 warmup。

2. Request throughput：
   `num_completed / (max(finish_time_s) - min(submit_time_s))`

   用于说明服务层请求完成速率。

对本次 paper-facing 占位图，优先用 decode token throughput，因为 KV cache / long output 系统优化更关心 token serving capacity。

## 模拟数据原则

KVCore 和 InfiniGen 暂无真实 end-to-end raw，因此占位图必须遵守：

- 图例使用 `KVCore (sim.)`、`InfiniGen (sim.)`。
- 文件名使用 `placeholder` 或 `simulated`。
- 图注或角标写明：`KVCore/InfiniGen are simulated placeholders; vLLM uses measured data.`
- 不把模拟数据写入 raw JSONL；只生成 plotting intermediate CSV/JSON。
- 模拟数据只用于版式、叙事和图形占位，不用于结论。

建议模拟趋势：

- KVCore：长输出和大 batch 下比 vLLM 更快，内存更低，tail latency 改善更明显。
- InfiniGen：比 vLLM 有一定内存优势，但吞吐/延迟收益弱于 KVCore，部分场景有额外调度或索引开销。
- OOM feasibility：KVCore 可以覆盖部分 vLLM OOM 点，InfiniGen 视叙事可覆盖一部分或全部。

## 推荐图组

### Figure 1：吞吐量随 batch size 扩展

目的：展示系统 serving capacity。

画法：

- x 轴：batch size，`1 / 8 / 16 / 24`
- y 轴：decode throughput，单位 `tok/s`，higher is better
- line plot，三条线：vLLM、InfiniGen (sim.)、KVCore (sim.)
- 分面：输出长度 `out1k / out2k / out6k`
- 模型：建议先固定 Llama-3.1-8B-Instruct；后续可做 3 模型平均版

占位数据：

- vLLM 使用真实 raw 聚合。
- KVCore 模拟为 vLLM 的 `1.15x-1.8x`，收益随 batch size 和 output length 增大。
- InfiniGen 模拟为 vLLM 的 `1.05x-1.35x`。
- vLLM OOM 点不连线，标 `OOM`；KVCore/InfiniGen 可用虚线延伸。

输出建议：

- `figure/experiments/end2end_serving_placeholder/throughput_vs_batch_llama31.png`
- 同时导出 PDF。

### Figure 2：P99 E2E latency 对比

目的：展示 tail latency，回答用户提到的 P99 需求。

画法：

- grouped bar chart
- x 轴：配置，例如 `out1k-bs8`、`out2k-bs8`、`out6k-bs8`、`out6k-bs16`
- y 轴：`P99 E2E latency (s, empirical)`，lower is better
- bar：vLLM、InfiniGen (sim.)、KVCore (sim.)
- 模型：先固定 Llama-3.1-8B-Instruct

占位数据：

- vLLM 从 raw 计算 empirical P99。
- KVCore 模拟 tail 改善比 median 更明显，例如 latency 乘以 `0.55-0.80`。
- InfiniGen 模拟乘以 `0.70-0.95`。

注意：

- 只有 32/48 请求，图注必须写 `empirical over 32/48 requests`。

### Figure 3：Latency breakdown proxy：TTFT vs TPOT

目的：把 prefill 和 decode 两部分分开，解释系统优化来自哪里。

画法：

- 两个 panel：
  - 左：median TTFT
  - 右：median TPOT
- x 轴：output length 或 batch size
- y 轴：TTFT 用 `s`，TPOT 用 `ms/token`
- 三个系统同色系固定顺序

建议配置：

- 固定 Llama，bs8，比较 out1k/out2k/out6k。
- 或固定 out2k，比较 bs1/bs8/bs16/bs24。

占位解读：

- vLLM TTFT 主要体现长 prompt prefill + batching。
- TPOT 更接近 decode 阶段效率。
- KVCore 模拟主要降低 TPOT，TTFT 也有轻微降低。
- InfiniGen 可能降低 memory pressure，但 TTFT/TPOT收益较温和。

### Figure 4：内存与可行性矩阵

目的：展示 A6000 单卡容量边界。

画法：

- heatmap 或 tile matrix
- 行：output length `1k / 2k / 6k`
- 列：batch size `1 / 8 / 16 / 24`
- 每个 tile 显示 vLLM `peak memory GB` 或 `OK/OOM`
- 可做三 panel：vLLM measured、InfiniGen sim.、KVCore sim.

推荐：

- 对 placeholder 先做 `OK/OOM` feasibility matrix，比 peak memory 更稳。
- vLLM out6k bs24 三个模型均 OOM，可以明确体现边界。

### Figure 5：三模型平均性能总览

目的：给组会或论文草图一个 compact summary。

画法：

- grouped bar chart
- x 轴：metric，`Throughput`、`Median Latency`、`P99 Latency`、`Peak Memory`
- y 轴：normalized to vLLM，vLLM = 1.0
- bar：vLLM、InfiniGen (sim.)、KVCore (sim.)
- 只聚合所有成功共同点，排除 vLLM OOM 点。

注意：

- Throughput higher is better，其余 lower is better。可以拆成两个 panel，避免方向混乱：
  - Panel A：Normalized throughput
  - Panel B：Normalized cost metrics，越低越好

### Figure 6：Latency-throughput tradeoff scatter

目的：展示 Pareto frontier。

画法：

- x 轴：P99 E2E latency，单位 s，lower is better
- y 轴：decode throughput，单位 tok/s，higher is better
- marker size：peak memory
- marker shape：输出长度
- color：system

用途：

- 适合放在组会或论文 evaluation overview。
- 如果点太多，先只画 Llama + bs8/bs16。

## 第一批占位图建议

为了快速用于汇报，建议先生成 4 张：

1. `throughput_vs_batch_llama31`
   - 最重要，展示 serving scaling。
2. `p99_latency_llama31`
   - 回答 P99 / tail latency。
3. `ttft_tpot_llama31`
   - 分解 prefill 和 decode。
4. `feasibility_matrix_all_models`
   - 展示容量边界和为什么 KVCore/InfiniGen 有意义。

这 4 张覆盖吞吐量、延迟、tail、内存/可行性，足够作为一版 placeholder。

## 绘图实现建议

新增脚本：

`figure/plot_end2end_serving_placeholder.py`

输入：

- vLLM raw directory：
  `evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/`
- 输出目录：
  `figure/experiments/end2end_serving_placeholder/`

脚本流程：

1. 读取 vLLM raw JSONL，只保留 `status == completed`。
2. 按 `(model_key, output_len_bucket, batch_size)` 聚合：
   - median / p95 / p99 of `ttft_s`
   - median / p95 / p99 of `e2e_latency_s`
   - median / p95 / p99 of `tpot_ms`
   - max or median `gpu_peak_memory_gb`
   - decode throughput
   - request throughput
3. 生成 KVCore / InfiniGen simulated dataframe。
4. 保存 intermediate：
   - `figure/experiments/end2end_serving_placeholder/placeholder_metrics.csv`
5. 导出 PNG + PDF。

## 颜色和样式

方法顺序固定：

1. vLLM：灰色 `#4D4D4D`
2. InfiniGen (sim.)：橙色 `#D55E00`
3. KVCore (sim.)：蓝色 `#0072B2`

图中单位：

- Throughput：`tok/s`
- TTFT：`s`
- E2E latency：`s`
- TPOT：`ms/token`
- Memory：`GB`

所有 placeholder 图需要角标或 subtitle：

`Measured: vLLM. Simulated placeholders: KVCore, InfiniGen.`

## 需要避免的误导

- 不要把模拟的 KVCore / InfiniGen 数据写成真实结果。
- P99 不要写成 production P99；当前请求数太少，只能写 empirical P99。
- 不要只画 normalized speedup；至少保留一张绝对 latency 或 throughput 图。
- vLLM OOM 点不要插值；应显示为 missing/OOM。
- 若某些系统模拟为可跑 vLLM OOM 点，要在图注说明这是 placeholder assumption。
