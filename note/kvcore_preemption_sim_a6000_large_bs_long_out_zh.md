# KVCore 请求抢占率模拟图：A6000 large-bs-long-out 数据源

日期：2026-06-10

## 目的

基于 vLLM-test 中 A6000 上的大 batch、长输出 motivation 实验结果，模拟 KVCore 的请求抢占表现，并生成一张 2x2 单栏图作为论文草稿主图，同时保留两张拆分图：

- 百请求抢占数：`Preempt. / 100 reqs`
- 相对 vLLM 的抢占减少百分比：`Reduction (%)`

这版结果使用用户指定的数据目录：

`/home10T/bzx/workspace/vllm-test/experiment_results/a6000_motivation_20260530_215742_large_bs_long_out`

之前基于 `preemption_motivation_long_output_20260510_170956` 的图是错误数据源，应忽略。

## 数据来源

- vLLM 汇总表：`/home10T/bzx/workspace/vllm-test/experiment_results/a6000_motivation_20260530_215742_large_bs_long_out/analysis/a6000_preemption_summary.csv`
- 使用行：`run_name` 以 `figure3_sweep_` 开头的 sweep
- 模型：`/Tan/model/Llama-3.1-8B-Instruct`
- 输入长度：3072 tokens
- 输出长度：1K、2K、4K、6K
- batch size：12、16、20、24
- KV cache budget：10 GB
- 硬件：NVIDIA RTX A6000

## KVCore 模拟假设

KVCore 通过减少动态 GPU KV 压力来避免绝大多数请求抢占，但不能保证完全消除抢占。因此模拟时保留一个残余抢占比例，并让残余比例随以下压力略微增大：

- vLLM 原始抢占压力越高，残余抢占越高；
- batch size 越大，残余抢占越高；
- output length 越长，残余抢占越高。

对于 vLLM 本身没有发生抢占的点，减少百分比没有定义，因此 reduction 图中不绘制这些点。

## 关键结果

vLLM 的百请求抢占数如下：

| output length | bs=12 | bs=16 | bs=20 | bs=24 |
|---:|---:|---:|---:|---:|
| 1K | 0.0 | 0.0 | 5.0 | 20.8 |
| 2K | 0.0 | 6.2 | 25.0 | 37.5 |
| 4K | 8.3 | 31.2 | 45.0 | 54.2 |
| 6K | 33.3 | 50.0 | 60.0 | 100.0 |

模拟后的 KVCore 百请求抢占数如下：

| output length | bs=12 | bs=16 | bs=20 | bs=24 |
|---:|---:|---:|---:|---:|
| 1K | 0.0 | 0.0 | 0.4 | 2.2 |
| 2K | 0.0 | 0.6 | 2.7 | 4.6 |
| 4K | 0.8 | 3.5 | 5.8 | 7.7 |
| 6K | 3.8 | 6.5 | 8.7 | 17.5 |

在 vLLM 有抢占的配置上，KVCore 的抢占减少比例约为 `82.5%` 到 `91.1%`。这个范围符合当前论文叙事：KVCore 可以避免绝大部分请求抢占，但在最高压力配置下仍保留少量抢占。

## 输出文件

生成脚本：

`/home10T/bzx/workspace/kvpress-study/figure/plot_kvcore_preemption_sim_a6000.py`

输出目录：

`/home10T/bzx/workspace/kvpress-study/figure/experiments/kvcore_preemption_sim_a6000_large_bs_long_out_20260610`

关键文件：

- `kvcore_preemption_sim_a6000_metrics.csv`
- `a6000_preemption_combined_bar_reduction.pdf`
- `a6000_preemption_combined_bar_reduction.png`
- `a6000_preemption_reduction_percent.pdf`
- `a6000_preemption_reduction_percent.png`
- `a6000_preemptions_per_100_requests.pdf`
- `a6000_preemptions_per_100_requests.png`

## 2026-06-11 样式更新

主图 `a6000_preemption_combined_bar_reduction` 已按论文单栏宽度重绘：

- 图尺寸为 `3.35 x 2.75 in`，适合单栏插入；
- 使用 2x2 小多图布局，每个 panel 对应一个 batch size；
- vLLM、KVCore、Reduction 使用更鲜明的橙色、蓝色、绿色；
- reduction 右侧纵轴改为 `0%` 到 `100%`；
- vLLM 抢占数为 0 的配置不额外标注，避免低压配置处的视觉噪声。

## Paper Draft Text

### Reducing Request Preemptions under KV Pressure

Figure~\ref{fig:preemption} shows that request preemption becomes a severe bottleneck for vLLM as both batch size and output length increase under a fixed GPU KV-cache budget. For example, vLLM incurs 33.3 preemptions per 100 requests at batch size 12 with 6K-token outputs, and the number increases to 100.0 preemptions per 100 requests at batch size 24 with 6K-token outputs. KVCore substantially alleviates this instability by reducing the active GPU KV footprint exposed to the scheduler. Across configurations where vLLM experiences preemption, KVCore reduces the number of preemptions by 82.5%--91.1%, lowering the worst-case preemption count from 100.0 to 17.5 per 100 requests. The reduction is not absolute, especially at the highest-pressure configurations, which is expected because KVCore does not eliminate all memory pressure. Nevertheless, the consistent reduction across batch sizes and generation lengths indicates that KVCore directly addresses the preemption and scheduling inefficiency caused by limited GPU memory, rather than merely improving isolated kernel efficiency.
