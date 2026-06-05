# Decode 阶段 Prompt KVCache 重要性热力图实验结果

## 1. 问题陈述

本实验验证 LongBench 长输出请求在不同 decode step 下，对 prompt KV cache 中不同 token position 的关注区域是否会变化。

核心问题：

> 如果每一步 query 关注的 prompt token 集合会漂移，那么 prefill 阶段一次性得到的静态 KV 重要性排序就不足以完全代表后续 decode 阶段。

## 2. 假设

假设 decode 阶段 prompt KV 重要性不是静态的：

- 相邻 step 的 retained token set 应较相似；
- 但随着 lag 增大，retained token set 的 Jaccard 应下降；
- 压缩率越高，保留集合越小，随 step 漂移带来的 overlap 下降应更明显。

## 3. 方法

模型：

- `/Tan/model/Llama-3.1-8B-Instruct`

数据集：

- `Xnhyacinth/LongBench`
- `gov_report`, row `66`, prompt tokens `7434`
- `qmsum`, row `76`, prompt tokens `12117`
- `multi_news`, row `115`, prompt tokens `9633`

采集方式：

- 对每条样本跑 `512` decode steps；
- `256` step 结果直接取前 256 step；
- prefill 使用 SDPA 建 KV cache，decode 单 token step 切到 eager attention 以读取 attention weights；
- 每一步只保留当前 query 对 prompt token positions 的 attention；
- 聚合最后 8 层、所有 heads 的 prompt attention probability；
- token 粒度 top-k，不做 block 聚合。

压缩率语义与仓库现有 KVPress 实验一致：

- `compression_ratio=0.3`：丢弃 30%，保留 attention score 最高的 70% prompt tokens；
- `compression_ratio=0.5`：丢弃 50%，保留 50%；
- `compression_ratio=0.7`：丢弃 70%，保留 30%。

## 4. 产物

运行脚本：

- `evaluation/ATC26_collect_decode_prompt_kvcache_importance_heatmap.py`
- `figure/ATC26_plot_decode_prompt_kvcache_importance_heatmap.py`

原始结果：

- `evaluation/results/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/artifacts/run_config.json`
- `evaluation/results/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/artifacts/sample_manifest.json`
- `evaluation/results/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/artifacts/summary_metrics.csv`
- `evaluation/results/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/artifacts/raw/*.npz`

图像：

- `figure/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/heatmap_*`
- `figure/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/keep_frequency_*`
- `figure/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/jaccard_adjacent_summary.png`

## 5. 结果

三条样本都完成了 `512` decode steps，没有因为 EOS 提前停止。

跨三条样本平均的 retained token set Jaccard：

| decode steps | compression ratio | adjacent | lag 16 | lag 64 | lag 128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.3 | 0.8489 | 0.7381 | 0.7085 | 0.6879 |
| 256 | 0.5 | 0.7605 | 0.6005 | 0.5647 | 0.5356 |
| 256 | 0.7 | 0.6613 | 0.4645 | 0.4255 | 0.3918 |
| 512 | 0.3 | 0.8549 | 0.7464 | 0.7235 | 0.7019 |
| 512 | 0.5 | 0.7704 | 0.6138 | 0.5872 | 0.5579 |
| 512 | 0.7 | 0.6749 | 0.4820 | 0.4558 | 0.4214 |

512-step 下各数据集 `lag=128` Jaccard：

| dataset | ratio 0.3 | ratio 0.5 | ratio 0.7 |
| --- | ---: | ---: | ---: |
| `gov_report` | 0.6845 | 0.5391 | 0.4066 |
| `qmsum` | 0.7294 | 0.5899 | 0.4538 |
| `multi_news` | 0.6918 | 0.5447 | 0.4038 |

## 6. 结论

确认发现：

- 相邻 step 的保留集合较稳定，但远距离 step 的 overlap 明显下降。
- 压缩率越高，下降越明显。512-step 下，`compression_ratio=0.7` 的平均 Jaccard 从 adjacent `0.6749` 降到 lag128 `0.4214`。
- 三个长输出任务趋势一致，说明这不是单条样本的偶发现象。

解释：

- token-level heatmap 和 lag-Jaccard 共同支持“decode 阶段 prompt KV 重要性会随生成进程变化”。
- 这为 decode-aware KV cache management 提供动机：固定 prefill-only 排序可能无法覆盖中后期 decode 的重要 token。
- 更强压缩下漂移更严重，因此动态刷新、hot/cold 分层或按阶段更新候选 token 的收益空间更大。

限制：

- 当前只跑了 `Llama-3.1-8B-Instruct` 和每个任务 1 条样本；
- attention probability 是解释性 proxy，不等价于真实删除 token 后的质量影响；
- 本轮聚合最后 8 层和所有 heads，后续可以补 layer/head 分解，确认漂移主要来自哪些层或 head group。

