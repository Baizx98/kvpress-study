# BlockWise Stage1 消融实验分析（RULER, ratio=0.7, fraction=0.2）

## 实验设置

- 运行脚本：
  - [run_blockwise_ablation_ratio70_stage1.sh](/home10T/bzx/workspace/kvpress-study/evaluation/run_blockwise_ablation_ratio70_stage1.sh)
- 结果目录：
  - [artifacts](/home10T/bzx/workspace/kvpress-study/results/experiments/blockwise_ablation_ratio70_stage1/artifacts)
  - [run.log](/home10T/bzx/workspace/kvpress-study/results/experiments/blockwise_ablation_ratio70_stage1/artifacts/run.log)
- 模型：
  - `/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：
  - `RULER (4096)`
  - 任务过滤：`niah_single_3, niah_multikey_3, qa_2`
- 压缩设置：
  - `compression_ratio=0.7`
  - `block_size=16`
  - `q_window_size=64`
  - `prefill_skip_first_layers=1`
  - `query_aware=true`
- 采样设置：
  - 不再使用 `samples_per_task`
  - 先按任务过滤，再对过滤后的 `1500` 条样本按 `fraction=0.2` 采样，最终得到 `300` 条样本

## 结果总览

以下表格给出本轮最重要的 13 组配置。这里用三项任务的简单平均作为便于比较的汇总指标：

| 配置 | niah_single_3 | niah_multikey_3 | qa_2 | avg(3 tasks) |
|---|---:|---:|---:|---:|
| `mean + norm-topk-mean` + `key_norm` + `max` + `uniform_mean` | 34.62 | 19.09 | 66.28 | **40.00** |
| `norm-topk-mean only` + `key_norm` + `mean` + `uniform_mean` | 23.08 | 20.00 | 62.79 | **35.29** |
| `multi_rep_max` + `key_norm` + `mean` + `uniform_mean` | 27.88 | 15.45 | 61.63 | **34.99** |
| baseline: `mean + norm-topk-mean` + `key_norm` + `mean` + `uniform_mean` | 23.08 | 15.45 | 60.47 | **33.00** |
| `tail_query_relevance` 替换 `key_norm` | 23.08 | 13.64 | 60.47 | 32.40 |
| `mean_only` | 22.12 | 14.55 | 60.47 | 32.38 |
| `random_topk (seed=43)` | 21.15 | 16.36 | 59.30 | 32.27 |
| `strength_weighted` | 21.15 | 15.45 | 59.30 | 31.97 |
| `random_topk (seed=42)` | 20.19 | 12.73 | 60.47 | 31.13 |
| `topr_mean` | 8.65 | 19.09 | 65.12 | 30.95 |
| `random_topk (seed=44)` | 19.23 | 15.45 | 58.14 | 30.94 |
| `top_head_only` | 2.88 | 18.18 | 66.28 | 29.11 |
| `Quest-prefill (minmax)` | 6.73 | 0.91 | 54.65 | **20.76** |

## 关键结论

## 1) 最有效的改动来自 query window aggregation：`max` 明显优于 `mean`

在 baseline 配置上，只把 `query_agg_mode` 从 `mean` 改成 `max`，三任务平均从：

- `33.00 -> 40.00`

具体提升：

- `niah_single_3`: `23.08 -> 34.62`
- `niah_multikey_3`: `15.45 -> 19.09`
- `qa_2`: `60.47 -> 66.28`

这说明在当前 question-aware prefill 设定下，“最后 query window 中最强的一部分 query”比“所有 query 平均”更能提供有效块热度信号。

一个合理解释是：

- `mean` 会把问题末尾少数高相关 query 稀释掉
- `max` 更接近“是否存在强匹配 query”的判别式
- 对检索型任务尤其是 `niah_single_3` 更有利

## 2) summary 形式上，`norm-topk-mean only` 比当前 baseline 更强

在其它维度固定时，A 组结果是：

- `mean_only`: `32.38`
- baseline `mean + norm-topk-mean`: `33.00`
- `norm-topk-mean only`: `35.29`
- `multi_rep_max`: `34.99`

结论比较清楚：

- 单纯 `mean_only` 不够强
- 当前 baseline 虽然比 `mean_only` 好，但不是最优
- `norm-topk-mean only` 反而是这一轮最好的 summary form

这说明块级 coarse summary 的主要有效信息，很可能来自“块内少数高强度 token”，而不是块整体 mean。

同时：

- `multi_rep_max` 在 `niah_single_3` 上有明显增益：`27.88`
- 但在 `niah_multikey_3` 上没有超过 `norm-topk-mean only`

所以它更像一个“对 single-key retrieval 更友好”的方向，而不是当前阶段最稳的默认配置。

## 3) representative selection 上，`tail_query_relevance` 没有打赢 `key_norm`

B 组结果：

- `key_norm`: `33.00`
- `tail_query_relevance`: `32.40`
- `random_topk`: `30.94 ~ 32.27`

这轮可以得到两个判断：

1. `key_norm` 已经是一个很强、很稳的 baseline
2. `tail_query_relevance` 在当前实现和当前 benchmark 子集上，没有带来可见收益

此外 `random_topk` 虽然整体略差，但并没有完全崩塌，说明：

- 当前 block-wise 方案的有效性并不只来自 representative 选择本身
- `summary + query aggregation + keep policy` 这条主链路仍然起主要作用

但 `random_topk` 依然稳定落后于 `key_norm`，所以 sanity check 是成立的。

## 4) head aggregation 上，复杂化没有收益，`top_head_only` 甚至明显有害

D 组结果：

- `uniform_mean`: `33.00`
- `strength_weighted`: `31.97`
- `top_head_only`: `29.11`

其中 `top_head_only` 的主要问题是：

- `niah_single_3` 直接掉到 `2.88`

这说明：

- 单头最强并不等于最稳
- 多头平均仍然是更鲁棒的块热度估计方式

所以在现阶段，`uniform_mean` 更适合作为默认 head aggregation，而不是继续往“挑头”方向复杂化。

## 5) Quest-style prefill block scorer 在这组实验上明显不如 summary-based blockwise

Quest-prefill 结果：

- `niah_single_3 = 6.73`
- `niah_multikey_3 = 0.91`
- `qa_2 = 54.65`
- `avg = 20.76`

相比当前 blockwise baseline 的 `33.00`，差距比较大。

这表明：

- 直接把 Quest 的 `min/max envelope` 思路迁移到 prefill block compression 上，在这组 RULER 子集上过于粗糙
- 它没有有效捕捉多 key 检索块和单 key 检索块的关键结构

至少从这轮结果看，Quest 更适合作为对照组，而不是当前主线方法。

## 对后续 stage2 的建议

如果要进入第二阶段组合验证，我建议优先带这两组：

1. 最稳主线候选
   - `summary_mode=norm_topk_mean_only`
   - `representative_mode=key_norm`
   - `query_agg_mode=max`
   - `head_agg_mode=uniform_mean`

2. 偏 single-key 检索增强的候选
   - `summary_mode=multi_rep_max`
   - `representative_mode=key_norm`
   - `query_agg_mode=max`
   - `head_agg_mode=uniform_mean`

不建议优先推进的方向：

- `tail_query_relevance`
- `strength_weighted`
- `top_head_only`
- 当前这版 `Quest-prefill`

## 限制与注意事项

这轮结论虽然已经有明显信号，但仍有几个限制：

- 只在 `RULER(4096)` 的三个子任务上做了 stage1 搜索
- 指标仍然是任务级 string match，不是更细粒度的 retrieval recall
- 这里是固定 `ratio=0.7` 的局部结论，不保证在 `0.3/0.5` 同样成立

所以更准确的表述应该是：

- 在“高压缩率 + RULER 子任务 + 当前实现”这个条件下，`query max` 和 `norm-topk summary` 是最值得继续推进的方向
- 而不是直接宣称它们是全局最优配置
