# BlockWise LongBench Stage1 消融实验分析（ratio=0.7, fraction=0.2）

## 实验设置

- 运行脚本：
  - [run_blockwise_ablation_ratio70_longbench_stage1.sh](/home10T/bzx/workspace/kvpress-study/evaluation/run_blockwise_ablation_ratio70_longbench_stage1.sh)
  - `triviaqa` 补跑：
    [rerun_blockwise_ablation_ratio70_longbench_stage1_triviaqa_missing.sh](/home10T/bzx/workspace/kvpress-study/evaluation/rerun_blockwise_ablation_ratio70_longbench_stage1_triviaqa_missing.sh)
- 结果目录：
  - [artifacts](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_ablation_ratio70_longbench_stage1/artifacts)
  - [run.log](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_ablation_ratio70_longbench_stage1/artifacts/run.log)
- 模型：
  - `/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：
  - `LongBench / hotpotqa`
  - `LongBench / multifieldqa_en`
  - `LongBench / triviaqa`
- 压缩设置：
  - `compression_ratio=0.7`
  - `block_size=16`
  - `q_window_size=64`
  - `summary_topk_keys=4`
  - `mean_key_weight=0.75`
  - `representative_k=4`
  - `multi_rep_k=4`
  - `query_topr=16`
  - `head_topk=1`
  - `query_aware=true`
- 采样设置：
  - 不使用 `samples_per_task`
  - 各任务直接按 `fraction=0.2` 采样

## 完整性说明

- `hotpotqa`：13/13 完整
- `multifieldqa_en`：13/13 完整
- `triviaqa`：13/13 完整

## 总体观察

- `hotpotqa`：最佳配置为 `mean_plus_norm_topk_mean` + `key_norm` + `max` + `uniform_mean`，分数 `56.27`
- `multifieldqa_en`：最佳配置为 `mean_plus_norm_topk_mean` + `key_norm` + `max` + `uniform_mean`，分数 `54.30`
- `triviaqa`：最佳配置为 `mean_plus_norm_topk_mean` + `tail_query_relevance` + `mean` + `uniform_mean`，分数 `98.00`

- `hotpotqa`：`query_agg=max` 相对 baseline 提升 `3.07`
- `multifieldqa_en`：`query_agg=max` 相对 baseline 提升 `10.68`
- `triviaqa`：`query_agg=max` 相对 baseline 提升 `0.83`

- `hotpotqa`：Quest 相对 baseline 变化 `-3.33`
- `multifieldqa_en`：Quest 相对 baseline 变化 `-6.97`
- `triviaqa`：Quest 相对 baseline 变化 `0.33`

- `hotpotqa` baseline：`53.20`
- `multifieldqa_en` baseline：`43.62`
- `triviaqa` baseline：`95.17`

## 数据集明细

## `hotpotqa`

| 配置 | 分数 |
|---|---:|
| `mean_plus_norm_topk_mean` + `key_norm` + `max` + `uniform_mean` | 56.27 |
| `mean_plus_norm_topk_mean` + `key_norm` + `topr_mean` + `uniform_mean` | 54.88 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `top_head_only` | 53.44 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `uniform_mean` | 53.20 |
| `norm_topk_mean_only` + `key_norm` + `mean` + `uniform_mean` | 53.20 |
| `mean_plus_norm_topk_mean` + `tail_query_relevance` + `mean` + `uniform_mean` | 53.20 |
| `random_topk(seed=42)` + baseline | 53.20 |
| `random_topk(seed=43)` + baseline | 53.20 |
| `random_topk(seed=44)` + baseline | 53.20 |
| `mean_only` + `key_norm` + `mean` + `uniform_mean` | 52.70 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `strength_weighted` | 52.37 |
| `multi_rep_max` + `key_norm` + `mean` + `uniform_mean` | 51.87 |
| `Quest-prefill (minmax)` | 49.87 |
## `multifieldqa_en`

| 配置 | 分数 |
|---|---:|
| `mean_plus_norm_topk_mean` + `key_norm` + `max` + `uniform_mean` | 54.30 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `top_head_only` | 52.97 |
| `mean_plus_norm_topk_mean` + `key_norm` + `topr_mean` + `uniform_mean` | 52.54 |
| `mean_plus_norm_topk_mean` + `tail_query_relevance` + `mean` + `uniform_mean` | 45.32 |
| `mean_only` + `key_norm` + `mean` + `uniform_mean` | 44.14 |
| `norm_topk_mean_only` + `key_norm` + `mean` + `uniform_mean` | 44.06 |
| `random_topk(seed=44)` + baseline | 43.68 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `uniform_mean` | 43.62 |
| `random_topk(seed=42)` + baseline | 43.06 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `strength_weighted` | 42.70 |
| `random_topk(seed=43)` + baseline | 41.21 |
| `multi_rep_max` + `key_norm` + `mean` + `uniform_mean` | 40.84 |
| `Quest-prefill (minmax)` | 36.65 |
## `triviaqa`

| 配置 | 分数 |
|---|---:|
| `mean_plus_norm_topk_mean` + `tail_query_relevance` + `mean` + `uniform_mean` | 98.00 |
| `random_topk(seed=42)` + baseline | 98.00 |
| `random_topk(seed=43)` + baseline | 98.00 |
| `multi_rep_max` + `key_norm` + `mean` + `uniform_mean` | 96.00 |
| `mean_plus_norm_topk_mean` + `key_norm` + `max` + `uniform_mean` | 96.00 |
| `norm_topk_mean_only` + `key_norm` + `mean` + `uniform_mean` | 95.50 |
| `Quest-prefill (minmax)` | 95.50 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `uniform_mean` | 95.17 |
| `random_topk(seed=44)` + baseline | 95.17 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `strength_weighted` | 95.17 |
| `mean_plus_norm_topk_mean` + `key_norm` + `topr_mean` + `uniform_mean` | 93.50 |
| `mean_only` + `key_norm` + `mean` + `uniform_mean` | 92.92 |
| `mean_plus_norm_topk_mean` + `key_norm` + `mean` + `top_head_only` | 90.91 |

## 阶段性结论

- `query_agg=max` 仍然是最值得优先关注的候选项；如果三个数据集都可比较，它通常会是最稳的增强方向。
- `Quest-prefill` 目前主要还是对照组，是否能追平 summary-based blockwise，要看 `triviaqa` 补齐后的完整对比。
- 如果本轮补跑后 `triviaqa` 也补齐，这份文档就可以直接作为 LongBench stage1 的正式归档说明；否则仍应把它视为“部分完成”的阶段性记录。

