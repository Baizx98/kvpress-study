# Blockwise Stage2 多数据集实验分析（ratio=0.7, fraction=0.2）

## 实验设置

- 运行脚本：
  - [run_blockwise_stage2_ratio70_fraction20_multidataset.py](/home10T/bzx/workspace/kvpress-study/evaluation/run_blockwise_stage2_ratio70_fraction20_multidataset.py)
- 结果目录：
  - [artifacts](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_stage2_ratio70_fraction20_multidataset/artifacts)
  - [run.log](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_stage2_ratio70_fraction20_multidataset/artifacts/run.log)
- 模型：
  - `/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：
  - `RULER / 4096 / niah_single_3, niah_multikey_3, qa_2`
  - `LongBench / qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, triviaqa`
  - `Needle in a Haystack / 16384 / [0,25,50,75,100]`
- 方法：
  - `blockwise_main`
  - `blockwise_norm_topk`
  - `blockwise_multi_rep`
  - `blockwise_tail_query_special`
  - `chunkkv_prefill_per_layer`

## 完整性说明

- `ruler:4096`：4/4
- `longbench:qasper`：5/5
- `longbench:multifieldqa_en`：5/5
- `longbench:hotpotqa`：5/5
- `longbench:2wikimqa`：5/5
- `longbench:musique`：5/5
- `longbench:triviaqa`：5/5
- `needle_in_haystack:16384`：4/4

## 主要观察

- `ruler:4096`：最佳方法为 `chunkkv_prefill_per_layer`，分数 `83.61`
- `longbench:qasper`：最佳方法为 `norm_topk_mean_only + key_norm + max + uniform_mean`，分数 `41.51`
- `longbench:multifieldqa_en`：最佳方法为 `multi_rep_max + key_norm + max + uniform_mean`，分数 `56.57`
- `longbench:hotpotqa`：最佳方法为 `mean_plus_norm_topk_mean + key_norm + max + uniform_mean`，分数 `56.27`
- `longbench:2wikimqa`：最佳方法为 `chunkkv_prefill_per_layer`，分数 `45.13`
- `longbench:musique`：最佳方法为 `chunkkv_prefill_per_layer`，分数 `35.63`
- `longbench:triviaqa`：最佳方法为 `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean`，分数 `98.00`
- `needle_in_haystack:16384`：最佳方法为 `multi_rep_max + key_norm + max + uniform_mean`，分数 `73.50`

## Blockwise vs ChunkKV

- `ruler:4096`：blockwise_main 相对 chunkkv 变化 `-44.47`（blockwise=`39.14`，chunkkv=`83.61`）
- `longbench:qasper`：blockwise_main 相对 chunkkv 变化 `0.54`（blockwise=`40.31`，chunkkv=`39.77`）
- `longbench:multifieldqa_en`：blockwise_main 相对 chunkkv 变化 `-0.11`（blockwise=`53.74`，chunkkv=`53.85`）
- `longbench:hotpotqa`：blockwise_main 相对 chunkkv 变化 `1.66`（blockwise=`56.27`，chunkkv=`54.61`）
- `longbench:2wikimqa`：blockwise_main 相对 chunkkv 变化 `-6.01`（blockwise=`39.12`，chunkkv=`45.13`）
- `longbench:musique`：blockwise_main 相对 chunkkv 变化 `-5.18`（blockwise=`30.45`，chunkkv=`35.63`）
- `longbench:triviaqa`：blockwise_main 相对 chunkkv 变化 `3.00`（blockwise=`96.00`，chunkkv=`93.00`）
- `needle_in_haystack:16384`：blockwise_main 相对 chunkkv 变化 `-2.75`（blockwise=`68.21`，chunkkv=`70.97`）

## 最终失败项

- `needle_in_haystack:16384__blockwise_main`: attempts=3, reason=unknown
- `needle_in_haystack:16384__blockwise_norm_topk`: attempts=3, reason=unknown
- `needle_in_haystack:16384__blockwise_multi_rep`: attempts=3, reason=unknown
- `needle_in_haystack:16384__chunkkv_prefill`: attempts=3, reason=unknown

## 数据集明细

## `ruler:4096`

| 方法 | 分数 | 说明 |
|---|---:|---|
| `mean_plus_norm_topk_mean + key_norm + max + uniform_mean` | 39.14 | avg=39.14; niah_multikey_3=20.00, niah_single_3=34.62, qa_2=62.79 |
| `norm_topk_mean_only + key_norm + max + uniform_mean` | 38.63 | avg=38.63; niah_multikey_3=16.36, niah_single_3=35.58, qa_2=63.95 |
| `multi_rep_max + key_norm + max + uniform_mean` | 40.62 | avg=40.62; niah_multikey_3=16.36, niah_single_3=40.38, qa_2=65.12 |
| `chunkkv_prefill_per_layer` | 83.61 | avg=83.61; niah_multikey_3=84.55, niah_single_3=100.00, qa_2=66.28 |

缺失方法：
- `blockwise_tail_query_special`

最佳方法：`chunkkv_prefill_per_layer`，分数 `83.61`
## `longbench:qasper`

| 方法 | 分数 | 说明 |
|---|---:|---|
| `mean_plus_norm_topk_mean + key_norm + max + uniform_mean` | 40.31 | 40.31 |
| `norm_topk_mean_only + key_norm + max + uniform_mean` | 41.51 | 41.51 |
| `multi_rep_max + key_norm + max + uniform_mean` | 40.61 | 40.61 |
| `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean` | 33.24 | 33.24 |
| `chunkkv_prefill_per_layer` | 39.77 | 39.77 |

最佳方法：`norm_topk_mean_only + key_norm + max + uniform_mean`，分数 `41.51`
## `longbench:multifieldqa_en`

| 方法 | 分数 | 说明 |
|---|---:|---|
| `mean_plus_norm_topk_mean + key_norm + max + uniform_mean` | 53.74 | 53.74 |
| `norm_topk_mean_only + key_norm + max + uniform_mean` | 54.14 | 54.14 |
| `multi_rep_max + key_norm + max + uniform_mean` | 56.57 | 56.57 |
| `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean` | 44.41 | 44.41 |
| `chunkkv_prefill_per_layer` | 53.85 | 53.85 |

最佳方法：`multi_rep_max + key_norm + max + uniform_mean`，分数 `56.57`
## `longbench:hotpotqa`

| 方法 | 分数 | 说明 |
|---|---:|---|
| `mean_plus_norm_topk_mean + key_norm + max + uniform_mean` | 56.27 | 56.27 |
| `norm_topk_mean_only + key_norm + max + uniform_mean` | 53.20 | 53.20 |
| `multi_rep_max + key_norm + max + uniform_mean` | 54.37 | 54.37 |
| `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean` | 53.20 | 53.20 |
| `chunkkv_prefill_per_layer` | 54.61 | 54.61 |

最佳方法：`mean_plus_norm_topk_mean + key_norm + max + uniform_mean`，分数 `56.27`
## `longbench:2wikimqa`

| 方法 | 分数 | 说明 |
|---|---:|---|
| `mean_plus_norm_topk_mean + key_norm + max + uniform_mean` | 39.12 | 39.12 |
| `norm_topk_mean_only + key_norm + max + uniform_mean` | 38.83 | 38.83 |
| `multi_rep_max + key_norm + max + uniform_mean` | 42.56 | 42.56 |
| `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean` | 42.87 | 42.87 |
| `chunkkv_prefill_per_layer` | 45.13 | 45.13 |

最佳方法：`chunkkv_prefill_per_layer`，分数 `45.13`
## `longbench:musique`

| 方法 | 分数 | 说明 |
|---|---:|---|
| `mean_plus_norm_topk_mean + key_norm + max + uniform_mean` | 30.45 | 30.45 |
| `norm_topk_mean_only + key_norm + max + uniform_mean` | 32.62 | 32.62 |
| `multi_rep_max + key_norm + max + uniform_mean` | 31.90 | 31.90 |
| `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean` | 26.24 | 26.24 |
| `chunkkv_prefill_per_layer` | 35.63 | 35.63 |

最佳方法：`chunkkv_prefill_per_layer`，分数 `35.63`
## `longbench:triviaqa`

| 方法 | 分数 | 说明 |
|---|---:|---|
| `mean_plus_norm_topk_mean + key_norm + max + uniform_mean` | 96.00 | 96.00 |
| `norm_topk_mean_only + key_norm + max + uniform_mean` | 96.00 | 96.00 |
| `multi_rep_max + key_norm + max + uniform_mean` | 96.00 | 96.00 |
| `mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean` | 98.00 | 98.00 |
| `chunkkv_prefill_per_layer` | 93.00 | 93.00 |

最佳方法：`mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean`，分数 `98.00`
## `needle_in_haystack:16384`

| 方法 | 分数 | 说明 |
|---|---:|---|
| `mean_plus_norm_topk_mean + key_norm + max + uniform_mean` | 68.21 | avg_rouge_l_f=68.21 |
| `norm_topk_mean_only + key_norm + max + uniform_mean` | 65.69 | avg_rouge_l_f=65.69 |
| `multi_rep_max + key_norm + max + uniform_mean` | 73.50 | avg_rouge_l_f=73.50 |
| `chunkkv_prefill_per_layer` | 70.97 | avg_rouge_l_f=70.97 |

缺失方法：
- `blockwise_tail_query_special`

最佳方法：`multi_rep_max + key_norm + max + uniform_mean`，分数 `73.50`

