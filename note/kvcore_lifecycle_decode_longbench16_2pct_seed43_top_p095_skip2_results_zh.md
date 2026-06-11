# KVCore Lifecycle Decode LongBench16 准确率消融结果：`kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2`

## 实验设置

- 模型：`/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：LongBench 16 个英文子数据集
- 采样：每个子数据集 `fraction=0.02`，`seed=43`
- 对比：`full_kv` vs `decode_qaware_blockwise_top_p`
- decode active block budget：block score softmax 后 top-p，`p=0.95`
- decode 前 `2` 层不压缩
- 运行设备：NVIDIA L40S

## 结论摘要

- 16-task macro delta：`-0.6100` 分。
- `full_kv` macro：`47.6581`。
- `decode_qaware_blockwise_top_p` macro：`47.0481`。

注意：这个实验测的是 decode 阶段 query-aware sparse active set 的质量影响；真实 offload/prefetch 如果能在 attention 前恢复所需 KV，数学上应与 full KV 一致。

## Per-task 结果

| Task | Category | Samples | Full KV | Decode top-p | Delta | Rel. delta |
|---|---|---:|---:|---:|---:|---:|
| `2wikimqa` | `multi_doc_qa` | 4 | 58.33 | 58.33 | 0.00 | 0.00% |
| `gov_report` | `summarization` | 4 | 31.90 | 24.34 | -7.56 | -23.70% |
| `hotpotqa` | `multi_doc_qa` | 4 | 12.50 | 12.50 | 0.00 | 0.00% |
| `lcc` | `code` | 10 | 49.40 | 56.10 | 6.70 | 13.56% |
| `multi_news` | `summarization` | 4 | 26.41 | 21.87 | -4.54 | -17.19% |
| `multifieldqa_en` | `single_doc_qa` | 3 | 28.92 | 28.92 | 0.00 | 0.00% |
| `musique` | `multi_doc_qa` | 4 | 86.11 | 86.11 | 0.00 | 0.00% |
| `narrativeqa` | `single_doc_qa` | 4 | 43.06 | 43.06 | 0.00 | 0.00% |
| `passage_count` | `synthetic` | 4 | 2.50 | 0.00 | -2.50 | -100.00% |
| `passage_retrieval_en` | `synthetic` | 4 | 100.00 | 100.00 | 0.00 | 0.00% |
| `qasper` | `single_doc_qa` | 4 | 56.28 | 58.08 | 1.80 | 3.20% |
| `qmsum` | `summarization` | 4 | 21.52 | 20.73 | -0.79 | -3.67% |
| `repobench-p` | `code` | 10 | 55.70 | 54.50 | -1.20 | -2.15% |
| `samsum` | `few_shot` | 4 | 39.90 | 38.23 | -1.67 | -4.19% |
| `trec` | `few_shot` | 4 | 50.00 | 50.00 | 0.00 | 0.00% |
| `triviaqa` | `multi_doc_qa` | 4 | 100.00 | 100.00 | 0.00 | 0.00% |

## 产物

- 汇总 CSV：`evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/summary.csv`
- 汇总 JSON：`evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/summary.json`
- 原始结果：`evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2/artifacts/`
