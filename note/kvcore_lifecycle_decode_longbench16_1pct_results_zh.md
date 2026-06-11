# KVCore Lifecycle Decode LongBench16 准确率消融结果：`kvcore_lifecycle_decode_longbench16_1pct`

## 实验设置

- 模型：`/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：LongBench 16 个英文子数据集
- 采样：每个子数据集 `fraction=0.01`，`seed=42`
- 对比：`full_kv` vs `decode_qaware_blockwise_top_p`
- decode active block budget：block score softmax 后 top-p，`p=0.9`
- decode 前 `0` 层不压缩
- 运行设备：NVIDIA L40S

## 结论摘要

- 16-task macro delta：`-2.3100` 分。
- `full_kv` macro：`39.3256`。
- `decode_qaware_blockwise_top_p` macro：`37.0156`。

注意：这个实验测的是 decode 阶段 query-aware sparse active set 的质量影响；真实 offload/prefetch 如果能在 attention 前恢复所需 KV，数学上应与 full KV 一致。

## Per-task 结果

| Task | Category | Samples | Full KV | Decode top-p | Delta | Rel. delta |
|---|---|---:|---:|---:|---:|---:|
| `2wikimqa` | `multi_doc_qa` | 2 | 0.00 | 0.00 | 0.00 | n/a |
| `gov_report` | `summarization` | 2 | 35.31 | 24.82 | -10.49 | -29.71% |
| `hotpotqa` | `multi_doc_qa` | 2 | 53.33 | 53.33 | 0.00 | 0.00% |
| `lcc` | `code` | 5 | 34.60 | 35.00 | 0.40 | 1.16% |
| `multi_news` | `summarization` | 2 | 18.97 | 15.99 | -2.98 | -15.71% |
| `multifieldqa_en` | `single_doc_qa` | 2 | 23.56 | 23.76 | 0.20 | 0.85% |
| `musique` | `multi_doc_qa` | 2 | 50.00 | 50.00 | 0.00 | 0.00% |
| `narrativeqa` | `single_doc_qa` | 2 | 11.11 | 11.11 | 0.00 | 0.00% |
| `passage_count` | `synthetic` | 2 | 0.00 | 0.00 | 0.00 | n/a |
| `passage_retrieval_en` | `synthetic` | 2 | 100.00 | 100.00 | 0.00 | 0.00% |
| `qasper` | `single_doc_qa` | 2 | 88.71 | 79.27 | -9.44 | -10.64% |
| `qmsum` | `summarization` | 2 | 32.23 | 27.24 | -4.99 | -15.48% |
| `repobench-p` | `code` | 5 | 39.40 | 41.60 | 2.20 | 5.58% |
| `samsum` | `few_shot` | 2 | 51.99 | 40.13 | -11.86 | -22.81% |
| `trec` | `few_shot` | 2 | 0.00 | 0.00 | 0.00 | n/a |
| `triviaqa` | `multi_doc_qa` | 2 | 90.00 | 90.00 | 0.00 | 0.00% |

## 产物

- 汇总 CSV：`evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_1pct/summary.csv`
- 汇总 JSON：`evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_1pct/summary.json`
- 原始结果：`evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_1pct/artifacts/`
