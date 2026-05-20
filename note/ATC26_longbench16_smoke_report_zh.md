# ATC26 LongBench-16 Smoke Test 报告

## 结论

LongBench-16 smoke test 已在 Device 2 / A6000 上跑通。当前结果覆盖：

- 3 个模型
- 3 个方法
- 16 个 LongBench 非中文子任务
- 1 个压缩率：`0.5`
- `fraction=0.01`

聚合表共有 144 条记录，符合预期：

`3 models × 3 methods × 16 tasks × 1 ratio = 144`

当前没有残留 `ATC26_run_longbench16_prefill_sweep.py` 或 `evaluation/evaluate.py --config_file /dev/null` 进程，A6000 已空闲。

## 执行命令

```bash
ATC26_GPUS=2 ATC26_MIN_FREE_MB=2:24000 MAX_RETRIES=2 POLL_SECONDS=60 \
  .venv/bin/python evaluation/ATC26_run_longbench16_prefill_sweep.py --mode smoke --resume
```

## 产物路径

| 类型 | 路径 |
|---|---|
| runner | `evaluation/ATC26_run_longbench16_prefill_sweep.py` |
| postprocess | `evaluation/ATC26_postprocess_longbench16_prefill_sweep.py` |
| result root | `evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/` |
| long table | `evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_metrics_long.csv` |
| progress | `evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_progress.md` |
| transient failures | `evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_failed_jobs.jsonl` |
| final failures | `evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_failed_jobs_final.jsonl` |

## 完整性检查

| 检查项 | 结果 |
|---|---:|
| records | 144 |
| `fraction=0.01` records | 144 |
| 每个模型 records | 48 |
| 每个方法 records | 48 |
| 每个 LongBench task records | 9 |
| `compression_ratio=0.5` records | 144 |
| BlockWise records with `prefill_skip_first_layers=2` | 48 |
| final failed jobs | 0 |

## 覆盖任务

- `narrativeqa`
- `qasper`
- `multifieldqa_en`
- `hotpotqa`
- `2wikimqa`
- `musique`
- `triviaqa`
- `gov_report`
- `qmsum`
- `multi_news`
- `samsum`
- `trec`
- `passage_count`
- `passage_retrieval_en`
- `lcc`
- `repobench-p`

## 观察到的问题

Smoke 过程中出现过 2 次 transient failure：

1. `llama31_8b_instruct__longbench_passage_retrieval_en__blockwise__r0p5`
2. `qwen3_8b__longbench_repobench_p__snapkv__r0p5`

两次失败日志均为 Hugging Face `ProxyError` / `RemoteDisconnected`，不是 OOM 或代码逻辑错误。两者第二次 attempt 均成功，最终失败数为 0。

已更新 runner：后续会把 `ProxyError` / `RemoteDisconnected` 归类为 `network`，便于正式实验时观察日志。

## 对正式实验的判断

从 smoke test 看，当前流程可以启动正式 LongBench-16 full test：

- 16 个 LongBench config 均能加载。
- QA、summarization、classification、retrieval/counting、code completion 的 scorer 都能跑通。
- `max_new_tokens=532` 的长输出任务在 A6000 上 smoke 未出现 OOM。
- BlockWise 的 `prefill_skip_first_layers=2` 已写入并在 48 条 BlockWise smoke 记录中全部生效。

正式测试前建议保留 `MAX_RETRIES=3`，因为 Hugging Face 数据加载偶发代理断连。
