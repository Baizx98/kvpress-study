# ATC26_longbench16 实验状态汇报

核查时间：2026-05-18 13:14 CST

## 总体结论

`ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv` 的当前恢复矩阵已经完成。

当前矩阵设置：

- 跳过数据集：`gov_report`
- 有效任务总数：675
- 模型：Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen3-8B
- 方法：blockwise、snapkv、chunkkv
- 压缩率：0.3、0.4、0.5、0.6、0.7

## 进程与 GPU

当前没有 longbench16 runner、watchdog 或 `evaluation/evaluate.py` 进程在运行。

GPU 状态：

- GPU0 L40S：14 MB used，45.3 GB free，util 0%
- GPU2 A6000：15 MB used，48.4 GB free，util 0%
- GPU1 3090：有其他用户任务，占用约 8.5 GB；longbench16 未使用 3090

watchdog 日志显示：

- `2026-05-18 11:35:25` runner 正常退出，`rc=0`
- progress 已完成，watchdog 正常退出

## Progress 文件状态

`ATC26_progress.md` 最新更新时间：2026-05-18 11:33:41

- Total：675
- Pending：0
- Running：0
- Success：343
- Failed：0
- Skipped：332

说明：`Skipped=332` 是 watchdog/runner 多次恢复后，`--resume` 识别已有完整结果并跳过，不表示结果缺失。

按模型：

- `llama31_8b_instruct`：skipped=225
- `mistral_7b_instruct_v03`：skipped=107, success=118
- `qwen3_8b`：success=225

## 原始结果完整性校验

按 `ATC26_config.yaml + ATC26_metrics.json + ATC26_predictions.csv` 扫描原始结果：

- full 结果总数：685
- 其中包含早先保留的 `gov_report` 结果：10
- 跳过 `gov_report` 后预期 full 结果：675
- 跳过 `gov_report` 后缺失结果：0

非 `gov_report` 675 任务分布：

按模型：

- `llama31_8b_instruct`：225
- `mistral_7b_instruct_v03`：225
- `qwen3_8b`：225

按子数据集：

- 每个非 `gov_report` LongBench 子数据集均为 45 条结果
- 共 15 个子数据集：675 = 15 datasets * 3 models * 3 methods * 5 ratios

按方法：

- `blockwise`：225
- `snapkv`：225
- `chunkkv`：225

按压缩率：

- 0.3：135
- 0.4：135
- 0.5：135
- 0.6：135
- 0.7：135

## 汇总产物

后处理已经写出：

- `evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_metrics_long.csv`

该 CSV 当前共 830 行，其中 1 行表头，829 条记录。记录数多于 675 是因为每个任务可能展开出不同 metric 记录，并且还包含历史保留结果。

## 失败记录

`ATC26_failed_jobs_final.jsonl` 仍只有 1 条：

- `llama31_8b_instruct__longbench_gov_report__snapkv__r0p4`
- 原因：L40S 上连续 3 次 OOM

该失败属于被跳过的 `gov_report`，不影响当前 675 任务矩阵的完整性。

## 后续建议

1. 当前非 `gov_report` 的 LongBench-16 full 实验可以进入结果分析和画图阶段。
2. `gov_report` 已保留部分原始结果，但不完整；若论文需要它，需要单独安排更保守显存策略补跑。
3. 后续分析应以原始结果完整性校验为准，而不是单看 `Success/Skipped` 的比例。
