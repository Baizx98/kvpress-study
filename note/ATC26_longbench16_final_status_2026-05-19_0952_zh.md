# ATC26_longbench16 最终实验状态

核查时间：2026-05-19 09:52 CST

## 总体结论

`ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv` 当前完整完成。

完整矩阵：

- 模型：3 个
  - `llama31_8b_instruct`
  - `mistral_7b_instruct_v03`
  - `qwen3_8b`
- LongBench 子数据集：16 个，包含此前补跑的 `gov_report`
- 方法：3 个
  - `blockwise`
  - `snapkv`
  - `chunkkv`
- 压缩率：5 个
  - 0.3、0.4、0.5、0.6、0.7

总任务数：720 = 3 models * 16 datasets * 3 methods * 5 ratios

原始结果完整性校验：

- full result rows：720
- expected full：720
- missing full：0

## 进程与 GPU

当前没有 longbench16 runner 或 `evaluation/evaluate.py` 进程。

GPU 状态：

- L40S：空闲
- A6000：空闲
- 3090：未被本实验使用

`gov_report` watchdog 已正常退出。此前用于处理 D-state 的 hang monitor 已停止。

## Progress 状态

`ATC26_progress.md` 最新更新时间：2026-05-19 09:06:01

- Total：45
- Pending：0
- Running：0
- Success：35
- Failed：0
- Skipped：10

注意：这是最后一轮 `gov_report` 专项补跑的 progress。完整 720 矩阵需要以原始结果完整性扫描为准。

## gov_report 补跑

`gov_report` 专项补跑已完成：

- Total：45
- Skipped：10
- Success：35
- Failed：0

按模型：

- `llama31_8b_instruct`：skipped=10, success=5
- `mistral_7b_instruct_v03`：success=15
- `qwen3_8b`：success=15

该轮覆盖了此前 `gov_report` 失败记录；当前 `ATC26_failed_jobs_final.jsonl` 为 0 行。

## 全矩阵分布

按模型：

- `llama31_8b_instruct`：240
- `mistral_7b_instruct_v03`：240
- `qwen3_8b`：240

按子数据集：

- 每个 LongBench 子数据集均为 45 条结果
- 16 个子数据集合计 720 条 full 任务结果

按方法：

- `blockwise`：240
- `snapkv`：240
- `chunkkv`：240

按压缩率：

- 0.3：144
- 0.4：144
- 0.5：144
- 0.6：144
- 0.7：144

## 汇总产物

后处理输出：

- `evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_metrics_long.csv`

当前 CSV：

- 行数：865
- 其中 1 行表头，864 条 metric 记录

## 失败记录

当前 final failure：

- `ATC26_failed_jobs_final.jsonl`：0 行

此前 gov_report OOM 失败记录已备份：

- `ATC26_failed_jobs.jsonl.bak_govrerun_20260518_131948`
- `ATC26_failed_jobs_final.jsonl.bak_govrerun_20260518_131948`

## 后续建议

1. 可以进入表格整理和可视化阶段。
2. 绘图和表格建议读取 `ATC26_metrics_long.csv`，必要时回查每个任务的 `ATC26_config.yaml` 与 `ATC26_predictions.csv`。
3. 由于 progress 文件只反映最后一轮 gov_report 补跑，论文统计不要直接使用 progress 的 success/skipped 字段；应以完整性扫描和 `ATC26_metrics_long.csv` 为准。
