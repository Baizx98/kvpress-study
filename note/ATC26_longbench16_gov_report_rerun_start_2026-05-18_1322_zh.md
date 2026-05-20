# ATC26_longbench16 gov_report 补跑启动记录

时间：2026-05-18 13:22 CST

## 操作目标

补跑此前跳过的 `gov_report` 子数据集，并覆盖旧的 `gov_report` 失败记录。

当前补跑矩阵：

- 数据集：`longbench:gov_report`
- 任务数：45 = 3 models * 3 methods * 5 ratios
- 模型：Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen3-8B
- 方法：blockwise、snapkv、chunkkv
- 压缩率：0.3、0.4、0.5、0.6、0.7
- GPU：L40S + A6000

## 已做修改

新增 `ATC26_ONLY_LONGBENCH_TASKS` 环境变量支持，使 runner 可以只调度指定 LongBench 子数据集。

新增 watchdog：

- `evaluation/ATC26_watch_longbench16_gov_report.sh`
- tmux session：`ATC26_longbench16_gov_watch`

新增 hang monitor：

- `evaluation/ATC26_monitor_gov_report_hangs.sh`
- tmux session：`ATC26_gov_hang_monitor`
- 作用：如果 gov_report evaluate 子进程长时间处于 `D / wait_on_page_bit_common` 且没有 GPU compute app，则杀掉子进程，让 runner 自动 retry。

## 失败记录处理

已备份并清理旧 gov_report 失败记录：

- `ATC26_failed_jobs.jsonl.bak_govrerun_20260518_131948`
- `ATC26_failed_jobs_final.jsonl.bak_govrerun_20260518_131948`

清理后：

- `ATC26_failed_jobs_final.jsonl`：0 行
- `ATC26_failed_jobs.jsonl`：3 行，均非当前 gov_report final failure

## 启动状态

dry-run 已确认只准备 45 个 gov_report jobs。

当前 watchdog 已启动，progress 显示：

- Total：45
- Skipped：10
- Running：2
- Pending：33
- Failed：0

当前运行：

- L40S / GPU0：
  - job：`llama31_8b_instruct__longbench_gov_report__chunkkv__r0p4`
  - PID：`1478482`
  - GPU 显存：约 17.9 GB

- A6000 / GPU2：
  - job：`llama31_8b_instruct__longbench_gov_report__snapkv__r0p4`
  - PID：`1478481`
  - GPU 显存：约 17.7 GB

3090 未被该补跑使用。

## 关键日志

- watchdog log：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_watch_gov_report.log`
- runner log：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_watch_gov_report.runner.log`
- hang monitor log：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_monitor_gov_report_hangs.log`
