# ATC26_longbench16 实验状态汇报

核查时间：2026-05-14 14:14 CST

## 总体状态

`ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv` 仍在运行，由 watchdog 托管：

- watchdog 进程：`1594545`
- runner 进程：`1594560`
- tmux session：`ATC26_longbench16_watch`
- 当前跳过数据集：`gov_report`
- 当前任务矩阵：`675` 个任务

watchdog 日志显示自 2026-05-12 20:09:13 启动后一直在托管 runner。runner 目前仍活着，没有进入自动重启循环。

## 当前进度

来自 `artifacts/ATC26_progress.md`，更新时间为 2026-05-14 14:14:40：

- Total：675
- Skipped：109
- Success：192
- Running：2
- Pending：372
- Failed：0

按模型：

- `llama31_8b_instruct`：skipped=109, success=116
- `mistral_7b_instruct_v03`：success=76, running=2, pending=147
- `qwen3_8b`：pending=225

当前 runner 已完成 Llama 的非 `gov_report` 部分，正在推进 Mistral。Qwen3 还没开始正式 full 结果。

## 当前运行任务

进程层面确认：

- L40S / worker0：
  - PID：`4179988`
  - job：`mistral_7b_instruct_v03__longbench_musique__blockwise__r0p4`
  - 状态：正在运行
  - GPU 显存：约 18.8 GB used

- A6000 / worker1：
  - PID：`4185388`
  - job：`mistral_7b_instruct_v03__longbench_musique__blockwise__r0p5`
  - 状态：进程存在，但处于 `D` / `wait_on_page_bit_common`
  - GPU 显存：暂未显示该进程显存占用，说明大概率还在模型/数据文件读取或页缓存等待阶段

GPU 状态：

- GPU0 L40S：18.8 GB used, 26.6 GB free, util 77%
- GPU2 A6000：15 MB used, 48.4 GB free, util 0%
- GPU1 3090：有其他用户进程，占用约 5.8 GB；当前 longbench16 没有使用 3090

## 原始结果落盘统计

按 `ATC26_config.yaml + ATC26_metrics.json + ATC26_predictions.csv` 重新扫描，当前 full 原始结果共 310 组。这个数量包含早先已产生但当前跳过的 `gov_report` 原始结果，因此会大于当前 progress 中 `success + skipped` 的即时计数。

按模型：

- `llama31_8b_instruct`：235
- `mistral_7b_instruct_v03`：75
- `qwen3_8b`：0

按方法：

- `blockwise`：106
- `snapkv`：104
- `chunkkv`：100

按压缩率：

- 0.3：63
- 0.4：61
- 0.5：62
- 0.6：62
- 0.7：62

## 失败情况

`ATC26_failed_jobs_final.jsonl` 目前只有 1 条 final failure：

- `llama31_8b_instruct__longbench_gov_report__snapkv__r0p4`
- 原因：L40S 上连续 3 次 OOM
- 当前策略：已跳过整个 `gov_report`，所以该失败不会阻塞后续非 `gov_report` 实验

当前跳过 `gov_report` 后的 675 任务矩阵中，progress 显示 failed=0。

## 风险与建议

1. A6000 上当前 worker1 处于 `wait_on_page_bit_common`，这通常是文件 IO 或页缓存等待。若长时间不进入 GPU 计算，需要检查模型/数据读取路径或存储压力。
2. watchdog 正常托管 runner，后续 runner 异常退出会自动 `--resume` 重启。
3. 目前不建议手动杀 runner；如果要处理 A6000 的 D 状态，先观察 10-20 分钟日志是否推进。
4. `gov_report` 已保留原始结果但被排除在当前恢复矩阵外，后续若要补齐，需要单独安排更保守的显存策略。

## 关键路径

- progress：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_progress.md`
- watchdog log：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_watch_skip_gov_report.log`
- runner log：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_watch_skip_gov_report.runner.log`
- final failures：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_failed_jobs_final.jsonl`
