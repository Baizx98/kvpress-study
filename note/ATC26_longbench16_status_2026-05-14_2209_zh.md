# ATC26_longbench16 实验状态汇报

核查时间：2026-05-14 22:09 CST

## 总体状态

`ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv` 仍由 watchdog 托管，runner 进程还活着：

- watchdog：`1594545`
- runner：`1594560`
- tmux session：`ATC26_longbench16_watch`
- 当前跳过数据集：`gov_report`
- 当前任务矩阵：`675`

watchdog 尚未触发重启循环，说明 runner 没有退出。

## Progress 状态

来自 `artifacts/ATC26_progress.md`，更新时间为 2026-05-14 22:09:05：

- Total：675
- Skipped：109
- Success：223
- Running：2
- Pending：341
- Failed：0

按模型：

- `llama31_8b_instruct`：skipped=109, success=116
- `mistral_7b_instruct_v03`：success=107, running=2, pending=116
- `qwen3_8b`：pending=225

当前状态说明：Llama 非 `gov_report` 部分已完成；Mistral 已完成 107 个 full job，正在 `qmsum`；Qwen3 还没开始。

## 当前运行项

progress 显示当前 running：

- `mistral_7b_instruct_v03__longbench_qmsum__blockwise__r0p5`
  - worker0 / GPU0 L40S
  - attempt 2
  - 对应进程处于 `D` / `wait_on_page_bit_common`
  - 当前 job log 为 0 字节，尚未进入正常日志输出阶段

- `mistral_7b_instruct_v03__longbench_qmsum__blockwise__r0p6`
  - worker1 / GPU2 A6000
  - attempt 1
  - 当前 job log 为 0 字节，尚未进入正常日志输出阶段

GPU 当前没有 ATC26 compute app：

- GPU0 L40S：17 MB used, 45.3 GB free, util 0%
- GPU2 A6000：18 MB used, 48.4 GB free, util 0%
- GPU1 3090：29 MB used, 24.0 GB free, util 0%

这说明两个 running 任务目前没有进入 GPU 推理阶段，更像是卡在文件 IO、页缓存或模型加载前的系统等待。

## 原始结果落盘统计

按 `ATC26_config.yaml + ATC26_metrics.json + ATC26_predictions.csv` 扫描，当前 full 原始结果共 342 组。该统计包含早先已经产生但当前跳过的 `gov_report` 原始结果。

按模型：

- `llama31_8b_instruct`：235
- `mistral_7b_instruct_v03`：107
- `qwen3_8b`：0

按方法：

- `blockwise`：117
- `snapkv`：114
- `chunkkv`：111

按压缩率：

- 0.3：70
- 0.4：68
- 0.5：68
- 0.6：68
- 0.7：68

## 失败与异常

`ATC26_failed_jobs_final.jsonl` 仍只有 1 条 final failure：

- `llama31_8b_instruct__longbench_gov_report__snapkv__r0p4`
- 原因：L40S 上 3 次 OOM
- 当前策略：`gov_report` 已整体跳过，不阻塞当前 675 任务矩阵

新增的非 final failure：

- `mistral_7b_instruct_v03__longbench_qmsum__blockwise__r0p5`
- attempt 1 在 2026-05-14 21:31:33 被 kill，return_code=-9
- runner 已自动重试 attempt 2

## 风险判断

当前最主要风险不是 GPU OOM，而是两个 running 任务没有 GPU 占用，且至少一个进程处于 `wait_on_page_bit_common`。这通常和文件系统 IO、页缓存、模型/数据读取等待有关。由于 runner 和 watchdog 仍活着，短时间内可以继续观察；如果 20-30 分钟后 progress 和 job log 仍没有推进，再考虑处理卡住的子进程，让 watchdog 自动 resume。

## 关键路径

- progress：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_progress.md`
- watchdog log：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_watch_skip_gov_report.log`
- runner log：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_watch_skip_gov_report.runner.log`
- final failures：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_failed_jobs_final.jsonl`
