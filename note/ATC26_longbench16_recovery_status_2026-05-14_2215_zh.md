# ATC26_longbench16 恢复状态

核查与恢复时间：2026-05-14 22:12-22:15 CST

## 触发原因

系统卡顿后重新检查发现，`ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv` 的 watchdog 和 runner 仍存在，但两个 evaluate 子进程都卡在 `D / wait_on_page_bit_common`：

- `mistral_7b_instruct_v03__longbench_qmsum__blockwise__r0p5`
- `mistral_7b_instruct_v03__longbench_qmsum__blockwise__r0p6`

当时 GPU0/GPU2 都没有 ATC26 compute app，两个当前 job 的日志文件大小为 0，说明实验处于“进程未退出但实际不推进”的挂住状态。

## 恢复动作

已停止旧的卡住进程树：

- watchdog：`1594545`
- runner：`1594560`
- stuck evaluate：`615795`
- stuck evaluate：`630160`

随后重新启动 tmux watchdog：

- tmux session：`ATC26_longbench16_watch`
- watchdog PID：`642507`
- runner PID：`642522`

恢复命令仍由 watchdog 托管，保持：

- 跳过 `gov_report`
- 只使用 L40S + A6000
- 使用 `--resume`
- 已完成结果自动跳过

## 恢复后状态

恢复后当前运行任务：

- L40S / GPU0：
  - PID：`642561`
  - job：`mistral_7b_instruct_v03__longbench_qmsum__blockwise__r0p5`
  - 状态：`R`
  - 显存：约 18.2 GB

- A6000 / GPU2：
  - PID：`642562`
  - job：`mistral_7b_instruct_v03__longbench_qmsum__blockwise__r0p6`
  - 状态：`R`
  - 显存：约 18.1 GB

GPU 利用率恢复：

- L40S：约 71%
- A6000：约 42%
- 3090：未被 longbench16 使用

## 当前进度快照

恢复后的 progress 文件更新时间：2026-05-14 22:14:25

- Total：675
- Skipped：332
- Running：2
- Pending：341
- Failed：0

当前 progress 中 `success=0` 是因为这轮 watchdog 重新启动后把已完成任务按 resume 逻辑标记为 `skipped`，不代表原始结果丢失。

## 说明

这次不是普通 runner 崩溃，而是子任务卡在内核不可中断等待，导致 watchdog 没有自动重启。手动清掉卡住进程树后，`--resume` 已经从未完成点重新拉起，当前 GPU 计算恢复。
