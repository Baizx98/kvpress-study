# ATC26_longbench16 gov_report 补跑状态

核查时间：2026-05-18 19:29 CST

## 总体状态

`gov_report` 补跑仍在运行，由 watchdog 和 hang monitor 托管：

- gov_report watchdog：`ATC26_longbench16_gov_watch`
- hang monitor：`ATC26_gov_hang_monitor`
- runner PID：`1478472`
- 任务矩阵：45
- 当前 failed final：0

## Progress

`ATC26_progress.md` 更新时间：2026-05-18 19:27:56

- Total：45
- Skipped：10
- Success：11
- Running：2
- Pending：22
- Failed：0

按模型：

- `llama31_8b_instruct`：skipped=10, success=5
- `mistral_7b_instruct_v03`：success=6, running=2, pending=7
- `qwen3_8b`：pending=15

按方法：

- `blockwise`：skipped=5, success=5, pending=5
- `snapkv`：skipped=4, success=2, running=2, pending=7
- `chunkkv`：skipped=1, success=4, pending=10

## 当前运行

- A6000 / GPU2：
  - job：`mistral_7b_instruct_v03__longbench_gov_report__snapkv__r0p4`
  - PID：`1928383`
  - 状态：运行中
  - GPU 显存：约 16.0 GB
  - GPU util：约 87%

- L40S / GPU0：
  - job：`mistral_7b_instruct_v03__longbench_gov_report__snapkv__r0p5`
  - PID：`1937268`
  - 状态：`D / wait_on_page_bit_common`
  - 当前没有出现在 `nvidia-smi` compute app 中
  - hang monitor 已开启，如果持续长时间 D 状态且无 GPU compute app，会杀掉该子进程触发 runner retry

3090 未被当前 gov_report 补跑使用。

## 已完成进展

自 13:20 启动后已经完成：

- Llama 的剩余 `snapkv r0p4` 与 `chunkkv r0p4-r0p7`
- Mistral 的 `blockwise r0p3-r0p7`
- Mistral 的 `snapkv r0p3`

当前正在推进 Mistral 的 `snapkv r0p4/r0p5`。

## 风险

L40S worker 当前又出现短期 `D / wait_on_page_bit_common`，但 hang monitor 已经在运行。A6000 侧正常计算。当前不需要手动干预，建议继续观察 monitor 是否触发 retry。

## 关键日志

- progress：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_progress.md`
- runner：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_watch_gov_report.runner.log`
- watchdog：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_watch_gov_report.log`
- hang monitor：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_monitor_gov_report_hangs.log`
