# ATC26 LongBench-16 Full Run 状态

## 启动信息

启动时间：`2026-05-11 18:25:43`

启动命令：

```bash
ATC26_GPUS=0,2 ATC26_MIN_FREE_MB=0:36000,2:24000 MAX_RETRIES=3 POLL_SECONDS=60 \
  .venv/bin/python evaluation/ATC26_run_longbench16_prefill_sweep.py --mode full --resume
```

使用 GPU：

- physical GPU 0: L40S
- physical GPU 2: RTX A6000
- physical GPU 1: RTX 3090 未用于本实验

## 实验矩阵

Full matrix：

`3 models × 3 methods × 5 ratios × 16 LongBench tasks = 720 jobs`

压缩率：

- `0.3`
- `0.4`
- `0.5`
- `0.6`
- `0.7`

BlockWise 设置：

- `prefill_skip_first_layers=2`

## 当前观察

首批任务：

- L40S: `llama31_8b_instruct__longbench_narrativeqa__blockwise__r0p3`
- A6000: `llama31_8b_instruct__longbench_narrativeqa__blockwise__r0p4`

截至 `2026-05-11 19:25:30`：

- `r0.3` 已完成并落盘。
- L40S 已继续执行 `llama31_8b_instruct__longbench_narrativeqa__blockwise__r0p5`。
- A6000 仍在执行 `llama31_8b_instruct__longbench_narrativeqa__blockwise__r0p4`。
- `ATC26_progress.md` 显示 full 进度：`success=1`，`running=2`，`failed=0`。

## 耗时风险

Full `narrativeqa/blockwise` 单 job 很慢。L40S 上首个 `r0.3` 任务耗时约 1 小时；A6000 上 `r0.4` 更慢。

这意味着完整 720-job 矩阵可能需要很长 wall-clock 时间。当前未调整实验方案，仍按用户确认的 full 设置继续运行。

## 监控路径

进度：

`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_progress.md`

日志：

`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/logs/`

原始结果：

`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/raw/`

聚合结果：

`ATC26_metrics_long.csv` 会在 runner 完整结束后由 postprocess 更新；运行中可能仍主要显示 smoke 聚合记录。
