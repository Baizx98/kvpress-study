# ATC26_longbench16 实验状态核查

核查时间：2026-05-12 14:30 CST

## 结论

当前 `ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv` 正式实验没有继续运行。进度文件停留在 2026-05-12 13:51:44，并显示 2 个 running job，但系统进程中已经没有 `ATC26_run_longbench16` 或 `evaluation/evaluate.py` 相关进程，因此这 2 个 running 状态是 stale 状态。

正式实验尚未完整完成。按原始结果目录中的 `ATC26_config.yaml`、`ATC26_metrics.json`、`ATC26_predictions.csv` 重新统计：

- 预期 full 任务数：720
- 已完成 full 结果：112
- 缺失 full 结果：608
- smoke 结果：144
- 有效结果目录总数：256

## 当前 GPU 状态

`nvidia-smi` 显示：

- GPU0 L40S：约 41.4 GB used，约 4.0 GB free，util 约 96%
- GPU1 RTX 3090：空闲，本实验未使用
- GPU2 RTX A6000：约 15 MB used，基本空闲

GPU0 上的占用不是当前 ATC26 实验进程，而是用户 `byh` 的任务：

- `python -m lib.Optimize_new2`
- `/home/byh/.conda/envs/byhpy310/bin/python -m lm_eval ... --device cuda:0 ...`

因此当前不应继续把 ATC26 任务调度到 GPU0，除非该占用释放。

## 已完成 full 结果分布

按模型：

- `llama31_8b_instruct`：112
- `mistral_7b_instruct_v03`：0
- `qwen3_8b`：0

按 LongBench 子数据集：

- `narrativeqa`：15
- `qasper`：15
- `multifieldqa_en`：15
- `hotpotqa`：15
- `2wikimqa`：15
- `musique`：15
- `triviaqa`：15
- `gov_report`：7

其余 8 个子数据集尚未进入正式结果：

- `qmsum`
- `multi_news`
- `samsum`
- `trec`
- `passage_count`
- `passage_retrieval_en`
- `lcc`
- `repobench-p`

按方法：

- `blockwise`：40
- `snapkv`：37
- `chunkkv`：35

按压缩率：

- 0.3：23
- 0.4：22
- 0.5：23
- 0.6：22
- 0.7：22

## 异常与中断点

最后的主运行日志记录到：

- 2026-05-12 13:34:57：`llama31_8b_instruct__longbench_gov_report__snapkv__r0p4` attempt 1 在 GPU0 上 OOM
- 2026-05-12 13:35:57：同一任务 attempt 2 在 GPU0 上重试
- 2026-05-12 13:51:44：`llama31_8b_instruct__longbench_gov_report__snapkv__r0p5` 完成
- 2026-05-12 13:51:44：`llama31_8b_instruct__longbench_gov_report__snapkv__r0p6` 在 GPU2 上启动

两个 stale running job 的单任务日志都停在 inference 过程中，没有看到正常完成记录：

- `llama31_8b_instruct__longbench_gov_report__snapkv__r0p4` attempt 2：停在 48/200 左右
- `llama31_8b_instruct__longbench_gov_report__snapkv__r0p6` attempt 1：停在 49/200 左右

`ATC26_failed_jobs_final.jsonl` 为空，说明目前没有被 runner 判定为最终失败的任务。`ATC26_failed_jobs.jsonl` 有 3 条，其中正式实验相关的是 `gov_report/snapkv/r0p4` 的一次 OOM attempt。

## 建议后续步骤

1. 暂时不要使用 GPU0 L40S，因为它已经被其他用户任务占满，且此前 `gov_report/snapkv/r0p4` 已在 GPU0 上出现 OOM。
2. 若要立即恢复，建议先只用 GPU2 A6000 执行：

```bash
ATC26_GPUS=2 ATC26_MIN_FREE_MB=2:24000 MAX_RETRIES=3 POLL_SECONDS=60 \
  .venv/bin/python evaluation/ATC26_run_longbench16_prefill_sweep.py --mode full --resume
```

3. 如果 GPU0 释放后再做双卡恢复，可使用原命令：

```bash
ATC26_GPUS=0,2 ATC26_MIN_FREE_MB=0:36000,2:24000 MAX_RETRIES=3 POLL_SECONDS=60 \
  .venv/bin/python evaluation/ATC26_run_longbench16_prefill_sweep.py --mode full --resume
```

4. 恢复前后都应以原始结果目录为准，而不是只看 stale progress 文件。恢复脚本的 `--resume` 会跳过已经有完整 `ATC26_config.yaml`、`ATC26_metrics.json`、`ATC26_predictions.csv` 的任务。
