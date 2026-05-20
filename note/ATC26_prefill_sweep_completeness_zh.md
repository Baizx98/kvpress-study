# ATC26 Prefill Sweep 完整性检查

## 结论

- ATC26 full 实验已补齐。当前 full-only 聚合结果为 432 条，覆盖 3 个模型、3 个方法、6 个压缩率和 8 个 dataset key。
- 当前没有 `ATC26_run_prefill_sweep.py`、`ATC26_monitor_full.py` 或 `evaluation/evaluate.py --config_file /dev/null` 残留进程。
- GPU 已空闲；本次补跑只使用 L40S 和 A6000，3090 未参与计算。
- runner 进度文件显示本轮补跑 `success=21`、`failed=0`、`skipped=411`。`ATC26_missing_full_jobs.csv` 已刷新为空表。
- 论文表格和折线图应使用 `ATC26_metrics_full_long.csv` / `ATC26_metrics_full_wide.csv`，不要直接使用包含 smoke 记录的 `ATC26_metrics_long.csv`。

## 当前产物

| 类型 | 路径 |
|---|---|
| full long table | `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_metrics_full_long.csv` |
| full wide table | `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_metrics_full_wide.csv` |
| all metrics, smoke + full | `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_metrics_long.csv` |
| missing full report | `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_missing_full_jobs.csv` |
| progress log | `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_progress.md` |
| figures | `figure/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/` |

## 完整性证据

| 项目 | 数量 | 说明 |
|---|---:|---|
| Manifest 期望 job | 432 | `ATC26_manifest.jsonl` |
| full-only metrics 记录 | 432 | `ATC26_metrics_full_long.csv` |
| 缺失 full job | 0 | `ATC26_missing_full_jobs.csv` 只有 header |
| 本轮补跑成功 | 21 | `ATC26_progress.md` |
| 本轮补跑失败 | 0 | `ATC26_progress.md` |

### full-only 覆盖情况

| 维度 | 覆盖 |
|---|---:|
| 每个模型 | 144 records |
| 每个 dataset key | 54 records |
| 每个方法 | 144 records |
| 每个压缩率 | 72 records |
| fraction | 全部为 `1.0` |

覆盖的 dataset key：

- `longbench:qasper`
- `longbench:multifieldqa_en`
- `longbench:hotpotqa`
- `longbench:2wikimqa`
- `longbench:musique`
- `longbench:triviaqa`
- `needle_in_haystack:16384`
- `pg19:test`

## 已修复的问题

之前完整性检查发现 21 个 full job 被 smoke 结果误判为已完成。根因是旧版断点判断只检查某个 `job_id` 目录下是否存在 `ATC26_metrics.json`，没有校验 `fraction`。

已完成的修复：

- `evaluation/ATC26_run_prefill_sweep.py` 的 `has_completed_results()` 现在读取 `ATC26_config.yaml`，并同时校验 `model`、`compression_ratio`、`fraction`。
- `evaluation/ATC26_postprocess_prefill_sweep.py` 额外输出 full-only 表：`ATC26_metrics_full_long.csv` 和 `ATC26_metrics_full_wide.csv`。
- `figure/ATC26_plot_prefill_sweep.py` 优先使用 `ATC26_metrics_full_long.csv` 画图。

## 后续步骤

1. 先基于 full-only 表做质量分析，不要混用 smoke 记录。
2. LongBench、Needle、PG19 分开分析：LongBench 和 Needle 是 higher-is-better，PG19 perplexity 是 lower-is-better。
3. 检查三张已生成图的趋势是否符合预期，再决定论文中使用表格、折线图，或二者结合。
4. 如果要进一步提升论文图质量，建议另写 publication plotting 脚本，固定颜色、marker、字体和输出 PDF/SVG。
