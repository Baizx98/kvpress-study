# ATC26 LongBench-16 实验方案

## 目标

单独补做一个 `ATC26_longbench16` 实验，用 16 个非中文 LongBench 子任务评估 prefill-only KVCache 压缩效果，避免把当前 `LongBench-QA-6` 结果误写成完整 LongBench。

本实验不包含 Needle 和 PG19，只覆盖 LongBench 16 个非中文任务。

## 实验命名和输出

实验名：

`ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv`

建议输出目录：

`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/`

建议图像目录：

`figure/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/`

所有新脚本、日志、结果、图片文件名继续使用 `ATC26` 前缀。

## 模型

保持当前 ATC26 设置不变：

- `/Tan/model/Llama-3.1-8B-Instruct`
- `/Tan/model/Mistral-7B-Instruct-v0.3`
- `/Tan/model/Qwen3-8B`

## 方法

保持当前 ATC26 设置不变：

- `blockwise`
- `snapkv`
- `chunkkv`

其中 `blockwise` 继续使用当前 SOTA BlockWisePress 参数，不启用 per-layer ratio；所有层使用同一个压缩率。

补充约束：

- `blockwise` 的前两层不压缩，即 `prefill_skip_first_layers=2`。
- 跳过前两层后，其余层仍使用同一个压缩率。
- `snapkv` 和 `chunkkv` 保持原设置，不额外跳过前两层。

## 压缩率

本实验改为 5 个压缩率：

- `0.3`
- `0.4`
- `0.5`
- `0.6`
- `0.7`

不跑 `0.8`。

## LongBench-16 任务

### QA 类

| task | max_new_tokens |
|---|---:|
| `narrativeqa` | 148 |
| `qasper` | 148 |
| `multifieldqa_en` | 84 |
| `hotpotqa` | 52 |
| `2wikimqa` | 52 |
| `musique` | 52 |
| `triviaqa` | 52 |

### Summarization 类

| task | max_new_tokens |
|---|---:|
| `gov_report` | 532 |
| `qmsum` | 532 |
| `multi_news` | 532 |
| `samsum` | 148 |

### Classification / Retrieval / Counting 类

| task | max_new_tokens |
|---|---:|
| `trec` | 84 |
| `passage_count` | 52 |
| `passage_retrieval_en` | 52 |

### Code 类

| task | max_new_tokens |
|---|---:|
| `lcc` | 84 |
| `repobench-p` | 84 |

这些 `max_new_tokens` 来自本地实际加载 `Xnhyacinth/LongBench` 对应 config 后读到的数据集字段。

## 实验规模

Full matrix：

`3 models × 3 methods × 5 ratios × 16 LongBench tasks = 720 jobs`

Smoke test：

同样生成 720 个 job，但 `fraction=0.01`，用于确认 16 个 task 全部能加载、推理、评测和落盘。

## 执行策略

### 1. 先做 smoke test

目标：

- 确认 16 个 LongBench task 都能加载。
- 确认摘要任务和代码任务的 scorer 正常。
- 确认 `max_new_tokens=532` 的长输出任务不会触发明显 OOM。
- 确认 `ATC26_config.yaml`、`ATC26_metrics.json`、`ATC26_predictions.csv` 全部落盘。

Smoke 参数：

- `--mode smoke`
- `fraction=0.01`
- 使用 L40S 和 A6000。
- 不使用 3090。

### 2. Smoke 通过后再 full

Full 参数：

- `--mode full`
- `fraction=1.0`
- 断点续跑开启。
- 每个 job 完成后更新进度文件。

### 3. GPU 策略

沿用当前约束：

- 使用物理 GPU 0：L40S
- 使用物理 GPU 2：A6000
- 尽量不使用物理 GPU 1：RTX 3090

建议运行环境变量：

```bash
ATC26_GPUS=0,2 ATC26_MIN_FREE_MB=0:36000,2:24000 MAX_RETRIES=3 POLL_SECONDS=60
```

runner 内部应继续用 GPU UUID 设置 `CUDA_VISIBLE_DEVICES`，避免物理 GPU 编号和进程内 `cuda:0` 混淆。

## 需要新增或调整的脚本

建议不要直接覆盖当前已完成的 ATC26 6-task runner，而是新增独立脚本：

- `evaluation/ATC26_run_longbench16_prefill_sweep.py`
- `evaluation/ATC26_postprocess_longbench16_prefill_sweep.py`
- `figure/ATC26_plot_longbench16_prefill_sweep.py`

原因：

- 当前 `ATC26_run_prefill_sweep.py` 同时包含 LongBench、Needle、PG19。
- LongBench-16 是单独实验，不应和已有 `LongBench-QA-6 + Needle + PG19` 结果混在一个 artifacts 目录。
- 独立 runner 便于在论文里区分 `LongBench-QA-6` 和 `LongBench-16`。

BlockWise job 的 CLI 需要额外带上：

```bash
--prefill_skip_first_layers 2
```

SnapKV / ChunkKV job 不添加该参数。

## 断点续跑设计

每个 job 的完成判断必须读取 `ATC26_config.yaml`，至少校验：

- `model`
- `dataset == longbench`
- `data_dir`
- `press_name`
- `compression_ratio`
- `fraction`
- 对 `blockwise`，还需要校验 `prefill_skip_first_layers == 2`

不能只检查 `ATC26_metrics.json` 是否存在，避免 smoke 结果被误认为 full 结果。

建议进度文件：

`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_progress.md`

建议失败文件：

`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_failures.jsonl`

## 聚合结果

Postprocess 应至少输出：

- `ATC26_metrics_long.csv`
- `ATC26_metrics_wide.csv`
- `ATC26_metrics_full_long.csv`
- `ATC26_metrics_full_wide.csv`
- `ATC26_missing_full_jobs.csv`
- `ATC26_longbench16_task_group_summary.csv`

其中 `ATC26_metrics_full_long.csv` 是论文画图和制表的默认输入。

## 建议图表

### 主图

1. `ATC26_longbench16_macro_by_model.png`
   - 每个模型一个子图。
   - x 轴为 compression ratio。
   - y 轴为 LongBench-16 macro average。
   - 三条线：BlockWise / SnapKV / ChunkKV。

2. `ATC26_longbench16_task_group_by_model.png`
   - 按 task group 聚合：QA / Summarization / Classification-Retrieval / Code。
   - 用于展示不同任务类型的趋势差异。

### 补充图

3. `ATC26_longbench16_task_grid.png`
   - 3 models × 16 tasks 的细分图。
   - 用于 appendix 或内部诊断。

## 风险和注意事项

1. 运行时间会明显增加。
   - 当前补全规模是 720 full jobs。
   - `gov_report/qmsum/multi_news` 的 `max_new_tokens=532`，会比 QA 任务慢很多。

2. 摘要任务对 decode 阶段更敏感。
   - 虽然本实验仍只压缩 prefill，但长输出任务的总耗时和质量会受 decode 影响更明显。
   - 分析时建议单独看 Summarization group，不要只看总平均。

3. Code 任务样本数是 500。
   - `lcc` 和 `repobench-p` 比多数自然语言任务样本更多。
   - Full 运行耗时会更高。

4. 不建议复用旧 6-task 结果直接拼表。
   - 为了可复现和目录清晰，建议 LongBench-16 独立跑完整 16 个任务。
   - 如果后续时间紧，可以再讨论是否将旧 6-task full 结果迁移/引用到新表，但这会增加 provenance 复杂度。

## 执行前检查清单

1. 确认 L40S 和 A6000 空闲。
2. 确认 Hugging Face LongBench 16 个 config 已可加载。
3. 先运行 smoke test。
4. 检查 smoke 后：
   - `ATC26_failures.jsonl` 为空或只有可解释失败。
   - 每个 task 至少有成功 metrics。
   - 长输出任务没有 OOM。
5. Smoke 通过后再启动 full。
6. Full 结束后只使用 `ATC26_metrics_full_long.csv` 做图和表格。
