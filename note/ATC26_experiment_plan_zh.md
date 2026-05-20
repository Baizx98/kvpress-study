# ATC26 Prefill 压缩实验方案

## 目标

为 ATC 2026 论文补充 prefill-only KVCache 压缩结果，比较 `BlockWisePress`、`SnapKV`、`ChunkKV` 在三类长上下文数据集和三种 8B 级模型上的质量退化、稳定性与可视化趋势。

所有本轮新增运行脚本、原始结果、运行日志、进度日志、分析表格和图片文件名都使用 `ATC26` 前缀。

## 实验矩阵

### 模型

- `/Tan/model/Llama-3.1-8B-Instruct`
- `/Tan/model/Mistral-7B-Instruct-v0.3`
- `/Tan/model/Qwen3-8B`

### 方法

- `block_wise`
  - 使用当前仓库 stage3 的 SOTA 主方案：`summary_mode=mean_plus_norm_topk_mean`、`representative_mode=key_norm`、`query_agg_mode=max`、`head_agg_mode=uniform_mean`
  - 固定 `block_size=16`、`q_window_size=64`、`summary_topk_keys=4`、`mean_key_weight=0.75`、`representative_k=4`、`multi_rep_k=4`、`query_topr=16`、`head_topk=1`
- `snapkv`
- `chunkkv`
  - 当前注册实现为 `ChunkKVPress(press=SnapKVPress(), chunk_length=20)`

本轮不使用 `PrefillPerLayerRatioPress` / per-layer ratio wrapper，所有层使用相同 `compression_ratio`。评测路径仍然只做 prefill 阶段压缩，不启用 `DecodingPress` 或 decode 阶段压缩。

### 压缩率

- `0.3`
- `0.4`
- `0.5`
- `0.6`
- `0.7`
- `0.8`

这里沿用当前代码的 `compression_ratio` 语义，表示压缩比例，而不是保留比例。执行前会用一个 smoke job 再确认结果目录、配置和输出中的 ratio 含义一致。

### 数据集

- `LongBench`
  - 第一版建议使用已有 stage3 主任务：`qasper`、`multifieldqa_en`、`hotpotqa`、`2wikimqa`、`musique`、`triviaqa`
  - 对应 `max_new_tokens` 沿用现有 runner：`qasper=148`，`multifieldqa_en=84`，其余 QA 任务 `52`
- `needle_in_haystack`
  - `max_context_length=16384`
  - `needle_depth=[0,25,50,75,100]`
  - `max_new_tokens=50`
- `PG19`
  - `max_context_length=4096`
  - `pg19_target_tokens=256`
  - `pg19_source_dataset=/Tan/dataset/pg19-test`

### 运行规模

完整矩阵规模：

- 模型：3
- 方法：3
- 压缩率：6
- 数据集项：LongBench 6 个子任务 + Needle 1 项 + PG19 1 项 = 8
- 总 job 数：`3 * 3 * 6 * 8 = 432`

smoke 矩阵按用户确认后的要求覆盖所有模型、所有数据集，数据比例控制为 `1%`：

- 模型：3 个都跑
- 方法：3 个都跑
- 压缩率：`0.5`
- 数据集：LongBench 6 个子任务 + Needle + PG19
- 数据比例：`fraction=0.01`
- 总 job 数：`3 * 3 * 1 * 8 = 72`

## 输出目录与命名

正式实验名：

`ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19`

结果目录：

`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/`

最小结构：

- `artifacts/`
  - 每个 job 的原始 `config.yaml`
  - 原始 `predictions*.csv`
  - 原始 `metrics*.json`
  - job stdout/stderr 子日志
  - controller 总日志
  - controller 进度日志
  - manifest 和断点状态
- `README.md`

图像目录：

`figure/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/`

计划图片：

- `ATC26_longbench_quality_vs_compression_by_model.png`
- `ATC26_longbench_quality_vs_compression_by_task.png`
- `ATC26_needle_accuracy_vs_compression_by_depth.png`
- `ATC26_pg19_ppl_vs_compression_by_model.png`
- `ATC26_method_delta_vs_snapkv_by_dataset.png`

分析文档：

`note/ATC26_prefill_sweep_analysis_zh.md`

## 脚本计划

### 1. Controller

新增：

`evaluation/ATC26_run_prefill_sweep.py`

职责：

- 生成完整 job manifest，不直接把矩阵散落在 shell 脚本里。
- 每个 job 包含 `job_id`、模型、数据集、任务、方法、压缩率、输出目录、命令行参数、状态。
- 使用两张卡并行跑 job：
  - worker 0：物理 GPU `0`，`NVIDIA L40S`
  - worker 1：物理 GPU `2`，`NVIDIA RTX A6000`
- 每个 worker 启动单独评测进程，进程内通过 GPU UUID 暴露单卡，因此子进程统一使用 `DEVICE=cuda:0`。不能使用数字 `CUDA_VISIBLE_DEVICES=<physical_gpu>`，因为 CUDA runtime 的数字顺序可能和 `nvidia-smi` 物理 index 不一致，容易误用 3090。
- 每个 job 启动前检查对应物理 GPU 可用显存。
  - L40S 默认阈值建议 `MIN_FREE_MB_L40S=36000`
  - A6000 默认阈值建议 `MIN_FREE_MB_A6000=40000`
- 用 `.venv/bin/python evaluation/evaluate.py ...` 运行单个 job。
- 每个 job 单独写 `ATC26_job_<job_id>.log`。
- 总日志写 `ATC26_run.log`。
- 进度日志写 `ATC26_progress.jsonl` 和 `ATC26_progress.md`。
- 失败记录写 `ATC26_failed_jobs.jsonl`、最终失败写 `ATC26_failed_jobs_final.jsonl`。

### 2. Manifest

新增：

`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_manifest.jsonl`

每行一个 job：

- `job_id`
- `model_name`
- `model_path`
- `dataset`
- `data_dir`
- `method_key`
- `press_name`
- `compression_ratio`
- `command`
- `result_base_dir`
- `status`
- `attempt`
- `started_at`
- `finished_at`

### 3. Progress 日志

新增：

- `ATC26_progress.jsonl`：机器可读，方便恢复和统计。
- `ATC26_progress.md`：人类可读，方便 `tail -f` 或直接打开观察。

`ATC26_progress.md` 每次 job 完成后刷新：

- 总进度：`finished / total`
- 按模型统计
- 按数据集统计
- 按方法统计
- 按 GPU / worker 统计
- 当前正在运行的 job
- 最近完成的 20 个 job
- 当前失败队列
- 预计剩余 job 数

### 4. Postprocess

新增：

`evaluation/ATC26_postprocess_prefill_sweep.py`

职责：

- 扫描 `artifacts/` 下所有 `metrics*.json` 和 `config.yaml`。
- 聚合为：
  - `ATC26_metrics_long.csv`
  - `ATC26_metrics_wide.csv`
  - `ATC26_job_status.csv`
- 对重复重跑目录保留原始结果，但聚合时选择最新成功结果。
- 对每个 job 标注 `success`、`failed`、`missing_metrics`、`duplicate_success`。

### 5. Plot

新增：

`figure/ATC26_plot_prefill_sweep.py`

职责：

- 从 `ATC26_metrics_long.csv` 读取，而不是直接扫原始目录。
- 按数据集类型分别画图，避免把 LongBench 分数、Needle accuracy、PG19 perplexity 混成一个指标。
- 输出到 `figure/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/`。

## 断点续跑设计

断点续跑优先级：

1. 如果 job 的 expected result dir 内已有 `metrics.json` 或 `metrics*.json`，判定为成功，直接跳过。
2. 如果 `ATC26_progress.jsonl` 中该 job 已记录 `success`，但文件缺失，以文件为准，重新跑。
3. 如果 job 曾失败但未超过 `MAX_RETRIES`，从失败点重试。
4. 如果 job 在中断时处于 `running`，下一次启动时改回 `pending` 并重新执行。

失败分类：

- `oom`
- `killed`
- `network`
- `cache_mismatch`
- `pg19_network`
- `missing_metrics`
- `unknown`

默认 `MAX_RETRIES=3`。OOM 或 killed 后等待对应 GPU 显存恢复；网络或 cache 问题短暂等待后重试。

## 关键实现检查点

执行前需要先做这些轻量检查：

1. 确认 L40S 是物理 `GPU_INDEX=0`，A6000 是物理 `GPU_INDEX=2`。
2. 确认三个模型路径存在，尤其用户写的 `Qwen3-8b` 在本机实际目录是 `/Tan/model/Qwen3-8B`。
3. 确认 `PRESS_REGISTRY` 中有：
   - `block_wise`
   - `snapkv`
   - `chunkkv`
4. 修改或绕开当前 `evaluate.py` 中的 `CUDA_VISIBLE_DEVICES="0,1"` 硬编码，让子进程尊重 controller 传入的 GPU UUID。A6000 必须用 UUID 绑定，不能用数字 `2` 绑定。
5. 确认普通 press 路径不会触发 decode 阶段压缩；本实验不使用 `DecodingPress`。
6. 先跑 1% smoke 矩阵，确认全部模型和全部数据集都能产出 `ATC26_metrics.json`。
7. smoke 通过后再启动完整 432-job 矩阵。

## 执行命令草案

只准备，不在未确认前运行：

```bash
ATC26_GPUS=0,2 ATC26_MIN_FREE_MB=0:36000,2:40000 MAX_RETRIES=3 \
  .venv/bin/python evaluation/ATC26_run_prefill_sweep.py --mode smoke
```

完整实验：

```bash
ATC26_GPUS=0,2 ATC26_MIN_FREE_MB=0:36000,2:40000 MAX_RETRIES=3 \
  .venv/bin/python evaluation/ATC26_run_prefill_sweep.py --mode full
```

观察进度：

```bash
tail -f evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_progress.md
```

重启续跑：

```bash
ATC26_GPUS=0,2 ATC26_MIN_FREE_MB=0:36000,2:40000 MAX_RETRIES=3 \
  .venv/bin/python evaluation/ATC26_run_prefill_sweep.py --mode full --resume
```

## 风险与处理

- 完整矩阵 432 个 job，两卡并行仍然可能耗时较长。先 smoke，再全量。
- PG19 是 perplexity 视角，不能和 LongBench/Needle 直接平均，只做单独图和单独表。
- Needle 固定 16K 可能区分度有限，因此保留 depth 维度图，不只看均值。
- 当前 `evaluate.py` 对已有结果目录会自动加 `/1`、`/2`，所以 controller 需要自己预测 expected result dir，并用 metrics 文件判断是否跳过，避免重复跑造成聚合混乱。
- Qwen3 可能有 tokenizer/chat template 或 remote code 差异，smoke 阶段必须覆盖。
