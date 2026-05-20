# ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19

## 实验目的

为 ATC26 论文补充 prefill-only KVCache 压缩实验，比较 BlockWise、SnapKV、ChunkKV 在 LongBench、needle_in_haystack、PG19 上的质量变化。

## 运行脚本

- `evaluation/ATC26_run_prefill_sweep.py`
- `evaluation/ATC26_postprocess_prefill_sweep.py`
- `figure/ATC26_plot_prefill_sweep.py`

## 数据集

- `longbench:2wikimqa`
- `longbench:hotpotqa`
- `longbench:multifieldqa_en`
- `longbench:musique`
- `longbench:qasper`
- `longbench:triviaqa`
- `needle_in_haystack:16384`
- `pg19:test`

## 方法

- `blockwise`
- `chunkkv`
- `snapkv`

## 模型

- `llama31_8b_instruct`
- `mistral_7b_instruct_v03`
- `qwen3_8b`

## 压缩率

`0.3`, `0.4`, `0.5`, `0.6`, `0.7`, `0.8`

## 采样比例

- smoke test: `fraction=0.01`
- full run: `fraction=1.0`
- 论文结果优先使用 full-only 聚合表，避免混入 smoke 记录。

## 产物位置

- 原始结果：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/raw/`
- 长表：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_metrics_long.csv`
- 宽表：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_metrics_wide.csv`
- full-only 长表：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_metrics_full_long.csv`
- full-only 宽表：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_metrics_full_wide.csv`
- LongBench 子数据集长表：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_longbench_subdataset_long.csv`
- LongBench 子数据集宽表：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_longbench_subdataset_wide.csv`
- Needle depth 长表：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_needle_depth_long.csv`
- Needle depth 宽表：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_needle_depth_wide.csv`
- 进度日志：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_progress.md`
- 缺失项检查：`evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_missing_full_jobs.csv`
- 完整性分析：`note/ATC26_prefill_sweep_completeness_zh.md`

## 推荐查看顺序

1. `note/ATC26_prefill_sweep_completeness_zh.md`
2. `artifacts/ATC26_metrics_full_long.csv`
3. `figure/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/README.md`
