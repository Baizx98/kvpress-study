# ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv

## 实验目的

为 ATC26 论文补充 prefill-only KVCache 压缩实验，比较 BlockWise、SnapKV、ChunkKV 在 LongBench、needle_in_haystack、PG19 上的质量变化。

## 运行脚本

- `evaluation/ATC26_run_prefill_sweep.py`
- `evaluation/ATC26_postprocess_prefill_sweep.py`
- `figure/ATC26_plot_prefill_sweep.py`

## 数据集

- `longbench:2wikimqa`
- `longbench:gov_report`
- `longbench:hotpotqa`
- `longbench:lcc`
- `longbench:multi_news`
- `longbench:multifieldqa_en`
- `longbench:musique`
- `longbench:narrativeqa`
- `longbench:passage_count`
- `longbench:passage_retrieval_en`
- `longbench:qasper`
- `longbench:qmsum`
- `longbench:repobench-p`
- `longbench:samsum`
- `longbench:trec`
- `longbench:triviaqa`

## 方法

- `blockwise`
- `chunkkv`
- `snapkv`

## 模型

- `llama31_8b_instruct`
- `mistral_7b_instruct_v03`
- `qwen3_8b`

## 压缩率

`0.3`, `0.4`, `0.5`, `0.6`, `0.7`

## 产物位置

- 原始结果：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/raw/`
- 长表：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_metrics_long.csv`
- 宽表：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_metrics_wide.csv`
- full-only 长表：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_metrics_full_long.csv`
- full-only 宽表：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_metrics_full_wide.csv`
- 进度日志：`evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv/artifacts/ATC26_progress.md`
