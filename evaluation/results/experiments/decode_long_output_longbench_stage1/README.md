# decode_long_output_longbench_stage1

## 实验目的

验证长输出场景下，decode 阶段 fixed-budget 策略是否能在保持质量的同时支持后续显存/计算管理设计。

## 运行脚本

- [`evaluation/run_decode_long_output_longbench_stage1.py`](/home10T/bzx/workspace/kvpress-study/evaluation/run_decode_long_output_longbench_stage1.py)

## 数据集

- `LongBench / gov_report`
- `LongBench / qmsum`
- `LongBench / multi_news`

## 方法

- `prefill_only_no_decode_pruning`
- `decode_permanent_eviction_fixed_budget`
- `decode_compute_cold_fixed_active_budget`

## 关键配置

- `compression_ratio=0.3`
- `block_size=16`
- `q_window_size=16`
- `query_agg_mode=max`
- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=key_norm`
- `head_agg_mode=uniform_mean`
- `protected_recent_blocks=2`
- `min_answer_tokens=64`
- `min_context_tokens=4000`
- `max_filtered_samples=20`

## 完整性

- `gov_report`: `3/3`
- `qmsum`: `3/3`
- `multi_news`: `3/3`

无最终失败任务。

## 产物位置

- [`artifacts`](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_long_output_longbench_stage1/artifacts)
- [`run.log`](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_long_output_longbench_stage1/artifacts/run.log)
- [`watchdog.log`](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_long_output_longbench_stage1/artifacts/watchdog.log)
- [`分析文档`](/home10T/bzx/workspace/kvpress-study/note/decode_long_output_longbench_stage1_analysis_zh.md)

## 主要结论

- 两种 decode fixed-budget 策略质量损失都不大。
- `permanent_fixed_budget` 和 `compute_cold_fixed_budget` 的宏平均非常接近。
- 当前更突出的瓶颈是运行开销，而不是质量崩溃。
- 下一步应优先补 `TPOT / peak memory / active blocks / live blocks` 等系统指标。
