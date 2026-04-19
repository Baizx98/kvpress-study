# decode_hybrid_final_stage

## 实验目的

这是最后一次 decode 算法探索，目标是验证：

- `dense_prefill + hybrid_decode`

能否比现有两条最强路线更稳：

- `dense_prefill + permanent_decode`
- `dense_prefill + compute_cold_decode`

## 运行脚本

- [run_decode_hybrid_final_stage.py](/home10T/bzx/workspace/kvpress-study/evaluation/run_decode_hybrid_final_stage.py)

## 数据集

- `LongBench / gov_report, qmsum, multi_news`
- `RULER / 4096 / niah_single_3, niah_multikey_2, niah_multikey_3, qa_2`

## 方法

- `Permanent 128`
- `Permanent 160`
- `Compute-Cold 128`
- `Compute-Cold 160`
- `Hybrid 128/96`
- `Hybrid 160/128`

说明：

- `Hybrid 128/96` 表示 `total_budget=128, active_budget=96`
- `Hybrid 160/128` 表示 `total_budget=160, active_budget=128`

## 关键配置

- `prefill compression_ratio = 0.0`
- `block_size = 16`
- `q_window_size = 16`
- `compression_interval = 16`
- `query_agg_mode = max`
- `summary_mode = mean_plus_norm_topk_mean`
- `representative_mode = key_norm`
- `head_agg_mode = uniform_mean`
- `protected_recent_blocks = 2`

LongBench 样本筛选：

- `min_answer_tokens = 64`
- `min_context_tokens = 4000`
- `max_filtered_samples = 20`

RULER 设置：

- `samples_per_task = 20`
- `max_new_tokens = 128`

## 完整性

- 逻辑配置总数：`24`
- `metrics.json`：`24`
- 有效完成结果：`24`

说明：

- `failed_jobs*.jsonl` 中有 6 条 `RULER` 假失败记录
- 这些记录的 `return_code=0`
- 真实结果已全部落盘，分析时按 `metrics.json` 为准

## 产物位置

- [artifacts](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_hybrid_final_stage/artifacts)
- [run.log](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_hybrid_final_stage/artifacts/run.log)
- [分析文档](/home10T/bzx/workspace/kvpress-study/note/decode_hybrid_final_stage_analysis_zh.md)
- [图像目录](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/README.md)

## 优先查看

1. [decode_hybrid_final_stage_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/decode_hybrid_final_stage_analysis_zh.md)
2. [longbench_hybrid_budget_lines.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/longbench_hybrid_budget_lines.png)
3. [longbench_hybrid_macro.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/longbench_hybrid_macro.png)
4. [ruler_hybrid_grouped.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/ruler_hybrid_grouped.png)
