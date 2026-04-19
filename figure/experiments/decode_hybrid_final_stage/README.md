# decode_hybrid_final_stage

本目录保存 `decode_hybrid_final_stage` 的可视化结果。

## 图表说明

- [longbench_hybrid_budget_lines.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/longbench_hybrid_budget_lines.png)
  - `LongBench` 三个长输出任务上，`Permanent / Compute-Cold / Hybrid` 的 budget 曲线
- [longbench_hybrid_macro.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/longbench_hybrid_macro.png)
  - `LongBench` 宏平均对比
- [ruler_hybrid_grouped.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/ruler_hybrid_grouped.png)
  - `RULER` 四个补充任务的 grouped bar 对比
- [summary.json](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/summary.json)
  - 本轮实验的结构化汇总数据

## 配套结果与分析

- [实验 README](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/decode_hybrid_final_stage/README.md)
- [分析文档](/home10T/bzx/workspace/kvpress-study/note/decode_hybrid_final_stage_analysis_zh.md)

## 读取规则

- 本轮 `24` 个逻辑配置均有 `metrics.json`
- `failed_jobs*.jsonl` 中的 `RULER` 记录是假失败，不影响图表数据
- 图表直接基于有效 `metrics.json` 生成，不受控制器误报影响
