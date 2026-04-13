# blockwise_ablation_ratio70_longbench_stage1

## 实验目的

将 `RULER stage1` 的 blockwise 逐轴消融迁移到 `LongBench`，在高压缩率 `ratio=0.7` 下比较：

- A. `block summary form`
- B. `block-internal representative selection`
- C. `query window aggregation`
- D. `head aggregation`
- 以及 `Quest-style prefill block scorer`

## 运行脚本

- 主脚本：
  [run_blockwise_ablation_ratio70_longbench_stage1.sh](/home10T/bzx/workspace/kvpress-study/evaluation/run_blockwise_ablation_ratio70_longbench_stage1.sh)
- `triviaqa` 缺失补跑：
  [rerun_blockwise_ablation_ratio70_longbench_stage1_triviaqa_missing.sh](/home10T/bzx/workspace/kvpress-study/evaluation/rerun_blockwise_ablation_ratio70_longbench_stage1_triviaqa_missing.sh)

## 数据集

- `LongBench / hotpotqa`
- `LongBench / multifieldqa_en`
- `LongBench / triviaqa`

## 方法

- `block_wise_prefill_per_layer`
- `quest_blockwise_prefill_per_layer`

## 关键 sweep 维度

- `compression_ratio=0.7`
- `fraction=0.2`
- `block_size=16`
- `q_window_size=64`
- 不设置 `skip_first`

## 采样比例

- 不使用 `samples_per_task`
- 各任务直接按 `fraction=0.2` 采样

## 产物位置

- 原始结果：
  [artifacts](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_ablation_ratio70_longbench_stage1/artifacts)
- 主运行日志：
  [run.log](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_ablation_ratio70_longbench_stage1/artifacts/run.log)
- 失败记录：
  - [failed_tasks.txt](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_ablation_ratio70_longbench_stage1/artifacts/failed_tasks.txt)
  - [failed_tasks_rerun.txt](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_ablation_ratio70_longbench_stage1/artifacts/failed_tasks_rerun.txt)

## 当前完整性

- `hotpotqa`：13/13，完整
- `multifieldqa_en`：13/13，完整
- `triviaqa`：13/13，完整

当前可以做公平比较的数据集：

- `hotpotqa`、`multifieldqa_en`、`triviaqa`

## 推荐优先查看

- 中文分析：
  [blockwise_ablation_ratio70_longbench_stage1_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/blockwise_ablation_ratio70_longbench_stage1_analysis_zh.md)

