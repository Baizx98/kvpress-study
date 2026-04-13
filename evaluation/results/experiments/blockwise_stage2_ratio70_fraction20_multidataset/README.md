# blockwise_stage2_ratio70_fraction20_multidataset

## 实验目的

基于 stage2 设计报告，验证 blockwise 主线推荐矩阵在多数据集上的稳定性，并加入 `chunkkv` 作为额外对照方法。

## 运行脚本

- 主总控脚本：
  [run_blockwise_stage2_ratio70_fraction20_multidataset.py](/home10T/bzx/workspace/kvpress-study/evaluation/run_blockwise_stage2_ratio70_fraction20_multidataset.py)

## 数据集

- `RULER / 4096 / niah_single_3, niah_multikey_3, qa_2`
- `LongBench / qasper`
- `LongBench / multifieldqa_en`
- `LongBench / hotpotqa`
- `LongBench / 2wikimqa`
- `LongBench / musique`
- `LongBench / triviaqa`
- `Needle in a Haystack / max_context_length=16384 / needle_depth=[0,25,50,75,100]`

## 方法

- `blockwise_main`
- `blockwise_norm_topk`
- `blockwise_multi_rep`
- `blockwise_tail_query_special`（仅 LongBench）
- `chunkkv_prefill_per_layer`

## 关键配置

- `compression_ratio=0.7`
- `fraction=0.2`
- `query_aware=true`
- 不设置 `prefill_skip_first_layers`

## 产物位置

- 原始结果：
  [artifacts](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_stage2_ratio70_fraction20_multidataset/artifacts)
- 主运行日志：
  [run.log](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_stage2_ratio70_fraction20_multidataset/artifacts/run.log)
- 失败记录：
  - [failed_jobs.jsonl](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_stage2_ratio70_fraction20_multidataset/artifacts/failed_jobs.jsonl)
  - [failed_jobs_final.jsonl](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/blockwise_stage2_ratio70_fraction20_multidataset/artifacts/failed_jobs_final.jsonl)

## 当前完整性

- `ruler:4096`：4/4
- `longbench:qasper`：5/5
- `longbench:multifieldqa_en`：5/5
- `longbench:hotpotqa`：5/5
- `longbench:2wikimqa`：5/5
- `longbench:musique`：5/5
- `longbench:triviaqa`：5/5
- `needle_in_haystack:16384`：4/4

## 最终失败项

- `needle_in_haystack:16384__blockwise_main`: attempts=3, reason=unknown
- `needle_in_haystack:16384__blockwise_norm_topk`: attempts=3, reason=unknown
- `needle_in_haystack:16384__blockwise_multi_rep`: attempts=3, reason=unknown
- `needle_in_haystack:16384__chunkkv_prefill`: attempts=3, reason=unknown

## 推荐优先查看

- 中文分析：
  [blockwise_stage2_ratio70_fraction20_multidataset_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/blockwise_stage2_ratio70_fraction20_multidataset_analysis_zh.md)

