# Blockwise Stage3 实验方案

## 1. 目标

本轮 `stage3` 的目标不是继续枚举旧的 `summary_mode` 组合，而是验证两个更聚焦的问题：

1. 能否用更统一的 summary 机制缩小 `blockwise_main / norm_topk / multi_rep` 之间的任务分化
2. 在当前主数据集上，新的 blockwise 方案相对 `chunkkv` 的位置是否改善

基于当前项目的数据集原则，本轮前期验证只使用：

- `LongBench`
- `needle_in_haystack`
- `PG19`

不使用：

- `longbench-v2`
- `infinitebench`
- `RULER`

## 2. 数据集与预算

### 2.1 主数据集

- `LongBench`
  - `qasper`
  - `multifieldqa_en`
  - `hotpotqa`
  - `2wikimqa`
  - `musique`
  - `triviaqa`
- `needle_in_haystack`
  - `max_context_length=16384`
  - `needle_depth in {0, 25, 50, 75, 100}`
- `PG19`
  - `test` split
  - 正式实验优先使用官方 `pg19`
  - smoke test 或网络不稳定时可退回 `emozilla/pg19-test`

### 2.2 统一预算

- `compression_ratio=0.7`
- `fraction=0.2`
- `device=cuda:0`

### 2.3 PG19 专用设置

- `max_context_length=4096`
- `pg19_target_tokens=256`

## 3. Stage3 候选方法矩阵

## 3.1 基线组

### B1. `blockwise_main`

- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

### B2. `blockwise_norm_topk`

- `summary_mode=norm_topk_mean_only`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

### B3. `blockwise_multi_rep`

- `summary_mode=multi_rep_max`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`

### B4. `chunkkv_prefill_per_layer`

- 作为强基线保留

### B5. `blockwise_tail_query_special`

- 仅在 `LongBench/triviaqa` 上运行
- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=tail_query_relevance`
- `query_agg_mode=mean`
- `head_agg_mode=uniform_mean`

## 3.2 Stage3 新方法组

### S1. `blockwise_adaptive_fusion_v1`

目标：

- 在不训练参数的情况下，用规则融合 `mean / norm-topk / multi-rep`

建议实现：

- 默认骨架沿用 `blockwise_main`
- 新增块内统计量：
  - top-k norm concentration
  - norm variance
  - representative diversity
- 根据统计量输出三路权重：
  - 集中度高时提高 `norm-topk` 权重
  - 峰值分散时提高 `multi-rep` 权重
  - 中间状态保留 `mean` 分量

### S2. `blockwise_multi_rep_diverse_v1`

目标：

- 检验“多峰块建模是否因为 representative 冗余而受限”

建议实现：

- 在 `multi_rep_max` 的 representative 选择中加入去冗余策略
- 第一版先做最轻量约束：
  - 位置最小间隔
  - 相似度去重二选一

### S3. `blockwise_adaptive_query_agg_v1`

目标：

- 在 `max` 为主的前提下，给极少数更适合 `mean` 的情况留出口

建议实现：

- 只在 `mean` 和 `max` 之间做规则切换
- 不引入训练
- 第一版门控信号建议：
  - query window top-gap
  - query score entropy

## 4. 建议实验批次

## 4.1 第一批：最小验证批

目的：

- 快速确认新方法是否有信号

运行：

- `B1 blockwise_main`
- `B3 blockwise_multi_rep`
- `B4 chunkkv_prefill_per_layer`
- `S1 blockwise_adaptive_fusion_v1`
- `S2 blockwise_multi_rep_diverse_v1`

数据集：

- `LongBench`: 全 6 个任务
- `needle_in_haystack`
- `PG19`

## 4.2 第二批：补充分支批

目的：

- 验证是否值得引入更复杂自适应

运行：

- `B2 blockwise_norm_topk`
- `S3 blockwise_adaptive_query_agg_v1`
- `B5 blockwise_tail_query_special`
  - 仅 `LongBench/triviaqa`

## 5. 自动化要求

本轮实验控制器需要满足：

- 固定使用 `cuda:0`
- 每个 job 单独记录状态
- 自动记录失败任务到：
  - `failed_jobs.jsonl`
  - `failed_jobs_final.jsonl`
- 遇到异常后自动分析原因并重试
- 若结果目录已有完整 `metrics.json`，自动跳过
- 全部结束后自动触发结果分析

## 6. 失败重试策略

### 6.1 可自动重试的错误

- `CUDA out of memory`
- 模型加载时显存不足
- Hugging Face 数据下载抖动
- 网络 `SSL/EOF/connection reset`
- 单次进程异常退出但无结果产物

### 6.2 重试策略

- 默认每个 job 最多重试 `3` 次
- 第一次失败后记录错误摘要
- 第二次失败前做：
  - `torch.cuda.empty_cache()` 对应进程级清理
  - 重新拉起独立子进程
- 若已有完整 `metrics.json`，即使日志里出现假失败记录，也以结果产物为准

### 6.3 不自动吞掉的错误

- 参数非法
- 方法实现 bug
- scorer 崩溃且结果结构缺失

这类错误应直接保留在最终失败清单中，避免无限重试。

## 7. 输出组织

建议实验名：

- `blockwise_stage3_ratio70_fraction20_primarybench`

建议产物目录：

- `evaluation/results/experiments/blockwise_stage3_ratio70_fraction20_primarybench/artifacts/`
- `figure/experiments/blockwise_stage3_ratio70_fraction20_primarybench/`
- `note/blockwise_stage3_ratio70_fraction20_primarybench_analysis_zh.md`

## 8. 预期判断标准

如果满足以下任一条件，就说明 `stage3` 新方法值得继续：

- `S1` 在 `LongBench` 宏观上优于 `B1/B2/B3`
- `S2` 在 `needle_in_haystack` 和 `multifieldqa_en` 上稳定优于 `B3`
- `S1` 或 `S2` 在 `PG19` 上相对 `chunkkv` 缩小 perplexity 劣势

如果三条都不成立，则说明：

- 当前 blockwise 的主要上限不在 summary 组合层
- 下一步更值得做的是 `blockwise` 与 `chunkkv` 的结构融合，而不是继续 blockwise 内部细调
