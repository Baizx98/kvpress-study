# KVCore Lifecycle Decode 压缩准确率消融实验方案

## 1. 问题定义

本实验要回答：在 KVCore 论文中的 `Sparsity-guided KV Lifecycle Management` 设计下，如果 decode 阶段依据当前 query 对历史 KV block 做 block-wise 稀疏选择，这种稀疏访问信号对 LongBench 准确率的影响有多大。

这里不测试真实 KVCore runtime，不测试真实 CPU offload、GPU prefetch、异步搬运、scheduler 或 block allocator。实验只在 `kvpress-study` 中用模拟方式评估质量影响。

## 2. 论文设计对应关系

`note/ATC26.tex` 中 lifecycle management 的核心语义是：

- KVCore 把 block-level sparse scores 从单步 attention pruning signal 提升为 runtime lifecycle hint。
- lifecycle manager 根据当前分数、历史分数、request state 和 memory pressure 决定 block 是否继续 GPU resident、变为 compute-cold、offload 到 CPU、prefetch 回 GPU，或 permanent eviction。
- dynamic sparse computation 是可逆的：未选中的 block 没有被删除，只是当前 attention 不可见。
- offload/prefetch 是可恢复的数据搬运动作：理论上只要被下一次 attention 需要前成功取回，不应改变模型输出。
- permanent eviction 是不可逆动作，风险最高，不适合作为本次“理论无损 lifecycle compression”的主实验对象。

因此本实验把“模拟压缩”定义为：用 decode query-aware BlockWisePress 在每个刷新间隔选择 active blocks，模拟 KVCore 在 decode 阶段识别热/冷 block 的 sparse active set。若讨论 offload/prefetch 的理论无损性，则需要强调：真实 offload/prefetch 本身不会改变数学计算；准确率变化只来自是否真的把未 active 的 KV 从 attention 中排除。当前 `compute_cold_fixed_budget` 属于后者，是质量压力测试，不等价于完全无损的数据搬运。

## 3. 实验假设

### H1: 可恢复 lifecycle 搬运本身应无损

如果 CPU-offloaded block 在被 attention 访问前被正确 prefetch 回 GPU，那么 attention 输入 KV 与 full KV 完全一致，输出应与 full KV 一致。这个结论不需要 LongBench 质量实验证明，主要是系统正确性语义。

### H2: decode q-aware block-wise sparse active set 对质量影响应较小

如果只让 query-aware 选择出的 active blocks 参与 decode attention，LongBench 准确率可能有小幅下降，但下降应足够小，可作为“稀疏算法用于 decode lifecycle hint 时不会显著破坏生成质量”的消融证据。

### H3: 任务差异会很明显

摘要类任务（`gov_report`、`qmsum`、`multi_news`）和代码/检索类任务可能更敏感，因为答案依赖长距离细节或格式；分类/短 QA 任务可能波动较小。由于每个数据集只跑 1% 请求，单任务结果只能作为初步消融，主结论应以 16 个数据集 macro average 和 paired delta 为主。

## 4. 实验范围

### 数据集

使用完整 LongBench 16 个英文子数据集：

- `narrativeqa`
- `qasper`
- `multifieldqa_en`
- `hotpotqa`
- `2wikimqa`
- `musique`
- `triviaqa`
- `gov_report`
- `qmsum`
- `multi_news`
- `samsum`
- `trec`
- `passage_count`
- `passage_retrieval_en`
- `lcc`
- `repobench-p`

每个子数据集使用 `fraction=0.01`，固定 `seed=42`。`evaluate.py` 中的 fraction 采样逻辑是 `max(1, round(N * fraction))`，因此小数据集至少会保留 1 条请求。

### 模型

只使用 `MODEL=/Tan/model/Llama-3.1-8B-Instruct`。原因是本实验是 lifecycle 稀疏算法对准确率影响的消融，不是跨模型鲁棒性主实验；本轮不扩展到 Qwen3-8B 或 Mistral-7B-Instruct-v0.3。

### 方法

只跑两种方法：

1. `full_kv`
   - `press_name=no_press`
   - `compression_ratio=0.0`
   - 作为完整 KV baseline。

2. `decode_qaware_blockwise`
   - `press_name=dual_phase_per_layer`
   - `dual_phase_mode=compute_cold_fixed_budget`
   - `compression_ratio=0.0`
   - prefill 不做物理压缩，保持 full prompt KV。
   - decode 阶段每隔 `compression_interval=16` tokens 用 BlockWisePress 根据当前 decode query 选择 active blocks。
   - active block budget 使用 block score top-p，`p=0.9`。实现上对 block scores 做 softmax，按分数降序保留累计概率达到 0.9 的最小 block 集合，同时保留 prefix sink、recent blocks 和 partial tail block。
   - 未 active blocks 通过 attention mask/fake-key 方式从当前 attention 中排除，但物理 KV 仍保留，用来模拟可恢复的 cold/offloaded block。

不跑 SnapKV、ChunkKV、permanent eviction、hybrid decode。原因是本次目标是说明 KVCore lifecycle 稀疏算法在 decode 使用时对准确率的影响，不是重新做通用 KV compression 方法横向比较。

## 5. 关键配置

推荐固定配置：

```text
model=/Tan/model/Llama-3.1-8B-Instruct
dataset=longbench
fraction=0.01
seed=42
block_size=16
q_window_size=16
compression_interval=16
decode_top_p_threshold=0.9
summary_topk_keys=4
mean_key_weight=0.75
representative_k=4
summary_mode=mean_plus_norm_topk_mean
representative_mode=key_norm
query_agg_mode=max
head_agg_mode=uniform_mean
protected_recent_blocks=2
prefix_sink_blocks=1
```

LongBench per-task `max_new_tokens` 复用已有 `evaluation/ATC26_run_longbench16_prefill_sweep.py` 中的设置：

```text
narrativeqa=148, qasper=148, multifieldqa_en=84,
hotpotqa=52, 2wikimqa=52, musique=52, triviaqa=52,
gov_report=532, qmsum=532, multi_news=532,
samsum=148, trec=84, passage_count=52,
passage_retrieval_en=52, lcc=84, repobench-p=84
```

### 关于压缩强度

本方案使用 `decode_top_p_threshold=0.9`，也就是 active block 数由当前 decode query 对 block scores 的分布自适应决定，而不是使用固定 block 数或固定压缩率。

如果需要更强的质量压力测试，可增加补充档位：

- `decode_top_p_threshold=0.8`
- `decode_top_p_threshold=0.7`
- `decode_top_p_threshold=0.5`

但这会超过“只跑 full KV 和 decode q 感知 block wise press”这一轮的最小范围，建议作为第二阶段。

## 6. 实现计划

### 新增 runner

新增：

```text
evaluation/run_kvcore_lifecycle_decode_longbench16_1pct.py
```

职责：

- 枚举 LongBench 16 个子数据集。
- 每个子数据集生成 2 个 job：`full_kv` 与 `decode_qaware_blockwise`。
- 复用 `evaluation/evaluate.py`，不要改核心评测入口。
- 输出到：

```text
evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_1pct/artifacts/
```

- 保存：
  - `run.log`
  - `progress.jsonl`
  - `manifest.jsonl`
  - `failed_jobs.jsonl`
  - 每个 job 的 `predictions.csv`、`metrics.json`、`config.yaml`

### 新增 postprocess

新增：

```text
evaluation/postprocess_kvcore_lifecycle_decode_longbench16_1pct.py
```

职责：

- 汇总每个子数据集的 LongBench score。
- 计算：
  - full KV score
  - decode q-aware BlockWise score
  - absolute delta = blockwise - full
  - relative delta = delta / full
  - 16-task macro average
  - 按任务类型聚合的 macro average：single-doc QA、multi-doc QA、summarization、few-shot、synthetic、code
- 检查每组样本数，避免 1% 采样导致某些任务只有 1 条样本时被误读。

输出：

```text
evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_1pct/summary.csv
evaluation/results/experiments/kvcore_lifecycle_decode_longbench16_1pct/summary.json
note/kvcore_lifecycle_decode_longbench16_1pct_results_zh.md
```

### 可选 figure

如果结果稳定，再新增：

```text
figure/plot_kvcore_lifecycle_decode_longbench16_1pct.py
figure/experiments/kvcore_lifecycle_decode_longbench16_1pct/
```

图建议只做一张：

- x 轴：16 个 LongBench 子数据集
- y 轴：score delta
- 横线：macro average delta
- 用颜色区分任务类型

## 7. 运行策略

首轮用 L40S 单 GPU 顺序跑，避免多个 LongBench job 同时加载 8B 模型造成 OOM。当前机器中 L40S 是 physical GPU 0；runner 会检查 GPU 名称包含 `L40S`，避免误跑到 3090 或 A6000。

推荐命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=/Tan/model/Llama-3.1-8B-Instruct \
DEVICE=cuda:0 \
GPU_INDEX=0 \
MIN_FREE_MB=38000 \
.venv/bin/python evaluation/run_kvcore_lifecycle_decode_longbench16_1pct.py
```

注意：`GPU_INDEX=0` 在 runner 中表示 physical GPU 0；`DEVICE=cuda:0` 表示进程内可见的第一张卡。设置 `CUDA_VISIBLE_DEVICES=0` 后，两者都指向 L40S。

## 8. 预期结果与解释边界

### 预期结果

- `full_kv` 是准确率上界。
- `decode_qaware_blockwise` 的 macro average 应接近 full KV。
- 如果 delta 很小，可以支持如下论文消融表述：

  “Using query-aware block-level sparse active sets during decode introduces negligible LongBench quality degradation under a 1% LongBench16 ablation, suggesting that KVCore's lifecycle hints can guide decode-time KV management without materially changing generation quality.”

### 不能过度解释的点

- 这个实验不能证明真实 KVCore offload/prefetch 的 latency、bandwidth、overlap 或 scheduler 收益。
- 如果采用 `compute_cold_fixed_budget`，它测的是“当前 attention 只访问 active blocks”的质量影响，不是严格数学无损的 offload/prefetch。
- 每个数据集只跑 1%，单任务分数方差可能较大；论文中应把它标成 ablation/sanity check，而不是最终 LongBench 主质量结果。
- 如果结果显示某些任务下降明显，需要查看 prediction diff 和样本长度，不能直接得出算法不适合该任务的强结论。

## 9. 最小验证计划

开始正式跑前先做 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=/Tan/model/Llama-3.1-8B-Instruct \
DEVICE=cuda:0 \
GPU_INDEX=0 \
ONLY_LONGBENCH_TASKS=trec \
SMOKE=1 \
.venv/bin/python evaluation/run_kvcore_lifecycle_decode_longbench16_1pct.py
```

smoke test 只检查：

- `no_press` 能正常生成并保存 metrics。
- `dual_phase_per_layer + compute_cold_fixed_budget + compression_ratio=0.0` 能正常生成并保存 metrics。
- 输出路径、config 去重、失败重试、postprocess 都正常。

通过后再跑完整 16 个 LongBench 子集。
