# PG19 Dense Position PPL 方案

## 目标

用 `Llama-3.1-8B-Instruct` 在 PG19 上比较 `snapkv`、`chunkkv`、`blockwise` 三种 prefill KV 压缩方法的长上下文困惑度退化趋势，生成一张类似论文常见 PPL-position curve 的密集折线图：

- 横坐标：`token length`，定义为被预测 token 在原文中的位置，也就是使用前 k 个 token 作为 context 后预测第 k+1 个真实 token
- 纵坐标：`PPL`，优先使用 `subword_perplexity`
- token length 上限：`32768`
- 主要问题：上下文长度增长时，三种方法在相同 KV 压缩预算下谁更稳定

## 修正版核心定义

本方案不是稀疏测试 `2k/4k/8k/...` 几个 context length，也不是对每个长度生成一段文本。它按论文里常见 PPL 曲线的方式，对 PG19 原文中的很多 token position 做 continuation likelihood：

```text
for k in [1024, 1536, 2048, ..., 32768]:
    context = tokens[0:k]
    target_window = tokens[k:k+256]
    nll(k) = mean -log p(target_window[1:] | compressed_cache(context), target_window[:-1])
    ppl(k) = exp(mean_book nll(k))
```

主实验固定：

```text
compression_ratio = 0.5
target_tokens = 256
primary_metric = PPL over target_window[1:], skipping the first continuation token
stride = 512
max_token_length = 32768
```

因此横坐标的每个点都是一个真实 token position，曲线点会比较密集。注意第一个 continuation token 的 logit 来自 context prefill 最后一位，尚未使用压缩后的 KV cache；为了比较 KV cache 压缩方法，主指标跳过这个 token，从第二个 continuation token 开始计算 PPL。

## 现有代码依据

- `evaluation/evaluate.py` 已支持核心参数：`model`、`press_name`、`compression_ratio`、`block_size`、`pg19_target_tokens`、`pg19_source_dataset`、`fraction`、`max_context_length`、`seed`。
- PG19 数据准备逻辑会从 `pg19_source_dataset` 读取 test split，并按 `max_context_length` 截取 context、按 `pg19_target_tokens` 截取 target。
- 但当前 `build_pg19_evaluation_dataframe()` 每本书只构造一个 `max_context_tokens` 点，不适合论文里点很密的 PPL-position curve。本轮需要新增一个专用 runner 或数据构造函数，对同一本书采样多个 context position。
- PG19 PPL 当前实现是 continuation likelihood：先对 context 做 prefill，随后对 target tokens 计算 NLL/PPL。本轮使用 `target_tokens=256`，主指标跳过第一个 target token，只统计真正依赖压缩后 cache 的 continuation tokens。
- registry 中已有三种方法：`snapkv`、`chunkkv`、`block_wise`；旧 ATC26 runner 也已经把这三者作为同组方法使用。

## 实验设置

### 模型与数据

- 模型：`/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：PG19 test
- 数据根：优先使用仓库已有路径约定 `/Tan/dataset/pg19-test`
- target tokens：固定 `256`。主指标跳过第一个 continuation token，统计后续 255 个真实 token 的平均 NLL/PPL。
- seed：`42`
- dtype/device：沿用当前 `evaluate.py` 默认 pipeline；如显存不足，再固定 `device` 和 `model_kwargs`，但第一版方案不改 evaluator 语义

### 方法

| 方法名 | `press_name` | 关键参数 | 说明 |
|---|---|---|---|
| SnapKV | `snapkv` | 无额外 block 参数 | 原始 SnapKV baseline |
| ChunkKV | `chunkkv` | `block_size=16` | registry 默认 `chunk_length=20`，runner 传 `block_size=16` 与历史脚本一致 |
| Blockwise | `block_wise` | `block_size=16`, `q_window_size=64`, `summary_topk_keys=4`, `mean_key_weight=0.75`, `representative_k=4`, `multi_rep_k=4`, `query_topr=16`, `head_topk=1`, `summary_mode=mean_plus_norm_topk_mean`, `representative_mode=key_norm`, `query_agg_mode=max`, `head_agg_mode=uniform_mean` | 与已有 ATC26 prefill sweep 的 blockwise 主配置对齐 |

### 压缩预算

第一张主图固定 `compression_ratio=0.5`。

原因：本轮目标是先看中等压缩强度下，PPL 随 context token length 增长的退化趋势；同时单图只比较 token length 维度，避免 compression ratio 和 token length 两个变量混在一起。

可选扩展：如果第一张图趋势正常，再补 `compression_ratio in {0.3, 0.7}`，每个 ratio 单独一张图或做 facet，不建议把所有 ratio 画进同一张图。

### Token Length / Position Sweep

论文式密集曲线不应该只取 `[2048, 4096, ...]` 这类稀疏点。建议用均匀 stride 生成 dense positions：

```text
token_lengths = [1024, 1536, 2048, 2560, ..., 32768]
stride = 512
```

如果 512 stride 仍太慢，降级为 `stride=1024`；如果需要更接近论文曲线，可在 full run 用 `stride=256`，再用 rolling mean 或 bin mean 平滑。

每个 token length 点都独立构造 PG19 样本：对每本书截取长度为 k 的 context，然后用紧随其后的 256 个真实 token 作为 continuation window。这样横坐标严格对应“压缩长度为 k 的上下文后，后续 continuation 的 PPL”：

```text
sample(book, k):
  context = tokens[0:k]
  target_window = tokens[k:k+256]
  primary_nll = mean -log p(target_window[1:] | compressed_cache(context), target_window[:-1])
```

每个 k 的聚合方式：

```text
PPL(k) = exp(mean_nll(target_window[1:] | compressed_cache(context_length=k)))
```

其中 `mean_nll` 在所有满足 `source_token_count > k + 256` 的 PG19 books 上平均。辅助表中可以同时保存包含第一个 target token 的 PPL，但论文主图用跳过第一个 token 的版本。

### 样本规模

分两阶段：

1. Smoke：`fraction=0.02` 或最多 8-16 本书，用于确认 32k 不 OOM、输出字段完整、三条曲线能画出来。
2. Full：`fraction=0.2` 起步；如果方差明显或曲线交叉不稳定，再提高到 `fraction=0.5/1.0`。

注意：PG19 book 长度分布会影响可用样本数。每个 token length 点必须记录实际保留下来的 book 数；如果 32k 有效样本过少，图中要标注 `n`，或把结论限制为“可用长书子集”。

## 任务矩阵

主实验 job 数：

- 方法：3
- token length：`ceil((32768 - 1024) / stride) + 1`，`stride=512` 时约 63 个位置点
- compression ratio：1
- 总计：建议不要把每个 k 拆成独立 job，而是每个方法一个 job，内部循环所有 dense positions；主实验共 3 个 job

建议额外跑 `no_press` 作为参考上界/无压缩曲线：

- `no_press` x dense positions = 1 个 job
- 主图可以用灰色虚线显示，不参与三方法排名

最终矩阵若包含 baseline：4 个 job，每个 job 内部产生约 63 个位置点。

## 结果产物设计

建议实验名：

```text
pg19_ppl_length_sweep_llama31_8b_snapkv_chunkkv_blockwise_ratio50
```

建议路径：

- 原始结果：`evaluation/results/experiments/pg19_ppl_length_sweep_llama31_8b_snapkv_chunkkv_blockwise_ratio50/artifacts/`
- 聚合表：`evaluation/results/experiments/pg19_ppl_length_sweep_llama31_8b_snapkv_chunkkv_blockwise_ratio50/artifacts/pg19_ppl_length_sweep_metrics.csv`
- 图片目录：`figure/experiments/pg19_ppl_length_sweep_llama31_8b_snapkv_chunkkv_blockwise_ratio50/`
- 主图：`figure/experiments/pg19_ppl_length_sweep_llama31_8b_snapkv_chunkkv_blockwise_ratio50/pg19_ppl_vs_token_length.pdf`
- 分析笔记：`note/pg19_ppl_length_sweep_llama31_8b_snapkv_chunkkv_blockwise_ratio50_analysis_zh.md`

聚合表字段建议：

| 字段 | 说明 |
|---|---|
| `method` | `snapkv` / `chunkkv` / `blockwise` / optional `no_press` |
| `model` | 模型路径或模型 key |
| `dataset` | `pg19:test` |
| `token_length` | context prefix length k，也等于被预测 token 的前缀长度 |
| `target_tokens` | 固定 `256` |
| `compression_ratio` | 固定 `0.5`，baseline 为 `0.0` 或空 |
| `subword_ppl` | 主指标，跳过第一个 continuation token |
| `avg_nll` | PPL 前的平均 NLL，跳过第一个 continuation token |
| `subword_ppl_all_targets` | 辅助指标，包含第一个 continuation token |
| `num_books` | 实际评测样本数 |
| `stride` | token length 采样间隔 |
| `result_dir` | 对应原始 job 输出目录 |

## 画图规范

- x 轴：`Token length`，曲线点来自 dense positions；主刻度显示 `2k, 4k, 8k, 16k, 24k, 32k`
- y 轴：`PG19 PPL`，默认线性坐标；如果 32k PPL 爆炸，再额外输出 log-y 版本，但主图仍先保留线性坐标
- 每条线一个方法：
  - `SnapKV`
  - `ChunkKV`
  - `Blockwise`
  - optional `No compression` 灰色虚线
- 图例中标注固定预算：`compression_ratio=0.5`
- caption 说明：PPL 是 continuation PPL，target window 固定 256，主指标跳过第一个不受压缩 cache 影响的 token

## 最小验证计划

执行前先做三个检查，但不改变主实验设置：

1. 路径检查：确认 `/Tan/model/Llama-3.1-8B-Instruct` 和 `/Tan/dataset/pg19-test` 存在。
2. 单点 smoke：只跑 `token_length=2048`，`fraction=0.02`，三种方法各一个 job，确认可以得到 continuation-window NLL/PPL。
3. dense smoke：使用 `stride=4096` 跑 `[4096, 8192, ..., 32768]`，优先 `blockwise` 和 `chunkkv`，确认显存和耗时可接受。
4. full dense：使用 `stride=512`，必要时降级到 `stride=1024`。

Smoke 全部通过后，再启动 3-method full dense sweep。

## 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| 高 token length 有效 PG19 样本太少 | 曲线末端方差大 | 聚合表记录每个 k 的 `num_books`；必要时只用能覆盖 32k 的长书子集，并在图注说明 |
| Llama3.1 8B 32k prefill 显存压力大 | OOM 或耗时很长 | 先跑 smoke；必要时固定 GPU、降低并发、保留 batch size=1 |
| 三种方法实际压缩语义不完全一致 | 公平性争议 | 固定 `compression_ratio=0.5`，并在方法表记录 block/chunk 参数 |
| 当前 evaluator 不支持 dense position curve | 不能直接复用旧 PG19 runner | 新增专用 dense PG19 PPL runner，输出 per-position NLL/PPL 表 |
| dense continuation PPL 单点噪声大 | 曲线抖动 | 对多个 books 和 255 个有效 continuation tokens 求均值；画图可额外输出 rolling mean，但原始点保留 |
| 只看 PPL 缺少效率指标 | 无法判断 cost-efficiency | 本轮主图只做质量曲线；后续可补 latency / peak memory 表，不混入本图 |

## 预期结论形式

最终分析按以下结构写：

1. 问题：长 context 下 KV 压缩是否显著破坏 PG19 continuation likelihood。
2. 假设：Blockwise 的 block-level summary 在更长 context 下 PPL 增幅低于 SnapKV/ChunkKV。
3. 方法：固定模型、数据、compression ratio 和 target tokens，只 sweep dense context positions。
4. 结果：报告每个方法在 2k-32k 的 PPL 曲线和相对 `no_press` 的 PPL gap。
5. 结论：只基于实际跑出的曲线判断，不提前宣称 Blockwise 更优。
