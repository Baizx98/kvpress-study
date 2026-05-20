# ATC26 细分数据集结果检查

## 结论

- 已新增 LongBench 子数据集级别图表，避免把 6 个 LongBench 子任务平均后掩盖差异。
- 已新增 Needle depth 级别图表，用于检查 `needle_depth=[0,25,50,75,100]` 下每个压缩率的表现。
- Llama-3.1-8B-Instruct 的 Needle 结果中，SnapKV 和 ChunkKV 的直线不是绘图聚合导致的；在 depth 级别也完全相同。
- 更具体地说，Llama + Needle 下 SnapKV 和 ChunkKV 各自只有 1 个唯一预测答案，5 个 depth 和 6 个压缩率的 ROUGE-L F1 都是 `70.967741`。这提示需要检查 Needle 任务上 SnapKV/ChunkKV 的压缩率是否真实生效，或输出是否被固定提示/答案前缀主导。

## LongBench 覆盖口径

当前 ATC26 实验只跑了 LongBench 的 6 个 QA 子任务：

- `qasper`
- `multifieldqa_en`
- `hotpotqa`
- `2wikimqa`
- `musique`
- `triviaqa`

这不是完整 LongBench。LongBench 原始 benchmark 一般按 21 个数据集描述：14 个英文自然语言任务、5 个中文任务、2 个代码任务。很多论文为了只比较英文/代码任务，会报告 16 个非中文任务；也有论文排除 `passage_count` 后报告 15 个 English subtasks。

因此当前 ATC26 结果在论文里应写成 `LongBench-QA-6` 或 `six LongBench QA tasks`，不能写成完整 `LongBench` 平均。若要和“16 个 LongBench 子任务”的论文严格对齐，需要补跑以下非中文任务：

- 已跑：`qasper`, `multifieldqa_en`, `hotpotqa`, `2wikimqa`, `musique`, `triviaqa`
- 未跑：`narrativeqa`, `gov_report`, `qmsum`, `multi_news`, `trec`, `samsum`, `passage_count`, `passage_retrieval_en`, `lcc`, `repobench-p`

其中 `gov_report/qmsum/multi_news/samsum` 是长输出/摘要类任务，对 decode 阶段更敏感；`lcc/repobench-p` 是代码任务；`passage_count/passage_retrieval_en` 是 synthetic retrieval/counting 任务。是否全部补跑取决于论文想要对齐“完整 LongBench English+Code”还是只强调 QA 任务。

## 新增图像

| 图像 | 说明 |
|---|---|
| `figure/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/ATC26_longbench_subdataset_quality_grid.png` | 3 个模型 x 6 个 LongBench 子数据集，每个子图画 3 个方法随压缩率变化 |
| `figure/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/ATC26_needle_depth_quality_grid.png` | 3 个模型 x 5 个 needle depth，每个子图画 3 个方法随压缩率变化 |

## 新增表格

| 表格 | 说明 |
|---|---|
| `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_longbench_subdataset_long.csv` | LongBench 子数据集 long format |
| `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_longbench_subdataset_wide.csv` | LongBench 子数据集 wide format，ratio 展成列 |
| `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_needle_depth_long.csv` | Needle depth long format，包含每个 depth 的 ROUGE-L F1 和预测答案 |
| `evaluation/results/experiments/ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19/artifacts/ATC26_needle_depth_wide.csv` | Needle depth wide format，ratio 展成列 |

## Llama Needle 细分检查

Llama-3.1-8B-Instruct 在 Needle 上的 SnapKV / ChunkKV 结果：

| depth | method | r0.3 | r0.4 | r0.5 | r0.6 | r0.7 | r0.8 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | chunkkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 0 | snapkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 25 | chunkkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 25 | snapkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 50 | chunkkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 50 | snapkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 75 | chunkkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 75 | snapkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 100 | chunkkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |
| 100 | snapkv | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 | 70.968 |

对照 BlockWise，Llama + Needle 的 BlockWise 至少出现了多个不同分数和预测答案，因此当前异常主要集中在 SnapKV / ChunkKV。

## 后续排查建议

1. 先抽查 Llama + Needle + SnapKV/ChunkKV 的 `ATC26_config.yaml`，确认 `compression_ratio` 已写入并被 press 构造读取。
2. 再检查这两个方法在 Needle 的输出目录名：当前路径中 SnapKV/ChunkKV 没有像 BlockWise 一样包含 `qwindow64/topk4`，这是正常方法差异，但需要确认不是走了无压缩或固定预算路径。
3. 对 Llama + Needle 单独重跑 2 个 ratio，例如 `0.3` 和 `0.8`，打开 debug 日志记录 prefill 后实际 KV cache token 数，确认压缩率是否改变。
4. 如果实际 KV 数量随 ratio 变化，但答案仍完全一致，可以在论文中说明 Needle 对该模型/任务不敏感；如果实际 KV 数量不变，则需要修 SnapKV/ChunkKV 的 ratio plumbing 后重跑 Needle。

## 为什么 Needle 结果整体变化很小

当前 Needle 设置本身对压缩率不敏感，主要原因有四个：

1. 样本数太少。当前 `needle_depth=[0,25,50,75,100]`，每个模型/方法/压缩率只有 5 条样本，平均值非常容易出现阶梯或直线。
2. Needle 文本固定且答案前缀泄露较强。预测文件中 `answer_prefix` 是 `Answer: The best thing to do in San Francisco is`，模型只需要续写 `eat a sandwich and sit in Dolores Park on a sunny day` 这类常见短句，就能拿到较高 ROUGE-L；这降低了必须从长上下文中精确检索 needle 的压力。
3. 指标是 ROUGE-L F1，不是 exact match。只要生成答案和 needle 的核心短语有较多重叠，即使没有完整复述 `Remember, ...`，也能得到 `70.967741`、`75.000` 这种稳定分数。
4. 对 query-aware prefill 压缩来说，这个任务的 query/answer-prefix 很强，压缩前模型已经有足够强的生成先验。压缩率改变不一定会改变最终短答案。

当前证据：

| model | method | unique answers | score std |
|---|---|---:|---:|
| `llama31_8b_instruct` | `blockwise` | 6 | 5.2952 |
| `llama31_8b_instruct` | `snapkv` | 1 | 0.0000 |
| `llama31_8b_instruct` | `chunkkv` | 1 | 0.0000 |
| `mistral_7b_instruct_v03` | `blockwise` | 15 | 19.8319 |
| `mistral_7b_instruct_v03` | `snapkv` | 5 | 7.9093 |
| `mistral_7b_instruct_v03` | `chunkkv` | 8 | 10.4577 |
| `qwen3_8b` | `blockwise` | 4 | 7.7227 |
| `qwen3_8b` | `snapkv` | 2 | 5.1675 |
| `qwen3_8b` | `chunkkv` | 2 | 8.1199 |

这说明“基本没变化”不是所有方法都完全相同：Mistral 和 Qwen 还有一定波动；最异常的是 Llama + SnapKV/ChunkKV，它们在 30 条细分记录中输出完全一样。

## 对论文使用的建议

- 不建议把当前 Needle 结果作为强结论，只适合作为补充 long-context retrieval sanity check。
- 主文图如果空间有限，优先放 LongBench 子任务和 PG19；Needle 可以放 appendix，并在图注中写清楚只有 5 个 depth 样本。
- 如果要让 Needle 更有区分度，建议重跑更强版本：增加 depth 数量、换多条 needle 文本、移除强 `answer_prefix`，并加入 exact match 或 contains-hit 指标。
