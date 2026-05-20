# ATC26 Needle Heatmap 实验方案

## 1. 修正说明

PG19 不应该用于大海捞针测试。PG19 的主要用途是 long-context language modeling / perplexity，用来回答压缩是否破坏长文本续写建模。

本实验改回常规 Needle in a Haystack 流程：使用长文本 haystack，在不同上下文长度和不同插入深度下插入一条可精确匹配的 needle，然后测试模型是否能找回该信息，并画二维 heatmap。

## 2. 实验目标

在 `Llama-3.1-8B-Instruct` 上评估 50% prefill KVCache 压缩对长上下文精确信息检索的影响。

核心问题：

- 随着 context length 增大，检索准确率是否下降？
- needle 位于上下文前部、中部、后部时，压缩方法是否有明显位置偏置？
- `BlockWisePress` 相比 `SnapKV` / `ChunkKV` 是否在更长上下文和更深 needle 位置上更稳定？

## 3. 实验设置

### 模型

- `/Tan/model/Llama-3.1-8B-Instruct`

### 方法

| method | compression_ratio | 作用 |
|---|---:|---|
| `block_wise` | `0.5` | 主方法 |
| `snapkv` | `0.5` | baseline |
| `chunkkv` | `0.5` | baseline |

如果时间有限，最小版本先跑：

| method | compression_ratio |
|---|---:|
| `block_wise` | `0.5` |
| `chunkkv` | `0.5` |

### BlockWise 参数

沿用 ATC26 当前主设置：

- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=key_norm`
- `query_agg_mode=max`
- `head_agg_mode=uniform_mean`
- `block_size=16`
- `q_window_size=64`
- `summary_topk_keys=4`
- `mean_key_weight=0.75`
- `representative_k=4`
- `multi_rep_k=4`
- `query_topr=16`
- `head_topk=1`
- `prefill_skip_first_layers=2`

仍然只做 prefill-only 压缩，不启用 decode 阶段压缩。

## 4. 常规 Needle 测试流程

### Haystack

使用常规 Needle in a Haystack 的长文本背景，而不是 PG19 perplexity 数据。

本项目当前已经支持该 benchmark，数据源在 `evaluation/evaluate_registry.py` 中注册为：

```python
"needle_in_haystack": "alessiodevoto/paul_graham_essays"
```

对应说明见 `evaluation/benchmarks/needle_in_haystack/README.md`：本项目跟随多数文献使用 Paul Graham essays 作为 haystack。

关键要求：

- haystack 文本只作为背景噪声，不参与 perplexity。
- needle 是人工插入的一条事实。
- question 只问 needle 中的精确信息。
- 评测关注是否找回 needle，不关注生成质量。

### 数据获取与缓存

不需要额外手工下载原始文件。运行评测时 Hugging Face `datasets` 会按 registry 自动加载：

```text
alessiodevoto/paul_graham_essays
```

本项目的 `evaluation/evaluate.py` 默认会把 Hugging Face 缓存放到：

```text
/Tan/dataset/hf_home
/Tan/dataset/hf_home/datasets
/Tan/dataset/hf_home/hub
```

因此推荐保持当前仓库默认路径，不要为 Needle 单独引入 PG19 或其它数据源。若要提前检查数据是否能加载，可以用最小 Python 片段：

```bash
.venv/bin/python - <<'PY'
from datasets import load_dataset
ds = load_dataset("alessiodevoto/paul_graham_essays", split="test")
print(ds)
print(ds[0].keys())
PY
```

### Needle

建议使用可精确匹配、不会被常识猜中的短事实：

```text
The special magic number for this experiment is 314159.
```

问题：

```text
What is the special magic number for this experiment?
```

标准答案：

```text
314159
```

## 5. Heatmap 网格

### 主实验网格

| 维度 | 取值 |
|---|---|
| context length | `4096, 8192, 16384, 32768, 65536` |
| needle depth | `0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100` |
| repeats per cell | `5` |
| seed | `42, 43, 44, 45, 46` |
| max_new_tokens | `50` |

每个方法样本数：

`5 context lengths * 11 depths * 5 repeats = 275 generations`

3 个方法总计：

`275 * 3 = 825 generations`

这个规模足够画 heatmap，比之前每组一个平均准确度更符合常规 Needle 论文图。

### Smoke 网格

正式跑之前先做小网格：

| 维度 | 取值 |
|---|---|
| context length | `4096, 16384` |
| needle depth | `0, 50, 100` |
| repeats per cell | `1` |
| methods | `block_wise, chunkkv` |

Smoke 样本数：

`2 * 3 * 1 * 2 = 12 generations`

Smoke 通过条件：

- 每个格子都有预测和指标。
- `block_wise` 和 `chunkkv` 都能稳定落盘。
- 输出中没有系统性空回答、重复模板回答或 metrics 缺失。

## 6. 评分方式

主指标使用 binary exact retrieval accuracy：

- 预测答案包含 `314159`：`correct = 1`
- 否则：`correct = 0`

每个 heatmap cell：

```text
accuracy = correct_count / repeats_per_cell
```

辅助保存：

- `predicted_answer`
- `correct`
- `rouge_l_f1`
- `context_length`
- `needle_depth`
- `seed`
- `method`
- `compression_ratio`
- `input_tokens`
- `generated_tokens`
- `latency_seconds`

主图使用 binary accuracy；ROUGE-L 只用于 debug。

## 7. 输出目录

实验名：

`ATC26_needle_heatmap_llama31_8b_ratio50`

结果目录：

`evaluation/results/experiments/ATC26_needle_heatmap_llama31_8b_ratio50/`

建议结构：

- `artifacts/ATC26_needle_heatmap_manifest.jsonl`
- `artifacts/ATC26_needle_heatmap_predictions.csv`
- `artifacts/ATC26_needle_heatmap_metrics_long.csv`
- `artifacts/ATC26_needle_heatmap_metrics_cell.csv`
- `artifacts/logs/`
- `artifacts/raw/`
- `README.md`

图目录：

`figure/experiments/ATC26_needle_heatmap_llama31_8b_ratio50/`

建议图片：

- `ATC26_needle_heatmap_block_wise_ratio50.png`
- `ATC26_needle_heatmap_snapkv_ratio50.png`
- `ATC26_needle_heatmap_chunkkv_ratio50.png`
- `ATC26_needle_heatmap_all_methods.png`

## 8. 图表规范

### 主图

画一个 1x3 或 3-row panel：

- 每个 panel 一个方法。
- x 轴：needle depth。
- y 轴：context length。
- 颜色：accuracy。
- colorbar 固定 `[0, 1]`。
- y 轴按真实 context length 标注，不要只写 index。

不画差值图。方法差异直接通过同一色标下的多个 heatmap panel 对比。

## 9. 实现计划

建议新增独立脚本，不复用 PG19 分支：

1. `evaluation/ATC26_run_needle_heatmap.py`
   - 生成 `context_length × needle_depth × seed × method` manifest。
   - 构造 haystack、插入 needle。
   - 调用现有模型和 press 评测链路。
   - 支持 `--mode smoke/full --resume`。

2. `evaluation/ATC26_postprocess_needle_heatmap.py`
   - 聚合预测。
   - 计算 per-sample `correct`。
   - 生成 per-cell accuracy 表。

3. `figure/ATC26_plot_needle_heatmap.py`
   - 读取 cell 表。
   - 输出 all-method heatmap。

当前 `evaluation/benchmarks/needle_in_haystack/utils.py` 已支持传入多个 `needle_depth`，但还不足以表达多个 context length 和多个 repeat。新 runner 可以复用插入逻辑，但需要外层 manifest 管理不同长度和 seed。

## 10. 最小验证计划

### Step 1: 只构造数据

不加载模型，先检查：

- 每个 context length 都能构造成功。
- needle depth 对应插入位置正确。
- tokenized context 不超过目标长度。
- 随机打印几个样本确认 needle 和 question 对齐。

### Step 2: 压缩 smoke

跑：

- `block_wise`
- `chunkkv`
- context length: `4096, 16384`
- depth: `0, 50, 100`

确认 50% 压缩下能稳定落盘，没有 OOM、空输出或 metrics 缺失。

### Step 3: Full heatmap

跑完整方法和完整网格。

只使用 full 表画图，不混用 smoke 结果。

## 11. 与 PG19 的关系

这组实验和 PG19 分工如下：

| 实验 | 回答的问题 | 指标 |
|---|---|---|
| Needle heatmap | 长上下文中精确信息是否还能找回 | retrieval accuracy |
| PG19 perplexity | 长文本续写建模是否被压缩破坏 | perplexity |

论文中不要把二者混成一个指标，也不要用 PG19 文本冒充常规 Needle 评测，除非明确写成一个额外变体。
