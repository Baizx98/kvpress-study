# ATC26 Needle 8K Token-Length x Depth Heatmap 实验规划

## 1. 目标

重新设计 Needle in a Haystack 实验，使图更接近论文中常见的丰富 heatmap：

- 固定最大上下文预算为 8K。
- 横轴表示实际输入 token length。
- 纵轴表示 needle depth。
- 每个格子表示该 token length 和 depth 下的检索准确率。
- 对比三种方法：`block_wise`、`snapkv`、`chunkkv`。

这版不再把横轴作为不同方法，也不再只看单一 8K 点；而是在 8K 范围内扫描多个 token length，使图能展示“随着上下文变长，检索是否退化”。

## 2. 实验边界

### 模型

- `/Tan/model/Llama-3.1-8B-Instruct`

### 数据集

使用项目现有 Needle benchmark：

```text
alessiodevoto/paul_graham_essays
```

该数据集已经在 `evaluation/evaluate_registry.py` 中注册为：

```python
"needle_in_haystack": "alessiodevoto/paul_graham_essays"
```

不使用 PG19。PG19 继续只用于 perplexity。

### 方法

| method | compression_ratio | 说明 |
|---|---:|---|
| `block_wise` | `0.5` | 当前主方法 |
| `snapkv` | `0.5` | baseline |
| `chunkkv` | `0.5` | baseline，按当前项目默认 `ChunkKVPress(press=SnapKVPress(), chunk_length=20)` |

不跑 `no_press`，不画差值图。

## 3. Heatmap 设计

### 横轴：token length

使用 8K 以内、按 256-token 步长采样的实际上下文长度：

```text
256, 512, 768, ..., 7936, 8192
```

解释：

- `8192` 是最大上下文预算。
- `256` 是最小上下文长度。
- 共有 `32` 个 token length 点。
- 256 步长能形成更密的横向趋势，适合生成视觉上更丰富的论文 heatmap。

### 纵轴：needle depth

使用 0 到 100 的 11 个位置：

```text
0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
```

### 每个格子

每个格子建议先跑 3 个 repeat：

```text
seed = 42, 43, 44
```

但注意：当前 `alessiodevoto/paul_graham_essays` test split 只有 1 条长 context。如果不改数据构造逻辑，repeat 只是重复同一条样本，不能提供真正独立样本。

因此这里建议实现一个轻量增强：

- 对每个 `seed`，从 Paul Graham 长 context 中选择不同的 token window。
- 每个 window 长度等于当前 token length。
- 在该 window 内按 depth 插入 needle。
- 这样同一个 token length/depth 下能得到 3 个不同背景窗口，heatmap 的 cell accuracy 更可信。

### 总规模

每个方法：

```text
32 token lengths * 11 depths * 3 repeats = 1056 generations
```

三种方法总计：

```text
1056 * 3 = 3168 generations
```

这比之前 `5 context length * 11 depth * 3 methods` 密很多，但仍然控制在 8K 内。需要注意总生成次数增加到 3168，wall-clock 时间会明显上升。

## 4. BlockWise 设置

先沿用 ATC26 主设置，保持和已有实验一致：

```text
compression_ratio=0.5
block_size=16
q_window_size=64
summary_topk_keys=4
mean_key_weight=0.75
summary_mode=mean_plus_norm_topk_mean
representative_mode=key_norm
query_agg_mode=max
head_agg_mode=uniform_mean
representative_k=4
multi_rep_k=4
query_topr=16
head_topk=1
```

本实验的目的不是立刻修 BlockWise，而是生成一个更标准、更丰富的 heatmap，明确展示它在 Needle 上的弱点。

## 5. 评分方式

沿用 Needle retrieval accuracy，但 scoring 要考虑 `answer_prefix`。

默认数据集的 needle 是：

```text
Remember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.
```

默认 `answer_prefix` 是：

```text
Answer: The best thing to do in San Francisco is
```

因此不能只检查 `predicted_answer` 是否包含完整 needle；应该检查：

```text
answer_prefix + predicted_answer
```

是否覆盖 needle 的核心内容。

主指标：

```text
accuracy = correct_count / repeats_per_cell
```

辅助保留：

- `rouge_l_f1`
- `predicted_answer`
- `window_start_token`
- `token_length`
- `needle_depth`
- `seed`
- `method`
- `compression_ratio`

## 6. 图表设计

### 主图

生成一个 1x3 panel：

- Panel 1：`BlockWise`
- Panel 2：`SnapKV`
- Panel 3：`ChunkKV`

每个 panel：

- x 轴：token length，`256 -> 8192`，步长 `256`
- y 轴：needle depth，`0 -> 100`
- 颜色：accuracy，固定 `[0, 1]`
- 每个格子显示数值，例如 `0.67`、`1.00`
- 使用统一 colorbar

推荐输出：

```text
figure/experiments/ATC26_needle_8k_token_length_depth_heatmap/
  ATC26_needle_8k_token_length_depth_all_methods.png
  ATC26_needle_8k_token_length_depth_all_methods.pdf
```

### 为什么比上一版更好

上一版是：

```text
context length: 4096, 8192, 16384, 32768, 65536
depth: 0..100
```

问题是：

- 3090 在 32K/64K 上 OOM，图会缺行。
- 每个 cell 只有 1 个样本，不够稳定。
- 横轴/纵轴不够密，视觉上不够丰富。

新方案是：

```text
token length: 256..8192, step=256
depth: 0..100
repeat: 3 windows
```

优点：

- 全部落在 8K 内，3090 更稳定。
- 横轴更密，图更像论文 heatmap。
- 每个 cell 有多个 window repeat，结果更可靠。

## 7. 实现计划

建议新增一套独立脚本，避免污染上一轮结果：

### 7.1 Runner

```text
evaluation/ATC26_run_needle_8k_heatmap.py
```

职责：

- 加载 `alessiodevoto/paul_graham_essays`。
- tokenize 原始 context。
- 按 `token_length` 和 `seed` 选择 window。
- 在 window 内按 depth 插入 needle。
- 调用现有 `kv-press-text-generation` pipeline。
- 逐 method 运行。
- 每个 method 单独加载模型，避免反复加载过多进程。
- 支持 `--mode smoke/full --resume`。

### 7.2 Postprocess

```text
evaluation/ATC26_postprocess_needle_8k_heatmap.py
```

职责：

- 聚合 predictions。
- 计算 per-sample correctness。
- 生成 cell-level accuracy。

输出：

```text
evaluation/results/experiments/ATC26_needle_8k_token_length_depth_heatmap/artifacts/
  ATC26_needle_8k_predictions.csv
  ATC26_needle_8k_metrics_long.csv
  ATC26_needle_8k_metrics_cell.csv
```

### 7.3 Plot

```text
figure/ATC26_plot_needle_8k_heatmap.py
```

职责：

- 读取 cell 表。
- 画 1x3 panel heatmap。
- 每个格子标数值。
- 输出 PNG 和 PDF。

## 8. Smoke 测试

先跑一个小网格：

```text
token_length = 256, 2048, 8192
needle_depth = 0, 50, 100
seed = 42
method = block_wise, chunkkv
```

总量：

```text
3 token lengths * 3 depths * 1 seed * 2 methods = 18 generations
```

通过标准：

- 无 OOM。
- 所有 predictions 和 metrics 落盘。
- 输出中能看到合理答案。
- postprocess 后 cell 表非空。
- heatmap 能正常生成。

## 9. Full 实验

Full 网格：

```text
token_length = 256, 512, 768, ..., 7936, 8192
needle_depth = 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
seed = 42, 43, 44
method = block_wise, snapkv, chunkkv
```

总量：

```text
3168 generations
```

建议先在 3090 上运行。如果 `block_wise` 8K 仍稳定，则可以直接得到完整 heatmap。

## 10. 预期结论形式

这张图应该能回答：

1. `block_wise` 是否在某些 token length 上突然退化。
2. `block_wise` 是否对前部、中部、后部 needle 有位置偏置。
3. `snapkv/chunkkv` 是否在 8K 范围内更稳定。
4. 当前 BlockWise 的 Needle 弱点是否来自 token length 增长、needle depth，还是二者交互。

如果 `block_wise` 在短 token length 也明显弱于 `chunkkv`，说明问题更偏向 block summary / representative selection；如果只在长 token length 退化，说明问题更偏向压缩预算和长上下文 block selection。
