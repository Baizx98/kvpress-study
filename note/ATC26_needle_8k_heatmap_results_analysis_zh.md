# ATC26 Needle 8K Token-Length x Depth Heatmap 结果分析

## 1. 实验完成情况

本轮实验已完成 full 网格：

- 模型：`/Tan/model/Llama-3.1-8B-Instruct`
- GPU：物理 `device1`，RTX 3090
- 数据集：`alessiodevoto/paul_graham_essays`
- 方法：`block_wise`、`snapkv`、`chunkkv`
- 压缩率：`0.5`
- token length：`256, 512, ..., 8192`
- needle depth：`0, 10, ..., 100`
- seed：`42, 43, 44`

进度文件显示 full 任务：

```text
success = 3168
failed = 0
```

注意：status 文件里还包含 smoke 的 18 条成功记录，因此总 status 记录数是 `3186`，其中 full 是 `3168`。

## 2. 关键产物

结果目录：

```text
evaluation/results/experiments/ATC26_needle_8k_token_length_depth_heatmap/
```

主要文件：

| 类型 | 路径 |
|---|---|
| progress | `evaluation/results/experiments/ATC26_needle_8k_token_length_depth_heatmap/artifacts/ATC26_needle_8k_progress.md` |
| raw predictions | `evaluation/results/experiments/ATC26_needle_8k_token_length_depth_heatmap/artifacts/ATC26_needle_8k_predictions.csv` |
| full long table | `evaluation/results/experiments/ATC26_needle_8k_token_length_depth_heatmap/artifacts/ATC26_needle_8k_metrics_long.csv` |
| full cell table | `evaluation/results/experiments/ATC26_needle_8k_token_length_depth_heatmap/artifacts/ATC26_needle_8k_metrics_cell.csv` |
| valid long table, depth 0-90 | `evaluation/results/experiments/ATC26_needle_8k_token_length_depth_heatmap/artifacts/ATC26_needle_8k_metrics_long_valid_depth0_90.csv` |
| valid cell table, depth 0-90 | `evaluation/results/experiments/ATC26_needle_8k_token_length_depth_heatmap/artifacts/ATC26_needle_8k_metrics_cell_valid_depth0_90.csv` |
| full heatmap | `figure/experiments/ATC26_needle_8k_token_length_depth_heatmap/ATC26_needle_8k_token_length_depth_all_methods.png` |
| valid heatmap, depth 0-90 | `figure/experiments/ATC26_needle_8k_token_length_depth_heatmap/ATC26_needle_8k_token_length_depth_valid_depth0_90.png` |

后续汇报和论文图建议优先使用 valid heatmap，即排除 depth=100 的版本。

## 3. 有效性检查：depth=100 不应进入主结论

重构输入后检查发现：

| depth | needle survives after pipeline truncation |
|---:|---:|
| 0-90 | 100% |
| 100 | 0% |

原因是当前 pipeline 对 context 采用前截断：

```text
context_ids = context_ids[:, :max_context_length]
```

当 needle 插在 100% 位置时，它位于上下文尾部；由于 chat template / wrapper 带来额外 token，最终会被 `max_context_length` 截断掉。因此 `depth=100` 全部为 0 不是方法失败，而是构造方式导致 needle 不在实际输入中。

所以主分析使用 `depth=0..90`。

## 4. 主结果，depth 0-90

按 full valid 样本统计：

| method | correct / total | accuracy |
|---|---:|---:|
| `block_wise` | 218 / 960 | 0.227 |
| `snapkv` | 208 / 960 | 0.217 |
| `chunkkv` | 337 / 960 | 0.351 |

结论：

- `chunkkv` 在这组 8K Needle heatmap 上最好。
- `block_wise` 略高于 `snapkv`，但明显低于 `chunkkv`。
- 三种方法整体准确率都不高，说明 50% 压缩下这个 Needle 设置比较苛刻。

## 5. 按 token length 看

选取部分 token length 汇总：

| token length | block_wise | snapkv | chunkkv |
|---:|---:|---:|---:|
| 256 | 0.100 | 0.233 | 0.333 |
| 1024 | 0.067 | 0.000 | 0.133 |
| 2048 | 0.267 | 0.200 | 0.333 |
| 3072 | 0.400 | 0.433 | 0.467 |
| 4096 | 0.267 | 0.200 | 0.267 |
| 5120 | 0.167 | 0.200 | 0.400 |
| 6144 | 0.167 | 0.100 | 0.233 |
| 7168 | 0.400 | 0.367 | 0.533 |
| 8192 | 0.300 | 0.133 | 0.267 |

观察：

- 准确率不是随 token length 单调下降。
- 这说明当前结果同时受 window 内容、needle depth、压缩策略影响，而不是单纯长度效应。
- `chunkkv` 在多数 token length 上更稳。
- `block_wise` 在 `3072/7168` 等位置有较好表现，但稳定性不如 `chunkkv`。

## 6. 按 needle depth 看

depth 0-90 平均结果：

| depth | block_wise | snapkv | chunkkv |
|---:|---:|---:|---:|
| 0 | 0.260 | 0.188 | 0.135 |
| 10 | 0.010 | 0.083 | 0.083 |
| 20 | 0.031 | 0.104 | 0.062 |
| 30 | 0.083 | 0.104 | 0.115 |
| 40 | 0.104 | 0.073 | 0.198 |
| 50 | 0.167 | 0.094 | 0.188 |
| 60 | 0.281 | 0.125 | 0.375 |
| 70 | 0.354 | 0.229 | 0.552 |
| 80 | 0.500 | 0.438 | 0.844 |
| 90 | 0.479 | 0.729 | 0.958 |

观察：

- `chunkkv` 在深位置 `70-90` 明显更强。
- `block_wise` 在浅层 `10-40` 表现很差。
- `snapkv` 在 `90` 处较强，但整体均值仍低于 `chunkkv`。
- 这个趋势说明本轮 Needle 任务存在明显位置偏置：needle 越靠后越容易被找回，尤其对 `chunkkv/snapkv` 更明显。

## 7. 对 BlockWise 的解释

本轮结果不再像上一轮那样显示 `block_wise` 全面落后于 `snapkv`，但仍明显落后于 `chunkkv`：

```text
block_wise = 0.227
snapkv     = 0.217
chunkkv    = 0.351
```

可能原因：

1. `block_wise` 的 block summary 仍会稀释短 needle。
2. `key_norm` representative 不保证保留 needle 关键 token。
3. `query_agg=max` 对少数 query token 敏感，可能选到表面相关但不含 needle 的 block。
4. `chunkkv` 的局部 chunk 结构可能更有利于保留连续短事实，因此在 Needle 上更稳。

## 8. 当前限制

1. `depth=100` 无效，应从主结论中排除。
2. 每个 cell 只有 3 个 window repeat，已经比上一轮更稳，但仍然偏少。
3. 没有 no-press 上界；因此当前结果只能比较压缩方法之间的相对表现，不能判断原模型在该构造下的绝对上限。
4. 由于 window 是从同一条 Paul Graham 长 context 中切出来的，样本独立性有限。

## 9. 后续建议

如果目标是让图更适合论文展示：

1. 主图使用 `ATC26_needle_8k_token_length_depth_valid_depth0_90.png`。
2. 图注明确写：
   - token length 从 256 到 8192，步长 256；
   - 每个 cell 是 3 个 window seeds 的平均准确率；
   - depth=100 因截断风险不纳入主图。
3. 如果要补强实验可信度，下一步增加 no-press 上界和更多 seed。
4. 如果要改进 BlockWise，优先测试：
   - `summary_mode=multi_rep_max`
   - `representative_mode=tail_query_relevance`
   - `block_size=8`
