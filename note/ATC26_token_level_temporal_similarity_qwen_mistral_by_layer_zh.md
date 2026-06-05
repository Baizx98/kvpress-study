# ATC26 Token-Level Temporal Similarity：Qwen / Mistral 分层结果

## 1. 实验目的

这次实验扩展上一版 token-level Figure 6 preview，在 Mistral 和 Qwen 上重新采集数据，并把每个 attention layer 的 `future-token oracle overlap` 分开绘制。

这次只保留 lag overlap：

```text
Overlap(T_i, T_{i+Delta}) =
  |T_i intersection T_{i+Delta}| / |T_{i+Delta}|
```

其中 `T_i` 是 decode step `i` 的 top-k historical KV token set。当前 self token 已排除。

`Fixed-refresh approximation` 没有进入这次绘图，因为它和 `future-token oracle overlap` 表现出相同趋势，主图保留 lag overlap 更直接。

## 2. 实验设置

```text
run_tag: cross_model_delta1024
script: evaluation/ATC26_collect_token_level_temporal_similarity.py
plot: figure/ATC26_plot_token_level_temporal_similarity_by_layer.py
dataset: PG19 test, /Tan/dataset/pg19-test
context_length: 8192
samples_per_model: 1
decode_steps: 2048
max Delta: 1024
lags: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
compression_ratios: 0.7, 0.5, 0.3
head aggregation: mean over heads
device: physical A6000 device2, CUDA_VISIBLE_DEVICES=2
```

模型：

```text
/Tan/model/Mistral-7B-Instruct-v0.3
/Tan/model/Qwen3-8B
```

注意：`compression_ratio=0.7` 表示保留约 30% tokens；`compression_ratio=0.3` 表示保留约 70% tokens。

## 3. 产物路径

原始结果：

```text
evaluation/results/experiments/ATC26_token_level_temporal_similarity/artifacts/cross_model_delta1024/
```

主要结果：

```text
evaluation/results/experiments/ATC26_token_level_temporal_similarity/artifacts/cross_model_delta1024/ATC26_token_level_temporal_similarity_aggregate.csv
evaluation/results/experiments/ATC26_token_level_temporal_similarity/artifacts/cross_model_delta1024/raw/ATC26_token_level_temporal_similarity_raw.jsonl
```

分层图目录：

```text
figure/experiments/ATC26_token_level_temporal_similarity/cross_model_delta1024/
```

分层 CSV：

```text
figure/experiments/ATC26_token_level_temporal_similarity/cross_model_delta1024/future_token_oracle_overlap_by_layer.csv
```

## 4. 关键整体趋势

### Mistral-7B-v0.3

| Keep budget | Delta=1 | Delta=128 | Delta=512 | Delta=1024 |
|---:|---:|---:|---:|---:|
| 70% | 0.9108 | 0.8062 | 0.7489 | 0.6934 |
| 50% | 0.8557 | 0.6979 | 0.6250 | 0.5598 |
| 30% | 0.7949 | 0.5871 | 0.4871 | 0.4068 |

### Qwen3-8B

| Keep budget | Delta=1 | Delta=128 | Delta=512 | Delta=1024 |
|---:|---:|---:|---:|---:|
| 70% | 0.8937 | 0.7997 | 0.7417 | 0.6834 |
| 50% | 0.8306 | 0.6858 | 0.6143 | 0.5490 |
| 30% | 0.7625 | 0.5656 | 0.4735 | 0.4007 |

## 5. 初步观察

1. 两个模型都复现了同一趋势：小 `Delta` overlap 高，长 `Delta` overlap 明显下降。
2. Keep budget 越小，曲线整体越低，远距离下降越明显。
3. Qwen3-8B 的整体 overlap 略低于 Mistral，尤其在 Keep 30% 下更明显。
4. 分层 heatmap 显示多数 attention layer 都遵循相同的随 `Delta` 下降趋势，但不同层的初始 overlap 和 long-horizon drift 强度有差别。

这说明 Observation 2/3 不只出现在 Llama preview 上，在 Mistral 和 Qwen 上也成立。

## 6. 推荐优先查看的图

Heatmap 更适合先看层间差异：

```text
figure/experiments/ATC26_token_level_temporal_similarity/cross_model_delta1024/mistral_7b_instruct_v03__ctx8192__keep30__layer_delta_overlap_heatmap.pdf
figure/experiments/ATC26_token_level_temporal_similarity/cross_model_delta1024/qwen3_8b__ctx8192__keep30__layer_delta_overlap_heatmap.pdf
```

Per-layer curves 更适合看每层是否单调下降：

```text
figure/experiments/ATC26_token_level_temporal_similarity/cross_model_delta1024/mistral_7b_instruct_v03__ctx8192__keep30__per_layer_overlap_curves.pdf
figure/experiments/ATC26_token_level_temporal_similarity/cross_model_delta1024/qwen3_8b__ctx8192__keep30__per_layer_overlap_curves.pdf
```

## 7. 当前限制

1. 这仍是 preview：每个模型只跑了 1 个 PG19 sample。
2. 当前只跑了 `context_length=8192`，还没有 16384 context。
3. 当前重要 token 定义来自 attention weight 的 head-mean top-k；如果论文主机制使用 blockwise score，需要明确 token-level 是更细粒度观察。
4. 这次采集脚本仍计算了一个最小 reuse interval 字段，但绘图和分析只使用 lag overlap。
