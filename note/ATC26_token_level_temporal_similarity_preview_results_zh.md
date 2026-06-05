# ATC26 Token-Level Temporal KV Similarity Preview 结果

## 1. 实验目的

这版实验用于测试论文 Figure 6 的 token-level 口径：

```text
T_i = decode step i 的 top-k historical KV token set
T_{i+Delta} = decode step i+Delta 的 top-k historical KV token set
```

主指标是：

```text
Overlap(T_i, T_{i+Delta}) =
  |T_i intersection T_{i+Delta}| / |T_{i+Delta}|
```

该指标表示：如果 runtime 使用当前 step 的重要 KV token set 近似未来 step 的 oracle-important token set，能覆盖多少未来重要 token。

## 2. 实验设置

本次是 preview 配置，目的是先看 Figure 6 曲线形态，不是最终 paper full run。

```text
script: evaluation/ATC26_collect_token_level_temporal_similarity.py
plot: figure/ATC26_plot_token_level_temporal_similarity.py
run_tag: preview_delta1024
model: /Tan/model/Llama-3.1-8B-Instruct
dataset: PG19 test, /Tan/dataset/pg19-test
context_length: 8192
samples_per_length: 1
decode_steps: 1024
compression_ratios: 0.7, 0.5, 0.3
lags: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
reuse_intervals: 2, 4, 8, 16, 32, 64, 128, 256, 512
head aggregation: mean over attention heads
metric scope: historical KV tokens only, excluding current self token
device: physical A6000 device2, CUDA_VISIBLE_DEVICES=2
```

注意：这里 `compression_ratio=0.7` 表示保留约 30% tokens，`compression_ratio=0.3` 表示保留约 70% tokens。

## 3. 产物路径

原始结果：

```text
evaluation/results/experiments/ATC26_token_level_temporal_similarity/artifacts/preview_delta1024/
```

主要文件：

```text
ATC26_token_level_temporal_similarity_aggregate.csv
ATC26_token_level_temporal_similarity_aggregate.json
raw/ATC26_token_level_temporal_similarity_raw.jsonl
```

图像：

```text
figure/experiments/ATC26_token_level_temporal_similarity/preview_delta1024/figure6_token_level_temporal_similarity.pdf
figure/experiments/ATC26_token_level_temporal_similarity/preview_delta1024/figure6_token_level_temporal_similarity.png
```

Paper-facing CSV：

```text
figure/experiments/ATC26_token_level_temporal_similarity/preview_delta1024/figure6_token_level_overlap_curve.csv
figure/experiments/ATC26_token_level_temporal_similarity/preview_delta1024/figure6_token_level_reuse_curve.csv
```

## 4. 关键结果

### 4.1 Token-level 相邻 step 仍然稳定

`Delta=1` 时，不同保留预算下 overlap 均较高：

| Keep budget | Compression ratio | Delta=1 overlap |
|---:|---:|---:|
| 70% | 0.3 | 0.9153 |
| 50% | 0.5 | 0.8634 |
| 30% | 0.7 | 0.8035 |

结论：即使把重要集合粒度从 block-level 收紧到 token-level，相邻 decode steps 的重要 KV token set 仍有明显重叠。这支持 Observation 2 的短期稳定性。

### 4.2 长距离 Delta 明显下降

`Delta=512` 时 overlap 下降：

| Keep budget | Compression ratio | Delta=512 overlap |
|---:|---:|---:|
| 70% | 0.3 | 0.7565 |
| 50% | 0.5 | 0.6423 |
| 30% | 0.7 | 0.5087 |

结论：重要 KV token set 随 decode horizon 增长出现漂移，且高压缩率下漂移更明显。这支持 Observation 3。

### 4.3 Fixed refresh 也体现预算敏感性

Reuse recall：

| Keep budget | R=32 | R=128 | R=512 |
|---:|---:|---:|---:|
| 70% | 0.8688 | 0.8405 | 0.7947 |
| 50% | 0.7902 | 0.7497 | 0.6866 |
| 30% | 0.7032 | 0.6510 | 0.5622 |

结论：token-level 口径下，固定刷新仍可利用短期稳定性，但默认刷新间隔需要比 block-level 更保守。尤其保留 30% token 时，`R=512` 已明显偏低。

## 5. 对 Figure 6 的启发

这版图能直接支撑论文文字：

1. 小 `Delta` 时 overlap 高，说明当前 sparse token set 对近未来访问有预测性。
2. 大 `Delta` 时 overlap 下降，说明 sparse pattern 会随 generation 漂移。
3. 更严格预算下曲线整体更低、下降更明显，说明 static compression 在高压缩场景下风险更高。

与 block-level 结果相比，token-level 结果更严格，因此数值更低，但趋势更清楚。

## 6. 当前限制

1. 这是 preview：只跑了 PG19 的 1 个样本和 8192 context。
2. 当前重要 token 定义来自 attention weight 的 head-mean top-k；如果论文最终使用 blockwise score，需要说明 token-level 是更细粒度补充证据。
3. 当前只排除了 current self token，没有额外排除 recent protected tokens。
4. 尚未跑 16384 context、多样本、跨模型或 LongBench/Needle 补充。

## 7. 下一步建议

如果这版图的曲线形态可接受，下一步 full run 建议：

```text
context_lengths: 8192, 16384
samples_per_length: 4
decode_steps: 1024
compression_ratios: 0.7, 0.5, 0.3
head_agg: mean
```

这会和已有 block-level `decode1024` 设置对齐，便于在论文里把 token-level 和 block-level 结果互相校验。
