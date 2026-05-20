# PG19 Dense Position PPL 结果检查

## 结论

本次 `PG19 dense position PPL` 实验结果完整，可以用于画图。

- 三种方法均完成：`snapkv`、`chunkkv`、`blockwise`
- 每种方法均有 `63` 个 token position
- token length 范围：`1024` 到 `32768`
- stride：`512`
- 压缩率：`compression_ratio=0.5`
- target window：`256`
- 主指标：跳过第一个 continuation token 后的 `subword_ppl`
- 运行设备：`NVIDIA GeForce RTX 3090`，GPU UUID `GPU-4eac01c5-47d1-3958-95bb-98d357b8b9c3`

## 产物路径

| 类型 | 路径 |
|---|---|
| 聚合 CSV | `evaluation/results/experiments/pg19_dense_position_ppl_llama31_8b_snapkv_chunkkv_blockwise_ratio50/artifacts/full_3090_f20_stride512_window256/pg19_dense_position_ppl_metrics.csv` |
| 运行日志 | `evaluation/results/experiments/pg19_dense_position_ppl_llama31_8b_snapkv_chunkkv_blockwise_ratio50/artifacts/full_3090_f20_stride512_window256/logs/run.log` |
| 主图 PDF | `figure/experiments/pg19_dense_position_ppl_llama31_8b_snapkv_chunkkv_blockwise_ratio50/pg19_dense_position_ppl.pdf` |
| 主图 PNG | `figure/experiments/pg19_dense_position_ppl_llama31_8b_snapkv_chunkkv_blockwise_ratio50/pg19_dense_position_ppl.png` |
| rolling 平滑图 PDF | `figure/experiments/pg19_dense_position_ppl_llama31_8b_snapkv_chunkkv_blockwise_ratio50/pg19_dense_position_ppl_rolling.pdf` |
| log-y 图 PDF | `figure/experiments/pg19_dense_position_ppl_llama31_8b_snapkv_chunkkv_blockwise_ratio50/pg19_dense_position_ppl_logy.pdf` |

## 完整性检查

| 检查项 | 结果 |
|---|---:|
| 总记录数 | `189` |
| 方法数 | `3` |
| 每方法记录数 | `63` |
| `subword_ppl` 空值 | `0` |
| `avg_nll` 空值 | `0` |
| `num_books` 范围 | `19-20` |
| 每点有效 target token 数范围 | `4845-5100` |

每个方法的 token length 覆盖一致：

| 方法 | min token length | max token length | 点数 |
|---|---:|---:|---:|
| `snapkv` | `1024` | `32768` | `63` |
| `chunkkv` | `1024` | `32768` | `63` |
| `blockwise` | `1024` | `32768` | `63` |

## 主要数值

| 方法 | mean PPL | min PPL | max PPL |
|---|---:|---:|---:|
| `snapkv` | `9.5650` | `8.1991` | `11.2282` |
| `chunkkv` | `9.6137` | `8.2228` | `11.3255` |
| `blockwise` | `9.7363` | `8.4047` | `11.2900` |

首尾 token length：

| token length | snapkv | chunkkv | blockwise |
|---:|---:|---:|---:|
| `1024` | `10.2679` | `10.3284` | `10.3935` |
| `32768` | `8.5165` | `8.5764` | `8.7770` |

逐点最低 PPL 计数：

| 方法 | 次数 |
|---|---:|
| `snapkv` | `58` |
| `chunkkv` | `5` |
| `blockwise` | `0` |

## 解释边界

本实验的横坐标是 dense token position。对每个 position `k`，先压缩长度为 `k` 的 context，再用后续 `256` 个真实 token 计算 continuation likelihood。主图使用跳过第一个 continuation token 的 PPL，因为第一个 token 的 logit 来自 prefill 最后一位，尚未真正使用压缩后的 KV cache。

本轮只检查质量指标 PPL，不包含 latency、峰值显存或吞吐率。结论应表述为：在 `Llama-3.1-8B-Instruct / PG19 / compression_ratio=0.5 / fraction=0.2` 设置下，`snapkv` 的平均 PPL 最低，`chunkkv` 略高，`blockwise` 最高。

