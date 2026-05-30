# ATC26 BlockWisePress 跨 step 稀疏索引相似性实验结果

## 1. 问题

本实验验证 BlockWisePress 在解码阶段选择的 KV block index 是否在相邻 step 之间高度相似，以及这种相似性随 step 间隔增大如何下降。目标是为“固定步长才重新计算一次稀疏索引，中间 step 复用上一次索引”的设计提供实验证据。

## 2. 实验设置

- 模型：`llama31_8b_instruct`
- 数据集：PG19 test，本地路径 `/Tan/dataset/pg19-test`
- 上下文长度：8192、16384
- 样本数：每个上下文长度 4 个样本，共 8 个 job
- Decode steps：1024
- Block size：16
- Window query size：16
- Compression ratio：0.5
- Lag sweep：1、2、4、8、16、32、64、128、256、512
- Reuse interval sweep：2、4、8、16、32、64、128、256、512
- GPU：物理 `device2`，通过 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2` 映射到进程内 `cuda:0`
- 运行脚本：`evaluation/ATC26_watch_blockwise_temporal_index_similarity.sh`
- 采集脚本：`evaluation/ATC26_collect_blockwise_temporal_index_similarity.py`
- 绘图脚本：`figure/ATC26_plot_blockwise_temporal_index_similarity.py --run-tag decode1024`

主结果路径：

- Raw JSONL：`evaluation/results/experiments/ATC26_blockwise_temporal_index_similarity/artifacts/decode1024/raw/ATC26_temporal_similarity_raw.jsonl`
- Aggregate CSV：`evaluation/results/experiments/ATC26_blockwise_temporal_index_similarity/artifacts/decode1024/ATC26_temporal_similarity_aggregate.csv`
- Aggregate JSON：`evaluation/results/experiments/ATC26_blockwise_temporal_index_similarity/artifacts/decode1024/ATC26_temporal_similarity_aggregate.json`
- Score arrays：`evaluation/results/experiments/ATC26_blockwise_temporal_index_similarity/artifacts/decode1024/scores/*.npz`
- Figures：`figure/experiments/ATC26_blockwise_temporal_index_similarity/decode1024/`

完整性检查：

- Watchdog heartbeat：`status=complete`
- 完成 job 数：8 / 8
- Aggregate CSV 行数：9728
- Raw JSONL 行数：8
- Score `.npz` 文件数：8
- Lag 覆盖：1 到 512，包含用户要求的 512
- Reuse interval 覆盖：2 到 512，包含用户要求的 512

说明：此前 `run_tag=full` 的 512-step 运行只覆盖到 lag=256，因为计算 lag=512 需要至少 1025 个 step 才能形成 `(t, t+512)` pair。因此主结果使用补跑的 `run_tag=decode1024`。

## 3. 指标定义

- `overlap`：两个 step 的 top-k block index 交集占 top-k 的比例。
- `jaccard`：两个 step 的 top-k block index 集合 Jaccard 相似度。
- `score_cosine`：两个 step 的 block score 向量余弦相似度。
- `reuse_recall`：以固定间隔刷新索引时，中间 step 复用刷新点 index 对当前 step top-k 的覆盖比例。
- `reuse_jaccard`：复用 index 与当前 step top-k 的 Jaccard 相似度。
- `refresh_reduction`：刷新次数减少比例，interval=512 时约为 99.8%。

`single` 表示逐 token 的严格对比；`window` 表示用最近 16 个 query token 的窗口打分，更接近 BlockWisePress 实际按窗口稳定打分的使用方式。

## 4. 主要结果

### 4.1 相邻 step 的 index 相似度很高

| Context | Mode | Lag=1 overlap | Lag=1 jaccard | Lag=1 score cosine |
|---:|---|---:|---:|---:|
| 8192 | single | 0.9112 | 0.8403 | 0.9964 |
| 8192 | window | 0.9839 | 0.9689 | 0.9999 |
| 16384 | single | 0.9168 | 0.8498 | 0.9966 |
| 16384 | window | 0.9854 | 0.9716 | 0.9999 |

结论：同一层相邻 decode step 的 block 选择高度稳定。严格的 `single` 模式下 overlap 也在 0.91 以上；更接近实际窗口打分的 `window` 模式下 overlap 接近 0.984-0.985，Jaccard 接近 0.97。

### 4.2 相似度随 step 间隔增大缓慢下降

| Context | Mode | Lag=1 overlap | Lag=32 overlap | Lag=128 overlap | Lag=512 overlap |
|---:|---|---:|---:|---:|---:|
| 8192 | single | 0.9112 | 0.8309 | 0.8167 | 0.7986 |
| 8192 | window | 0.9839 | 0.8777 | 0.8569 | 0.8340 |
| 16384 | single | 0.9168 | 0.8377 | 0.8185 | 0.7979 |
| 16384 | window | 0.9854 | 0.8829 | 0.8590 | 0.8329 |

| Context | Mode | Lag=1 cosine | Lag=128 cosine | Lag=512 cosine |
|---:|---|---:|---:|---:|
| 8192 | single | 0.9964 | 0.9929 | 0.9918 |
| 8192 | window | 0.9999 | 0.9956 | 0.9943 |
| 16384 | single | 0.9966 | 0.9917 | 0.9901 |
| 16384 | window | 0.9999 | 0.9943 | 0.9921 |

结论：index overlap 会随 lag 增大下降，但下降是渐进的；score cosine 在 lag=512 仍然大于 0.99，说明 block 重要性分布本身非常稳定。对论文来说，overlap/Jaccard 更能直接说明“选择的 block index 相似”，cosine 更适合作为补充证据说明“注意力块分布连续”。

### 4.3 固定步长复用 index 的覆盖率仍然较高

| Context | Mode | R=2 recall | R=32 recall | R=128 recall | R=512 recall | R=512 refresh reduction |
|---:|---|---:|---:|---:|---:|---:|
| 8192 | single | 0.9557 | 0.8511 | 0.8293 | 0.8045 | 0.9980 |
| 8192 | window | 0.9920 | 0.9100 | 0.8716 | 0.8294 | 0.9980 |
| 16384 | single | 0.9587 | 0.8593 | 0.8351 | 0.8144 | 0.9980 |
| 16384 | window | 0.9927 | 0.9169 | 0.8777 | 0.8405 | 0.9980 |

| Context | Mode | R=32 jaccard | R=128 jaccard | R=512 jaccard |
|---:|---|---:|---:|---:|
| 8192 | single | 0.7488 | 0.7169 | 0.6835 |
| 8192 | window | 0.8401 | 0.7790 | 0.7176 |
| 16384 | single | 0.7621 | 0.7269 | 0.6984 |
| 16384 | window | 0.8518 | 0.7899 | 0.7353 |

结论：固定步长刷新是有空间的。以 `window` 模式为主，R=32 时 recall 仍有 0.91-0.917，刷新次数减少 96.88%；R=128 时 recall 仍有 0.872-0.878，刷新次数减少 99.22%；R=512 时 recall 仍有 0.829-0.841，刷新次数减少 99.80%。如果论文主张要保守，建议主文强调 R=32 或 R=128，R=512 作为极限 case。

## 5. 论文可用结论

1. BlockWisePress 的同层 KV block 选择在相邻 decode step 间高度稳定。PG19 长上下文上，lag=1 的 index overlap 在 `single` 模式达到 0.91+，在 `window` 模式达到 0.98+。
2. 这种稳定性不是只存在于相邻 step。即使 lag=512，index overlap 仍保持在约 0.80-0.83，score cosine 仍大于 0.99。
3. 因此，稀疏 index 不必每个 decode step 都重新计算。按固定间隔刷新 index 可以显著减少 index computation；R=32/R=128 是更稳妥的默认候选，R=512 可以作为展示上界和极限复用能力的结果。
4. 推荐图：优先使用 `window` 模式的 lag curve 和 reuse curve，因为它更接近实际实现中的窗口打分；`single` 模式可作为更严格的补充。

## 6. 限制与下一步

已确认：

- 这组结果证明的是 sparse index / score distribution 的时间稳定性。
- 结果来自真实模型、真实 PG19 长上下文、device2 A6000、完整 watchdog 运行。
- 原始数据、聚合数据和 `.npz` score artifact 都已保留，可复画。

尚未证明：

- 固定步长复用 index 后的下游生成质量不下降。
- 固定步长刷新能在端到端实现里带来多少实际 latency/throughput 改善。
- 不同任务类型和更长上下文下是否维持同样趋势。

建议下一步：

1. 选 R=32、R=128、R=512 做真实 fixed-refresh BlockWisePress decode，对 LongBench / Needle / PG19 ppl 或 task score 验证质量。
2. 加入端到端 latency breakdown，将 index computation 的减少量和总 decode latency 联系起来。
3. 主文图使用 `window` 模式：一张 lag-overlap 曲线说明 temporal locality，一张 reuse-recall vs refresh-reduction 曲线说明 fixed-step refresh 的收益。
