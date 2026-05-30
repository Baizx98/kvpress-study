# ATC26 BlockWisePress Ranked Top-k 跨 step 相似性实验结果

## 1. 实验状态

实验已完成。

- 运行目录：`evaluation/results/experiments/ATC26_blockwise_ranked_topk_temporal_similarity/artifacts/decode1024/`
- Watchdog heartbeat：`status=complete`
- 完成 job：8 / 8
- Raw JSONL：8 行
- Ranked index `.npz`：8 个
- Aggregate CSV：`ATC26_ranked_topk_temporal_similarity_aggregate.csv`
- Aggregate JSON：`ATC26_ranked_topk_temporal_similarity_aggregate.json`
- 原始 index artifact 总大小：约 4.3G

说明：运行过程中曾出现一次 watchdog 误判 heartbeat stale 并重启；修复 heartbeat 原子写后，实验通过 `--resume` 完整跑完。最终结果目录中的 8 个 job 均已完成。

## 2. 实验设置

- 模型：`llama31_8b_instruct`
- 数据集：PG19 test，本地路径 `/Tan/dataset/pg19-test`
- 上下文长度：8192、16384
- 样本数：每个上下文长度 4 个样本
- Decode steps：1024
- Block size：16
- Window query size：16
- 压缩率：0.7、0.5、0.3
- Lag sweep：1、2、4、8、16、32、64、128、256、512
- Reuse interval sweep：2、4、8、16、32、64、128、256、512
- GPU：物理 device2 A6000，通过 `CUDA_VISIBLE_DEVICES=2` 映射到进程内 `cuda:0`

这里的压缩率沿用 BlockWisePress 语义：`compression_ratio=0.7` 表示保留约 30% blocks，`0.5` 表示保留约 50% blocks，`0.3` 表示保留约 70% blocks。

## 3. 与上一版实验的关键区别

这次实验按新的定义采集：

1. 每个 decode step 使用完整 KV cache 计算 importance score。
2. 不做实际压缩，不修改 `past_key_values`。
3. 每个 step 记录按 score 降序排列的 ranked top-k block indices。
4. 记录 top-k block 的来源：prefill、decode、mixed tail。
5. 单独统计 decode 阶段产生的 block 进入 top-k 的比例。

因此这次结果更接近要验证的问题：完整 KV 中每一步重新计算 sparse index 时，top-k index 是否随 step 间隔变大而变化。

## 4. 主要结果

### 4.1 相邻 step 仍然高度相似

`window` 模式下，lag=1 的 overlap 非常高：

| Context | Compression | Lag=1 overlap | Lag=1 Jaccard |
|---:|---:|---:|---:|
| 8192 | 0.3 | 0.9895 | 0.9796 |
| 8192 | 0.5 | 0.9849 | 0.9708 |
| 8192 | 0.7 | 0.9804 | 0.9622 |
| 16384 | 0.3 | 0.9905 | 0.9814 |
| 16384 | 0.5 | 0.9858 | 0.9724 |
| 16384 | 0.7 | 0.9809 | 0.9631 |

结论：相邻 step 的 top-k block index 仍然高度稳定。这个结果支持短间隔复用 sparse index。

### 4.2 lag=512 时，高压缩率下 index 差异明显增大

`window` 模式下：

| Context | Compression | Lag=512 overlap | Lag=512 Jaccard | Decode-new ratio |
|---:|---:|---:|---:|---:|
| 8192 | 0.3 | 0.8290 | 0.7479 | 0.5224 |
| 8192 | 0.5 | 0.7646 | 0.6560 | 0.5291 |
| 8192 | 0.7 | 0.6827 | 0.5526 | 0.6065 |
| 16384 | 0.3 | 0.8600 | 0.7799 | 0.3763 |
| 16384 | 0.5 | 0.7961 | 0.6881 | 0.3591 |
| 16384 | 0.7 | 0.7286 | 0.6015 | 0.3963 |

结论：当压缩率更高、top-k 更小，远距离 step 的 index 差异明显放大。尤其在 `compression_ratio=0.7` 时，lag=512 的 overlap 降到 0.68-0.73，Jaccard 降到 0.55-0.60。这个趋势符合“长间隔不能无限复用 index”的预期。

### 4.3 8192 上下文比 16384 更容易看到 decode block 进入 top-k

`window` 模式下 decode-origin block 在 top-k 中的比例：

| Context | Compression | Mean decode-in-topk | Last-128 decode-in-topk | Best decode global rank |
|---:|---:|---:|---:|---:|
| 8192 | 0.3 | 0.0829 | 0.1483 | 3.1721 |
| 8192 | 0.5 | 0.1132 | 0.2005 | 3.1721 |
| 8192 | 0.7 | 0.1721 | 0.2988 | 3.1721 |
| 16384 | 0.3 | 0.0423 | 0.0777 | 12.6889 |
| 16384 | 0.5 | 0.0556 | 0.1024 | 12.6889 |
| 16384 | 0.7 | 0.0805 | 0.1468 | 12.6889 |

结论：decode 阶段新产生的 block 确实会进入 top-k，并且越到后期比例越高。8192 上下文中这个趋势更强：高压缩率 0.7 下，最后 128 step 的 top-k 中约 29.9% 来自 decode/mixed blocks。16384 上下文中比例较低，说明更长 prefill 会稀释 decode block 的相对占比。

### 4.4 固定刷新间隔的可用范围

`window` 模式下 reuse recall：

| Context | Compression | R=32 recall | R=128 recall | R=512 recall |
|---:|---:|---:|---:|---:|
| 8192 | 0.3 | 0.9421 | 0.9119 | 0.8596 |
| 8192 | 0.5 | 0.9155 | 0.8715 | 0.8002 |
| 8192 | 0.7 | 0.8881 | 0.8290 | 0.7346 |
| 16384 | 0.3 | 0.9470 | 0.9191 | 0.8836 |
| 16384 | 0.5 | 0.9188 | 0.8767 | 0.8247 |
| 16384 | 0.7 | 0.8904 | 0.8341 | 0.7628 |

结论：固定刷新是可行的，但刷新间隔应随压缩率收紧。对于 `compression_ratio=0.7`，R=512 的 recall 已降到 0.73-0.76，不适合作为默认策略；R=32 仍有约 0.89 recall，更稳妥。对于低压缩率 0.3，R=128/R=512 仍较稳。

## 5. 论文可用结论

1. 相邻 decode step 的 BlockWise top-k index 高度相似，说明每一步都重新计算 sparse index 存在冗余。
2. 随 step 间隔增大，top-k index 相似度显著下降；高压缩率下下降更明显。
3. decode 阶段新增 block 会逐渐进入 top-k，尤其在较短上下文和较高压缩率下更明显。这说明长间隔复用会错过新的重要 block。
4. 推荐系统设计：采用固定间隔刷新，但刷新间隔不应过大；可把 R=32 作为保守默认，R=128 作为中等压缩率的候选，R=512 只作为低压缩率或附录上界。

## 6. 限制

- 当前实验只证明 sparse index 的时间变化，不直接证明 fixed-refresh 后的任务质量。
- 优化后的脚本为了降低 CPU 后处理开销，`rank_biased_overlap` 和 `common_rank_delta` 暂未计算，主结论基于 overlap、Jaccard、decode-new ratio、reuse recall。
- 当前只跑 PG19；后续应在 LongBench / Needle 上验证质量和端到端 latency。

## 7. 下一步

1. 基于这组结果选择 `compression_ratio=0.7, 0.5, 0.3` 下的 R=32/R=128/R=512 做真实 fixed-refresh decoding。
2. 测任务质量或 PPL，确认 index 复用不会显著损害输出。
3. 测端到端 latency breakdown，量化 index computation 减少带来的实际收益。
