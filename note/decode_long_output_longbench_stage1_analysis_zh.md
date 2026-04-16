# Decode Long-Output LongBench Stage1 实验分析

## 1. 实验目的

本轮实验验证 decode 阶段两类 fixed-budget 策略在长输出任务上的表现：

- `decode_permanent_eviction_fixed_budget`
  - decode 阶段物理删除未保留块
- `decode_compute_cold_fixed_active_budget`
  - decode 阶段保留全部 KV，但只让 active blocks 参与计算

对照组为：

- `prefill_only_no_decode_pruning`
  - prefill 使用 `blockwise_main`
  - decode 不做额外压缩或计算稀疏

本轮重点不是继续做 blockwise 内部消融，而是回答：

> 在长输出场景下，decode 阶段的固定预算策略是否能在保持质量的同时带来可解释的内存/计算管理能力？

---

## 2. 实验设置

数据集只使用 `LongBench` 中长输出任务：

- `gov_report`
- `qmsum`
- `multi_news`

统一设置：

- `compression_ratio=0.3`
- `block_size=16`
- `q_window_size=16`
- `query_agg_mode=max`
- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=key_norm`
- `head_agg_mode=uniform_mean`
- `protected_recent_blocks=2`
- `min_answer_tokens=64`
- `min_context_tokens=4000`
- `max_filtered_samples=20`

运行设备：

- 物理 GPU 1 A6000
- 通过 `CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0` 映射到进程内 `cuda:0`

说明：

- `compression_ratio=0.3` 表示 prefill 阶段压掉约 `30%` 的块
- decode 阶段每 `block_size=16` 个生成 step 刷新一次
- decode fixed budget 以 prefill 后保留块数为锚点

---

## 3. 完整性与异常状态

实验完整性：

| 数据集 | 方法数 | 状态 |
| --- | ---: | --- |
| `gov_report` | `3/3` | 完成 |
| `qmsum` | `3/3` | 完成 |
| `multi_news` | `3/3` | 完成 |

异常状态：

- 无 `failed_jobs.jsonl`
- 无 `failed_jobs_final.jsonl`
- `watchdog` 未发现运行中 traceback / OOM / CUDA assert

曾出现过一次设备映射问题：

- 直接使用 `DEVICE=cuda:1` 时，PyTorch 的 `cuda:1` 映射到了被 `sglang` 占用的 3090
- 后续改为 `CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0`，实际使用物理 GPU 1 A6000
- 本轮最终结果均来自修正后的运行

---

## 4. 质量结果

| 数据集 | `prefill_only_no_decode_pruning` | `permanent_fixed_budget` | `compute_cold_fixed_budget` |
| --- | ---: | ---: | ---: |
| `gov_report` | `34.79` | `34.71` | `34.32` |
| `qmsum` | `25.17` | `25.06` | `25.07` |
| `multi_news` | `25.95` | `24.89` | `25.30` |
| Macro Avg | `28.64` | `28.22` | `28.23` |

相对 `prefill_only_no_decode_pruning` 的变化：

| 数据集 | `permanent_fixed_budget` | `compute_cold_fixed_budget` |
| --- | ---: | ---: |
| `gov_report` | `-0.08` | `-0.47` |
| `qmsum` | `-0.11` | `-0.10` |
| `multi_news` | `-1.06` | `-0.65` |
| Macro Avg | `-0.42` | `-0.41` |

结论：

- 两种 decode 策略整体质量损失都不大
- `qmsum` 上二者几乎等价
- `gov_report` 上永久驱逐略优
- `multi_news` 上计算冷块明显优于永久驱逐
- 宏平均上二者非常接近，`compute_cold` 只比 `permanent` 高 `0.01`

---

## 5. 运行时间观察

根据 `run.log` 中 controller 的 job 起止时间估算：

| 数据集 | `prefill_only_no_decode_pruning` | `permanent_fixed_budget` | `compute_cold_fixed_budget` |
| --- | ---: | ---: | ---: |
| `gov_report` | `7.28 min` | `40.23 min` | `29.30 min` |
| `qmsum` | `4.68 min` | `13.35 min` | `9.20 min` |
| `multi_news` | `6.83 min` | `30.90 min` | `21.62 min` |

结论：

- 当前实现下，decode 策略的额外开销非常明显
- `permanent_fixed_budget` 比 `compute_cold_fixed_budget` 更慢
- 主要原因很可能是：
  - 每 `16` 个 decode step 重新做块级打分
  - 永久驱逐还会物理 gather / 写回 cache
  - 当前实现更偏 correctness prototype，而不是性能优化版

这意味着：

- 质量结果已经说明 decode 策略“可行”
- 但系统价值还没有被当前实现释放出来
- 下一步不能只看质量分数，必须补 runtime / memory / active-block trace

---

## 6. 方法判断

## 6.1 永久驱逐

优点：

- `gov_report` 上最接近 prefill-only baseline
- 行为简单，物理 KV 确实减少
- 更适合作为显存 cap 机制

风险：

- `multi_news` 上质量损失最大
- 一旦误删，后续 decode 没有恢复机会
- 当前物理裁剪实现开销偏大

## 6.2 计算冷块

优点：

- `multi_news` 上比永久驱逐稳
- 保留了后续重新变热的可能性
- 更符合未来 offload / hot-cold 调度叙事

风险：

- 当前没有真正减少物理 KV
- 需要依赖 `masked_key_indices` / fake key path
- 当前实现仍然有较大 runtime 开销

## 6.3 当前主结论

从质量上看：

- 两条 decode fixed-budget 路径都能保持接近 baseline 的结果
- 暂时没有出现“compute cold 全面显著优于 permanent eviction”的证据
- 也没有出现“永久驱逐明显不可用”的证据

从系统角度看：

- 当前最突出的问题不是质量，而是 decode 刷新和块重评分的开销
- `compute_cold` 比永久驱逐更快一些，但仍比 prefill-only 慢很多

因此，本轮结论更适合表述为：

> fixed-budget decode 策略在长输出任务上质量可行，但当前实现的重评分和状态更新开销过高。下一步应先做过程指标与刷新开销优化，再决定是否扩大任务或 budget sweep。

---

## 7. 下一步建议

优先级 1：补过程统计

- `generated_length`
- `TPOT`
- `peak_gpu_memory`
- `decode_refresh_count`
- `avg_live_blocks`
- `avg_active_blocks`
- `masked_token_ratio`

优先级 2：降低 decode refresh 成本

- 从每 `16` step 刷新改成：
  - `32`
  - `64`
- 或者只在新 decode block 形成时更新 summary，非必要不重建全部块摘要

优先级 3：小规模 budget sweep

- `0.75 * B_prefill_keep`
- `1.00 * B_prefill_keep`
- `1.25 * B_prefill_keep`

优先级 4：方法选择

- 如果目标是显存 cap，继续优化 `permanent_fixed_budget`
- 如果目标是未来 GPU/CPU hot-cold 调度，继续优化 `compute_cold_fixed_budget`

当前更推荐：

- 短期继续保留两条路径
- 不立刻淘汰任何一条
- 先补系统指标，再做下一轮选择

---

## 8. 限制

本轮仍有几个限制：

- 每个任务只取 `20` 条样本
- 只看 LongBench 长输出子集
- 还没有记录真实 TPOT 和 peak memory
- 还没有导出每层 block state trace
- 当前 quality scorer 仍是 LongBench 自动指标，不能完全代表摘要可读性

这些限制不影响本轮作为 stage1 feasibility test 的价值，但不应直接作为最终系统 claim。
