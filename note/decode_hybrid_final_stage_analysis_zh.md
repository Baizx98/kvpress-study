# Decode Hybrid Final Stage 分析

## 1. 实验目的

这是 decode 算法探索的最后一轮验证，目标很收敛：

- 在 `dense_prefill` 主线下，比较
  - `permanent_decode`
  - `compute_cold_decode`
  - `hybrid_decode`
- 回答 `hybrid` 是否能把前两者的优点合起来，成为最终推理框架的更优统一方案

本轮 `hybrid` 的定义是：

- `permanent core + cold fringe`

即：

- 先按 `total_budget` 物理保留一部分历史块
- 再在保留块内部按 `active_budget` 做 cold masking

---

## 2. 配置与数据

### 2.1 统一设置

- `prefill` 不压缩：
  - `compression_ratio = 0.0`
- `block_size = 16`
- `q_window_size = 16`
- `refresh_interval = 16`
- `protected_recent_blocks = 2`
- `query_agg_mode = max`
- `summary_mode = mean_plus_norm_topk_mean`
- `representative_mode = key_norm`
- `head_agg_mode = uniform_mean`

### 2.2 LongBench 主验证

- `gov_report`
- `qmsum`
- `multi_news`

筛选：

- `min_answer_tokens = 64`
- `min_context_tokens = 4000`
- `max_filtered_samples = 20`

### 2.3 RULER 补充验证

- `data_dir = 4096`
- `task_filter = niah_single_3, niah_multikey_2, niah_multikey_3, qa_2`
- `samples_per_task = 20`
- `max_new_tokens = 128`

### 2.4 方法矩阵

- `Permanent 128`
- `Permanent 160`
- `Compute-Cold 128`
- `Compute-Cold 160`
- `Hybrid 128/96`
- `Hybrid 160/128`

其中：

- `Hybrid 128/96` 表示 `total_budget=128, active_budget=96`
- `Hybrid 160/128` 表示 `total_budget=160, active_budget=128`

---

## 3. 完整性与异常说明

这轮原始产物共有：

- `24` 个 `config.yaml`
- `24` 个 `metrics.json`

所以从结果完整性上看：

- 所有逻辑配置都已完成
- 没有真实缺失结果

但控制器日志里仍有 `6` 条 `failed_jobs*.jsonl` 记录，全部来自 `RULER`，并且都是：

- `return_code = 0`
- `reason = missing_metrics`

这不是模型或算法失败，而是控制器匹配完成结果时的类型不一致：

- `match_fields` 里 `data_dir` 使用的是字符串 `"4096"`
- `config.yaml` 里 `data_dir` 实际落成了整数 `4096`

因此本轮分析采用的规则是：

- 直接以 `metrics.json` 是否存在作为有效结果标准
- 忽略这 6 条假失败记录

---

## 4. LongBench 主结果

## 4.1 分任务结果

### `gov_report`

- `Permanent 128 = 31.25`
- `Permanent 160 = 32.65`
- `Compute-Cold 128 = 31.32`
- `Compute-Cold 160 = 31.92`
- `Hybrid 128/96 = 29.89`
- `Hybrid 160/128 = 31.17`

最优：

- `Permanent 160 = 32.65`

结论：

- `hybrid` 两档都没有接近最优
- 尤其 `Hybrid 128/96` 明显偏低

### `qmsum`

- `Permanent 128 = 22.53`
- `Permanent 160 = 24.87`
- `Compute-Cold 128 = 22.48`
- `Compute-Cold 160 = 23.10`
- `Hybrid 128/96 = 21.44`
- `Hybrid 160/128 = 23.21`

最优：

- `Permanent 160 = 24.87`

结论：

- `Hybrid 160/128` 相比 `Compute-Cold 160` 略有改善
- 但仍明显低于 `Permanent 160`
- `Hybrid 128/96` 仍然是最低档

### `multi_news`

- `Permanent 128 = 23.89`
- `Permanent 160 = 23.96`
- `Compute-Cold 128 = 24.47`
- `Compute-Cold 160 = 25.38`
- `Hybrid 128/96 = 24.27`
- `Hybrid 160/128 = 23.90`

最优：

- `Compute-Cold 160 = 25.38`

结论：

- `Hybrid 128/96` 能略好于 `Permanent`
- 但仍低于 `Compute-Cold`
- `Hybrid 160/128` 反而退化到接近 `Permanent`

## 4.2 LongBench 宏平均

- `Permanent 128 = 25.89`
- `Permanent 160 = 27.16`
- `Compute-Cold 128 = 26.09`
- `Compute-Cold 160 = 26.80`
- `Hybrid 128/96 = 25.20`
- `Hybrid 160/128 = 26.09`

宏平均最优：

- `Permanent 160 = 27.16`

关键信息：

- `Hybrid 160/128` 只追平了 `Compute-Cold 128`
- 仍低于：
  - `Compute-Cold 160`
  - `Permanent 160`
- `Hybrid 128/96` 是全组最差宏平均

---

## 5. RULER 补充结果

### 5.1 宏平均

- `Permanent 128 = 80.0`
- `Permanent 160 = 87.5`
- `Compute-Cold 128 = 80.0`
- `Compute-Cold 160 = 87.5`
- `Hybrid 128/96 = 75.0`
- `Hybrid 160/128 = 80.0`

结果非常直接：

- `Hybrid 128/96` 明显差于同量级基线
- `Hybrid 160/128` 也只回到了 `128` 档基线水平
- 它没有接近 `160` 档的 `Permanent/Compute-Cold`

### 5.2 分任务观察

`Hybrid` 的主要损失集中在：

- `niah_multikey_2`
- `niah_multikey_3`

具体表现：

- `Hybrid 128/96`
  - `niah_multikey_2 = 85`
  - `niah_multikey_3 = 55`
- 对比 `Permanent 128 / Compute-Cold 128`
  - `95`
  - `65`

`Hybrid 160/128` 则表现为：

- `niah_multikey_2 = 95`
- `niah_multikey_3 = 65`

它只恢复到 `128` 档基线水平，没有吃到 `160 total budget` 理应带来的检索收益。

这说明：

- 在这组检索型任务上，`hybrid` 把一部分预算留给 cold fringe 并没有产生收益
- 反而削弱了真正决定命中的核心历史块保留能力

---

## 6. 核心结论

## 6.1 `hybrid` 没有成为更优统一方案

这是这轮最重要的结论。

从 LongBench 和 RULER 两边看，`hybrid` 都没有表现出“统一吸收两者优点”的迹象：

- 在 `gov_report` 上输给 `Permanent`
- 在 `multi_news` 上输给 `Compute-Cold`
- 在 `qmsum` 上也没超过 `Permanent`
- 在 RULER 上整体更差

所以当前没有证据支持：

- `hybrid decode` 是比 `permanent` / `compute-cold` 更好的最终路线

## 6.2 `Permanent 160` 仍然是当前最稳主线

LongBench 宏平均最优仍然是：

- `Permanent 160 = 27.16`

而且它还同时拿下：

- `gov_report`
- `qmsum`

因此如果必须收敛成一个默认最终框架，目前最稳的答案仍是：

- `dense_prefill + permanent_decode @ 160 blocks`

## 6.3 `Compute-Cold 160` 仍是 `multi_news` 特化分支

如果你后面愿意接受“按任务分流”，那：

- `multi_news` 更适合 `Compute-Cold 160`

这个判断在上一轮 fixed-budget stage1 就已经出现，这轮 `hybrid` 没有改变它。

---

## 7. 对后续工作的建议

这轮结果的意义不是“`hybrid` 还可以再改一改”，而是：

- decode 算法树基本可以停止继续扩展了

原因很明确：

- `Permanent`
- `Compute-Cold`
- `Hybrid`

三条主干已经都做过了，而且：

- `Hybrid` 没有带来结构性收益

所以后续更合理的路线是：

1. 停止继续扩 decode 算法
2. 将默认最终框架定为：
   - `dense_prefill + permanent_decode @ 160`
3. 若保留任务特化说明，可以补充：
   - `multi_news` 上 `dense_prefill + compute_cold_decode @ 160` 更优
4. 之后把精力转向：
   - 分层 budget
   - decode 永久驱逐与稀疏策略的更强故事组织
   - 更大规模主实验与论文叙事

---

## 8. 一句话总结

最后一次 `hybrid decode` 实验已经给出足够明确的答案：

> `hybrid` 没有成为更优统一方案；当前最稳的最终推理框架仍然是 `dense_prefill + permanent_decode @ 160 blocks`，可以停止继续扩展 decode 算法树了。
