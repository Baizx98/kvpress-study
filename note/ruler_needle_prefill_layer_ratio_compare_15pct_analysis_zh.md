# RULER + Needle 分层 Prefill 压缩实验分析（15%）

## 实验设置

- 方法：
  - `block_wise_prefill_per_layer`
  - `chunkkv_prefill_per_layer`
- 压缩率：`0.3 / 0.5 / 0.7`
- 分层策略：
  - `skip_first=0`
  - `skip_first=1`
  - `skip_first=2`
- 采样比例：`fraction=0.15`
- 数据集：
  - `RULER (4096)`，任务过滤：`niah_single_3, niah_multikey_3, qa_2`，`samples_per_task=6`
  - `Needle in a Haystack`，`needle_depth=50`，`max_context_length=16384`

图像产物：

- [ruler_compare.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_needle_prefill_layer_ratio_compare_15pct/ruler_compare.png)
- [needle_compare.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_needle_prefill_layer_ratio_compare_15pct/needle_compare.png)
- [summary.json](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_needle_prefill_layer_ratio_compare_15pct/summary.json)

---

## 关键结果

## 1) RULER：`BlockWise` 随压缩率明显退化，`ChunkKV` 几乎全程满分

### BlockWise（宏平均）

- `r=0.3`: `100`（skip0/1/2 全部一致）
- `r=0.5`: `75`（skip0/1/2 全部一致）
- `r=0.7`: `25`（skip0/1/2 全部一致）

细看子任务可见，退化主要来自：

- `niah_single_3`: `100 -> 100 -> 0`（随压缩率升高崩塌）
- `niah_multikey_3`: 基本稳定在 `50`（这轮样本上）

### ChunkKV

- `r=0.3 / 0.5 / 0.7` 均为 `100`（skip0/1/2 全一致）

结论：

- 在这组 RULER 子任务上，`ChunkKV` 明显强于 `BlockWise`。
- `skip_first=0/1/2` 对两种方法都几乎没有影响，主导因素是压缩方法本身与压缩率，而不是是否跳过前层。

## 2) Needle：所有配置分数完全一致

- 两种方法在 `0.3/0.5/0.7`、`skip=0/1/2` 下全部为同一个值：
  - `ROUGE-L F = 0.709677...`

结论：

- 这轮 Needle 配置下，看不出分层策略差异，也看不出方法差异。
- 当前设置更像是“通过性检查”，不是区分策略优劣的有效实验。

---

## 为什么 `skip_first` 在这轮几乎没效果

这轮和 LongBench 的现象不同，主要原因是任务结构与样本规模：

1. `RULER` 只测了 2 个有效子任务信号（`qa_2` 在 metrics 中未出现）
2. `Needle` 在 `fraction=0.15` 下本质上仍是单样本单深度，统计方差极低
3. 在这种设置下，分层策略的小差异很容易被“任务本身难度 + 评分离散性”掩盖

---

## 可执行建议

1. 如果目标是验证“前层不压缩是否有效”，`RULER+Needle` 这组设置不够敏感。  
   建议延续 `LongBench` 那组做主验证，`RULER/Needle` 放在补充实验。

2. 若还想在 `RULER` 上看 `skip_first`，建议：
   - 增加任务覆盖（至少加入 `niah_single_2`, `niah_multikey_2`）
   - 增加每任务样本量（`samples_per_task` 从 6 提升）

3. 若还想在 `Needle` 上拉开差异，建议：
   - 多个 `needle_depth`（例如 `10,30,50,70,90`）
   - 不同上下文长度组（例如 `8k/16k/32k`）

---

## 总结

这轮结果非常清楚地说明：

- 对 `RULER` 当前子集而言，`BlockWise` 的主要问题是高压缩率鲁棒性，而不是是否跳过前层；
- 对 `Needle` 当前设置而言，实验分辨率不足，无法支持分层策略结论；
- “前层不压缩”的收益，仍应以你前面的 `LongBench` 结果作为主证据。
