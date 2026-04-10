# LongBench 分层 Prefill 压缩实验分析（15%）

## 实验设置

- 方法：
  - `block_wise_prefill_per_layer`
  - `chunkkv_prefill_per_layer`
- 压缩率：`0.3 / 0.5 / 0.7`
- 分层策略：
  - `skip_first=0`（不跳过前层）
  - `skip_first=1`（第 1 层不压缩）
  - `skip_first=2`（前 2 层不压缩）
- 数据集：`LongBench` 的
  - `hotpotqa`
  - `multifieldqa_en`
  - `triviaqa`
- 采样比例：`fraction=0.15`

图像产物：

- [dataset_grouped_bar.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/longbench_prefill_layer_ratio_compare_15pct/dataset_grouped_bar.png)
- [ratio_trend_by_dataset.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/longbench_prefill_layer_ratio_compare_15pct/ratio_trend_by_dataset.png)
- [summary.json](/home10T/bzx/workspace/kvpress-study/figure/experiments/longbench_prefill_layer_ratio_compare_15pct/summary.json)

---

## 关键结论

### 1) `hotpotqa`：不跳过前层更稳

- `BlockWise` 在三种压缩率下都更偏向 `skip_first=0` 最优或近最优：
  - `r=0.3`: `60.21 > 57.22 > 54.56`
  - `r=0.5`: `60.58 > 59.26 > 57.78`
  - `r=0.7`: `58.36 > 57.04 ≈ 57.04`
- `ChunkKV` 在 `r=0.7` 下 `skip_first=2` 最优（`60.74`），但整体增益不大。

结论：该任务上“前层不压缩”不是稳定收益项，特别是对 `BlockWise` 会明显掉分。

### 2) `multifieldqa_en`：跳过前 1 层收益显著

- `BlockWise`：
  - `r=0.3`: `58.82 -> 65.10`（`skip0 -> skip1`，显著提升）
  - `r=0.5`: `62.69 -> 62.99`（小幅提升）
  - `r=0.7`: `56.06 -> 52.70`（高压缩下反而下降）
- `ChunkKV`：
  - `r=0.3`: `55.30 -> 65.11`
  - `r=0.5`: `55.55 -> 64.44`
  - `r=0.7`: `60.84 -> 63.71`

结论：这是最支持“前层不压缩”假设的数据集，尤其 `ChunkKV` 收益非常稳定。

### 3) `triviaqa`：整体接近饱和，策略敏感度低

- `BlockWise` 基本在 `96.67` 附近（最高 `97.33`）
- `ChunkKV` 在 `skip0` 略低（`93~94`），`skip1/2` 提升到 `96.67`

结论：该任务上大多数配置都能达到高分，分层策略的边际影响有限。

---

## 方法对比观察

### `BlockWise` 的行为

- 在 `hotpotqa` 更偏向保留前层压缩（`skip0` 更好）
- 在 `multifieldqa_en` 的中低压缩率下可从 `skip1` 获益
- 说明 `BlockWise` 对“是否跳过前层”的最优策略具有任务依赖性

### `ChunkKV` 的行为

- 在 `multifieldqa_en` 和 `triviaqa` 上，`skip1/2` 常常优于 `skip0`
- 在 `hotpotqa` 上收益不稳定，但不会像 `BlockWise` 那样明显退化
- 整体看 `ChunkKV` 对“前层保护”更容易受益

---

## 可执行建议

1. 若要统一一套默认策略，建议先用 `skip_first=1`，再按任务微调。  
   理由：它在 `multifieldqa_en` 收益大，在 `triviaqa` 不吃亏，在 `hotpotqa` 只是小幅下降。

2. 若允许按任务定制：
   - `hotpotqa`: `skip_first=0`
   - `multifieldqa_en`: `skip_first=1`
   - `triviaqa`: `skip_first=1`（或 `0`，差异很小）

3. 后续如做论文主结论，建议把这轮最优候选（每方法 2~3 组）用更高比例（如 `fraction=0.5`）复验一次，验证趋势稳定性。
