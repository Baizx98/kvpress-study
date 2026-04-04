# RULER 0.7 跨层分数残差定向验证解读

## 1. 实验目的

这轮实验用于验证一个很轻的假设：

- `BlockWisePress` 在高压缩检索任务上的主要问题，可能不是单层块打分完全错误
- 而是块分数在层间不够稳定，导致远距离关键块与支持块的排序抖动较大

因此本轮只做一个最小改动：

- 在 `BlockWisePress` 中引入跨层块分数残差
- 形式为：
  `score_l = 0.8 * current + 0.2 * prev_layer`

同时，为了让结论更干净：

- 上一轮没有效果的 `token correction` 默认关闭
- 只单独测试跨层残差

## 2. 实验设置

- 数据集：`RULER`
- 压缩率：`0.7`
- 采样比例：`0.5`
- 设备：`cuda:0`
- 残差权重：`0.2`

比较对象：

- 旧版 `BlockWisePress`
- `BlockWisePress + token correction`
- `BlockWisePress + cross-layer residual`
- `ChunkKV`

结果目录：

- 当前实验：
  [ruler_cross_layer_residual_50pct](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/ruler_cross_layer_residual_50pct)
- 历史对照：
  [prefill_compare_50pct_blockwise_chunkkv](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/prefill_compare_50pct_blockwise_chunkkv)
  [ruler_token_correction_50pct](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/ruler_token_correction_50pct)

图：

- 全子任务图：
  [ruler_cross_layer_residual_vs_baselines.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_cross_layer_residual_50pct/ruler_cross_layer_residual_vs_baselines.png)
- 重点子任务图：
  [ruler_cross_layer_residual_vs_baselines_focus.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_cross_layer_residual_50pct/ruler_cross_layer_residual_vs_baselines_focus.png)

## 3. 结果汇总

### 3.1 宏平均

- 旧版 `BlockWisePress`: `74.68`
- `BlockWisePress + token correction`: `74.10`
- `BlockWisePress + cross-layer residual`: `74.73`
- `ChunkKV`: `92.17`

结论：

- 跨层残差比旧版 `BlockWisePress` 略有提升
- 也明显好于上一轮的 `token correction`
- 但提升幅度很小，还远不足以缩小与 `ChunkKV` 的大差距

### 3.2 重点子任务

- `niah_multikey_2`
  - old: `46.22`
  - token correction: `44.89`
  - cross-layer residual: `43.56`
  - chunkkv: `99.56`
- `niah_multikey_3`
  - old: `7.83`
  - token correction: `7.39`
  - cross-layer residual: `7.83`
  - chunkkv: `80.87`
- `niah_single_3`
  - old: `59.66`
  - token correction: `60.08`
  - cross-layer residual: `63.87`
  - chunkkv: `99.58`
- `qa_1`
  - old: `68.53`
  - token correction: `66.93`
  - cross-layer residual: `67.33`
  - chunkkv: `87.25`
- `qa_2`
  - old: `61.69`
  - token correction: `61.29`
  - cross-layer residual: `60.89`
  - chunkkv: `60.89`

## 4. 结果解读

### 4.1 跨层残差确实比 token correction 更对路

从结果看：

- `token correction` 基本没有提升
- 跨层残差至少在方向上更合理

最明显的改善体现在：

- `niah_single_3`: `59.66 -> 63.87`
- `niah_multivalue`: `78.25 -> 79.07`
- `niah_multiquery`: `85.59 -> 86.08`

这说明：

- 对单 key 深层检索
- 对部分需要层间稳定召回的任务

跨层分数平滑是有帮助的。

### 4.2 但它没有解决真正最难的 multikey 问题

`niah_multikey_2/3` 基本没有改善：

- `niah_multikey_2` 还略有下降
- `niah_multikey_3` 完全没有回升

这说明：

- 当前主要短板之一仍然是多支持证据块的召回和排序
- 仅靠层间平滑，不能凭空恢复那些在当前层已经没有进入高分集合的远距离关键块

换句话说：

- 残差能稳定已有热点
- 但不能替代“更强的远距离关键块识别能力”

### 4.3 这个结果非常像“稳定性改善了，但召回能力没变”

这是我对这轮结果最核心的判断。

跨层残差带来的收益主要是：

- 降低层间排序抖动
- 让某些已经偏高的关键块更稳定地保留下来

但它没有改变当前块摘要的表达边界：

- 对 `multikey_3` 这类需要多个远距离支持块的任务
- 当前摘要机制仍然不够强

所以最终表现为：

- 一些单点任务略有改善
- 但真正拉大差距的 hardest cases 依然没动

## 5. 和前一轮 token correction 对比的结论

如果把这两轮放在一起看，可以得到一个非常清晰的排序：

1. `token correction` 不是当前最值得继续挖的方向  
   它基本没有带来新信息。

2. `cross-layer residual` 比 `token correction` 更值得保留  
   它至少改善了一部分层间稳定性问题。

3. 但 `cross-layer residual` 也不是决定性解法  
   它无法解决多 key / 多支持块召回不足的问题。

## 6. 下一步该怎么做

我认为现在最合理的路线是：

### 路线 1：保留跨层残差，但把它视为“稳定器”

跨层残差不应该被当成主改进，而应该被当成：

- 一个默认开启的小稳定器

因为它：

- 开销低
- 没有明显副作用
- 对部分任务有小幅帮助

### 路线 2：真正的下一步仍然要增强远距离关键块召回

既然：

- `token correction` 不够
- `cross-layer residual` 只能稳定已有高分块

那么下一步真正应该解决的是：

- 如何让远距离关键块更容易进入高分集合

这可以考虑两种方向：

1. `top hot blocks`
   - 在 recent blocks 之外
   - 显式保留少量高热远距离块

2. 更 query-aware 的 block summary
   - 不是继续按 `key norm` 选 token
   - 而是让少量代表 token 的选择更贴近 query 响应

## 7. 结论

这轮跨层残差实验的结论是：

- 它比轻量 `token correction` 更有效
- 说明“层间排序稳定性”确实是问题的一部分
- 但它只能带来小幅改善
- 无法解决 `RULER 0.7` 下最核心的 `multikey` 召回短板

因此，更合理的策略是：

- 把跨层残差保留为一个轻量稳定器
- 下一步把主要精力转到远距离关键块召回机制上
