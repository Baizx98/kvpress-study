# RULER 0.7 轻量 Token Correction 定向验证解读

## 1. 实验目的

这轮实验的目标很明确：

- 在不改变 `BlockWisePress` 总体框架的前提下
- 加入一个低开销的 `lightweight token correction`
- 只针对最薄弱的场景做最小验证

本轮只测试：

- 数据集：`RULER`
- 压缩率：`0.7`
- 采样比例：`0.5`
- 方法：
  - 旧版 `BlockWisePress`
  - 新版 `BlockWisePress + token correction`
  - `ChunkKV`

其中：

- 旧版 `BlockWisePress` 和 `ChunkKV` 的结果来自上一轮已完成实验：
  [prefill_compare_50pct_blockwise_chunkkv](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/prefill_compare_50pct_blockwise_chunkkv)
- 新版 `BlockWisePress` 的结果来自本轮实验：
  [ruler_token_correction_50pct](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/ruler_token_correction_50pct)

## 2. 可视化

- 全子任务对比图：
  [ruler_token_correction_vs_baselines.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_token_correction_50pct/ruler_token_correction_vs_baselines.png)
- 重点子任务对比图：
  [ruler_token_correction_vs_baselines_focus.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_token_correction_50pct/ruler_token_correction_vs_baselines_focus.png)

## 3. 结果汇总

### 3.1 宏平均

- 旧版 `BlockWisePress`: `74.68`
- 新版 `BlockWisePress + token correction`: `74.10`
- `ChunkKV`: `92.17`

结论很直接：这版轻量 `token correction` 没有带来提升，整体还略有下降。

### 3.2 重点子任务

- `niah_multikey_2`
  - old: `46.22`
  - new: `44.89`
  - chunkkv: `99.56`
- `niah_multikey_3`
  - old: `7.83`
  - new: `7.39`
  - chunkkv: `80.87`
- `niah_single_3`
  - old: `59.66`
  - new: `60.08`
  - chunkkv: `99.58`
- `qa_1`
  - old: `68.53`
  - new: `66.93`
  - chunkkv: `87.25`
- `qa_2`
  - old: `61.69`
  - new: `61.29`
  - chunkkv: `60.89`

从这些关键子任务看：

- `niah_single_3` 只有非常轻微的回升
- `niah_multikey_2/3` 没有改善，反而略退
- `qa_1` 也略退
- `qa_2` 基本持平

这说明当前这版 correction 没有命中真正的瓶颈。

## 4. 为什么这版 correction 没有效果

我认为主要有三个原因。

### 4.1 correction token 的选取方式和摘要分支高度同质

当前实现里，`token_correction_keys` 仍然是按块内 `key norm top-k` 选出来的。

这会导致一个问题：

- `topk_key_mean`
- `token_correction_keys`

二者都依赖同一种“高范数 token”偏置，信息来源高度重合。

因此新增的 correction 分支并没有真正带来新的判别力，只是对原有摘要分支做了轻微重复加权。

### 4.2 correction 聚合方式过于局部，但仍然不够尖锐

当前 correction 分数是：

- query 和少量 correction tokens 做相似度
- 在 correction token 维度取 `max`
- 再在 query 维度取 `mean`

这在形式上是“轻量”的，但实际有两个问题：

- 对 `multikey` 任务来说，仅靠少量高范数 token 不能覆盖多个支持证据
- 对 `single key` 任务来说，query 维度平均又会把最关键 query 的峰值冲淡

所以它既没有像 `ChunkKV` 那样真正恢复 token 级 sharpness，也没有保持原摘要分支那种稳定性。

### 4.3 当前主要短板不只是“答案块是否被看见”，而是“支持块召回不稳定”

前一轮失败 case 分析已经说明：

- 很多失败样本里，答案块本身并没有完全被删掉
- 问题在于远距离关键块、支持块的保留不稳定

而这版 correction 只是在单个块内部做了一个非常轻的补丁，并没有改善：

- 远距离热点块的召回
- 多支持块之间的稳定排序

所以它对 `niah_multikey_3` 这类任务几乎没有帮助。

## 5. 这轮实验带来的启发

这轮结果虽然是负结果，但信息价值很高，因为它排除了一个看起来合理、实际上不足够有效的方向。

可以确认的结论是：

1. 只在当前摘要框架上，补一个“同样按 key norm 选出来的少量 token correction”，不够。
2. 主要问题不是“缺少一个很小的块内补丁”这么简单。
3. 真正需要增强的，是：
   - 远距离关键块的稳定召回
   - 多 key / 多支持证据场景下的块排序能力

## 6. 下一步更值得做什么

相比继续微调这版 correction，我更建议下面两个方向。

### 方向 1：让 correction token 的选取和摘要分支解耦

如果还想保留 correction 路线，更值得尝试的是：

- 不再按 `key norm top-k` 选 correction tokens
- 改为：
  - 块内和最后 `q_window` 的 query 响应最强的 token
  - 或者块内 top-1 `query-aware token`

这样 correction 才会真正带来新信息，而不是重复摘要分支。

### 方向 2：优先尝试跨层热度稳定化

你刚刚提到的“像残差那样把上一层分数连接到下一层”，我认为比继续堆当前这版 token correction 更值得优先尝试。

原因是：

- 当前失败更像“跨层排序不稳定”
- 而不是单层打分完全错

一个非常轻量的形式就可以先验证：

- `score_l = alpha * current_score + (1 - alpha) * prev_layer_score`

这不会明显增加元数据开销，但有机会缓解高压缩下远距离块排序抖动的问题。

## 7. 结论

这轮 `RULER 0.7` 定向验证表明：

- 当前实现的 `lightweight token correction` 没有改善 `BlockWisePress`
- 它没有缩小与 `ChunkKV` 的差距
- 反而说明仅靠块内高范数 token 的轻量补丁是不够的

因此，下一步不建议继续在这版 correction 上做小修小补。  
更值得转向：

- query-aware 的 correction token 选择
- 或者你提出的跨层分数残差/平滑机制
