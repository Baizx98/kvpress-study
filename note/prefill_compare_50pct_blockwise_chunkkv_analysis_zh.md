# BlockWisePress vs ChunkKV 50% 数据集对比实验解读

## 1. 实验设置

- 对比方法：`BlockWisePress`、`ChunkKV`
- 压缩阶段：仅 `prefill`
- 压缩率：`0.3`、`0.7`
- 数据集：`RULER`、`LongBench(triviaqa)`、`Needle in a Haystack`
- 数据集采样比例：`0.5`
- 设备：仅 `cuda:0`

本轮 `BlockWisePress` 使用的是消融实验后更新的默认超参数：

- `q_window_size = 32`
- `summary_topk_keys = 4`
- `protected_recent_blocks = 2`
- `mean_key_weight = 0.75`

详细图：

- `RULER`：[prefill_compare_50pct_blockwise_chunkkv_detailed_ruler.png](/home10T/bzx/workspace/kvpress-study/figure/prefill_compare_50pct_blockwise_chunkkv_detailed_ruler.png)
- `LongBench`：[prefill_compare_50pct_blockwise_chunkkv_detailed_longbench.png](/home10T/bzx/workspace/kvpress-study/figure/prefill_compare_50pct_blockwise_chunkkv_detailed_longbench.png)
- `Needle in a Haystack`：[prefill_compare_50pct_blockwise_chunkkv_detailed_needle_in_haystack.png](/home10T/bzx/workspace/kvpress-study/figure/prefill_compare_50pct_blockwise_chunkkv_detailed_needle_in_haystack.png)

结果目录：

- [results_prefill_compare_50pct_blockwise_chunkkv](/home10T/bzx/workspace/kvpress-study/evaluation/results_prefill_compare_50pct_blockwise_chunkkv)

## 2. 结果汇总

### 2.1 RULER

- `BlockWisePress`
  - `ratio=0.3`: 宏平均约 `91.10`
  - `ratio=0.7`: 宏平均约 `74.68`
- `ChunkKV`
  - `ratio=0.3`: 宏平均约 `95.57`
  - `ratio=0.7`: 宏平均约 `92.17`

几个关键子任务：

- `BlockWisePress`
  - `ratio=0.3`: `qa_1=84.46`, `qa_2=63.31`, `niah_multikey_3=68.70`, `niah_single_3=98.74`
  - `ratio=0.7`: `qa_1=68.53`, `qa_2=61.69`, `niah_multikey_3=7.83`, `niah_single_3=59.66`
- `ChunkKV`
  - `ratio=0.3`: `qa_1=87.65`, `qa_2=62.90`, `niah_multikey_3=100.00`, `niah_single_3=99.58`
  - `ratio=0.7`: `qa_1=87.25`, `qa_2=60.89`, `niah_multikey_3=80.87`, `niah_single_3=99.58`

### 2.2 LongBench(triviaqa)

- `BlockWisePress`
  - `ratio=0.3`: `89.19`
  - `ratio=0.7`: `84.58`
- `ChunkKV`
  - `ratio=0.3`: `89.19`
  - `ratio=0.7`: `89.72`

### 2.3 Needle in a Haystack

- `BlockWisePress`
  - `ratio=0.3`: 平均 `rouge-l_f = 0.709677`
  - `ratio=0.7`: 平均 `rouge-l_f = 0.699944`
- `ChunkKV`
  - `ratio=0.3`: 平均 `rouge-l_f = 0.709677`
  - `ratio=0.7`: 平均 `rouge-l_f = 0.709677`

`BlockWisePress` 在 `ratio=0.7` 下各 depth 的 `rouge-l_f` 为：

- `[0.750000, 0.620690, 0.709677, 0.709677, 0.709677]`

这说明下降主要来自单个 depth 点，而不是所有 depth 同时退化。

## 3. 结果解读

### 3.1 默认参数更新是有效的

本轮默认参数来自前一轮 `RULER` 消融：

- 更短的 `q_window`
- 更大的 `summary_topk_keys`
- 更小的 `protected_recent_blocks`
- 更高的 `mean_key_weight`

从结果看，这组参数让 `BlockWisePress` 在中低压缩率下已经比较稳：

- 在 `LongBench 0.3` 上与 `ChunkKV` 持平
- 在 `Needle 0.3` 上与 `ChunkKV` 完全持平
- 在 `RULER 0.3` 上也已经达到 `91+`

这说明当前这版简化后的 `BlockWisePress` 已经不再是“方向错误”，而是“高压缩时仍存在明显短板”。

### 3.2 当前短板仍然集中在 RULER 的高压缩检索任务

最明显的问题出现在 `RULER 0.7`：

- `niah_multikey_3` 从 `68.70` 直接掉到 `7.83`
- `niah_single_3` 从 `98.74` 掉到 `59.66`

而 `ChunkKV` 在同样 `0.7` 下仍然保持很高：

- `niah_multikey_3 = 80.87`
- `niah_single_3 = 99.58`

这基本说明：

- 你的块摘要方案在高压缩下，对“块内极少数关键 token”的保真仍然不够
- 尤其是多 key 检索和深层单 key 检索任务，当前块级摘要仍然会把关键 token 稀释掉

### 3.3 LongBench 和 Needle 说明当前方法在“整体语义保持”上已经接近可用

`LongBench(triviaqa)` 的结果很有代表性：

- `ratio=0.3` 完全不输 `ChunkKV`
- `ratio=0.7` 虽然有差距，但不是灾难性下跌

`Needle in a Haystack` 也类似：

- `ratio=0.3` 完全持平
- `ratio=0.7` 只是个别 depth 点下降

这说明当前 `BlockWisePress` 的问题不是普遍性的语义损坏，而是：

- 对极端检索任务
- 对极少数关键 token 主导的块
- 在高压缩下仍然缺少足够尖锐的校正能力

## 4. 方法层面的结论

这轮结果支持下面这个判断：

- `BlockWisePress` 的“块摘要常驻 + query 与块摘要交互”的总体方向是对的
- 这条路线已经在 `LongBench` 和 `Needle` 上展现出比较好的性价比
- 但如果要在 `RULER` 的高压缩检索任务上进一步逼近 `ChunkKV`，只靠当前摘要还不够

因此，下一步最合理的改进仍然是此前讨论过的轻量方向，而不是重新把系统做复杂：

- 保持当前块摘要主路径不变
- 增加一个 very-cheap 的 `lightweight token correction`
- 只对每个块极少量代表 token 做额外 query 校正

这样仍然可以保持未来块卸载系统需要的低元数据开销，同时有机会补足 `RULER 0.7` 这类高压缩检索场景。

## 5. 下一步建议

建议优先做下面两步：

1. 保持当前默认超参数不变，先把这版结果作为新的基线版本固定下来。

2. 在当前实现上增加轻量 token correction，并只重点重跑以下点：

- `RULER`
- `ratio=0.7`
- 重点观察：
  - `niah_multikey_2`
  - `niah_multikey_3`
  - `niah_single_3`
  - `qa_1`
  - `qa_2`

如果这些点能显著回升，再考虑是否扩展到完整多数据集评测。
