# RULER 残差权重快速消融结果

## 1. 实验设置

本轮是一个速度优先的快速消融，只为了判断 `cross_layer_score_residual_weight` 的大致合理范围。

- 数据集：`RULER`
- 压缩率：`0.7`
- 子任务：
  - `niah_multikey_2`
  - `niah_multikey_3`
  - `niah_single_3`
  - `qa_1`
  - `qa_2`
- 每个任务样本数：`6`
- 总样本数：`30`
- 设备：`cuda:0`
- 残差权重：
  - `0.0`
  - `0.1`
  - `0.2`
  - `0.3`
  - `0.5`

结果目录：

- [ruler_residual_ablation_fast](/home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/ruler_residual_ablation_fast)

可视化图：

- [ruler_residual_ablation_fast.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/ruler_residual_ablation_fast/ruler_residual_ablation_fast.png)

## 2. 结果表

### `weight = 0.0`

- `niah_multikey_2 = 66.67`
- `niah_multikey_3 = 16.67`
- `niah_single_3 = 50.0`
- `qa_1 = 83.33`
- `qa_2 = 50.0`

### `weight = 0.1`

- `niah_multikey_2 = 66.67`
- `niah_multikey_3 = 16.67`
- `niah_single_3 = 33.33`
- `qa_1 = 83.33`
- `qa_2 = 50.0`

### `weight = 0.2`

- `niah_multikey_2 = 66.67`
- `niah_multikey_3 = 16.67`
- `niah_single_3 = 50.0`
- `qa_1 = 83.33`
- `qa_2 = 33.33`

### `weight = 0.3`

- `niah_multikey_2 = 50.0`
- `niah_multikey_3 = 16.67`
- `niah_single_3 = 50.0`
- `qa_1 = 83.33`
- `qa_2 = 33.33`

### `weight = 0.5`

- `niah_multikey_2 = 66.67`
- `niah_multikey_3 = 16.67`
- `niah_single_3 = 50.0`
- `qa_1 = 83.33`
- `qa_2 = 50.0`

## 3. 结论

这轮结果非常直接：

1. 残差权重不是当前性能瓶颈。  
   在这组关键子任务小样本上，除了少数点的小波动，整体几乎不随权重变化。

2. `0.1` 明显不好。  
   它把 `niah_single_3` 从 `50.0` 拉低到了 `33.33`，没有带来任何其它收益。

3. `0.3` 也不理想。  
   它把 `niah_multikey_2` 和 `qa_2` 都拉低了。

4. `0.0 / 0.2 / 0.5` 基本打平。  
   在这组快速样本上，这三者几乎没有可区分的优势。

## 4. 建议

如果只是为了保留一个轻量稳定器，我建议暂时用：

- `cross_layer_score_residual_weight = 0.2`

原因不是它在这轮里明显最好，而是：

- 它在完整 `RULER 0.7` 实验里比 `0.0` 略有提升
- 在这轮快速消融里没有明显副作用
- 比较符合“弱残差、当前层主导”的设计初衷

但更重要的结论是：

- 继续细调残差权重的收益很可能很小
- 下一步更应该转向新的关键块召回机制，而不是在残差权重上继续花时间
