# RULER Ablation 10pct

## 目的

在 `RULER` 上对 `BlockWisePress` 做参数消融，测试：

- `q_window_size`
- `summary_topk_keys`
- `protected_recent_blocks`
- `mean_key_weight`

## 相关结果

- 结果目录：
  `evaluation/results/experiments/ruler_ablation_10pct/artifacts/`
- 图像：
  - `ruler_ablation_10pct_q_window.png`
  - `ruler_ablation_10pct_summary_topk.png`
  - `ruler_ablation_10pct_protected_recent.png`
  - `ruler_ablation_10pct_mean_key_weight.png`

## 用途

这组实验用于选择当前 `BlockWisePress` 的默认超参数。
