# Decode 阶段 Prompt KVCache 重要性热力图实验方案

## 1. 问题陈述

目标是验证：

> 在同一条请求的不同 decode step 下，模型对 prompt KV cache 中不同 token position 的关注区域会变化，因此 KV cache 重要性不是静态不变的。

本实验只做解释性 trace，不直接比较任务分数。核心产物是若干张二维热力图：

- x 轴：prompt token position
- y 轴：decode step
- 颜色：粉色表示该 prompt token 在该 decode step 被判定为应该保留；蓝色表示丢弃

## 2. 实验假设

如果 prompt KV cache 的重要性随 decode 阶段变化，那么同一条 LongBench 请求在不同 decode step 的 retained token set 应该出现明显漂移：

- 早期 decode 可能更关注问题、指令、局部上下文或答案起始相关位置；
- 中后期 decode 可能重新关注 prompt 中不同 evidence span；
- 长输出任务中这种漂移应更明显，因为生成过程会覆盖更多推理/摘要阶段。

若热力图中粉色区域随 step 发生移动、扩散或周期性变化，就能支持“decode 阶段不能只依赖一次 prefill 静态压缩决策”的叙事。

## 3. 数据集与样本选择

数据集使用本仓库已有 LongBench 路径：

- Hugging Face dataset: `Xnhyacinth/LongBench`
- 缓存默认沿用仓库设置：`/Tan/dataset/hf_home`

优先选择长输出 LongBench 子任务，因为需要 256/512 decode steps：

- `gov_report`
- `qmsum`
- `multi_news`

样本选择策略：

- 每个子任务先筛选 `prompt_tokens >= 4000` 的样本；
- 排除很可能早停的异常短输出样本；
- 每个子任务取 1 条代表样本，总计 3 条；
- 固定 `seed=42`，保存 sample manifest，记录 dataset row index、prompt token length、context SHA1、问题/输入摘要。

如果 512 step 下某条样本提前 EOS，则保留实际 step 数，同时在 manifest 标注 `early_eos=true`；必要时从同一子任务补选下一条长样本。

## 4. 模型与设备

默认先使用一个模型完成解释性实验：

- `/Tan/model/Llama-3.1-8B-Instruct`

理由：

- 本地 checkpoint 已存在；
- 与仓库前面多组 ATC26 / decode 实验一致；
- 8B 模型在 A6000 上跑 512-step attention trace 更可控。

设备：

- 物理 GPU：`NVIDIA RTX A6000`
- 运行时使用：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`
- 进程内 device：`cuda:0`

当前检查到物理 GPU 2 是 A6000，但显存几乎已占满。正式执行前需要先确认占用来源；如果是之前的显存占位进程，需要停止或切换空闲 A6000 后再跑。

## 5. 重要性分数定义

对每个 decode step `t`，取当前生成 token 的 query 对 prompt KV token 的 attention score：

```text
A_t[p] = aggregate_attention(query at decode step t, prompt key position p)
```

聚合方式：

- 只统计 prompt token，不统计已生成 token；
- layer 聚合：默认取中后层平均，建议先用最后 8 层均值；
- KV head / query head 聚合：对 head 取均值；
- score 使用 softmax 后 attention probability；
- 可选保存 raw score，用于后续改阈值或画连续热力图。

二值保留规则保持 token 粒度，不做 block 聚合：

```text
keep_t[p] = A_t[p] in top K prompt positions at decode step t
```

默认设置：

- `compression_ratio=0.3, 0.5, 0.7`
- 语义与仓库现有 KVPress 实验保持一致：`compression_ratio=0.3` 表示丢弃约 30% prompt token，保留 attention score 最高的 70% prompt token。
- 每个 decode step 单独排序并得到 token-level keep mask。

## 6. 实验矩阵

主实验矩阵：

| 维度 | 设置 |
| --- | --- |
| dataset | `gov_report`, `qmsum`, `multi_news` |
| samples | 每个 dataset 1 条，总计 3 条 |
| model | `Llama-3.1-8B-Instruct` |
| max decode steps | `256`, `512` |
| keep granularity | token-level score, token-level display |
| compression ratio | `0.3`, `0.5`, `0.7` |
| seed | `42` |

总计：

```text
3 samples x 2 decode lengths x 3 compression ratios = 18 heatmaps
```

如果首轮结果清晰，再扩展：

- 再加 `keep_ratio=0.5` 对比压缩强度；
- 再加 `Mistral-7B-Instruct-v0.3` 或 `Qwen3-8B` 验证跨模型一致性。

## 7. 采集实现计划

新增采集脚本：

```text
evaluation/ATC26_collect_decode_prompt_kvcache_importance_heatmap.py
```

脚本职责：

1. 加载 LongBench 指定子任务；
2. 按 token length 和 seed 选择样本；
3. 使用 `model.generate` 或手写 greedy decode 循环生成 256/512 step；
4. 每一步打开 `output_attentions=True` 或 attention hook；
5. 提取当前 decode query 对 prompt positions 的 attention；
6. 聚合 layer/head；
7. 计算 token-level top-k keep mask；
8. 保存每个 step 的二值 mask 和必要 metadata。

优先采用手写 greedy decode 循环，而不是一次性 `generate`：

- 更容易逐 step 拿 attention；
- 更容易控制只保存 prompt attention，避免完整 attention tensor 爆显存；
- 每步保存 CPU numpy，及时释放 GPU tensor。

## 8. 绘图计划

新增绘图脚本：

```text
figure/ATC26_plot_decode_prompt_kvcache_importance_heatmap.py
```

每个样本和 decode length 输出一张主图：

- 蓝色：discard
- 粉色：keep
- x 轴：prompt token position
- y 轴：decode step
- title 中标注 dataset、row index、prompt length、decode steps、keep ratio

如果 prompt 太长，主图会同时输出两个版本：

- full prompt heatmap
- downsampled overview heatmap，例如每 4 或 8 个 token position 聚合显示

额外保存 token/block 变化指标：

- adjacent-step Jaccard：`J(keep_t, keep_{t+1})`
- lag Jaccard：`lag=16,32,64,128`
- per-position keep frequency：每个 prompt token 在所有 decode step 中被保留的比例

这些指标用于辅助解释热力图，不抢主图叙事。

## 9. 结果目录

实验名：

```text
ATC26_decode_prompt_kvcache_importance_heatmap_longbench
```

原始结果：

```text
evaluation/results/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/
```

建议文件：

- `artifacts/sample_manifest.json`
- `artifacts/raw/*.npz`
- `artifacts/summary_metrics.csv`
- `artifacts/run_config.json`
- `run.log`

图像目录：

```text
figure/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/
```

建议图像：

- `heatmap_<dataset>_row<idx>_decode256.png`
- `heatmap_<dataset>_row<idx>_decode512.png`
- `keep_frequency_<dataset>_row<idx>_decode512.png`
- `jaccard_lag_summary.png`

## 10. 执行命令草案

正式执行时建议使用后台日志：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
setsid .venv/bin/python evaluation/ATC26_collect_decode_prompt_kvcache_importance_heatmap.py \
  --model /Tan/model/Llama-3.1-8B-Instruct \
  --datasets gov_report,qmsum,multi_news \
  --sample-count-per-dataset 1 \
  --decode-steps 256,512 \
  --min-prompt-tokens 4000 \
  --compression-ratios 0.3,0.5,0.7 \
  --seed 42 \
  --device cuda:0 \
  > evaluation/results/experiments/ATC26_decode_prompt_kvcache_importance_heatmap_longbench/run.log 2>&1 &
```

绘图：

```bash
.venv/bin/python figure/ATC26_plot_decode_prompt_kvcache_importance_heatmap.py
```

## 11. 预期结论形式

如果现象明显，论文/报告中可表述为：

> LongBench 长输出样本中，同一 prompt token 在不同 decode step 的保留状态呈现明显阶段性变化。静态 prefill-only KV 重要性排序无法覆盖完整 decode 过程，因此 decode-aware KV cache management 有必要动态刷新或采用 hot/cold 分层机制。

如果现象不明显，也有价值：

> 部分长输出任务中 prompt token 重要性在 512 step 内高度稳定，说明这类请求更适合一次性 prefill 压缩；动态 decode 管理应优先用于热区漂移明显的请求，而不是所有请求无差别启用。

## 12. 风险与控制

主要风险：

- `output_attentions=True` 可能显著拖慢并增加显存；
- 512 step 可能遇到 EOS 早停；
- token-level top-k 图可能太碎；
- LongBench 某些子任务输出不够长。

控制方式：

- 手写 step decode，每步只保留 prompt attention 的聚合结果；
- 首轮只跑 3 条样本；
- 坚持 token-level score、token-level 展示；如果视觉过碎，只在绘图阶段额外提供 downsampled overview，不改变原始 mask；
- 样本筛选偏向 `gov_report/qmsum/multi_news` 这类长输出任务；
- 保存 raw score，后续阈值变化无需重跑模型。
