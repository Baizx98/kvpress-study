---
name: experiment-results-hygiene
description: Project-scoped skill for kvpress-study experiment output organization, regrouping figure/results artifacts by experiment, writing per-experiment README summaries, and standardizing how future evaluation runs archive scripts, logs, figures, and raw metrics.
---

# Experiment Results Hygiene

用于规范 `kvpress-study` 仓库实验结果的归档、整理和说明文档。

仅在 `/home10T/bzx/workspace/kvpress-study` 内使用；离开该仓库后不应触发此技能。

## 何时使用

在以下场景触发：

- 新实验跑完后，需要整理 `figure/` 和 `evaluation/results/`
- 旧实验结果零散，需要重新归并
- 需要给每组实验补 `README.md`
- 需要建立或维护实验结果总索引

## 目标结构

### 图像结果

`figure/`

- 绘图脚本保留在根目录
- 所有实验图放在 `figure/experiments/<experiment_name>/`
- 每个实验目录必须包含一个 `README.md`
- 根目录维护 `EXPERIMENT_INDEX.md`

### 评测结果

`evaluation/results/`

- 正式实验统一放在 `evaluation/results/experiments/<experiment_name>/`
- 每个实验目录必须包含：
  - `artifacts/`
  - `README.md`
- 历史零散结果统一放在 `evaluation/results/ad_hoc_baselines/`
- 根目录维护 `EXPERIMENT_INDEX.md`

## 每组实验的最小 README 内容

每个实验目录下的 `README.md` 至少写清楚：

- 实验目的
- 运行脚本
- 数据集
- 方法
- 压缩率或其它关键 sweep 维度
- 采样比例
- 产物位置
- 推荐优先查看的图或解读文档

## 整理步骤

1. 识别一组实验对应的：
   - 运行脚本
   - 原始结果目录
   - 图像文件
   - 配套说明文档
2. 为该实验创建：
   - `figure/experiments/<experiment_name>/`
   - `evaluation/results/experiments/<experiment_name>/`
3. 将结果目录整体移动到：
   - `evaluation/results/experiments/<experiment_name>/artifacts/`
4. 将图像移动到：
   - `figure/experiments/<experiment_name>/`
5. 分别在图像目录和结果目录写 `README.md`
6. 更新两个总索引：
   - `figure/EXPERIMENT_INDEX.md`
   - `evaluation/results/EXPERIMENT_INDEX.md`

## 命名规范

`<stage>_<compare_or_ablation>_<fraction_or_scale>_<methods>`

例如：

- `prefill_compare_50pct_blockwise_chunkkv`
- `ruler_ablation_10pct`

要求：

- 小写
- 下划线分词
- 名称中显式包含最关键实验条件

## 注意事项

- 不要把绘图脚本混进某个实验子目录
- `artifacts/` 中保留原始 `run.log`、`config.yaml`、`predictions.csv`、`metrics.json`
- 若有重复重跑目录如 `/1`、`/2`、`/3`，保留原始产物，但在分析脚本中做去重
- 若结果目录路径改变，后续图脚本和分析脚本要改成从新路径读取

## 推荐做法

- 先整理目录，再做结果分析
- 新实验一结束就归档，不要堆积
- 分析文档优先放 `note/`，在对应实验 `README.md` 中引用
