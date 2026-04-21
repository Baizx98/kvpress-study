# 组会汇报PPT大纲

## 1. 研究背景与本周目标

- 项目背景：
  - 当前研究主题是 `KV Cache` 压缩与 decode 阶段稀疏/驱逐策略
  - 前期已经完成多轮 `blockwise` 消融，但结果显示内部排列组合收益开始收敛
- 当前瓶颈：
  - 需要从“方法细节优化”转向“最终推理框架选择”
  - 核心问题变为：
    - 最终框架里是否还需要 `prefill compression`
    - decode 更适合 `permanent eviction` 还是 `compute-cold`
    - 是否存在能统一二者优势的 `hybrid decode`
- 本周目标：
  - 完成 `DualPhasePerLayerPress` 的专项重构
  - 做 fixed-budget decode 框架实验
  - 做最后一次 `hybrid decode` 验证，并判断是否停止继续扩展 decode 算法树

图片预留位置：
- 本页不放实验图，建议放一张“研究问题示意图”
- 如果要放项目演进图，可后续补一张自绘流程图

---

## 2. 本周完成工作总览

- 工作 1：完成 `PG19` benchmark 适配，并明确早期主验证数据集优先级
- 工作 2：完成 `blockwise stage3` 结果整理，确认继续做 blockwise 细碎消融的价值有限
- 工作 3：重构 `DualPhasePerLayerPress`，只保留当前 decode 长输出实验真正需要的能力
- 工作 4：完成 `decode_long_output_longbench_stage1`
  - 验证 `permanent_fixed_budget` 与 `compute_cold_fixed_budget` 在长输出任务上的基本可行性
- 工作 5：完成 `decode_final_framework_fixed_budget_stage1`
  - 比较 `dense_prefill + decode` 与 `blockwise_prefill + decode`
- 工作 6：完成最后一次 `decode_hybrid_final_stage`
  - 验证 `hybrid decode` 是否值得继续

已确认：
- 本周提交主线集中在 `2026-04-13 ~ 2026-04-19`
- 关键提交包括：
  - `063ac9c refactor: simplify dual-phase decode compression`
  - `11e1be0 feat: add fixed-budget and hybrid decode experiment pipeline`
  - `9b5c276 doc: archive decode framework experiment analyses`

图片预留位置：
- 建议右侧放提交时间线截图或手工整理时间线
- 当前仓库无现成时间线图，建议本页以文字为主

---

## 3. 工作一：从 blockwise 内部消融转向最终推理框架

- 做了什么：
  - 完成 `blockwise stage3` 结果复盘
  - 明确当前不应再继续主攻 `summary / representative / aggregation` 的排列组合
- 目的：
  - 给后续 decode 框架实验收缩方向
  - 判断是否继续在 prefill/blockwise 细节上投入
- 方法/改动点：
  - 基于 `LongBench + needle` 的 stage3 结果做跨任务分析
  - 重点比较：
    - `blockwise_main`
    - `blockwise_multi_rep`
    - `chunkkv`
- 当前效果：
  - 得到明确结论：
    - `blockwise` 内部继续消融收益下降
    - 下一步应转向：
      - 分层 budget
      - decode 永久驱逐
      - decode 计算稀疏

已确认：
- 结论来自 [blockwise_stage3_current_results_analysis_and_next_steps_zh.md](/home10T/bzx/workspace/kvpress-study/note/blockwise_stage3_current_results_analysis_and_next_steps_zh.md)

图片预留位置：
- 当前无专门图像文件
- 建议本页使用 1 个表格总结：
  - `blockwise_main / multi_rep / chunkkv` 的任务最优分布

---

## 4. 工作二：DualPhasePerLayerPress 重构

- 做了什么：
  - 将 `DualPhasePerLayerPress` 重构为当前阶段专用实现
- 目的：
  - 去掉早期探索遗留逻辑，减少实现噪音
  - 让 decode 实验只围绕：
    - `permanent_fixed_budget`
    - `compute_cold_fixed_budget`
- 方法/改动点：
  - 只保留 prefill physical compression + decode fixed-budget 两段式流程
  - 删除 per-layer ratio tables、score reuse、offload/prefetch 模拟等冗余支持
  - 补充针对 `DualPhase` 的专项测试
- 当前效果：
  - `DualPhase` 实现与当前实验目标完全对齐
  - 后续 fixed-budget / hybrid 实验的代码基础清晰稳定

已确认：
- 架构说明见 [dual_phase_per_layer_refactor_architecture_zh.md](/home10T/bzx/workspace/kvpress-study/note/dual_phase_per_layer_refactor_architecture_zh.md)

图片预留位置：
- 本页建议不放实验图
- 建议放一张架构示意图：
  - `prefill -> decode refresh -> permanent/cold/hybrid`
- 当前无现成图，建议答辩前补绘

---

## 5. 实验一：LongBench 长输出 decode stage1 可行性验证

- 实验设置：
  - 数据集：
    - `LongBench / gov_report, qmsum, multi_news`
  - 方法：
    - `prefill_only_no_decode_pruning`
    - `decode_permanent_eviction_fixed_budget`
    - `decode_compute_cold_fixed_active_budget`
  - 特点：
    - 这是 decode-only feasibility test
    - 主要验证质量是否会崩
- 结果：
  - 两条 decode 路线都没有明显质量崩坏
  - 说明：
    - decode 阶段做 fixed-budget 压缩是可行的
- 结论：
  - 这一步证明方向可以继续，但还不能决定最终框架
  - 下一步需要把实验扩展到：
    - 不同 fixed budget
    - decode-only vs prefill+decode 联合框架

已确认：
- 分析文档见 [decode_long_output_longbench_stage1_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/decode_long_output_longbench_stage1_analysis_zh.md)

图片预留位置：
- 当前该实验没有现成图像
- 明确写在 [figure/experiments/decode_long_output_longbench_stage1/README.md](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_long_output_longbench_stage1/README.md) 中：尚未生成图表
- 本页建议直接放结果表格，不预留图片

---

## 6. 实验二：最终推理框架 fixed-budget 对比

- 实验设置：
  - 主验证：
    - `LongBench / gov_report, qmsum, multi_news`
  - 补充验证：
    - `RULER / 4096 / niah_single_3, niah_multikey_2, niah_multikey_3, qa_2`
  - 比较路线：
    - `dense_prefill + permanent_decode`
    - `dense_prefill + compute_cold_decode`
    - `blockwise_prefill + permanent_decode`
    - `blockwise_prefill + compute_cold_decode`
  - fixed budget：
    - `96 / 128 / 160 blocks`
- 结果：
  - LongBench 最优宏平均：
    - `dense_prefill + permanent_decode @ 160 = 27.16`
  - 分任务最优：
    - `gov_report`: `dense + permanent @ 160`
    - `qmsum`: `dense + permanent @ 160`
    - `multi_news`: `dense + compute-cold @ 160`
  - `blockwise_prefill + decode` 未超过 `dense_prefill + decode`
- 结论：
  - 当前最终框架不应默认带 prefill compression
  - 主要候选收缩为：
    - `dense + permanent`
    - `dense + compute-cold`

已确认：
- 分析文档见 [decode_final_framework_fixed_budget_stage1_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/decode_final_framework_fixed_budget_stage1_analysis_zh.md)

图片预留位置：
- 左侧主图：
  - [longbench_fixed_budget_lines.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_final_framework_fixed_budget_stage1/longbench_fixed_budget_lines.png)
- 右侧辅助图：
  - [longbench_fixed_budget_macro.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_final_framework_fixed_budget_stage1/longbench_fixed_budget_macro.png)
- 页脚补充图：
  - [ruler_fixed_budget_grouped.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_final_framework_fixed_budget_stage1/ruler_fixed_budget_grouped.png)

---

## 7. 实验三：最后一次 Hybrid Decode 验证

- 实验设置：
  - 只保留 `dense_prefill` 主线
  - 比较：
    - `Permanent 128 / 160`
    - `Compute-Cold 128 / 160`
    - `Hybrid 128/96`
    - `Hybrid 160/128`
  - `Hybrid` 定义：
    - `permanent core + cold fringe`
- 结果：
  - LongBench 宏平均：
    - `Permanent 160 = 27.16`
    - `Compute-Cold 160 = 26.80`
    - `Hybrid 160/128 = 26.09`
    - `Hybrid 128/96 = 25.20`
  - RULER 宏平均：
    - `Permanent 160 = 87.5`
    - `Compute-Cold 160 = 87.5`
    - `Hybrid 160/128 = 80.0`
    - `Hybrid 128/96 = 75.0`
- 结论：
  - `hybrid` 没有成为更优统一方案
  - 最稳主线仍然是：
    - `dense_prefill + permanent_decode @ 160`

已确认：
- 分析文档见 [decode_hybrid_final_stage_analysis_zh.md](/home10T/bzx/workspace/kvpress-study/note/decode_hybrid_final_stage_analysis_zh.md)

图片预留位置：
- 左侧主图：
  - [longbench_hybrid_budget_lines.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/longbench_hybrid_budget_lines.png)
- 右侧主图：
  - [longbench_hybrid_macro.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/longbench_hybrid_macro.png)
- 下方补充图：
  - [ruler_hybrid_grouped.png](/home10T/bzx/workspace/kvpress-study/figure/experiments/decode_hybrid_final_stage/ruler_hybrid_grouped.png)

---

## 8. 本周结论收敛

- 已确认 1：
  - `blockwise` 内部继续做细碎消融的边际收益已经很低
- 已确认 2：
  - `DualPhasePerLayerPress` 已重构为面向 decode 专项实验的稳定实现
- 已确认 3：
  - fixed-budget 结果表明最终框架更偏向 `dense_prefill + decode`
- 已确认 4：
  - 最后一次 `hybrid` 验证失败，说明继续扩 decode 算法树的必要性已经很低
- 当前最稳默认框架：
  - `dense_prefill + permanent_decode @ 160 blocks`
- 可保留的任务特化说明：
  - `multi_news` 上 `compute-cold 160` 更优

图片预留位置：
- 建议这一页放 1 张总结表，而不是再放新图
- 表格内容建议：
  - “方法是否继续推进 / 结论 / 证据页码”

---

## 9. 当前问题与风险

- 问题 1：
  - `RULER` 控制器仍有假失败记录
  - 但不影响本周主要结论，因为 `metrics.json` 已完整落盘
- 问题 2：
  - 当前结论主要基于质量，不包含时间/显存等系统指标
- 问题 3：
  - `multi_news` 与 `gov_report/qmsum` 对 decode 机制偏好不一致
  - 后面需要考虑：
    - 是否做任务感知分流
    - 或者接受一个“最稳默认配置 + 个别任务例外”的叙事
- 风险判断：
  - 如果继续扩算法树，时间会花在局部启发式上，论文主线会变散
  - 当前更应该进入框架定型和主实验阶段

图片预留位置：
- 本页不放图片
- 建议用“风险-影响-应对”三列表格

---

## 10. 下周计划

- 下一步 1：
  - 固化最终默认框架：
    - `dense_prefill + permanent_decode @ 160`
- 下一步 2：
  - 以该框架为主线，规划更大规模主实验
- 下一步 3：
  - 如果论文需要，可把 `multi_news` 上 `compute-cold 160` 作为任务特化补充分析
- 预期收益：
  - 停止算法发散，开始收敛故事线
  - 让论文主线从“方法搜索”转为“最终推理框架设计”
- 验证方式：
  - 扩大正式 benchmark 覆盖
  - 做最终表格与主要图
  - 开始组织论文中的方法图、系统图和主结果表

图片预留位置：
- 本页建议不放实验图
- 可放一张“后续研究路线图”示意图

---

## 附录页建议（可选）

### A. 本周关键提交时间线

- `2026-04-13`
  - `PG19` benchmark 支持与数据集优先级澄清
- `2026-04-14`
  - `blockwise stage3` 控制器、结果与分析归档
- `2026-04-16`
  - `DualPhasePerLayerPress` 重构
  - `decode_long_output_longbench_stage1` 结果归档
- `2026-04-19`
  - fixed-budget + hybrid decode 实验流水线
  - 最终 decode 框架分析归档

### B. 讲述主线建议

- 第一段：
  - 为什么不再继续做 blockwise 内部细碎消融
- 第二段：
  - 为什么要把问题重写成“最终推理框架选择”
- 第三段：
  - fixed-budget 证明什么
- 第四段：
  - hybrid 为什么最终没成立
- 第五段：
  - 为什么现在该收敛到主框架并准备更大规模实验
