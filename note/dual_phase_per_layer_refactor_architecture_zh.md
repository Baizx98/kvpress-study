# DualPhasePerLayerPress 重构架构说明

## 1. 重构目标

本次重构的目标是让 `DualPhasePerLayerPress` 只服务当前阶段的 decode 长输出实验，而不是继续承载早期探索阶段的多种兼容逻辑。

当前实验问题是：

- prefill 阶段固定使用 `blockwise_main`
- prefill 压缩率固定为 `0.3`
- decode 阶段每 `block_size` 个 step 刷新一次
- decode 打分使用固定窗口内的 `q-max`
- 比较两种 decode 策略：
  - 永久驱逐
  - 计算冷块

因此，新版实现只保留与这两个策略直接相关的能力。

---

## 2. 主要文件

- `kvpress/presses/dual_phase_per_layer_press.py`
  - 新版 dual-phase press 主实现
- `kvpress/presses/block_wise_press.py`
  - 增加 `keep_budget` 参数，使 block selection 支持 fixed block budget
- `kvpress/attention_patch.py`
  - 对 `masked_key_indices` 做边界过滤，避免 stale mask 在 decode 中触发 CUDA 越界
- `evaluation/evaluate.py`
  - 增加 long-output 样本筛选参数
  - 增加 `dual_phase_mode` 与 fixed-budget decode 参数接线
- `evaluation/run_decode_long_output_longbench_stage1.py`
  - 当前 long-output decode stage1 控制脚本
- `tests/test_dual_phase_per_layer_press.py`
  - 新版 dual-phase 专项测试

---

## 3. 新版 DualPhasePerLayerPress 的核心接口

新版 `DualPhasePerLayerPress` 只保留两个 decode mode：

- `permanent_fixed_budget`
- `compute_cold_fixed_budget`

关键参数：

- `prefill_press`
  - prefill 阶段使用的 `BlockWisePress`
- `decode_press`
  - decode 阶段用于块摘要和打分的 `BlockWisePress`
- `decode_mode`
  - 选择永久驱逐或计算冷块
- `block_size`
  - 块大小，也是本轮实验的 decode refresh interval
- `compression_interval`
  - decode 每隔多少 token 刷新一次，当前实验设为 `block_size`
- `decode_hidden_states_buffer_size`
  - 最近 query window 大小，当前实验也设为 `block_size`
- `decode_block_budget`
  - 永久驱逐显式 block budget，可选
- `decode_cold_block_budget`
  - 计算冷块显式 active block budget，可选
- `decode_budget_scale`
  - 从 prefill budget 派生永久驱逐 budget 时的缩放
- `decode_cold_budget_scale`
  - 从 prefill budget 派生 active budget 时的缩放

默认情况下，decode budget 从 prefill 后保留块数派生：

- `B_prefill_keep = ceil(prefill_compressed_seq_len / block_size)`
- `B_decode = B_prefill_keep * scale + protected_recent_blocks`

这样可以保持 decode 阶段是 fixed-budget 设定，而不是随着输出长度增长的 ratio budget。

---

## 4. 执行流程

## 4.1 Prefill 阶段

prefill 阶段调用：

- `prefill_press.compress(...)`

并记录：

- 每层 prefill 后保留的 block 数
- `layer_prefill_kept_blocks[layer_idx]`

这一步是后续 fixed decode budget 的锚点。

## 4.2 Decode 阶段公共逻辑

decode 阶段每次 forward hook 会：

1. 保存最近 decode hidden states
2. 累计当前层 decode step 数
3. 若还没到 `compression_interval`，复用当前 decode state
4. 若到达刷新点，使用最近 query window 重新对所有块摘要打分

当前实验中：

- `compression_interval = block_size`
- `decode_hidden_states_buffer_size = block_size`

这意味着每形成一个 decode block，就刷新一次块状态。

## 4.3 永久驱逐模式

`permanent_fixed_budget` 的行为是：

1. 根据 prefill 保留块数得到固定历史 budget
2. 用 `decode_press.build_block_plan(..., keep_budget=...)` 选择保留块
3. 通过 `gather_by_token_indices` 物理删除未保留 KV
4. 写回 cache

特点：

- 真正减少 live KV
- 决策不可逆
- 适合检验固定显存预算下的永久回收能力

## 4.4 计算冷块模式

`compute_cold_fixed_budget` 的行为是：

1. 根据 prefill 保留块数得到固定 active budget
2. 用 `decode_press.build_block_plan(..., keep_budget=...)` 选择 active blocks
3. 不物理删除 KV
4. 通过 `module.masked_key_indices` 让 cold blocks 不参与 attention 计算

特点：

- 物理 KV 全保留
- 当前步只计算 active blocks
- 冷块后续可以重新变热
- 适合验证可逆计算稀疏是否比永久驱逐更稳

---

## 5. 已删除或不再支持的旧功能

新版实现刻意删除了以下旧探索功能：

- `layer_phase_ratios`
- `layer_phase_cold_ratios`
- `default_phase_ratios`
- `default_phase_cold_ratios`
- `score_refresh_interval`
- `score_reuse_mode`
- `score_reuse_weight`
- `history_momentum`
- `resident_gpu_ratio`
- `prefetch_ratio`
- offload / prefetch 状态模拟
- 同时混合永久驱逐和 cold ratio 的旧路径

删除原因：

- 当前实验不验证这些变量
- 它们会增加解释复杂度
- 容易让 long-output decode 结果混入历史兼容逻辑

保留的兼容点只有：

- `init_class_vars(...)`

它仍然存在，是为了不破坏 `evaluate_registry.py` 的注册方式，但传入的旧 ratio 字段不会再代表完整旧语义。

---

## 6. 与评测入口的关系

`evaluation/evaluate.py` 现在支持：

- `dual_phase_mode=permanent_fixed_budget`
- `dual_phase_mode=compute_cold_fixed_budget`
- `decode_block_budget`
- `decode_cold_block_budget`
- `decode_budget_scale`
- `decode_cold_budget_scale`
- `min_answer_tokens`
- `min_context_tokens`
- `max_filtered_samples`

当前 stage1 long-output decode 实验使用：

- `compression_ratio=0.3`
- `block_size=16`
- `q_window_size=16`
- `compression_interval=16`
- `min_answer_tokens=64`
- `min_context_tokens=4000`
- `max_filtered_samples=20`

---

## 7. 验证情况

代码级验证：

- `python -m py_compile`
  - `kvpress/attention_patch.py`
  - `kvpress/presses/block_wise_press.py`
  - `kvpress/presses/dual_phase_per_layer_press.py`
  - `evaluation/evaluate.py`
  - `evaluation/run_decode_long_output_longbench_stage1.py`
- `pytest tests/test_dual_phase_per_layer_press.py -q`
  - `7 passed`

端到端 smoke test：

- `LongBench / qmsum`
- `max_filtered_samples=1`
- `max_new_tokens=32`
- `dual_phase_mode=permanent_fixed_budget`
- `dual_phase_mode=compute_cold_fixed_budget`

两种模式都成功完成：

- 模型加载
- 数据筛选
- 推理
- `predictions.csv` 写出
- `metrics.json` 写出

---

## 8. 当前限制

当前版本是为 decode long-output stage1 实验定制的最小实现，暂不支持：

- per-layer 不同 decode budget
- ratio-based decode budget
- offload / prefetch 模拟
- dense oracle regret 统计
- 详细 TPOT / active-block trace 导出

这些功能不是不能做，而是建议等第一轮 long-output 结果出来后，再按论文叙事需要逐步加回。

当前最重要的是先回答：

> 在长输出任务上，fixed-budget 永久驱逐和 fixed-budget 计算冷块，哪一个更稳、更值得继续推进？
