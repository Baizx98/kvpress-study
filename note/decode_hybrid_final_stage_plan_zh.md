# Final Hybrid Decode 实验方案

## 1. 目标

这是最后一次算法探索，只回答一个问题：

- `dense_prefill + hybrid_decode` 能否同时吸收 `permanent` 与 `compute-cold` 的优点，并在长输出任务上形成更稳的质量表现？

如果答案是否定的，我们就停止 decode 算法继续扩展，转入最终框架定型。

---

## 2. 设计原则

- `prefill` 阶段不做压缩：
  - `compression_ratio = 0.0`
- 只测试 `decode` 阶段策略
- `block_size = 16`
- `refresh_interval = 16`
- `query aggregation = q-max`
- `summary = mean_plus_norm_topk_mean`
- `representative = key_norm`
- `head aggregation = uniform_mean`
- `protected_recent_blocks = 2`

---

## 3. 方法矩阵

只保留 3 条路线：

1. `dense_prefill + permanent_decode`
2. `dense_prefill + compute_cold_decode`
3. `dense_prefill + hybrid_decode`

其中 `hybrid_decode` 的定义是：

- 先按 `total_budget` 物理保留一部分块
- 再在保留块内部按 `active_budget` 做 cold masking
- 形式上对应：
  - `permanent core + cold fringe`

---

## 4. Budget 设计

### 4.1 对照路线

- `permanent`:
  - `128`
  - `160`
- `compute-cold`:
  - `128`
  - `160`

### 4.2 Hybrid 路线

- `total=128, active=96`
- `total=160, active=128`

设计动机：

- `permanent` 负责稳定保住核心块
- `compute-cold` 负责给边缘块保留再激活机会
- `32` 个 block 的 fringe 是当前最小、也最容易解释的混合预算

---

## 5. 数据集

### 5.1 主验证

只使用 LongBench 长输出任务：

- `gov_report`
- `qmsum`
- `multi_news`

筛选：

- `min_answer_tokens = 64`
- `min_context_tokens = 4000`
- `max_filtered_samples = 20`

### 5.2 补充验证

RULER 只做补充确认，不做主搜索：

- `niah_single_3`
- `niah_multikey_2`
- `niah_multikey_3`
- `qa_2`

设置：

- `data_dir = 4096`
- `samples_per_task = 20`
- `max_new_tokens = 128`

---

## 6. 总运行规模

### 6.1 LongBench

- `3 tasks x 6 configs = 18 runs`

### 6.2 RULER

- `6 configs = 6 runs`

### 6.3 总计

- `24 runs`

---

## 7. 成功判据

### 7.1 如果 hybrid 明显更好

说明：

- 最终框架应收敛为 `dense_prefill + hybrid_decode`

### 7.2 如果 hybrid 没有明显更好

说明：

- 没必要继续扩 decode 算法树
- 最终框架直接在：
  - `dense_prefill + permanent_decode`
  - `dense_prefill + compute_cold_decode`
 之间做最终选择

---

## 8. 执行要求

- 在 A6000 上执行
- 自动记录失败任务
- 自动跳过已完成任务
- 对 `OOM / network / killed` 做自动重试
- 运行过程中持续检查日志
- 若早期出现实现错误，立即停止、修复、续跑
