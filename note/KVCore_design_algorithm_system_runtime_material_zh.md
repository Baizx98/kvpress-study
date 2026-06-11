# KVCore Design 写作材料：算法、系统与运行流程

本文档面向后续论文 Design 部分写作。目标不是直接生成最终英文论文段落，而是把网页端 ChatGPT 无法读取的本地论文初稿、实验 note、`kvpress-study` 原型代码，以及 `/home10T/bzx/workspace/KVCore` 框架设计汇总成一份足够完整的中文背景材料。

建议后续写作时重点覆盖三个 Design 小节：

1. `Overview`
2. `Block-level Sparse Scoring`
3. `Sparsity-guided KV Lifecycle Management`

## 0. 材料边界

### 0.1 主要来源

当前仓库 `kvpress-study`：

- 论文初稿：`note/ATC26.tex`
- BlockWisePress 主算法：`kvpress/presses/block_wise_press.py`
- BlockWise 打分组件：`kvpress/presses/blockwise_components.py`
- 论文算法伪代码草稿：`note/paper_blockwise_press_algorithm_latex_zh.md`
- 稀疏索引开销实验：`note/sparse_index_overhead_first_results_zh.md`
- 跨 decode step 稀疏索引相似性实验：`note/ATC26_blockwise_ranked_topk_temporal_similarity_results_zh.md`
- decode 策略探索：`note/decode_long_output_longbench_stage1_analysis_zh.md`、`note/decode_hybrid_final_stage_analysis_zh.md`、`note/permanent_eviction_vs_compute_cold_blocks_zh.md`

实际 KVCore 框架仓库：

- 用户给出的路径是 `/home10T/bzx/workspace/kvcore`，但当前环境真实路径是 `/home10T/bzx/workspace/KVCore`
- 项目 README：`/home10T/bzx/workspace/KVCore/README.md`
- 当前架构流程：`/home10T/bzx/workspace/KVCore/notes/current-architecture-flow.md`
- KVManager 边界设计：`/home10T/bzx/workspace/KVCore/notes/kv-manager-interface-design.md`
- Sparse KV 当前实现：`/home10T/bzx/workspace/KVCore/notes/current-sparse-kv-implementation-2026-05-08.md`
- 关键代码：`kvcore/config.py`、`kvcore/kv/kv_manager.py`、`kvcore/kv/sparse.py`、`kvcore/model/block_summary.py`、`kvcore/model/block_score.py`、`kvcore/model/blockwise_scoring.py`、`kvcore/model/model_runner.py`、`kvcore/sched/scheduler.py`、`kvcore/engine/engine_core.py`

### 0.2 当前代码快照与投稿目标版本的区别

已经在 `kvpress-study` 中较完整实现和验证的是：

- 块级摘要打分算法，即 BlockWisePress / KVCore scoring primitive。
- prefill 阶段块级永久压缩。
- decode 阶段 fixed-budget 策略原型，包括 permanent eviction 和 compute-cold 两类思路。
- 稀疏 index 跨 step 稳定性实验。
- 稀疏 index 构造开销 benchmark。

截至当前代码快照，`/home10T/bzx/workspace/KVCore` 中已经实现的 runtime 能力是：

- vLLM 风格但更精简的 `EngineCore -> Scheduler -> ModelRunner -> Attention backend` 执行骨架。
- block-based KVManager、全局 BlockPool、per-layer logical block table。
- paged KV tensor、block table、slot mapping、paged attention metadata。
- sparse KV 的 `disabled` / `dynamic` / `permanent` 三种模式。
- dynamic 模式：只改变当前 forward 的 read block table，不释放物理 KV。
- permanent 模式：prefill 完成后把未选择 logical block 替换为 null block，并释放对应 physical block。
- block summary manager、query collector、block score update、EMA score 写回。

截至当前代码快照，仍处在实现中或尚未完整接入端到端评测的能力是：

- CPU offload / GPU prefetch 的真实数据搬运。
- GPU/CPU 双层 KV block state 的完整状态机。
- preemption-aware 或 pressure-aware 的最终 serving policy。
- 高性能 offload/prefetch overlap、transport abstraction、bandwidth-aware scheduling。
- 完整质量/性能/系统评测闭环。

但论文投稿时的描述对象不是这个中间代码快照，而应是投稿前将完成的 KVCore 完整系统。因此后续写 Design 正文时，可以把 CPU offload、GPU prefetch、pressure-aware lifecycle policy 作为 KVCore 的完整系统能力来描述；只是在本材料中保留当前快照状态，方便区分“已有代码证据”和“投稿目标版本需要补齐的实现”。

## 1. 论文当前叙事主线

当前 `note/ATC26.tex` 已经形成的核心故事是：

1. 现代 LLM serving runtime 已经用 block granularity 管理 KV cache，例如 paged attention / block pool / continuous batching。
2. 这个 block abstraction 解决了连续内存分配和碎片问题，但没有解决 decode 阶段 KV cache 持续增长的问题。
3. 调度器为了吞吐通常维持较高 KV block utilization，但请求输出长度未知。活跃请求继续生成时会持续消耗剩余 block，最终可能触发 preemption、swap 或 recomputation。
4. Attention sparsity 提供了一个更细粒度的视角：每个 decode step 实际只需要历史 KV 的一个重要子集。
5. 如果 runtime 事先知道未来每个 step 的最小 KV 子集 `S_i^*`，就可以从 oracle sparse-access trace 中推导理想生命周期决策：
   - 永远不会再被访问的 block 可以 permanent eviction。
   - 暂时不用但以后会再用的 block 可以 offload 到 CPU。
   - 很快会再用的 block 应保留在 GPU 或提前 prefetch。
6. 真实 runtime 不知道未来 query，也不知道未来 oracle set。
7. 关键 observation 是 practical sparse scoring 可以近似未来 sparse-access trace，因为相邻 decode step 的重要 KV set 具有短期稳定性，长 horizon 上又会逐渐漂移。
8. 因此 KVCore 把 sparse score 从“当前 attention 的 pruning signal”提升为“runtime lifecycle hint”。

这个故事和传统 KV compression 的区别是：

- 传统方法问的是：当前应该保留哪些 token/chunk/block 来减少 attention 计算或 KV footprint。
- KVCore 问的是：一个 KV block 在未来一段 decode horizon 中的生命周期价值是什么，它应该继续 resident、暂时 offload、被 prefetch，还是 permanent eviction。

## 2. Design Overview 建议内容

### 2.1 KVCore 的系统定位

KVCore 是一个面向 block-based LLM serving runtime 的稀疏引导 KV 生命周期管理层。它不替代已有 scheduler、block allocator 或 attention backend，而是在它们之间增加一个 sparse-aware control plane。

可以这样描述：

- `Scheduler` 仍负责请求队列、continuous batching、chunked prefill、decode token budget，以及 logical KV block 分配。
- `KVManager` 仍负责 request/layer 的 logical block table、physical block allocation、prefix cache、free、permanent eviction 等 tensor-free 生命周期操作。
- `ModelRunner` 仍负责 runner-side 输入 batch、KV cache tensor、block table、slot mapping、paged attention metadata，以及模型 forward。
- `Attention backend` 仍负责把 K/V 写入 paged KV tensor，并基于 block table 读取历史 KV 做 causal attention。
- KVCore 新增的是：
  - block summary maintenance；
  - query-aware block scoring；
  - sparse plan generation；
  - lifecycle state update；
  - 后续 CPU offload / GPU prefetch / eviction policy。

因此 KVCore 的设计边界是：它提供 runtime-level block value signal 和 lifecycle transition，不要求模型层或 attention kernel 暴露 full attention score matrix。

### 2.2 两条路径：data path 与 control path

KVCore 中同一份 block score 同时服务两条路径。

第一条是 attention data path：

- 根据当前 sparse plan，只让 selected KV blocks 进入本次 attention read block table。
- 在 `KVCore` 当前实现中，这对应 `dynamic` mode。
- dynamic mode 不修改 KVManager 的 logical block table，不释放 physical block，只影响当前 forward 的 visible blocks。

第二条是 runtime control path：

- 根据 score history 和 runtime pressure 更新每个 block 的 lifecycle state。
- 当前已经实现的 control action 是 permanent eviction：把未选择 logical block 替换为 null block，并释放物理 block。
- 投稿目标版本中，这条路径还包括 CPU offload 和 GPU prefetch：低价值但可恢复的 block 被迁移到 CPU，预测即将变热的 block 被异步预取回 GPU。

这两条路径的区别很重要：

- compute sparsity 是 per-forward metadata choice。
- permanent eviction / offload / prefetch 是 KV lifecycle operation，会影响后续 block table、memory residency 或数据搬运。

### 2.3 当前 KVCore 运行骨架

`/home10T/bzx/workspace/KVCore` 当前执行流程可以概括为：

```text
LLMEngine / AsyncLLMEngine
  -> EngineCore.step()
    -> Scheduler.schedule()
       - 从 waiting/running 队列选请求
       - 分配新 KV slots
       - 根据已有 block score 生成 SparseKVPlan
    -> ModelRunner.execute_model()
       - 更新 InputBatch
       - 构造 read block table 和 slot mapping
       - 注入 PagedAttentionMetadata 到 forward context
       - 模型 forward，attention backend 写/读 paged KV cache
       - 采样 logits
       - 刷新完整 block 的 summaries
       - 收集 query-window block scores
    -> Scheduler.update_from_outputs()
       - 写回 block score / EMA
       - 推进 request computed tokens
       - cache full blocks
       - permanent mode 下在 prefill 完成点执行 eviction
       - 更新输出 token / finish 状态
```

这套骨架的核心价值是把 KV cache 当作 first-class runtime object，而不是 attention 的副产物。

## 3. Block-level Sparse Scoring

### 3.1 设计动机

KVCore 需要一个 block-level sparse scoring method，原因有三点。

第一，现代 serving runtime 的管理单位是 KV block，而不是 token。调度、allocation、reclamation、offload、prefetch 都天然围绕 block table 和 physical block pool 发生。如果 scoring 输出是 token-level selection，runtime 还需要额外聚合到 block-level，可能造成碎片和搬运不对齐。

第二，直接 materialize full token-level attention score 代价高，也和 attention kernel 强绑定。KVCore 需要在 attention 前后都能使用的轻量 signal，而不是只能在 attention kernel 内部看到的 score。

第三，实验观察显示 sparse eviction 具有空间连续性，邻近 token 的重要性高度相关，因此用 block summary 近似 token-level sparse evidence 是合理的。

### 3.2 块划分

对每一层的历史 KV cache：

```text
K, V: [num_kv_heads, seq_len, head_dim]
block_size = B
num_blocks = ceil(seq_len / B)
block_i = tokens [i * B, min((i + 1) * B, seq_len))
```

`kvpress-study` 和 `KVCore` 默认 block size 都围绕 `16` 展开。`compression_ratio = rho` 表示丢弃比例，实际 keep budget 是：

```text
keep_budget = ceil(num_valid_blocks * (1 - rho))
```

例如 `rho = 0.7` 表示保留约 30% blocks。

### 3.3 块摘要

当前主算法应优先写低成本版本，而不是把所有探索过的 summary variant 放进主文。

`kvpress-study` 当前 BlockWisePress 主线使用：

- `mean_keys`：每个 block 内 key 的均值，表示稳定的语义中心。
- `topk_key_means`：按 key norm 选择少量 high-norm keys，再求均值，表示 block 内少数高响应 anchor。
- `token_counts`：处理尾部 partial block。

推荐主文默认配置：

| 配置项 | 默认值 |
|---|---:|
| block size | 16 |
| query window | 64 或 16，按实验设置区分 |
| summary mode | `mean_plus_norm_topk_mean` |
| representative mode | `key_norm` |
| summary top-k keys | 4 |
| mean-key weight | 0.75 |
| query aggregation | `max` 或 `mean`，论文主算法建议和当前默认保持一致 |
| head aggregation | `uniform_mean` |
| protected prefix blocks | 1 |
| protected recent blocks | 2 |

注意：`KVCore` 框架代码中还保留了 `multi_rep_keys`、`multi_rep_max`、`adaptive_fusion_v1` 等探索性路径；但 `kvpress-study` 当前开销实验已经将主线 BlockWisePress 收敛到低成本 key summary。论文主算法建议写 `mean + top-k mean`，把 multi-rep/adaptive 作为 ablation 或 future variant，不要让主设计显得过复杂。

### 3.4 代表 key 选择

主线代表 key 选择是 `key_norm`：

```text
for each block b and KV head h:
  score token k by ||K[h, k]||_2
  choose top m tokens
  topk_key_mean[b, h] = mean(selected keys)
```

直觉是 high-norm keys 往往是 block 内更可能产生显著 dot-product response 的 anchor。它比 query-dependent representative 更便宜，也更容易缓存和复用。

`KVCore` 代码中还支持：

- `key_norm_diverse`：在 high-norm 基础上增加位置多样性。
- `tail_query_relevance`：用 tail query 选择 representative。
- `random_topk`：用于对照或接口验证。

主文不建议把这些都写进算法主体。

### 3.5 Query-aware scoring

KVCore 不试图重建整个 prefix 的信息，而是估计哪些 block 对当前及近未来 decode query 有用。因此 scoring 使用最近 query window：

```text
Q_window = last w query states
```

在 GQA/MQA 模型中，query heads 多于 KV heads。`KVCore` 的 collector 会按 KV head group 对 query heads 做平均，使 query 与 KV heads 对齐：

```text
num_query_heads = num_kv_heads * groups
Q_kv_head[h] = mean(Q_query_heads[h * groups : (h + 1) * groups])
```

然后计算 query 与 block summary anchor 的 dot product：

```text
score_mean[h, q, b] = <Q[h, q], mean_key[h, b]> / sqrt(head_dim)
score_topk[h, q, b] = <Q[h, q], topk_key_mean[h, b]> / sqrt(head_dim)
```

对 query window 聚合：

- `mean`：平均所有 query 的响应，稳定但可能平滑掉尖峰。
- `max`：只看 window 内最大响应，更适合捕捉“某个 query 强烈需要该 block”的情况。
- `topr_mean`：取 top-r query response 平均。
- `adaptive_mean_max_v1`：在 mean/max 之间自适应融合。

主线建议用 `max` 或和实验表一致的默认配置，强调 query-aware 和 spike-sensitive。

融合 mean anchor 和 top-k anchor：

```text
s_head[b] = lambda * agg_query(score_mean[:, b])
          + (1 - lambda) * agg_query(score_topk[:, b])
```

默认 `lambda = 0.75`，也就是更偏稳定 mean anchor，但保留 high-norm anchor 对局部关键信息的纠偏。

最后对 KV heads 聚合：

```text
s_block[b] = mean_h s_head[h, b]
```

`KVCore` 还支持 `strength_weighted` 和 `top_head_only`，但主文可先写 uniform mean，突出低成本和稳定性。

### 3.6 Block selection

给定每层 block scores 后，KVCore 不是简单全局 top-k，而是先保护结构性关键 block：

- prefix sink blocks：最前面的少量 block，作为全局 anchor。
- recent blocks：最近生成或最近上下文 block，保存局部连续性。
- current/tail partial block：当前正在写入或未满的尾块。

选择逻辑：

```text
valid_blocks = all non-null logical blocks
keep_budget = ceil(len(valid_blocks) * (1 - compression_ratio))
protected = prefix_sink ∪ recent ∪ current/tail

if |protected| <= keep_budget:
  keep = protected ∪ top_score_blocks(valid_blocks - protected, keep_budget - |protected|)
else:
  keep = top_score_blocks(valid_blocks, keep_budget)
```

这个 fallback 很重要：在极端压缩率下，如果 protected blocks 已经超过预算，系统不能让 cache size 失控，因此回退到 score-based global selection。

### 3.7 Score history 与 refresh

KVCore 不是每次都只看当前 instant score。当前框架已有 EMA：

```text
ema_score = score if previous is None
          else alpha * previous + (1 - alpha) * score
```

默认 `score_ema_alpha = 0.8`。EMA 的意义是：

- 平滑 query 抖动。
- 避免某一步偶然低分立即触发危险动作。
- 给 lifecycle manager 一个更稳定的 near-future value estimate。

Temporal similarity 实验支持 sparse index 可以固定间隔刷新：

- 相邻 decode step 的 top-k block index overlap 很高。
- 随 lag 增大，overlap/Jaccard 会下降。
- 高压缩率下下降更明显。
- R=32 是较保守的默认候选，R=128 可作为中等压缩率候选，R=512 不适合高压缩率默认。

因此论文可以把 refresh interval 作为系统参数：

```text
refresh_interval R
  - every R decode steps recompute sparse scores / selected set
  - intermediate steps reuse previous sparse plan or blended score
```

注意：当前 `KVCore` 框架支持 `selection_interval = step / block / n_tokens`，但最终论文可以抽象成 periodic refresh，并解释它由 temporal stability 支撑。

## 4. Sparse Scoring Runtime Pipeline

### 4.1 Summary refresh

在 `/home10T/bzx/workspace/KVCore` 中，summary 不是每次从全量 KV 重算，而是在 forward 后对刚完整写完的 block 生成或刷新 summary。

流程：

1. `ModelRunner` 在 sparse enabled 时创建 `KVBlockSummaryManager`。
2. 每次 forward 后调用 `_refresh_block_summaries()`。
3. `KVBlockSummaryManager.refresh_from_scheduler_output()` 遍历本 step touched blocks。
4. 只总结已经完整写完的 block：
   - 未完整结束的 partial block 跳过。
   - block id 为 0 的 null block 跳过。
   - 已经 valid 且 layer 匹配的 summary 跳过。
5. 从 paged KV tensor 取 key cache：

```text
block_keys = kv_cache_tensor[0, block_id]
```

其中 `kv_cache_tensor` 的 layout 是：

```text
[2, num_blocks, block_size, num_kv_heads, head_dim]
```

第一维 `0` 是 key，`1` 是 value。

当前 summary 主要基于 key cache，不使用 value cache。这应在论文限制或实现细节中说明。

### 4.2 Query collection

每个 attention layer 在 forward 时会把当前 query 交给 `BlockScoreCollector.record_query()`。这发生在通用 Attention wrapper 中，因此模型层不需要显式传 sparse metadata。

forward 完成后，collector 根据 attention metadata 中的 `token_request_indices` 找到每个 request 在本 step 的 query tokens，并更新 `(request_id, layer_idx)` 的 rolling query window。

这个设计的好处是：

- scoring 与模型结构低耦合；
- 支持 continuous batching，因为同一个 flattened token batch 中不同 token 可以映射回不同 request；
- 支持 GQA/MQA，因为 collector 内部把 query heads 转成 KV-head-aligned query。

### 4.3 Score update 写回

`BlockScoreCollector.collect()` 对每个 request/layer：

1. 从 scheduler output 得到该 request/layer 的 full block ids。
2. 从 `KVBlockSummaryManager.gather(block_ids)` 拿到 valid block summaries。
3. 调用 `compute_blockwise_scores()` 得到 block scores。
4. 生成 `BlockScoreUpdate`：

```text
request_id
layer_idx
logical_block_indices
scores
score_kind
step_id
```

然后：

```text
ModelRunnerOutput.block_score_updates
  -> EngineCore.step()
  -> Scheduler.update_from_outputs()
  -> KVManager.update_block_scores()
  -> SingleTypeKVManager.update_block_scores()
```

最终每个 logical block 的 `BlockSparseState` 更新：

- `score`
- `ema_score`
- `last_scored_step`

如果 block 已经 permanently evicted，score update 会跳过。

## 5. Sparsity-guided KV Lifecycle Management

### 5.1 设计目标

KVCore 的 lifecycle manager 要解决的问题不是单纯“丢哪些 block”，而是为每个 block 选择合适的生命周期状态：

```text
Hot / GPU resident / active:
  当前或近未来高价值，参与 attention。

GPU resident but compute-cold:
  仍在 GPU 上，但当前 forward 不参与 attention，可在未来重新激活。

CPU offloaded:
  当前不用，但未来可能会用，转移到 CPU 保留恢复能力。

Prefetching:
  根据 score trend 或即将调度的 request，把 CPU block 提前搬回 GPU。

Permanently evicted:
  长期低价值，错误风险较低，释放 GPU/CPU 状态，不再恢复。
```

当前 `KVCore` 已实现其中两类简化动作：

- dynamic sparse compute：相当于 compute-cold，只是不改变物理 residency。
- permanent eviction：释放 physical block，用 null block 表示 logical hole。

投稿目标版本的 KVCore 应把 offload/prefetch 也纳入同一套 lifecycle state machine。也就是说，论文 Design 正文可以直接描述完整的 GPU/CPU residency transition；本材料中单独说明当前代码快照尚未接入真实搬运，只是为了给实现排期和代码证据留边界。

### 5.2 为什么需要多状态，而不是二元 keep/drop

practical sparse score 不是 oracle。一个低分 block 有几种可能：

- 它真的长期无用，适合 permanent eviction。
- 它只是当前 query 不需要，未来 query 会重新需要，适合 offload 或 compute-cold。
- 它分数低但不确定性高，应该暂时保守 resident。

不同动作的代价和风险不同：

- Keep resident：质量风险最低，但占 GPU block。
- Compute-cold：不释放 GPU memory，但减少当前 attention 计算，保留恢复机会。
- CPU offload：释放 GPU memory，但需要 PCIe/NVLink 传输，预测错会造成 stall。
- Prefetch：可隐藏未来 transfer latency，但错误 prefetch 浪费带宽和 GPU block。
- Permanent eviction：释放最彻底，但误删不可恢复。

因此 lifecycle manager 应该结合：

- 当前 score；
- EMA score；
- score 连续低分时长；
- 最近是否被 selected；
- 当前 GPU block pressure；
- transfer bandwidth pressure；
- request priority / remaining budget；
- protected block rules。

### 5.3 当前 KVCore 的 dynamic mode

`dynamic` mode 的语义：

- 每个调度 step，`Scheduler.schedule()` 调用 `KVManager.build_sparse_plan()`。
- `build_sparse_plan()` 对每层选择 selected logical block indices。
- 这些 selection 被封装进 `SparseKVPlan`。
- `ModelRunner._materialize_read_block_table()` 根据 sparse plan 构造本次 forward 的 read block table。
- attention backend 只读取 selected physical block ids。

它不做的事情：

- 不修改 `KVManager` 的 logical block table。
- 不释放 physical block。
- 不影响 prefix cache ref count。
- 不改变下一次 forward 默认 dense 可见性，除非下一次又生成 sparse plan。

因此 dynamic mode 是安全的、可逆的、适合作为 compute sparsity 或 compute-cold 的原型。

### 5.4 当前 KVCore 的 permanent mode

`permanent` mode 当前触发很窄：只在 request 刚完成 prefill 时自动驱逐。

触发顺序：

1. forward 完成。
2. block score updates 先写回 KVManager。
3. request `num_computed_tokens` 推进。
4. `cache_blocks()` 把 full blocks 放入 prefix cache。
5. 如果上一步还在 prompt 内，而本步后已经完成 prefill，则调用 `evict_unselected_sparse_blocks()`。

这个顺序避免了“block 还没写完就被驱逐”的错误。

驱逐过程：

```text
selected = _select_selected_block_indices(...)
evict_indices = valid_indices - selected
evict_request_blocks(LayerBlockSelection(...))
```

在每层 `SingleTypeKVManager.evict_blocks()` 中：

- 将对应 logical block position 替换成 global null block。
- 记录 `permanently_evicted_blocks`。
- 更新 `BlockSparseState.is_permanently_evicted = True`。
- 将原 physical block 放回 `BlockPool.free_blocks()`。

驱逐后：

- logical block table 保留空洞，block id 变成 0。
- read block table 构造时跳过 block id 0。
- freed physical block 可被其他 request 复用。
- prefix cache 不会把这些 evicted holes 重新 cache。

论文可以将这个机制描述为 permanent lifecycle transition 的 prototype。

### 5.5 论文中的完整 lifecycle state machine 建议

可以把最终 KVCore 设计写成一个更完整的状态机：

```text
Allocated / Filling
  -> block 尚未完整写入，不参与 summary，也不能 eviction/offload。

GPU-Resident Hot
  -> 高 score 或 protected，参与 sparse attention。

GPU-Resident Cold
  -> 低 score，但仍在 GPU，可被 dynamic plan 暂时跳过。

CPU-Offloaded
  -> 长期低分但仍有未来复用可能，GPU block 被释放，CPU 保存 K/V 内容。

Prefetching
  -> 根据 score 上升、近期 selected 需求、scheduler 预测，把 CPU block 异步搬回 GPU。

GPU-Resident Warm
  -> prefetch 完成但未必马上参与 attention，等待下一次 sparse plan。

Permanently Evicted
  -> 连续低分、低风险、或 memory pressure 极高时不可恢复删除。
```

核心 transition 可写为：

- `Filling -> GPU-Resident Hot`：block 完整写入后生成 summary，默认保护最近 block。
- `Hot -> Cold`：score/EMA 降低，且不在 protected set。
- `Cold -> Hot`：score 回升或被 sparse plan selected。
- `Cold -> CPU-Offloaded`：GPU pressure 高，且该 block 低分但不满足 permanent eviction 置信度。
- `CPU-Offloaded -> Prefetching`：预测近未来需要，比如 score trend 上升、相邻 step reuse、request 即将调度。
- `Prefetching -> Warm/Hot`：DMA 完成并分配 GPU physical block。
- `Cold/CPU-Offloaded -> Permanently Evicted`：长期低分、长期未被 selected，或者策略判断未来访问概率足够低。

### 5.6 Pressure-aware policy 建议

论文 Design 可以把 policy 写成分层决策：

第一层，保守保护：

- prefix sink blocks；
- recent/current blocks；
- 未完整写入 block；
- 用户设定不可驱逐 block；
- 可能还有 prefix-shared blocks 或高优先级 request blocks。

第二层，score-based active set：

- 每个 refresh interval 根据 block score/EMA 选择 active blocks。
- active blocks 进入 attention read block table。

第三层，residency/offload：

- 如果 GPU block pool pressure 低，低分 block 可留在 GPU resident cold，避免搬运。
- 如果 pressure 中等，低分但可能未来复用的 block offload 到 CPU。
- 如果 pressure 高，连续低分 block permanent eviction。

第四层，prefetch：

- 在下一次调度前，根据上一轮 score、reuse interval、request position、CPU-resident block 的 predicted value，发起 prefetch。
- prefetch 应尽量和 decode compute overlap。

这种 policy 可以表达 KVCore 的核心理念：sparse score 不是直接等于 eviction decision，而是 lifecycle hint，最终动作由风险和系统压力共同决定。

## 6. 与现有实验的对应关系

### 6.1 稀疏索引时间稳定性

`ATC26_blockwise_ranked_topk_temporal_similarity_results_zh.md` 的关键结果：

- 模型：Llama-3.1-8B-Instruct。
- 数据集：PG19 test。
- context length：8192、16384。
- decode steps：1024。
- block size：16。
- compression ratio：0.7、0.5、0.3。
- lag sweep：1 到 512。
- reuse interval：2 到 512。

论文可用事实：

- `window` 模式下 lag=1 overlap 很高：
  - 8192 context 下，compression 0.3/0.5/0.7 的 lag=1 overlap 分别约 0.9895 / 0.9849 / 0.9804。
  - 16384 context 下分别约 0.9905 / 0.9858 / 0.9809。
- lag=512 时高压缩率差异明显：
  - 8192, compression 0.7：overlap 约 0.6827，Jaccard 约 0.5526。
  - 16384, compression 0.7：overlap 约 0.7286，Jaccard 约 0.6015。
- decode-new blocks 会逐渐进入 top-k，特别是 8192 context 和高压缩率下更明显。
- fixed refresh 可行，但 interval 要随压缩率收紧：
  - compression 0.7 下 R=32 recall 约 0.888 到 0.890；
  - R=512 recall 降到约 0.735 到 0.763，不适合作为默认。

结论写法：

> These results support periodic sparse-index refresh: nearby decoding steps reuse similar important block sets, so recomputing sparse indices every step is redundant. However, the degradation at larger lags shows that the sparse approximation must be refreshed online rather than treated as a static prefill decision.

### 6.2 稀疏索引构造开销

`sparse_index_overhead_first_results_zh.md` 的关键结果：

- GPU：NVIDIA L40S。
- 模型：Llama-3.1-8B-Instruct。
- 真实 32 层 Q/K projection 权重。
- timed region 包括 importance score、top-k/block selection/index construction。
- 不包含 K/V gather，不包含真实 transformer forward，不包含 q_proj/k_proj。
- BlockWise summary 只构造 `mean_keys`、`topk_key_means`、`token_counts`。

长度 sweep 中，`B=1, ratio=0.5, reuse_steps=64`：

- L=2048：BlockWise amortized 约 0.524 ms，ChunkKV 约 1.356 ms。
- L=8192：BlockWise amortized 约 0.761 ms，SnapKV 约 0.695 ms，ChunkKV 约 4.144 ms。
- L=16384：BlockWise amortized 约 1.131 ms，SnapKV 约 2.101 ms，ChunkKV 约 8.673 ms。
- L=32768：BlockWise amortized 约 2.108 ms，SnapKV 约 4.596 ms，ChunkKV 约 19.243 ms。

batch sweep 中，`L=8192, ratio=0.5, reuse_steps=64`：

- B=8：BlockWise amortized 约 3.235 ms，SnapKV 约 8.953 ms，ChunkKV 约 36.648 ms。

summary amortization：

- summary build 一次性成本约 0.37 到 0.40 ms。
- reuse=1 时 BlockWise amortized 约 1.173 ms。
- reuse>=4 后约 0.90 ms。
- reuse=64 时约 0.841 ms。

结论写法：

> Block-level scoring converts expensive token-level score construction into compact query-summary matching. Its one-time summary construction is sub-millisecond and can be amortized across refresh intervals. The current benchmark supports the claim about sparse-index construction overhead, not end-to-end serving speed by itself.

### 6.3 Decode lifecycle 策略探索

`decode_long_output_longbench_stage1_analysis_zh.md` 验证了两类 decode fixed-budget 策略：

- `decode_permanent_eviction_fixed_budget`：decode 阶段物理删除未保留块。
- `decode_compute_cold_fixed_active_budget`：保留全部 KV，但当前只让 active blocks 参与计算。

LongBench 长输出任务：

- `gov_report`
- `qmsum`
- `multi_news`

结果相对 prefill-only baseline：

- permanent fixed budget 宏平均下降约 0.42。
- compute-cold fixed budget 宏平均下降约 0.41。
- 两者质量损失都不大。
- `multi_news` 上 compute-cold 明显优于 permanent。
- 当前实现 runtime 开销明显，主要来自 decode 重评分、物理 gather/写回等 correctness prototype 路径。

`decode_hybrid_final_stage_analysis_zh.md` 后续比较：

- `permanent_decode`
- `compute_cold_decode`
- `hybrid_decode = permanent core + cold fringe`

当前结论：

- hybrid 没有成为更优统一方案。
- 在这轮结果中，`dense_prefill + permanent_decode @ 160 blocks` 是最稳路线。
- `compute_cold` 仍可作为 `multi_news` 特化分支。

论文使用建议：

- 可以用这些结果说明多状态 lifecycle 的必要性和策略空间。
- 不建议把 hybrid 写成 KVCore 默认方案。
- 不建议把当前 `kvpress-study` Python correctness prototype 的 runtime 时间作为最终系统性能结论；最终论文性能应以后续完整 KVCore 系统实现和端到端评测为准。

## 7. 推荐 Design 组织方式

### 7.1 Overview 小节可以写这些点

1. KVCore 的输入是 block-based runtime 中已有的 logical block table、physical KV tensor、current query states 和 runtime pressure。
2. KVCore 输出两类结果：
   - sparse attention plan：本次 forward 读哪些 blocks。
   - lifecycle actions：哪些 blocks 保持 resident、offload、prefetch、permanent eviction。
3. KVCore 由三个组件组成：
   - Block Summary Manager：维护 GPU-resident block summaries。
   - Sparse Scorer：用 query window 与 summaries 计算 block value。
   - Lifecycle Manager：根据 score history 和 pressure 更新 block state。
4. 设计不依赖 full attention score materialization，不绑定具体 attention kernel。
5. 设计和 paged KV runtime 对齐，管理单位是 block。

### 7.2 Block-level Sparse Scoring 小节可以写这些点

1. Block partition and summary construction。
2. Query window collection and KV-head alignment。
3. Summary-query scoring formula。
4. Query aggregation and head aggregation。
5. Protected block mechanism。
6. Budget-based top-k selection。
7. Score EMA and periodic refresh。
8. Complexity and overhead intuition。

可用伪代码高层版本：

```text
Algorithm: Block-level Sparse Scoring
Input: per-layer K/V blocks, current query window, compression ratio, protected block config
Output: block scores and selected block set

1. For each completed KV block:
     build or reuse mean key and top-k key mean summaries.
2. Convert query heads to KV-head groups if using GQA/MQA.
3. Compute query-summary dot products.
4. Aggregate over query window.
5. Fuse mean-anchor and top-k-anchor scores.
6. Aggregate over KV heads to obtain one score per block.
7. Smooth scores with EMA.
8. Protect sink/recent/current blocks.
9. Select top scored blocks under keep budget.
10. Return selected logical block indices and scores.
```

### 7.3 Sparsity-guided KV Lifecycle Management 小节可以写这些点

1. Sparse score approximates near-future sparse-access value, not just current-step attention utility。
2. Lifecycle states:
   - hot GPU resident；
   - compute-cold GPU resident；
   - CPU offloaded；
   - prefetching；
   - permanently evicted。
3. Different actions have different risk/cost：
   - permanent eviction 最大化回收但风险最高；
   - offload 可恢复但有搬运开销；
   - prefetch 可隐藏延迟但可能浪费带宽；
   - resident 最安全但占 GPU memory。
4. Policy uses score level, score history, protected rules, and memory pressure。
5. Runtime integration:
   - Scheduler builds sparse/lifecycle plan before forward。
   - ModelRunner materializes read block table。
   - Attention backend reads only selected/resident blocks。
   - After forward, summaries and scores are refreshed。
   - KVManager applies safe lifecycle transitions。
6. Safety rules：
   - incomplete current block cannot be evicted/offloaded；
   - protected recent/sink blocks are retained unless extreme fallback；
   - permanent eviction requires high confidence or high pressure；
   - offloaded blocks remain recoverable；
   - null block keeps logical positions stable。

## 8. 写作边界

- 在最终论文正文中，可以按“投稿前完成的 KVCore 完整系统”描述 CPU offload、GPU prefetch 和 pressure-aware lifecycle manager。
- 在内部材料或实现进度说明中，应区分当前代码快照与投稿目标版本。
- 不要用当前 `kvpress-study` prototype 的 runtime 时间直接证明最终 end-to-end serving throughput。
- 不要写 sparse score 等价于 oracle minimal KV set。
- 不要写 permanent eviction 一定优于 compute-cold。
- 不要写 block-level summary 永远不损失 token-level 信息。

可以写：

- KVCore uses sparse scores as an online approximation of future sparse access。
- Current block-level scoring is low overhead and aligns with block-based KV runtimes。
- Temporal stability supports periodic refresh instead of every-step index recomputation。
- Dynamic and permanent modes validate two ends of the lifecycle design space。
- CPU offload and prefetch are natural lifecycle actions enabled by the same block-level score signal。

## 9. 可以直接给网页端 ChatGPT 的简短任务提示

如果要让网页端继续写论文，可以把本文档连同下面提示一起给它：

```text
请基于这份本地代码与实验整理材料，继续撰写 KVCore 论文 Design 部分，包含 Overview、Block-level Sparse Scoring、Sparsity-guided KV Lifecycle Management 三个小节。请使用系统论文风格，强调 KVCore 如何把 block-level sparse score 从 attention pruning signal 提升为 runtime lifecycle hint。论文正文应描述投稿前完成的 KVCore 完整系统版本，其中 CPU offload、GPU prefetch、permanent eviction 和 dynamic sparse compute 都属于统一 lifecycle manager 的动作；同时不要把 sparse score 写成 oracle，也不要把当前 prototype 的 runtime 时间当作最终端到端系统性能。输出英文 LaTeX 正文草稿。
```

## 10. 关键术语映射

| 中文理解 | 英文建议 |
|---|---|
| 块级稀疏打分 | block-level sparse scoring |
| 块摘要 | block summary |
| 查询感知 | query-aware |
| 稀疏访问轨迹 | sparse-access trace |
| 未来理想稀疏集合 | future ideal sparse set |
| 生命周期管理 | lifecycle management |
| 常驻 GPU | GPU-resident |
| 计算冷块 | compute-cold block |
| 暂时卸载 | temporary CPU offload |
| 预取 | prefetch |
| 永久驱逐 | permanent eviction |
| 读块表 | read block table |
| 逻辑块 | logical block |
| 物理块 | physical block |
| 空块 | null block |
| 稀疏计划 | sparse plan |
| 刷新间隔 | refresh interval |
