# ATC26 BlockWisePress Ranked Top-k Index Temporal Similarity 实验重设计

## 1. 问题

上一版实验结果显示 lag=512 时 BlockWisePress 的 block index overlap 仍然较高，这不符合预期。复查采集逻辑后发现两个设计问题：

1. 当前 hook 中只使用 `initial_len` 以内的 KV：
   `keys = keys[:, :, :initial_len]`
   因此 decode 阶段新生成的 KV block 没有参与重要性计算。
2. 当前相似度主要使用 `selected_scored_only`，它排除了 protected sink/recent blocks，并且保存的是排序后的集合，不保留 top-k rank order。

这会让实验更像是在问“固定 prompt prefill blocks 的重要性是否稳定”，而不是问“完整 KV 中每个 decode step 的 top-k block index 是否发生变化”。

## 2. 新假设

在每个 decode step 使用完整 KV 计算 block importance 后：

- 相邻 step 的 top-k ranked indices 仍应高度相似。
- lag 较大时，尤其 lag=256/512，top-k ranked indices 应出现明显变化。
- decode 阶段新增 block 会逐步进入 top-k 或 decode-only high-importance list，从而拉低远距离 step 的 ranked similarity。

如果实验结果符合该假设，就可以更准确地支持一个分层结论：

- 短间隔：index 可复用。
- 长间隔：必须刷新，否则会错过新出现的重要 decode blocks。

## 3. 实验原则

1. 不做实际压缩。
   - 不调用 `compress()`。
   - 不修改 `past_key_values`。
   - Decode 始终基于完整 KV cache 继续。
2. 每个 step 计算重要性分数时使用完整 KV。
   - step `t` 的 KV 长度应为 `context_length + t + 1`。
   - 包含 prefill tokens 和已经生成/teacher-forced decode tokens。
3. 每个 step 记录 ranked top-k indices。
   - 保存按 score 从高到低排列的 block index。
   - 不只保存排序后的集合。
4. 单独记录 decode 阶段产生的 block 的重要性排名。
   - 每个 block 记录 origin：`prefill`、`decode`、`mixed_tail`。
   - 保存 decode-origin blocks 在全局 score ranking 中的位置。
5. 每次计算都强制刷新 block summary / score。
   - 不复用上一 step 的 summary。
   - 保证相似性来自模型注意力/score 分布，而不是缓存的 index。

## 4. 采集对象

每个样本、每层、每个 decode step 保存以下数组：

### 4.1 完整 KV top-k

- `ranked_topk_indices_all`
  - shape: `[layers, steps, topk]`
  - 按 block score 降序排列。
- `ranked_topk_scores_all`
  - shape: `[layers, steps, topk]`
  - 对应 top-k block score。
- `ranked_topk_origins_all`
  - shape: `[layers, steps, topk]`
  - `0=prefill, 1=decode, 2=mixed_tail`。

### 4.2 Decode block 专项记录

- `decode_block_indices_ranked`
  - 每个 step 中所有 decode-origin 或 mixed-tail block，按 score 降序。
- `decode_block_global_ranks`
  - 这些 decode blocks 在完整 block score 排名中的 rank。
- `decode_block_scores`
  - decode blocks 的 score。
- `num_decode_blocks`
  - 当前 step 已产生的 decode/mixed blocks 数。
- `best_decode_global_rank`
  - 最重要 decode block 在全局 ranking 中的最好名次。
- `decode_blocks_in_topk_count`
  - 完整 top-k 中有多少 block 来自 decode 阶段。
- `decode_blocks_in_topk_ratio`
  - `decode_blocks_in_topk_count / topk`。

### 4.3 元数据

- `kv_len`
- `num_blocks`
- `topk`
- `context_length`
- `block_size`
- `step`
- `layer`

## 5. Top-k 设置

为避免“大 top-k 集合天然稳定”掩盖变化，建议同时记录多组 top-k：

- `topk = 16`
- `topk = 32`
- `topk = 64`
- `topk = 128`
- `topk_ratio = 0.5` 对应旧 BlockWisePress compression ratio 的 keep budget，作为和旧实验对齐的设置。

论文主图优先使用 `topk=32` 或 `topk=64`：

- top-k 太小可能噪声大。
- top-k 太大可能因为集合覆盖过宽而显得过稳。
- `topk_ratio=0.5` 放附录或补充，用于解释旧实验为何 lag=512 仍较高。

## 6. Query 模式

保留两种模式，但主结论要分开解释：

1. `single`
   - 当前 decode token 的 hidden state 作为 query。
   - 更严格，更容易反映 step-to-step 排名变化。
2. `window`
   - 最近 `window_query_size=16` 个 decode tokens 聚合为 query。
   - 更接近实际 BlockWisePress 的窗口式重要性估计。

如果目标是证明“固定步长刷新”，主文可用 `window`；如果目标是证明“长间隔确实会变化”，需要同时展示 `single`。

## 7. 相似性指标

### 7.1 集合相似度

- `topk_overlap@K`
  - `|topK_t ∩ topK_t+lag| / K`
- `topk_jaccard@K`
  - `|A ∩ B| / |A ∪ B|`

### 7.2 排名敏感相似度

上一版只看集合，不看排序。新版必须加入排名敏感指标：

- `rank_biased_overlap@K`
  - 更重视前几名的 index 是否一致。
- `weighted_rank_overlap@K`
  - 对 rank 赋权，例如 `1/log2(rank+1)`。
- `common_index_rank_delta`
  - 共同出现的 blocks 的平均 rank 位移。

推荐主文指标：

- `topk_overlap@K`：直观说明 index 复用覆盖率。
- `rank_biased_overlap@K`：说明 top-ranked block 是否真的稳定。
- `decode_in_topk_ratio`：说明新 decode blocks 何时开始影响 top-k。

### 7.3 Decode block 动态

新增以下随 step 的曲线：

- `decode_blocks_in_topk_ratio` vs step。
- `best_decode_global_rank` vs step。
- `new_topk_entry_ratio` vs lag：
  - `1 - overlap@K`，表示 lag 后 top-k 中有多少 index 是新的。
- `decode_new_entry_ratio` vs lag：
  - lag 后新增 top-k entries 中 decode-origin blocks 的比例。

## 8. Lag 与 reuse sweep

保持用户要求的 step 间隔：

- Lag：`1, 2, 4, 8, 16, 32, 64, 128, 256, 512`
- Reuse interval：`2, 4, 8, 16, 32, 64, 128, 256, 512`
- Decode steps：至少 `1024`

说明：

- lag=512 需要至少 1025 个有效 step pair；实际跑 `decode_steps=1024` 时，如果 step 从 0 到 1023，则 lag=512 可形成 512 对。
- 不应再用 `decode_steps=512` 声称覆盖 lag=512。

## 9. Raw artifact 设计

不要只保留 aggregate。每个 job 保存：

- `raw/*.jsonl`
  - 每个 job 一行 summary，包含路径、配置、完整性统计。
- `indices/*.npz`
  - 保存 ranked top-k index、score、origin、decode block rank。
- `aggregate/*.csv`
  - lag/reuse/layer/sample 聚合结果。
- `logs/*.log`
  - watchdog 和 run log。

推荐 `.npz` keys：

- `{mode}_topk{K}_ranked_indices_all`
- `{mode}_topk{K}_ranked_scores_all`
- `{mode}_topk{K}_ranked_origins_all`
- `{mode}_topk{K}_decode_in_topk_count`
- `{mode}_topk{K}_best_decode_global_rank`
- `{mode}_topk{K}_decode_block_indices_ranked`
- `{mode}_topk{K}_decode_block_global_ranks`
- `{mode}_topk{K}_kv_len`
- `{mode}_topk{K}_num_blocks`

## 10. 推荐运行矩阵

第一阶段先验证趋势：

- Context length：8192、16384
- Samples per length：4
- Decode steps：1024
- Top-k：16、32、64、128、ratio=0.5
- Modes：single、window
- Dataset：PG19
- Model：Llama-3.1-8B-Instruct
- GPU：device2 A6000

第二阶段用于论文增强：

- 增加 LongBench/Needle 的 representative prompts。
- 增加上下文长度 32768。
- 对 R=32/128/512 做真实 fixed-refresh quality/latency 验证。

## 11. 预期图

主文推荐 3 张图：

1. `overlap@K vs lag`
   - 展示相邻 step 高，远距离下降。
2. `rank_biased_overlap@K vs lag`
   - 展示排名靠前的 blocks 在长间隔下变化更明显。
3. `decode_in_topk_ratio / best_decode_global_rank vs step`
   - 展示 decode 阶段新增 blocks 逐步进入重要 block 集合。

附录图：

- layer-wise heatmap：`layer x lag` 的 overlap@K。
- 不同 K 的 sensitivity。
- `single` vs `window` 对比。
- `topk_ratio=0.5` 与固定 K 的对比，用于解释旧实验结果。

## 12. 最小实现改动

建议新建脚本而不是覆盖旧脚本：

- `evaluation/ATC26_collect_blockwise_ranked_topk_temporal_similarity.py`
- `evaluation/ATC26_watch_blockwise_ranked_topk_temporal_similarity.sh`
- `figure/ATC26_plot_blockwise_ranked_topk_temporal_similarity.py`

关键代码改动：

1. 删除 `keys[:, :, :initial_len]` 截断，改为使用完整 KV：
   `keys, values = extract_keys_and_values(cache, layer_idx)`
2. 用当前 `kv_len` 计算 block origin：
   - block end `<= context_length`：prefill
   - block start `>= context_length`：decode
   - 否则：mixed_tail
3. 从 `block_scores` 直接取 ranked top-k：
   - 不使用 `selected_scored_only`
   - 不排除 protected blocks
   - 不把 top-k indices 再按 index 排序
4. 保存 top-k rank order 和 score。
5. 计算 similarity 时按每个 K 独立聚合。

## 13. 判断标准

如果新实验符合预期，应该看到：

- lag=1 的 overlap@32 / overlap@64 仍较高。
- lag=512 的 overlap@32 / RBO@32 明显低于 lag=1。
- decode_in_topk_ratio 随 step 增大而上升，或 best_decode_global_rank 随 step 增大逐渐靠前。
- `topk_ratio=0.5` 可能仍然较高，但固定小 K 会显示更明显变化。

如果仍然不符合预期，需要进一步排查：

1. BlockWise score 是否天然偏向 prefill 中的 sink/global blocks。
2. top-k 是否被过大的 K 稀释。
3. PG19 teacher-forced decode 是否导致局部注意力稳定性强于开放式生成。
4. 是否需要按 head / KV-head group 记录，而不是只看 layer-level 聚合。
