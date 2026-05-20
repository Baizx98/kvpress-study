# SnapKV / ChunkKV 块重要性打分开销实验设计

## 1. 问题定义

本实验要回答的问题是：

> 在不同请求长度下，`SnapKV` / `ChunkKV` 这类现有 KV 重要性打分方法的时间开销，相对真正的 attention kernel 计算开销是否已经足够大，以至于会削弱 KV 压缩方法的系统收益？

这里的 attention kernel 计算只统计已经拿到 Q/K/V 之后的 attention kernel 时间，建议使用 `flash_attn` 作为对照，不把 tokenizer、模型加载、QKV projection、MLP、采样等无关开销混进来。

当前仓库中的关键实现边界：

- `kvpress/presses/snapkv_press.py`：`SnapKVPress.score()` 在没有 `attentions` 时，会用 recent-window query 和全部 key 重新计算一遍 attention weight。
- `kvpress/presses/chunkkv_press.py`：`ChunkKVPress.compress()` 先调用底层 `ScorerPress.score()` 得到 token 级分数，再聚合成 chunk 分数，然后做 top-k chunk 选择和 gather。
- 因此，本地 `chunkkv` 的主要打分成本本质上仍来自 `SnapKVPress.score()`，chunk 聚合只是额外成本，不能把二者混为一谈。

## 2. 核心假设

### Hypothesis A：SnapKV 打分不是“免费”的

`SnapKV` 的打分近似包含：

```text
Q_window [B, Hq, W, D] x K_all [B, Hq, L, D]
softmax over L
mean over W
avg_pool1d over L
head group reduce
```

其中 `W` 是 observe window，默认可取 `64`；`L` 是请求长度。即使理论复杂度是 `O(W * L)`，它通常由若干 PyTorch eager ops 组成，不像 FlashAttention 那样是高度融合 kernel，所以在中短上下文和 decode-like 场景里可能占据很高比例。

### Hypothesis B：ChunkKV 的额外开销应拆开看

本地 `ChunkKVPress` 的总时间可以拆成：

```text
chunkkv_total =
  snapkv_score
  + token_score_to_chunk_score
  + topk_chunks
  + build_token_indices
  + gather_kv
```

如果只报告 `chunkkv_total`，会误以为 ChunkKV 的 chunk 逻辑很贵；但更可能的真实结论是：ChunkKV 贵在继承了 token-level SnapKV 打分，chunk 选择和 gather 是次级开销。

### Hypothesis C：要同时比较 prefill attention 和 decode attention

如果只拿 `L x L` 的 full prefill FlashAttention 做对照，超长上下文下 attention kernel 会因 `O(L^2)` 占优，容易弱化打分开销的论点。

更有说服力的设计是同时报告：

1. `prefill_fa_ms`：`q_len=L, kv_len=L` 的 causal FlashAttention。
2. `decode_fa_ms`：`q_len=1, kv_len=L` 的 single-token FlashAttention。
3. `window_fa_ms`：`q_len=W, kv_len=L` 的 observe-window FlashAttention，可作为“如果打分也能被高度融合，理论上应该接近什么水平”的参考。
4. `snap_score_ms` / `chunkkv_total_ms`：当前实现路径的真实打分与选择时间。

最终重点不是声称“打分永远比 full prefill attention 更贵”，而是证明：

- 在真实 serving 的 decode / refresh 场景，`snap_score_ms / decode_fa_ms` 会非常高；
- 在中等长度 prefill 中，`snap_score_ms / prefill_fa_ms` 也可能不是小数；
- `snap_score_ms / window_fa_ms` 可以暴露当前打分实现没有享受到 FlashAttention 级融合优化。

## 3. 实验一：Synthetic Per-layer Microbenchmark

### 3.1 目的

隔离模型其它模块，只测单层 attention kernel 和单层重要性打分的 GPU 时间。

### 3.2 Tensor 形状

优先使用 Llama-3.1-8B 近似配置：

| 参数 | 建议值 | 说明 |
|---|---:|---|
| batch size | `1` | 先消除 batch padding 干扰 |
| dtype | `bf16` | 与本地长上下文推理设置一致 |
| query heads | `32` | Llama-3.1-8B 典型配置 |
| kv heads | `8` | GQA 配置 |
| head dim | `128` | Llama 系列常见配置 |
| observe window | `32, 64, 128` | 主结果用 `64`，另做敏感性分析 |
| chunk length | `16, 20, 64` | 本地 registry 中 `chunkkv` 默认 `20`，ATC sweep 曾用 `16` |
| compression ratio | `0.5` | 只影响 top-k/gather，不影响 SnapKV 打分主成本 |

请求长度建议：

```text
L = 1024, 2048, 4096, 8192, 16384, 32768
```

如果单卡显存不足，`32768` 可降级为只测 `decode_fa_ms`、`window_fa_ms`、`snap_score_ms`，跳过 `prefill_fa_ms`。

### 3.3 计时对象

每个长度都输出下面这些列：

| 指标 | 内容 |
|---|---|
| `prefill_fa_ms` | `flash_attn_func(q[L], k[L], v[L], causal=True)` |
| `decode_fa_ms` | `flash_attn_func(q[1], k[L], v[L], causal=False)` |
| `window_fa_ms` | `flash_attn_func(q[W], k[L], v[L], causal=False)` |
| `snap_qproj_ms` | 当前 `SnapKVPress.compute_window_attention()` 里从 hidden states 重算 last-window Q 的时间；如果 benchmark 直接传 Q，则该列记为 `0` 或单独实现 |
| `snap_qk_softmax_ms` | recent-window QK、mask、softmax、截掉 observe window 的时间 |
| `snap_reduce_pool_ms` | mean、avg_pool、GQA group reduce、pad recent window 的时间 |
| `snap_score_ms` | 上三项总和 |
| `chunk_score_agg_ms` | token score 聚合到 chunk score 的时间 |
| `chunk_topk_index_ms` | top-k chunk、构造 token indices、排序的时间 |
| `chunk_gather_ms` | gather K/V 的时间 |
| `chunkkv_total_ms` | `snap_score_ms + chunk_*` |

必须额外输出这些比例：

```text
snap_vs_prefill_fa = snap_score_ms / prefill_fa_ms
snap_vs_decode_fa  = snap_score_ms / decode_fa_ms
snap_vs_window_fa  = snap_score_ms / window_fa_ms
chunk_vs_prefill_fa = chunkkv_total_ms / prefill_fa_ms
chunk_vs_decode_fa  = chunkkv_total_ms / decode_fa_ms
chunk_extra_ratio   = (chunkkv_total_ms - snap_score_ms) / chunkkv_total_ms
```

### 3.4 计时方法

使用 CUDA event，不用 Python wall time 直接包 GPU kernel：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
fn()
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

每个配置：

- warmup：`20` 次；
- repeat：`100` 次；
- 报告：`median / p10 / p90`；
- 固定随机种子：`torch.manual_seed(0)`；
- 每轮结束清空大 tensor 引用，并记录 `torch.cuda.max_memory_allocated()`。

## 4. 实验二：真实 `SnapKVPress` / `ChunkKVPress` 路径计时

### 4.1 目的

Synthetic benchmark 说明纯算子比例，但论文/报告里还需要证明这个比例不是伪造出来的。第二个实验直接在当前 `kvpress` 的 press 路径上插计时。

### 4.2 方法

在不改变算法输出的前提下，给以下函数加临时 profiling wrapper：

- `SnapKVPress.score()`
- `SnapKVPress.compute_window_attention()`
- `ChunkKVPress.compress()`
- `ScorerPress.compress()`

每层记录：

```json
{
  "method": "snapkv|chunkkv",
  "layer_idx": 0,
  "seq_len": 8192,
  "window_size": 64,
  "chunk_length": 20,
  "compression_ratio": 0.5,
  "score_ms": 0.0,
  "chunk_agg_ms": 0.0,
  "topk_index_ms": 0.0,
  "gather_ms": 0.0,
  "kept_tokens": 4096,
  "gpu": "...",
  "dtype": "bf16"
}
```

输出建议放到：

```text
evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/raw/
```

然后生成汇总表：

```text
evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/scoring_overhead_summary.csv
```

### 4.3 请求构造

不要先上 LongBench 全量，先构造固定长度 synthetic prompt，避免数据集和 tokenizer 差异干扰：

```text
L = 1024, 2048, 4096, 8192, 16384
```

每个长度构造 `input_ids`，只跑 prefill forward，不生成长 decode：

```text
max_new_tokens = 1
do_sample = false
```

模型优先用：

```text
/Tan/model/Llama-3.1-8B-Instruct
```

如果加载成本太高，可以先用 `/Tan/model/Qwen3-8B` 或本地已有小模型做 smoke，但最终主图应使用一个固定 7B/8B 级模型。

## 5. 实验三：长度敏感性与窗口敏感性

### 5.1 长度敏感性

主图横轴用请求长度 `L`，纵轴用时间和比例：

1. `snap_score_ms`、`chunkkv_total_ms`、`prefill_fa_ms`、`decode_fa_ms` 折线图。
2. `snap_vs_prefill_fa`、`chunk_vs_prefill_fa` 柱状图。
3. `snap_vs_decode_fa`、`chunk_vs_decode_fa` 柱状图，建议用 log-scale。

预期现象：

- `prefill_fa_ms` 随 `L` 更接近二次增长；
- `snap_score_ms` 在固定 `W` 下接近线性增长；
- `decode_fa_ms` 也接近线性增长，但常数明显小于 `snap_score_ms`；
- 因此 `snap_vs_decode_fa` 应该显著大于 `snap_vs_prefill_fa`。

这个结果可以支持一个更准确的系统结论：

> 对 prefill 全量 attention 来说，现有打分方法未必永远主导；但对在线 refresh / decode-like 选择来说，当前 SnapKV/ChunkKV 的打分开销很可能已经超过真正的 attention kernel，成为压缩收益兑现的瓶颈。

### 5.2 Window 敏感性

固定 `L = 8192` 和 `16384`，扫：

```text
W = 16, 32, 64, 128, 256
```

观察：

```text
snap_score_ms ~ O(W)
snap_vs_decode_fa ~ O(W)
```

这个实验可以说明：如果为了精度增大 observe window，系统成本会直接放大；所以“更准的 token-level query-aware 打分”并不是免费增强。

### 5.3 Chunk length 敏感性

固定 `L = 8192/16384`、`W = 64`、`compression_ratio = 0.5`，扫：

```text
chunk_length = 8, 16, 20, 32, 64, 128
```

重点看：

```text
chunk_extra_ratio = (chunkkv_total_ms - snap_score_ms) / chunkkv_total_ms
```

预期：

- chunk 越小，chunk 数越多，top-k/index/gather 开销会上升；
- 但只要 `chunk_extra_ratio` 仍远小于 `snap_score_ms / chunkkv_total_ms`，就能证明 ChunkKV 的主要系统瓶颈不是 chunk 这个思想，而是 token-level scoring 继承自 SnapKV。

## 6. 对照组与公平性

### 6.1 必须有的对照

| 对照 | 作用 |
|---|---|
| `flash_attn` prefill | 代表当前层 full attention kernel 成本 |
| `flash_attn` decode | 代表单步 decode attention kernel 成本 |
| `flash_attn` window | 代表 `W x L` 级别 fused attention 的参考成本 |
| `snapkv` scoring | 当前 token-level query-aware scoring 成本 |
| `chunkkv` scoring+selection | 当前 chunk wrapper 的实际总成本 |
| `blockwise` scoring 可选 | 如果要引出自己的方法，可加入低开销块摘要打分对照 |

### 6.2 不建议混入的开销

主实验不要混入：

- tokenizer；
- dataset loading；
- model loading；
- full generation；
- sampling；
- CPU 日志写入；
- 首次 kernel 编译和 CUDA lazy init；
- 跨进程调度时间。

这些会让结论变得不干净。端到端实验可以另做，但不能替代 microbenchmark。

## 7. 结果呈现方式

建议最终报告至少包含 4 张图：

1. **Absolute time vs request length**：`prefill_fa_ms`、`decode_fa_ms`、`snap_score_ms`、`chunkkv_total_ms`。
2. **Scoring / prefill FlashAttention ratio**：说明 prefill 中打分占比。
3. **Scoring / decode FlashAttention ratio**：说明 decode 或在线 refresh 中打分可能远大于 attention kernel。
4. **ChunkKV breakdown**：`snap_score`、`chunk_agg`、`topk_index`、`gather` stacked bar，说明 ChunkKV 贵在哪里。

表格建议包含：

| L | W | prefill_fa_ms | decode_fa_ms | window_fa_ms | snap_score_ms | chunkkv_total_ms | snap/prefill | snap/decode | chunk extra |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## 8. 最小可验证版本

第一阶段不要直接做完整评测，只需要跑一个脚本：

```text
evaluation/bench_scoring_overhead_snapkv_chunkkv.py
```

最低配置：

```text
L = 2048, 4096, 8192, 16384
W = 64
chunk_length = 20
compression_ratio = 0.5
dtype = bf16
repeat = 50
warmup = 10
```

通过标准：

1. 每个配置都有 `median/p10/p90`；
2. 所有 timing 用 CUDA event；
3. 所有结果保存到 `evaluation/results/experiments/scoring_overhead_snapkv_chunkkv/artifacts/`；
4. 至少能画出 `snap_score_ms / decode_fa_ms` 随长度变化的图；
5. 如果 `snap_score_ms / decode_fa_ms` 长期大于 `1`，就足以支持“当前重要性打分对在线选择非常贵”这个主张；
6. 如果 `snap_score_ms / prefill_fa_ms` 在 `2K-8K` 仍达到明显比例，例如 `>10%-20%`，则可进一步支持“即使在 prefill 中也不是可忽略开销”。

## 9. 结论应该如何表述

推荐结论口径：

> SnapKV / 当前 ChunkKV 的重要性估计依赖 recent-window token-level attention scoring。虽然该打分理论上比 full prefill attention 低一阶，但当前实现路径需要额外的 QK、softmax、pooling、top-k 与 gather，且不能直接复用 FlashAttention 主 kernel 的输出。因此，在中等长度 prefill 和 decode-like refresh 场景中，打分时间相对真正 attention kernel 已经不可忽略，甚至可能超过单步 decode attention kernel。这个结果说明，后续 KV 压缩如果要获得系统收益，需要从 token-level scoring 转向更低开销的 block-native scoring、score reuse 或 lazy refresh。

不建议写成：

> SnapKV / ChunkKV 一定比 FlashAttention 更慢。

这个说法太强，因为 full prefill FlashAttention 的复杂度随 `L^2` 增长，而固定 window 的 scoring 更接近 `W * L`。更严谨的说法应该区分 prefill 和 decode/refresh 场景。

