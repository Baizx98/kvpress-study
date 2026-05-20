# ATC26 Needle Heatmap 中 BlockWise 效果偏差分析

## 1. 现象

本轮实验使用：

- 模型：`/Tan/model/Llama-3.1-8B-Instruct`
- benchmark：`needle_in_haystack`
- haystack：`alessiodevoto/paul_graham_essays`
- 压缩率：`0.5`
- 方法：`block_wise`、`snapkv`、`chunkkv`
- depth：`0,10,...,100`
- 运行设备：物理 GPU 1，RTX 3090

full 结果中，`block_wise` 在已成功长度上的 exact retrieval accuracy 明显低于 `snapkv` 和 `chunkkv`：

| method | context length | correct / total | accuracy |
|---|---:|---:|---:|
| `block_wise` | 4096 | 7 / 11 | 0.636 |
| `block_wise` | 8192 | 6 / 11 | 0.545 |
| `block_wise` | 16384 | 7 / 11 | 0.636 |
| `snapkv` | 4096 | 11 / 11 | 1.000 |
| `snapkv` | 8192 | 11 / 11 | 1.000 |
| `snapkv` | 16384 | 10 / 11 | 0.909 |
| `snapkv` | 32768 | 11 / 11 | 1.000 |
| `chunkkv` | 4096 | 11 / 11 | 1.000 |
| `chunkkv` | 8192 | 11 / 11 | 1.000 |
| `chunkkv` | 16384 | 10 / 11 | 0.909 |
| `chunkkv` | 32768 | 11 / 11 | 1.000 |

同时，`block_wise` 在 32768 和 65536 上 OOM；`snapkv` 和 `chunkkv` 只在 65536 上 OOM。

## 2. 先排除的因素

### 2.1 不是数据集不匹配

本项目的 `needle_in_haystack` benchmark 明确使用 `alessiodevoto/paul_graham_essays`，字段包括：

- `context`
- `needle`
- `question`
- `answer_prefix`
- `max_new_tokens`

本轮实验使用的就是这个数据源。

### 2.2 不是完全检索不到答案

默认 needle 是：

```text
Remember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.
```

默认 `answer_prefix` 是：

```text
Answer: The best thing to do in San Francisco is
```

很多 `block_wise` 的错误输出不是完全跑偏，而是只输出了：

```text
eat a sandwich and sit in Dolores Park on a sunny day.
```

或：

```text
to eat a sandwich and sit in Dolores Park on a sunny day.
```

这类输出语义上已经找到了核心地点和动作，但 exact 判定要求把 `answer_prefix + predicted_answer` 拼回完整 needle。修正后仍然存在 `block_wise` 明显低于 baseline 的现象，说明不是单纯 scoring bug。

### 2.3 不是 64k 失败导致的唯一问题

OOM 确实造成了长上下文缺失：

- `block_wise`: 32768、65536 OOM
- `snapkv`: 65536 OOM
- `chunkkv`: 65536 OOM

但 `block_wise` 在 4096、8192、16384 这些成功运行的长度上也低于 `snapkv/chunkkv`，所以质量问题不能只归因于 OOM。

## 3. 最可能原因

### 3.1 当前 BlockWise 的块级摘要会稀释短 needle

当前参数：

- `block_size=16`
- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=key_norm`
- `summary_topk_keys=4`
- `mean_key_weight=0.75`

`BlockWisePress` 先把上下文切成 16-token block，然后为每个 block 构造摘要。当前摘要主要由：

- block 内 key 的均值；
- block 内 key-norm top-k token 的均值；
- 二者按 `mean_key_weight=0.75` 融合。

Needle 是 26 tokens，跨越约 2 个 block。它不是一段持续很长的主题，而是一条局部短事实。用 block mean 做主导摘要时，needle token 很容易被同一个 block 内其它背景 token 稀释。

这对 LongBench QA 可能没那么致命，因为答案相关证据往往不是单个极短 passkey；但对 Needle 这种“必须保留一个短事实”的任务非常敏感。

### 3.2 `key_norm` 代表 token 不一定等价于 needle token

当前代表 token 选择：

```text
representative_mode=key_norm
```

这会优先选择 key norm 大的 token，而不是和 question 最相关的 token。Needle 里真正关键的是：

```text
eat a sandwich and sit in Dolores Park on a sunny day
```

这些 token 未必在每一层、每个 head 上都是 key norm 最大的 token。结果是：needle 所在 block 即使没有被整块丢掉，它的 block summary 也可能没有充分代表 needle 中最关键的词。

### 3.3 当前评分只看 tail query window，位置检索容易受问题局部词影响

当前运行参数：

- `query_aware=True`
- `q_window_size=64`
- `query_agg_mode=max`
- `query_topr=16`

`query_aware=True` 会把 question 拼进 context 后再压缩。`BlockWisePress` 用最后 `q_window_size` 个 query token 和各 block summary 打分。

这个机制对一般问答是合理的，但 Needle 的问题很短且模板化：

```text
Based on the content of the book, what is the best thing to do in San Francisco?
```

它可能强烈匹配一些和 San Francisco / best thing / book content 表面相似的 block，而不是精确匹配人工插入的 needle block。`query_agg_mode=max` 又会放大少数 query token 的最大相似度，容易选择“看起来相关”的背景块。

这可以解释为什么 `block_wise` 的错误输出常常仍包含 Dolores Park 或 sandwich 的一部分，但不稳定地缺失完整形式。

### 3.4 50% 压缩下块级 top-k 对 Needle 更激进

`compression_ratio=0.5` 表示删掉约一半 KV。对 token-level 方法来说，某些 needle token 可能仍然被保留；对 block-level 方法来说，一旦 needle 所在 block 的排序稍低，整个 16-token block 都可能被丢弃。

Needle 的关键事实只有约 26 tokens，跨 2 个 block。若两个 block 中任意一个丢失，生成时就容易出现：

- 只记住地点，缺动作；
- 只记住动作，缺尾部；
- 生成 prefix 后半句不完整；
- 被背景文本干扰。

### 3.5 `block_wise` 的 OOM 说明当前实现额外中间张量较重

日志显示 `block_wise` 在 32768 上已经 OOM：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 894.00 MiB.
```

而 `snapkv/chunkkv` 能跑到 32768。说明当前 `block_wise` 在长上下文下的临时张量和 summary scoring 开销更大。这个问题不直接解释 4k/8k 的准确率差，但会限制后续 heatmap 覆盖范围。

## 4. 当前结论

基于本轮数据，`block_wise` 在 Needle 上差，最可能不是因为模型完全不会做检索，而是因为当前 BlockWise 的设计目标更偏向“块级语义保留”，不适合“短局部事实必须完整保留”的 Needle。

具体来说：

1. `mean_plus_norm_topk_mean` 会稀释短 needle。
2. `key_norm` 代表 token 不保证选中 needle 关键 token。
3. `query_agg=max` + tail query window 可能偏向表面相关背景块。
4. block-level 选择的粒度较粗，needle 跨 block 时特别脆弱。
5. 当前实现长上下文显存开销更高，导致 32768+ 无法完整比较。

## 5. 下一步验证计划

### 5.1 先做 Needle 专用消融，不要直接改主方法

建议只在 Needle 上跑小规模消融：

| variant | 改动 | 目的 |
|---|---|---|
| `blockwise_multi_rep` | `summary_mode=multi_rep_max` | 减少 mean summary 对短事实的稀释 |
| `blockwise_tail_query_rep` | `representative_mode=tail_query_relevance` | 让代表 token 更贴近 question |
| `blockwise_top_head` | `head_agg_mode=top_head_only` | 避免 uniform mean 稀释少数检索 head |
| `blockwise_smaller_block` | `block_size=8` | 降低 needle 跨 block 和整块误删风险 |
| `blockwise_lower_ratio` | `compression_ratio=0.3/0.4` | 检查是否只是 50% 过激 |

最小验证网格：

- context length：`4096, 8192, 16384`
- depth：`0,20,50,80,100`
- 方法：上述 variant + 当前 `block_wise`

### 5.2 记录 block selection，而不是只看最终答案

为了确认 needle 是否被保留，下一步应在 `BlockWisePress` 中导出每层 `last_kept_token_indices` 或至少每层 kept block indices，并计算：

```text
needle token kept ratio
needle block kept ratio
```

如果答案错误时 needle block 经常被丢掉，说明是 selection 问题；如果 needle block 被保留但答案仍错，说明是 block 内代表/值压缩或生成阶段问题。

### 5.3 需要补一个 no-compression 上界

本轮按要求没有跑 `no_press`。但要严肃解释 Needle，最好至少跑一次 no-compression 上界，确认：

- Llama-3.1-8B-Instruct 在当前 prompt 和 scoring 下本身能稳定做这个任务；
- 当前 exact 判定不会误伤正常答案。

这不一定放主图，但应作为实验 sanity check。

## 6. 建议

短期不要把当前 `block_wise` 的 Needle 结果作为主方法结论。更合理的写法是：

- 当前 BlockWise 在 LongBench/PG19 等任务上看整体语义和建模质量；
- Needle heatmap 暴露出短事实精确保留不足；
- 下一步需要增加 Needle-aware 的 block selection 诊断和消融。

如果目标是让 BlockWise 在 Needle 上也好看，优先尝试：

1. `summary_mode=multi_rep_max`
2. `representative_mode=tail_query_relevance`
3. `head_agg_mode=top_head_only`
4. `block_size=8`
5. 降低 Needle 场景下压缩率到 `0.3/0.4`

其中最值得先跑的是 `multi_rep_max + tail_query_relevance`，因为它最直接针对“短 needle 被 mean summary 稀释”的问题。
