# ATC26 LongBench Prefill 驱逐连续性观察实验结果

## 1. 实验目的

本实验用于论文 observation 小节：验证长请求在 prefill 阶段按 token-level attention importance 做 KV 压缩时，被驱逐的 token 是否呈现成片连续分布。

重要限制：这里不是直接画 `BlockWisePress` 的 block-level eviction 结果，而是先用 SnapKV-style token-level attention score 生成 kept/evicted mask，再离线分析这些 token-level 决策是否能被 block 粒度近似。因此该实验用于支撑“block 粒度管理 KV 是可行的系统接口”，避免把 block 连续性直接写进方法再拿来证明方法。

## 2. 配置

- 模型：`/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：`Xnhyacinth/LongBench`
- 子任务：`hotpotqa`, `multifieldqa_en`, `qasper`, `gov_report`
- 抽样：每个子任务 2 条，`seed=42`
- 压缩率：`0.3`, `0.5`, `0.7`
- 最大上下文：`max_context_length=16384`
- token-level scorer：`SnapKVPress` 的 recent-window attention score
  - `q_window_size=64`
  - `kernel_size=5`
- protected region：
  - sink: 前 64 tokens
  - recent: 后 64 tokens
  - run length 和 block projection 统计排除 protected region
- block projection：`block_size=16,32,64`

## 3. 产物

- 原始结果目录：`evaluation/results/experiments/ATC26_longbench_prefill_eviction_contiguity/`
- summary CSV：`evaluation/results/experiments/ATC26_longbench_prefill_eviction_contiguity/artifacts/ATC26_eviction_contiguity_summary.csv`
- summary JSON：`evaluation/results/experiments/ATC26_longbench_prefill_eviction_contiguity/artifacts/ATC26_eviction_contiguity_summary.json`
- raw JSONL：`evaluation/results/experiments/ATC26_longbench_prefill_eviction_contiguity/artifacts/raw/ATC26_eviction_contiguity_raw.jsonl`
- score arrays：`evaluation/results/experiments/ATC26_longbench_prefill_eviction_contiguity/artifacts/scores/`
- 图目录：`figure/experiments/ATC26_longbench_prefill_eviction_contiguity/`

图文件：

- `ATC26_eviction_mask_heatmap_main.png`
- `ATC26_evicted_run_length_vs_random.png`
- `ATC26_block_projection_mismatch.png`

## 4. 样本

| dataset | row index | effective tokens at ratio=0.5 |
|---|---:|---:|
| hotpotqa | 66 | 13090 |
| hotpotqa | 187 | 7626 |
| multifieldqa_en | 81 | 7011 |
| multifieldqa_en | 4 | 7576 |
| qasper | 15 | 16384 |
| qasper | 190 | 7964 |
| gov_report | 178 | 6249 |
| gov_report | 71 | 7147 |

其中 `qasper row=15` 原始长度超过 16k，被截断到 `16384` tokens。

## 5. 主要结果

### 5.1 Evicted run 明显长于随机 baseline

在同样压缩率、同样 protected region、同样 evicted token 数量下，attention-based mask 的连续驱逐段明显长于随机 mask。

| compression ratio | runs | mean evicted run | random mean evicted run | gain |
|---:|---:|---:|---:|---:|
| 0.3 | 8 | 7.99 | 1.44 | 5.55x |
| 0.5 | 8 | 13.04 | 2.03 | 6.42x |
| 0.7 | 8 | 25.50 | 3.46 | 7.37x |

结论：token-level attention score 产生的 cold KV 不是独立随机散点，而是沿序列维度形成明显连续区间。压缩率越高，连续驱逐段越长。

### 5.2 Block projection 的误差在 block_size=16 时最低

将 token-level mask 投影到 block 粒度时，`block_size=16` 的平均 mismatch 最低。随着 block size 变大，mismatch 上升，但趋势仍然平滑。

| compression ratio | block size | mismatch | false eviction | false keep | pure block | majority-pure block |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 16 | 0.144 | 0.085 | 0.279 | 0.465 | 0.670 |
| 0.3 | 32 | 0.173 | 0.110 | 0.318 | 0.338 | 0.590 |
| 0.3 | 64 | 0.197 | 0.108 | 0.399 | 0.246 | 0.539 |
| 0.5 | 16 | 0.154 | 0.161 | 0.147 | 0.431 | 0.638 |
| 0.5 | 32 | 0.185 | 0.209 | 0.162 | 0.271 | 0.562 |
| 0.5 | 64 | 0.213 | 0.243 | 0.185 | 0.165 | 0.482 |
| 0.7 | 16 | 0.117 | 0.252 | 0.063 | 0.566 | 0.721 |
| 0.7 | 32 | 0.137 | 0.325 | 0.061 | 0.400 | 0.709 |
| 0.7 | 64 | 0.153 | 0.382 | 0.060 | 0.244 | 0.679 |

解释：

- `block_size=16` 与当前 BlockWisePress 默认 block 粒度一致，mismatch 大约在 `11.7% ~ 15.4%`。
- `ratio=0.7` 时 false eviction 较高，但 false keep 很低。这说明高压缩下 block projection 更倾向于整块驱逐，会牺牲一部分 token-level kept token。
- `majority-pure block` 在 `block_size=16` 下为 `0.64 ~ 0.72`，说明多数 block 内部决策具有明显同质性。

## 6. 论文可用结论

可以写成：

> We compute token-level KV importance using prefill attention scores and evict the lowest-scored tokens under a fixed budget. The resulting eviction masks are highly clustered along the sequence dimension: across LongBench requests, the average evicted-run length is 5.55x to 7.37x longer than a random mask with the same eviction budget. Moreover, projecting token-level decisions to 16-token blocks introduces only about 11.7% to 15.4% token-decision mismatch, suggesting that block/page-level KV eviction is a practical approximation to fine-grained attention-based eviction while matching the memory-management granularity of existing serving systems.

## 7. 边界与注意事项

1. 当前只使用 `Llama-3.1-8B-Instruct`。如果论文审稿风险较高，可以补 Mistral/Qwen 作为 appendix 统计。
2. scorer 是 SnapKV-style recent-window attention score，不是 BlockWise scorer。这个选择有利于 observation 的独立性，但论文 caption 里必须明确。
3. 该实验只证明压缩决策的空间连续性，不直接证明 block eviction 的质量无损；质量结论仍应引用 LongBench/Needle/PG19 的正式压缩评测。
4. protected sink/recent 区域从统计中排除，避免连续性主要来自强制保留策略。

