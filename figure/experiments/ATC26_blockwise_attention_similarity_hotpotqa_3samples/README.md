# ATC26_blockwise_attention_similarity_hotpotqa_3samples

## 说明

本目录保存 ATC26 BlockWise attention similarity 实验的热力图。

配套结果目录：

`evaluation/results/experiments/ATC26_blockwise_attention_similarity_hotpotqa_3samples/`

配套说明：

`note/ATC26_attention_similarity_experiment_plan_zh.md`

## 预期图像

- `ATC26_layer_similarity_grid.png`
- `ATC26_head_similarity_grid.png`
- `ATC26_head_score_cosine_grid.png`
- `<model>__r<ratio>__layer_similarity.png`
- `<model>__r<ratio>__head_similarity.png`
- `<model>__r<ratio>__head_score_cosine.png`

## 推荐查看顺序

1. 先看 `ATC26_layer_similarity_grid.png`，确认 layer 间保留 block 是否稳定。
2. 再看 `ATC26_head_similarity_grid.png`，确认 KV head 间 top-k block 选择是否相似。
3. 如果 head Jaccard 较低，再看 `ATC26_head_score_cosine_grid.png`，判断是否只是 top-k 边界导致离散化差异。
