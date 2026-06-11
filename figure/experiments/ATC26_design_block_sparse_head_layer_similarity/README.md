# ATC26_design_block_sparse_head_layer_similarity

## 图像说明

本目录保存 ATC26 Design 中块摘要稀疏方法的独立 head/layer 相似度热力图。

## 数据来源

- Source experiment: `ATC26_blockwise_attention_similarity_hotpotqa_3samples`
- Aggregate JSON: `evaluation/results/experiments/ATC26_blockwise_attention_similarity_hotpotqa_3samples/artifacts/ATC26_attention_similarity_aggregate.json`
- 原始模型输出未重跑，本脚本只复用旧相似度数据重新绘图。

## 绘图配置

- Model: `Llama-3.1-8B`
- Compression ratio: `0.5`
- Dataset: LongBench HotpotQA, 3 samples
- Colormap/range: `viridis`, fixed `[0, 1]`
- Export: PDF and PNG

## 输出文件

- `ATC26_design_llama31_8b_instruct__r0p5__kv_head_score_cosine.pdf`
- `ATC26_design_llama31_8b_instruct__r0p5__kv_head_score_cosine.png`
- `ATC26_design_llama31_8b_instruct__r0p5__layer_block_index_jaccard.pdf`
- `ATC26_design_llama31_8b_instruct__r0p5__layer_block_index_jaccard.png`

## 运行脚本

- `figure/ATC26_plot_design_block_sparse_head_layer_similarity.py`

## 指标

- `KV-head score cosine`: 不同 KV head group 的 block score vector 余弦相似度。
- `Layer block-index Jaccard`: 不同 layer 最终 kept block index 集合的 Jaccard 相似度。
