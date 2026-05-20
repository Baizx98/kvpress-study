# 稀疏索引开销正文描述草稿

## 时间开销

如图 X 所示，我们首先比较不同方法构建稀疏索引的时间开销。该开销只包含重要性分数计算和 top-k/index 构建，不包含后续 K/V gather。结果表明，SnapKV 和 ChunkKV 都需要在 token 粒度上计算重要性分数，因此其开销会随着请求长度和 batch size 增长而明显上升；其中 ChunkKV 还需要在 token score 之上进一步构建 chunk 级索引，因此整体开销最高。相比之下，KVCore 将选择对象从 token 转换为 block，并复用预先构建的 block summary，因此在线阶段只需要在更少的 block 候选上完成打分和选择。随着请求长度或 batch size 增大，KVCore 的在线索引构建开销增长更慢；在考虑 summary 复用后，其摊销开销也保持在较低水平。

## 空间开销

KVCore 的额外空间主要来自可复用的 block summary 和每次压缩生成的 block index。KVCore 为每个 block 保留两个 key summary；当 block size 为 16 时，基础 summary 规模约为原始 K/V cache 的 `2/16 = 1/8`。考虑到 attention heads 和 layers 之间的 block 重要性具有相似性，summary 还可以分别沿 head 和 layer 维度进一步合并，整体压缩约 `4x` 和 `3x`。因此，KVCore 只需要用很小的常驻 GPU 空间保存摘要，而 token count 和 block index 只带来少量整数元数据开销。

SnapKV 和 ChunkKV 不需要额外保存 block summary，这是它们在空间上的直接优势。然而，当完整 KV cache 已经被卸载到 CPU 或更低层级存储时，它们若要在 GPU 上重新计算所有 token/chunk 的重要性分数，就必须把大量 KV 数据重新搬回 GPU，或者在低带宽设备上完成打分。KVCore 的 block summary 更轻量，可以常驻 GPU；即使完整 KV cache 被卸载，调度器仍然可以直接在 GPU 上用 summary 完成 block-level scoring 和索引构建。换言之，KVCore 用少量常驻摘要空间换取了低开销、低数据搬运的在线选择能力，这一点对长上下文、KV offload 和批处理场景尤其重要。
