# BlockWisePress 论文算法伪代码 LaTeX 草稿

## 依据与边界

本草稿基于当前仓库中的 `BlockWisePress` 实现与 stage3 主线配置整理：

- 实现路径：`kvpress/presses/block_wise_press.py`
- 组件路径：`kvpress/presses/blockwise_components.py`
- stage3 配置脚本：`evaluation/run_blockwise_stage3_ratio70_fraction20_primarybench.py`
- stage3 结果分析：`note/blockwise_stage3_ratio70_fraction20_primarybench_analysis_zh.md`

当前结果显示，`blockwise_main` 不是所有 benchmark 上的单点最优，但它用最低的摘要成本取得了接近最优的效果，并且在 `hotpotqa`、`triviaqa` 等任务上表现稳定。因此论文主算法建议只写 `Main` 变体，把重点放在 **低成本块摘要 + query-window scoring + protected top-block selection** 这个系统设计，而不是把所有探索过的 scoring 分支都写进主文算法。

| 配置项 | Main 变体 |
|---|---|
| `summary_mode` | `mean_plus_norm_topk_mean` |
| `representative_mode` | `key_norm` |
| `query_agg_mode` | `max` |
| `head_agg_mode` | `uniform_mean` |
| `summary_topk_keys` | `4` |
| `mean_key_weight` | `0.75` |

推荐论文主文使用下面的 `Algorithm~\ref{alg:blockwise-press}`。`MultiRep` 和 `AdaptiveFusion` 可以留在实验消融或附录中解释，不进入主算法。

## LaTeX 包

双栏系统论文中建议使用紧凑的 `algorithm` + `algpseudocode`，不要用过宽的 `algorithm*`，除非会议模板允许跨栏浮动。

```latex
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{amsmath}
```

## 伪代码

```latex
\begin{algorithm}[t]
\caption{\textsc{BlockWisePress}: Low-Cost Query-Guided Block Compression}
\label{alg:blockwise-press}
\small
\begin{algorithmic}[1]
\Require Key/value cache $K,V \in \mathbb{R}^{H \times T \times d}$ at one layer,
hidden states $X$, block size $B$, compression ratio $\rho$,
query window $w$, protected prefix blocks $p$, protected recent blocks $r$.
\Ensure Compressed cache $\widetilde{K},\widetilde{V}$.
\State Partition $K,V$ into $n=\lceil T/B\rceil$ contiguous blocks
       $\{(K_i,V_i)\}_{i=1}^{n}$.
\State Obtain the last-$w$ pre-RoPE query states
       $Q \leftarrow \mathrm{Proj}_{Q}(X_{T-w:T})$.
\For{$i=1$ \textbf{to} $n$}
    \State Compute mean anchor
       $\mu_i \leftarrow \frac{1}{|K_i|}\sum_{k \in K_i} k$.
    \State Select $m$ high-norm keys
       $R_i \leftarrow \mathrm{TopM}_{k\in K_i} \|k\|_2$,
       and compute top-key anchor $\tau_i \leftarrow \frac{1}{m}\sum_{r \in R_i} r$.
    \State Score the mean anchor:
       $a_{h,i} \leftarrow
       \max_{q \in Q}
       \frac{\langle q_h,\mu_{i,h}\rangle}{\sqrt{d}}$.
    \State Score the top-key anchor:
       $c_{h,i} \leftarrow
       \max_{q \in Q}
       \frac{\langle q_h,\tau_{i,h}\rangle}{\sqrt{d}}$.
    \State Fuse the two anchor scores:
       $s_{h,i} \leftarrow \lambda a_{h,i} + (1-\lambda)c_{h,i}$.
    \State Aggregate heads:
       $s_i \leftarrow \frac{1}{H}\sum_{h=1}^{H}s_{h,i}$.
\EndFor
\State Set block budget $b \leftarrow \lceil n(1-\rho)\rceil$.
\State Initialize protected set
       $\mathcal{P} \leftarrow \{1,\ldots,p\}
       \cup \{n-r+1,\ldots,n\}$.
\If{the last block is partial}
    \State $\mathcal{P} \leftarrow \mathcal{P}\cup\{n\}$.
\EndIf
\If{$|\mathcal{P}| \le b$}
    \State Select $\mathcal{S} \leftarrow \mathcal{P}
    \cup \mathrm{Top}_{b-|\mathcal{P}|}
    \left(\{s_i: i \notin \mathcal{P}\}\right)$.
\Else
    \State Select $\mathcal{S} \leftarrow
    \mathrm{Top}_{b}\left(\{s_i: 1\le i\le n\}\right)$.
\EndIf
\State Expand selected blocks to token indices
       $\mathcal{I}\leftarrow \bigcup_{i\in\mathcal{S}}\{(i-1)B,\ldots,\min(iB,T)-1\}$.
\State \Return $\widetilde{K}\leftarrow K[:,\mathcal{I},:]$,
       $\widetilde{V}\leftarrow V[:,\mathcal{I},:]$.
\end{algorithmic}
\end{algorithm}
```

## 推荐正文描述

可以把算法前后的正文写成下面这段：

```latex
BlockWisePress compresses the prefill KV cache at block granularity.
For each contiguous KV block, it builds two low-cost anchors: the mean key
and the mean of a few high-norm keys. The last query window probes these
anchors to estimate block utility, and the system keeps only a fixed budget
of high-scoring blocks while preserving prefix sink blocks, recent blocks,
and the partial tail block when present. This avoids token-level cache
fragmentation and keeps the compressed cache layout friendly to batched
inference.
```

如果需要中文理解版：

> BlockWisePress 的核心不是 token-level pruning，而是用非常便宜的块摘要代表连续 KV block：一个 mean key 加一个 high-norm top-key mean。末尾 query window 只和这些块摘要交互来估计块重要性。最终只保留高分块，同时强制保留 prefix sink、最近块和未满尾块。这让压缩后的 KV cache 仍然保持块级连续布局，更适合 batch inference 和系统实现。

## Main 配置表 LaTeX

```latex
\begin{table}[t]
\centering
\small
\caption{BlockWisePress main configuration.}
\label{tab:blockwise-config}
\begin{tabular}{lc}
\toprule
Parameter & Value \\
\midrule
Block size $B$ & 16 \\
Query window $w$ & 64 \\
Summary & Mean+TopK \\
Top-key selector & KeyNorm \\
Top keys per block $m$ & 4 \\
Mean-key weight $\lambda$ & 0.75 \\
Query aggregation & Max \\
Head aggregation & Mean \\
Protected prefix blocks & 1 \\
Protected recent blocks & 2 \\
\bottomrule
\end{tabular}
\end{table}
```

建议在表注或正文中补充固定超参：

```latex
Unless otherwise stated, BlockWisePress uses block size $B=16$,
query window $w=64$, compression ratio $\rho=0.7$, four high-norm
keys per block summary, one protected prefix block, and two protected
recent blocks.
```

## 写作建议

1. 算法标题用 `Low-Cost Query-Guided Block Compression`，突出这是系统友好的低开销块级压缩，而不是复杂 token scorer。
2. 主算法不要写太多工程状态缓存，例如 `last_block_heat`、EMA、summary cache，这些属于实现优化或诊断工具，不适合放入主算法伪代码。
3. 双栏版面中，算法应控制在半栏以内；现在只保留 Main 变体，长度已经更适合 ATC/OSDI/ASPLOS 风格。
4. 当前实验证据显示 Main 变体不一定单点最优，但它成本最低、机制最清楚、效果接近最优。论文中建议表述为“we choose the low-cost Main configuration as the default system design and study stronger summary variants in ablations”，避免把探索性变体写成主算法。
