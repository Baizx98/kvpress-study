# KV Cache Offload + Block-Wise Compression: Story Draft

## Thesis

Long-context serving fails in batch settings not only because KV cache grows linearly, but because the *resident* KV working set becomes unstable under mixed request lengths. A pure compression policy reduces footprint, but it may throw away reusable context that would have been cheap to keep on CPU. A pure offload policy preserves KV, but it still leaves the GPU vulnerable to bursty growth, preemption, and latency spikes. The right system is a *joint memory policy*:

1. Compress KV at block granularity with query-aware importance.
2. Use the block score as a heat signal for offload/prefetch.
3. Keep some blocks permanently deleted and some blocks temporarily cold but recoverable.
4. Specialize the policy by layer and phase (`prefill` vs `decode`).

This is the story we want to tell.

## What the Repository Already Confirms

The current repo already establishes the experimental substrate:

- A unified press abstraction and hook-based KV cache compression pipeline.
- A broad set of compression baselines in `kvpress/presses/`, including `KnormPress`, `SnapKVPress`, `FinchPress`, `KeyDiffPress`, `PyramidKVPress`, `ChunkKVPress`, `BlockPress`, `DuoAttentionPress`, and phase wrappers such as `DecodingPress` and `PrefillDecodingPress`.
- An evaluation harness under `evaluation/` that supports long-context benchmarks such as `ruler`, `longbench`, `longbench-v2`, `loogle`, `zero_scrolls`, `infinitebench`, and `needle_in_haystack`.
- A quick testing knob via `fraction` in `evaluate_config.yaml`, which is already intended for short runs.

These are enough to support a systems-style study without changing the benchmark infrastructure.

## Why Compression Alone Is Not Enough

### Confirmed from the repo and current design

- Existing presses mostly optimize *which tokens to keep* under a fixed cache budget.
- The evaluation harness already treats compression ratio as the main knob.
- `BlockPress` and `BlockWisePress` operate on blocks, but their current forms are still only cache reducers, not memory schedulers.

### Hypothesis

For batch serving, the main pain point is not only cache size, but cache *volatility*. When request lengths are skewed, some sequences overgrow and preempt others. Compression alone can reduce this pressure, but it cannot distinguish between:

- blocks that are truly dead and can disappear,
- blocks that are cold for the current iteration but may be needed again soon,
- blocks that are better kept on CPU and prefetched back.

That distinction is exactly what an offload-aware policy needs.

## Why Offload Alone Is Not Enough

### Confirmed from the system framing

- CPU offload increases capacity, but it does not remove GPU-side pressure caused by active working-set growth.
- Offload decisions need a good heat signal to prioritize what should stay resident or be prefetched.

### Hypothesis

If the score used for compression is also used as offload heat, then one cheap score can drive both eviction and prefetch. This creates a single memory-policy layer instead of two disconnected heuristics.

## Relation to Prior Work

### Within KVPress baselines

The most relevant reference points are:

- `KnormPress`: query-free, low-cost, key-norm-based pruning.
- `SnapKVPress` and `FinchPress`: query-aware, but centered on last-query windows.
- `KeyDiffPress`: key-similarity driven, good for redundancy but not directly memory-scheduling aware.
- `PyramidKVPress`: layer-sensitive budgeting.
- `BlockPress`: block-wise iterative pruning.
- `DuoAttentionPress`: head-specialized retention, useful as a conceptual baseline for head weighting.

### Distinction of the proposed method

Our method differs in three ways:

1. It scores blocks rather than individual tokens.
2. It treats head contribution as non-uniform.
3. It distinguishes permanent deletion from temporary cold storage.

That makes the method closer to a *memory policy* than a pure KV compressor.

## Core Design

### 1. Block score

Each block gets a cheap query-aware score that combines:

- overall block importance,
- a small number of extreme tokens inside the block,
- head weights or head filtering,
- an adaptive query window instead of a fixed last-`N` window.

The score must be cheap enough to compute during serving and stable enough to reuse as a heat metric.

### 2. Two block states

We want two logical block states:

- `deleted`: removed from the current active KV set;
- `cold`: physically retained, but not participating in the current iteration's active computation; can be reactivated later.

This supports both compression and recoverable offload.

### 3. Per-layer, per-phase specialization

The prefill phase and decode phase do not have the same importance structure. Early layers and late layers also behave differently. So the policy should allow:

- different block sizes or thresholds by layer,
- different compression ratios by phase,
- different head weights or query-window functions by phase.

## Story Skeleton

### Problem statement

Batch long-context serving suffers from unstable KV residency under mixed-length requests, causing memory spikes, preemption, and latency inflation.

### Hypothesis

A block-wise query-aware score can simultaneously guide compression and offload if it captures both block-average importance and rare critical tokens, while remaining cheap enough for serving.

### Method

Implement a layer- and phase-aware block policy with:

- adaptive query selection,
- head weighting / head filtering,
- block-level top-k pruning,
- temporary cold blocks for recoverable memory pressure,
- score reuse as an offload heat signal.

### Experiment

Use the repo's `evaluation/` harness on small fractions of:

- `ruler` for long-context synthetic stress,
- `needle_in_haystack` for retrieval robustness,
- `longbench` / `zero_scrolls` for downstream QA and reasoning,
- optionally `infinitebench` for heterogeneous long-context behavior.

Run with a reduced `fraction` during iteration, then scale up on the strongest settings.

### Result

Expected outcome to validate:

- lower peak GPU residency than pure compression baselines,
- fewer preemption events or less latency variance under batch pressure,
- better accuracy than naive block deletion at the same effective memory budget,
- low scoring overhead.

### Conclusion

The key claim is not "better compression alone," but "a unified KV memory policy that jointly decides what to delete, what to cool down, and what to prefetch."

## Suggested Baselines

Within `kvpress/presses/`, the first comparison set should be:

- `KnormPress`
- `SnapKVPress`
- `FinchPress`
- `KeyDiffPress`
- `PyramidKVPress`
- `BlockPress`
- `ChunkKVPress`
- `DuoAttentionPress`

For phase-aware comparisons, `DecodingPress` and `PrefillDecodingPress` are useful wrappers.

## Suggested Figures and Tables

1. A system overview figure showing GPU active KV, CPU cold KV, and the block score driving both eviction and prefetch.
2. A block-score diagram showing block-average importance plus rare-token preservation.
3. A layer-by-phase heatmap of compression ratios or cold-block ratios.
4. A memory timeline under batch load that shows reduced spikes and fewer preemptions.
5. A quality-vs-memory table comparing the proposed method to the baselines above.
6. A cost table reporting scoring overhead, peak memory, and throughput.

## Assumptions and Limits

- This draft assumes block granularity stays fixed during a run.
- The temporary cold-block mechanism is logical, not physical deletion.
- The current repo validates compression infrastructure, but the final offload scheduler still needs code-level integration before claims about end-to-end memory scheduling can be made.
- Any final performance claim must separate confirmed benchmark results from hypothesis-driven expectations.

