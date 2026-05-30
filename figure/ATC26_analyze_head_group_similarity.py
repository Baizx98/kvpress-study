from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENT = "ATC26_blockwise_attention_similarity_hotpotqa_3samples"
EXPERIMENT_NAME = "ATC26_blockwise_head_group_similarity_hotpotqa_3samples"
SOURCE_ARTIFACTS = ROOT / "evaluation" / "results" / "experiments" / SOURCE_EXPERIMENT / "artifacts"
ARTIFACT_DIR = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME
NOTE_PATH = ROOT / "note" / "ATC26_head_group_similarity_analysis_zh.md"

MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B",
    "mistral_7b_instruct_v03": "Mistral-7B-v0.3",
    "qwen3_8b": "Qwen3-8B",
}
MODEL_ORDER = ["llama31_8b_instruct", "mistral_7b_instruct_v03", "qwen3_8b"]
RATIO_ORDER = [0.3, 0.5, 0.7]
GROUP_SIZES = [1, 2, 4, 8]


def parse_score_file(path: Path) -> tuple[str, float, int]:
    match = re.match(r"(.+)__r([0-9]p[0-9]+)__sample([0-9]+)\.npz$", path.name)
    if not match:
        raise ValueError(f"Unexpected score file name: {path}")
    model_key, ratio_text, sample_idx = match.groups()
    return model_key, float(ratio_text.replace("p", ".")), int(sample_idx)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    set_a = set(int(x) for x in a.tolist())
    set_b = set(int(x) for x in b.tolist())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def pairwise_jaccard(sets: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(sets), len(sets)), dtype=np.float64)
    for i in range(len(sets)):
        for j in range(len(sets)):
            out[i, j] = jaccard(sets[i], sets[j])
    return out


def upper_mean(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    return float(matrix[np.triu_indices(matrix.shape[0], k=1)].mean())


def select_top_blocks(scores: np.ndarray, keep_count: int) -> np.ndarray:
    if keep_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if keep_count >= scores.shape[0]:
        return np.arange(scores.shape[0], dtype=np.int64)
    indices = np.argpartition(scores, -keep_count)[-keep_count:]
    return np.sort(indices.astype(np.int64))


def average_rank_correlation(score_vectors: np.ndarray) -> float:
    ranks = np.argsort(np.argsort(score_vectors, axis=-1), axis=-1).astype(np.float64)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(ranks, axis=-1, keepdims=True)
    normed = ranks / np.clip(denom, 1e-12, None)
    corr = np.clip(normed @ normed.T, -1.0, 1.0)
    return upper_mean(corr)


def analyze_file(path: Path) -> list[dict]:
    model_key, ratio, sample_idx = parse_score_file(path)
    arrays = np.load(path)
    per_head_scores = arrays["per_head_scores"]
    kept_layer_blocks = arrays["kept_layer_blocks"]
    kept_head_blocks = arrays["kept_head_blocks"]
    n_layers, n_heads, _ = per_head_scores.shape
    records = []

    for layer_idx in range(n_layers):
        keep_count = kept_layer_blocks[layer_idx].shape[0]
        head_jaccard = upper_mean(pairwise_jaccard([kept_head_blocks[layer_idx, h] for h in range(n_heads)]))
        score_vectors = per_head_scores[layer_idx]
        score_cos = np.corrcoef(score_vectors)
        score_cos_upper = upper_mean(score_cos)
        rank_corr = average_rank_correlation(score_vectors)

        for group_size in GROUP_SIZES:
            if n_heads % group_size != 0:
                continue
            group_sets = []
            for start in range(0, n_heads, group_size):
                group_scores = score_vectors[start : start + group_size].mean(axis=0)
                group_sets.append(select_top_blocks(group_scores, keep_count))
            group_pairwise = pairwise_jaccard(group_sets)
            overlaps = [jaccard(group_set, kept_layer_blocks[layer_idx]) for group_set in group_sets]
            union = np.unique(np.concatenate(group_sets)) if group_sets else np.empty((0,), dtype=np.int64)
            baseline = kept_layer_blocks[layer_idx]
            baseline_set = set(int(x) for x in baseline.tolist())
            union_set = set(int(x) for x in union.tolist())
            union_recall = len(union_set & baseline_set) / max(1, len(baseline_set))
            union_expansion = len(union_set) / max(1, len(baseline_set))

            records.append(
                {
                    "model_key": model_key,
                    "ratio": ratio,
                    "sample_idx": sample_idx,
                    "layer_idx": layer_idx,
                    "n_heads": n_heads,
                    "group_size": group_size,
                    "group_count": n_heads // group_size,
                    "head_jaccard_upper_mean": head_jaccard,
                    "head_score_corr_upper_mean": score_cos_upper,
                    "head_rank_corr_upper_mean": rank_corr,
                    "group_vs_all_mean": float(np.mean(overlaps)),
                    "group_vs_all_min": float(np.min(overlaps)),
                    "group_pairwise_upper_mean": upper_mean(group_pairwise),
                    "union_recall_vs_all": float(union_recall),
                    "union_expansion_vs_all": float(union_expansion),
                }
            )
    return records


def aggregate(records: list[dict]) -> list[dict]:
    keys = ["model_key", "ratio", "group_size", "group_count"]
    metrics = [
        "head_jaccard_upper_mean",
        "head_score_corr_upper_mean",
        "head_rank_corr_upper_mean",
        "group_vs_all_mean",
        "group_vs_all_min",
        "group_pairwise_upper_mean",
        "union_recall_vs_all",
        "union_expansion_vs_all",
    ]
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        grouped.setdefault(tuple(record[k] for k in keys), []).append(record)

    rows = []
    for key, items in sorted(grouped.items()):
        row = dict(zip(keys, key))
        row["n"] = len(items)
        for metric in metrics:
            values = np.asarray([item[metric] for item in items], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_std"] = float(finite.std()) if finite.size else float("nan")
        rows.append(row)
    return rows


def aggregate_by_layer(records: list[dict]) -> list[dict]:
    keys = ["model_key", "ratio", "group_size", "group_count", "layer_idx"]
    metrics = ["group_vs_all_mean", "group_pairwise_upper_mean", "head_jaccard_upper_mean"]
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        grouped.setdefault(tuple(record[k] for k in keys), []).append(record)

    rows = []
    for key, items in sorted(grouped.items()):
        row = dict(zip(keys, key))
        row["n"] = len(items)
        for metric in metrics:
            values = np.asarray([item[metric] for item in items], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
        rows.append(row)
    return rows


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def plot_group_vs_all(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True, constrained_layout=True)
    colors = {0.3: "#0072B2", 0.5: "#D55E00", 0.7: "#009E73"}
    for ax, model_key in zip(axes, MODEL_ORDER):
        model_rows = [r for r in rows if r["model_key"] == model_key]
        for ratio in RATIO_ORDER:
            ratio_rows = sorted([r for r in model_rows if r["ratio"] == ratio], key=lambda x: x["group_size"])
            ax.plot(
                [r["group_size"] for r in ratio_rows],
                [r["group_vs_all_mean_mean"] for r in ratio_rows],
                marker="o",
                linewidth=1.8,
                color=colors[ratio],
                label=f"r={ratio:g}",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(GROUP_SIZES)
        ax.set_xticklabels([str(x) for x in GROUP_SIZES])
        ax.set_ylim(0, 1.02)
        ax.set_title(MODEL_LABELS[model_key])
        ax.set_xlabel("Merged KV heads per group")
        ax.grid(True, axis="y", alpha=0.35)
    axes[0].set_ylabel("Jaccard vs. all-head selection")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.savefig(OUTDIR / "ATC26_head_group_vs_all_selection.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "ATC26_head_group_vs_all_selection.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_group_pairwise(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True, constrained_layout=True)
    colors = {0.3: "#0072B2", 0.5: "#D55E00", 0.7: "#009E73"}
    for ax, model_key in zip(axes, MODEL_ORDER):
        model_rows = [r for r in rows if r["model_key"] == model_key]
        for ratio in RATIO_ORDER:
            ratio_rows = sorted([r for r in model_rows if r["ratio"] == ratio], key=lambda x: x["group_size"])
            x = [r["group_size"] for r in ratio_rows if r["group_count"] > 1]
            y = [r["group_pairwise_upper_mean_mean"] for r in ratio_rows if r["group_count"] > 1]
            ax.plot(x, y, marker="s", linewidth=1.8, color=colors[ratio], label=f"r={ratio:g}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4])
        ax.set_xticklabels(["1", "2", "4"])
        ax.set_ylim(0, 1.02)
        ax.set_title(MODEL_LABELS[model_key])
        ax.set_xlabel("Merged KV heads per group")
        ax.grid(True, axis="y", alpha=0.35)
    axes[0].set_ylabel("Jaccard among head groups")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.savefig(OUTDIR / "ATC26_head_group_pairwise_similarity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "ATC26_head_group_pairwise_similarity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_layer_curves(layer_rows: list[dict]) -> None:
    for model_key in MODEL_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True, constrained_layout=True)
        for ax, ratio in zip(axes, RATIO_ORDER):
            for group_size, color in [(1, "#9E9E9E"), (2, "#0072B2"), (4, "#D55E00"), (8, "#009E73")]:
                rows = sorted(
                    [
                        r
                        for r in layer_rows
                        if r["model_key"] == model_key and r["ratio"] == ratio and r["group_size"] == group_size
                    ],
                    key=lambda x: x["layer_idx"],
                )
                ax.plot(
                    [r["layer_idx"] for r in rows],
                    [r["group_vs_all_mean_mean"] for r in rows],
                    linewidth=1.5,
                    label=f"g={group_size}",
                    color=color,
                )
            ax.set_title(f"r={ratio:g}")
            ax.set_xlabel("Layer")
            ax.set_ylim(0, 1.02)
            ax.grid(True, axis="y", alpha=0.35)
        axes[0].set_ylabel("Jaccard vs. all-head selection")
        axes[-1].legend(frameon=False, loc="lower right")
        fig.suptitle(f"{MODEL_LABELS[model_key]} per-layer head-group stability", fontsize=12)
        fig.savefig(OUTDIR / f"{model_key}__per_layer_group_vs_all.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUTDIR / f"{model_key}__per_layer_group_vs_all.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_per_layer_head_heatmaps() -> None:
    selected = {
        ("llama31_8b_instruct", 0.5, 0),
        ("mistral_7b_instruct_v03", 0.5, 0),
        ("qwen3_8b", 0.5, 0),
    }
    per_layer_dir = OUTDIR / "per_layer_head_similarity"
    per_layer_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((SOURCE_ARTIFACTS / "scores").glob("*.npz")):
        model_key, ratio, sample_idx = parse_score_file(path)
        key = (model_key, ratio, sample_idx)
        if key not in selected:
            continue
        arrays = np.load(path)
        kept_head_blocks = arrays["kept_head_blocks"]
        matrices = [pairwise_jaccard([kept_head_blocks[layer_idx, h] for h in range(kept_head_blocks.shape[1])]) for layer_idx in range(kept_head_blocks.shape[0])]
        n_layers = len(matrices)
        cols = 6
        rows = int(np.ceil(n_layers / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12.0, 2.0 * rows), constrained_layout=True)
        axes = np.asarray(axes).reshape(rows, cols)
        last_im = None
        for layer_idx in range(rows * cols):
            ax = axes.flat[layer_idx]
            if layer_idx >= n_layers:
                ax.axis("off")
                continue
            arr = np.asarray(matrices[layer_idx], dtype=np.float64)
            last_im = ax.imshow(arr, vmin=0, vmax=1, cmap="viridis", origin="lower")
            ax.set_title(f"L{layer_idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, label="Jaccard")
        fig.suptitle(f"{MODEL_LABELS[model_key]} r={ratio:g} sample={sample_idx} per-layer KV-head Jaccard", fontsize=12)
        out = per_layer_dir / f"{model_key}__r{str(ratio).replace('.', 'p')}__sample{sample_idx:02d}__all_layers_head_similarity.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_note(rows: list[dict]) -> None:
    def find_row(model_key: str, ratio: float, group_size: int) -> dict:
        for row in rows:
            if row["model_key"] == model_key and row["ratio"] == ratio and row["group_size"] == group_size:
                return row
        raise KeyError((model_key, ratio, group_size))

    lines = [
        "# ATC26 Head Group Similarity 分析",
        "",
        "## 结论先行",
        "",
        "原始 `ATC26_head_similarity_grid` 的 head-head Jaccard 低，不能直接说明不同 head 的注意力分布完全不同。它更说明：当每个 head 单独做 hard top-k block selection 时，top-k 边界附近的小排序差异会被放大成不同的 kept block set。",
        "",
        "`ATC26_head_score_cosine_grid` 很高，说明不同 KV head 的 block score 向量方向高度一致。也就是说，大部分 head 对“哪些 block 大体重要”判断相近，但单个 head 的 top-k 离散选择不稳定。",
        "",
        "因此更适合论文叙述的实验不是“单 head 是否能独立决定 KVCache”，而是“把若干 KV head 合并成 group 后，group-level selection 是否接近全头平均 selection”。这和最终方法更一致。",
        "",
        "## 新实验定义",
        "",
        "输入来自已有原始分数，不重新跑模型：",
        "",
        "- `per_head_scores`: 每层每个 KV head 的 block score",
        "- `kept_layer_blocks`: 当前 `head_agg_mode=uniform_mean` 的全头平均 block selection",
        "- `kept_head_blocks`: 单 head 独立选择的 block set",
        "",
        "对 KV heads 做固定连续分组，group size 为 `1/2/4/8`。每个 group 内对 score 取平均，再按相同 keep budget 选择 block。",
        "",
        "主要指标：",
        "",
        "- `Jaccard vs. all-head selection`: 每个 head group 的选择与当前全头平均选择的 Jaccard。",
        "- `Jaccard among head groups`: 不同 head group 之间的 Jaccard。",
        "- `union recall vs. all-head`: 多个 group 的 union 对全头平均选择的覆盖率。",
        "",
        "## 关键数值",
        "",
        "| model | ratio | g=1 vs all | g=2 vs all | g=4 vs all | g=8 vs all |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_key in MODEL_ORDER:
        for ratio in RATIO_ORDER:
            vals = [find_row(model_key, ratio, g)["group_vs_all_mean_mean"] for g in GROUP_SIZES]
            lines.append(
                f"| {MODEL_LABELS[model_key]} | {ratio:g} | "
                + " | ".join(f"{v:.3f}" for v in vals)
                + " |"
            )

    lines += [
        "",
        "## 解释",
        "",
        "1. `g=1` 对应每个 KV head 独立选择，和原来的 head Jaccard 一样容易受到 top-k 边界扰动影响。",
        "2. `g=2/4` 会明显稳定选择结果；它衡量的是“合并部分 head 后是否仍接近全头平均”。",
        "3. `g=8` 是所有 KV head 合成一个 group，本质上退化为当前 `uniform_mean`，因此和 all-head selection 一致。",
        "4. 这支持一个更合理的系统设计方向：不要选择单个 head，也不要让每个 head 单独维护完全独立的 KV block set；应该将相似 score distribution 的 head 合并成少量 group，再做 group-level block selection。",
        "",
        "## 图像",
        "",
        "- `figure/experiments/ATC26_blockwise_head_group_similarity_hotpotqa_3samples/ATC26_head_group_vs_all_selection.png`",
        "- `figure/experiments/ATC26_blockwise_head_group_similarity_hotpotqa_3samples/ATC26_head_group_pairwise_similarity.png`",
        "- `figure/experiments/ATC26_blockwise_head_group_similarity_hotpotqa_3samples/<model>__per_layer_group_vs_all.png`",
        "- `figure/experiments/ATC26_blockwise_head_group_similarity_hotpotqa_3samples/per_layer_head_similarity/*all_layers_head_similarity.png`",
        "",
        "## 论文建议",
        "",
        "主文不要展示原始 `head_similarity_grid` 作为主要证据，因为它回答的是“单 head hard top-k 是否一致”，而不是最终方法需要的“head group 是否可共享选择”。更推荐展示：",
        "",
        "- 一张 `head_score_cosine` 或文字说明 score distribution 高相似。",
        "- 一张 `head_group_vs_all_selection`，说明合并 2/4 个 KV heads 后选择已经接近全头平均。",
        "- appendix 放 per-layer head Jaccard，说明单 head top-k 的不稳定性来自 hard selection 边界。",
        "",
    ]
    NOTE_PATH.write_text("\n".join(lines))


def write_readmes() -> None:
    result_readme = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "README.md"
    figure_readme = OUTDIR / "README.md"
    result_readme.parent.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    result_readme.write_text(
        f"""# {EXPERIMENT_NAME}

## 实验目的

基于 `{SOURCE_EXPERIMENT}` 的原始 per-head block scores，评估将多个 KV heads 合并成 group 后，group-level block selection 是否接近当前 all-head uniform-mean selection。

## 运行脚本

- `figure/ATC26_analyze_head_group_similarity.py`

## 数据集

- `Xnhyacinth/LongBench`
- config: `hotpotqa`
- samples: 3

## 方法

- BlockWise per-head score 后处理
- group size: `1`, `2`, `4`, `8`

## 产物位置

- 聚合 JSON：`artifacts/ATC26_head_group_similarity_aggregate.json`
- per-layer JSON：`artifacts/ATC26_head_group_similarity_by_layer.json`
- 分析文档：`note/ATC26_head_group_similarity_analysis_zh.md`
"""
    )
    figure_readme.write_text(
        f"""# {EXPERIMENT_NAME}

## 图像说明

本目录保存 KV head group selection 的相似性分析图。

配套结果目录：

`evaluation/results/experiments/{EXPERIMENT_NAME}/`

配套分析：

`note/ATC26_head_group_similarity_analysis_zh.md`

## 推荐查看

1. `ATC26_head_group_vs_all_selection.png`
2. `ATC26_head_group_pairwise_similarity.png`
3. `<model>__per_layer_group_vs_all.png`
4. `per_layer_head_similarity/*all_layers_head_similarity.png`
"""
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((SOURCE_ARTIFACTS / "scores").glob("*.npz")):
        records.extend(analyze_file(path))
    rows = aggregate(records)
    layer_rows = aggregate_by_layer(records)
    write_json(ARTIFACT_DIR / "ATC26_head_group_similarity_records.json", records)
    write_json(ARTIFACT_DIR / "ATC26_head_group_similarity_aggregate.json", rows)
    write_json(ARTIFACT_DIR / "ATC26_head_group_similarity_by_layer.json", layer_rows)
    plot_group_vs_all(rows)
    plot_group_pairwise(rows)
    plot_layer_curves(layer_rows)
    plot_per_layer_head_heatmaps()
    write_note(rows)
    write_readmes()
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment": SOURCE_EXPERIMENT,
        "figures": sorted(str(p.relative_to(OUTDIR)) for p in OUTDIR.rglob("*.png")),
    }
    write_json(OUTDIR / "summary.json", summary)
    print(OUTDIR)


if __name__ == "__main__":
    main()
