from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_blockwise_attention_similarity_hotpotqa_3samples"
ARTIFACT_DIR = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
AGGREGATE_JSON = ARTIFACT_DIR / "ATC26_attention_similarity_aggregate.json"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME

MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B",
    "mistral_7b_instruct_v03": "Mistral-7B-v0.3",
    "qwen3_8b": "Qwen3-8B",
}


def load_payload() -> dict:
    if not AGGREGATE_JSON.exists():
        raise FileNotFoundError(
            f"Missing aggregate file: {AGGREGATE_JSON}. "
            "Run evaluation/ATC26_collect_attention_similarity.py first."
        )
    return json.loads(AGGREGATE_JSON.read_text())


def plot_matrix(matrix: list[list[float]], title: str, output: Path, xlabel: str, ylabel: str) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
    tick_step = max(1, arr.shape[0] // 8)
    ticks = np.arange(0, arr.shape[0], tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_grid(configs: list[dict], matrix_key: str, title: str, output: Path, xlabel: str, ylabel: str) -> None:
    model_keys = sorted({cfg["model_key"] for cfg in configs})
    ratios = sorted({float(cfg["compression_ratio"]) for cfg in configs})
    lookup = {(cfg["model_key"], float(cfg["compression_ratio"])): cfg for cfg in configs}

    fig, axes = plt.subplots(
        len(model_keys),
        len(ratios),
        figsize=(4.2 * len(ratios), 3.8 * len(model_keys)),
        constrained_layout=True,
        squeeze=False,
    )
    last_im = None
    for row_idx, model_key in enumerate(model_keys):
        for col_idx, ratio in enumerate(ratios):
            ax = axes[row_idx][col_idx]
            cfg = lookup[(model_key, ratio)]
            arr = np.asarray(cfg[matrix_key], dtype=np.float64)
            last_im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
            tick_step = max(1, arr.shape[0] // 8)
            ticks = np.arange(0, arr.shape[0], tick_step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            if row_idx == len(model_keys) - 1:
                ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            short_model = MODEL_LABELS.get(model_key, model_key)
            mean_key = matrix_key.replace("_matrix_mean", "_upper_mean")
            ax.set_title(f"{short_model}, r={ratio:g}\nmean={cfg[mean_key]:.3f}")

    fig.suptitle(title, fontsize=14)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payload = load_payload()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    configs = payload["configs"]

    plot_grid(
        configs,
        "layer_similarity_matrix_mean",
        "BlockWise Layer-to-Layer Kept-Block Similarity",
        OUTDIR / "ATC26_layer_similarity_grid.png",
        xlabel="Layer",
        ylabel="Layer",
    )
    plot_grid(
        configs,
        "head_similarity_matrix_mean",
        "BlockWise KV-Head-to-KV-Head Kept-Block Similarity",
        OUTDIR / "ATC26_head_similarity_grid.png",
        xlabel="KV head",
        ylabel="KV head",
    )
    plot_grid(
        configs,
        "head_score_cosine_matrix_mean",
        "BlockWise KV-Head Score-Vector Cosine Similarity",
        OUTDIR / "ATC26_head_score_cosine_grid.png",
        xlabel="KV head",
        ylabel="KV head",
    )

    for cfg in configs:
        model_key = cfg["model_key"]
        ratio_tag = str(cfg["compression_ratio"]).replace(".", "p")
        prefix = f"{model_key}__r{ratio_tag}"
        label = MODEL_LABELS.get(model_key, model_key)
        plot_matrix(
            cfg["layer_similarity_matrix_mean"],
            f"{label} r={cfg['compression_ratio']:g} layer kept-block Jaccard",
            OUTDIR / f"{prefix}__layer_similarity.png",
            xlabel="Layer",
            ylabel="Layer",
        )
        plot_matrix(
            cfg["head_similarity_matrix_mean"],
            f"{label} r={cfg['compression_ratio']:g} KV-head kept-block Jaccard",
            OUTDIR / f"{prefix}__head_similarity.png",
            xlabel="KV head",
            ylabel="KV head",
        )
        plot_matrix(
            cfg["head_score_cosine_matrix_mean"],
            f"{label} r={cfg['compression_ratio']:g} KV-head score cosine",
            OUTDIR / f"{prefix}__head_score_cosine.png",
            xlabel="KV head",
            ylabel="KV head",
        )

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "source": str(AGGREGATE_JSON.relative_to(ROOT)),
        "figures": sorted(path.name for path in OUTDIR.glob("*.png")),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(OUTDIR)


if __name__ == "__main__":
    main()
