from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_needle_heatmap_llama31_8b_ratio50"
ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
CELL_TABLE = ARTIFACTS_DIR / "ATC26_needle_heatmap_metrics_cell.csv"

METHOD_ORDER = ["block_wise", "snapkv", "chunkkv"]
METHOD_LABELS = {
    "block_wise": "BlockWise",
    "snapkv": "SnapKV",
    "chunkkv": "ChunkKV",
}
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536]
NEEDLE_DEPTHS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _plot_matrix(ax, subset: pd.DataFrame, method: str):
    pivot = subset.pivot(index="context_length", columns="needle_depth", values="accuracy")
    pivot = pivot.reindex(index=CONTEXT_LENGTHS, columns=NEEDLE_DEPTHS)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
    ax.set_title(METHOD_LABELS.get(method, method))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(v)) for v in pivot.index])
    ax.set_xlabel("Needle depth (%)")
    ax.set_ylabel("Context length")
    return image


def main() -> None:
    df = pd.read_csv(CELL_TABLE)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    methods = [method for method in METHOD_ORDER if method in set(df["method"])]
    if not methods:
        raise SystemExit("No methods found in cell table.")

    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.2), constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]

    image = None
    for ax, method in zip(axes, methods):
        subset = df[df["method"].eq(method)]
        image = _plot_matrix(ax, subset, method)

    if image is not None:
        fig.colorbar(image, ax=axes, label="Retrieval accuracy")
    output = FIGURE_DIR / "ATC26_needle_heatmap_all_methods.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    for method in methods:
        fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
        image = _plot_matrix(ax, df[df["method"].eq(method)], method)
        fig.colorbar(image, ax=ax, label="Retrieval accuracy")
        fig.savefig(FIGURE_DIR / f"ATC26_needle_heatmap_{method}_ratio50.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
