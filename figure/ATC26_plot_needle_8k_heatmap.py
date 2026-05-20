from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_needle_8k_token_length_depth_heatmap"
ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
CELL_TABLE = ARTIFACTS_DIR / "ATC26_needle_8k_metrics_cell.csv"
CELL_TABLE_VALID = ARTIFACTS_DIR / "ATC26_needle_8k_metrics_cell_valid_depth0_90.csv"

METHOD_ORDER = ["block_wise", "snapkv", "chunkkv"]
METHOD_LABELS = {
    "block_wise": "BlockWise",
    "snapkv": "SnapKV",
    "chunkkv": "ChunkKV",
}
TOKEN_LENGTHS = list(range(256, 8192 + 1, 256))
DEPTHS = list(range(0, 101, 10))


def _plot_method(ax, df: pd.DataFrame, method: str, depths: list[int]):
    pivot = df[df["method"].eq(method)].pivot(index="needle_depth", columns="token_length", values="accuracy")
    pivot = pivot.reindex(index=depths, columns=TOKEN_LENGTHS)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
    ax.set_title(METHOD_LABELS.get(method, method))
    ax.set_xlabel("Token length")
    ax.set_ylabel("Needle depth (%)")
    xtick_positions = list(range(0, len(TOKEN_LENGTHS), 4))
    if len(TOKEN_LENGTHS) - 1 not in xtick_positions:
        xtick_positions.append(len(TOKEN_LENGTHS) - 1)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(TOKEN_LENGTHS[i]) for i in xtick_positions], rotation=45, ha="right")
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([str(v) for v in depths])

    for y_idx, depth in enumerate(depths):
        for x_idx, token_length in enumerate(TOKEN_LENGTHS):
            row = df[
                df["method"].eq(method)
                & df["needle_depth"].eq(depth)
                & df["token_length"].eq(token_length)
            ]
            if row.empty:
                continue
            value = float(row["accuracy"].iloc[0])
            if x_idx % 2 == 0:
                color = "white" if value < 0.55 else "black"
                ax.text(x_idx, y_idx, f"{value:.1f}", ha="center", va="center", fontsize=4.5, color=color)
    return image


def _plot(df: pd.DataFrame, depths: list[int], output_stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if method in set(df["method"])]
    if not methods:
        raise SystemExit("No methods found in cell table.")

    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 4.6), constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]
    image = None
    for ax, method in zip(axes, methods):
        image = _plot_method(ax, df, method, depths)
    if image is not None:
        fig.colorbar(image, ax=axes, label="Retrieval accuracy")

    png = FIGURE_DIR / f"{output_stem}.png"
    pdf = FIGURE_DIR / f"{output_stem}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CELL_TABLE)
    _plot(df, DEPTHS, "ATC26_needle_8k_token_length_depth_all_methods")
    if CELL_TABLE_VALID.exists():
        valid_df = pd.read_csv(CELL_TABLE_VALID)
        _plot(valid_df, list(range(0, 100, 10)), "ATC26_needle_8k_token_length_depth_valid_depth0_90")


if __name__ == "__main__":
    main()
