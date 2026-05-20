from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "sparse_index_overhead_snapkv_chunkkv_blockwise"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
SUMMARY_CSV = ARTIFACTS_DIR / "sparse_index_overhead_summary.csv"
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"


METHOD_LABELS = {
    "snapkv": "SnapKV",
    "chunkkv": "ChunkKV",
    "blockwise_online": "KVCore online",
    "blockwise_amortized": "KVCore amort",
}
METHOD_COLORS = {
    "snapkv": "#0072B2",
    "chunkkv": "#E69F00",
    "blockwise_online": "#009E73",
    "blockwise_amortized": "#009E73",
}
METHOD_MARKERS = {
    "snapkv": "o",
    "chunkkv": "s",
    "blockwise_online": "^",
    "blockwise_amortized": "v",
}
REUSE_STEPS_FOR_LONG_CONTEXT = [1, 4, 16, 64, 256]
ACM_SINGLE_COLUMN_WIDTH_IN = 3.33


def format_ms_tick(value: float, _pos: int | None = None) -> str:
    if value <= 0:
        return "0"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def setup_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.45, alpha=0.28)
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.8)
    ax.minorticks_off()
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms_tick))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def plot_line(ax: plt.Axes, x, y, key: str, *, linestyle: str = "-") -> None:
    ax.plot(
        x,
        y,
        label=METHOD_LABELS[key],
        color=METHOD_COLORS[key],
        marker=METHOD_MARKERS[key],
        markersize=2.4,
        linewidth=1.35,
        linestyle=linestyle,
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )


def method_series(df: pd.DataFrame, value_col: str, method: str) -> pd.Series:
    return df[df["method"] == method].sort_values(df.columns[0])[value_col]


def plot_length_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df[df["sweep"] == "length"].copy()
    sub = sub.sort_values(["method", "length"])
    x = sorted(sub["length"].unique())
    snap = sub[sub["method"] == "snapkv"].sort_values("length")
    chunk = sub[sub["method"] == "chunkkv"].sort_values("length")
    block = sub[sub["method"] == "blockwise"].sort_values("length")

    plot_line(ax, x, snap["online_index_cuda_ms_mean"], "snapkv")
    plot_line(ax, x, chunk["online_index_cuda_ms_mean"], "chunkkv")
    plot_line(ax, x, block["online_index_cuda_ms_mean"], "blockwise_online")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v // 1024)}k" for v in x])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Request length")
    add_panel_label(ax, "(a)")
    style_axis(ax)


def plot_batch_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df[df["sweep"] == "batch"].copy()
    sub = sub.sort_values(["method", "batch_size"])
    x = sorted(sub["batch_size"].unique())
    snap = sub[sub["method"] == "snapkv"].sort_values("batch_size")
    chunk = sub[sub["method"] == "chunkkv"].sort_values("batch_size")
    block = sub[sub["method"] == "blockwise"].sort_values("batch_size")

    plot_line(ax, x, snap["online_index_cuda_ms_mean"], "snapkv")
    plot_line(ax, x, chunk["online_index_cuda_ms_mean"], "chunkkv")
    plot_line(ax, x, block["online_index_cuda_ms_mean"], "blockwise_online")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Batch size")
    add_panel_label(ax, "(b)")
    style_axis(ax)


def plot_reuse_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df[(df["sweep"] == "length") & (df["length"] == 4096)].set_index("method")
    reuse_steps = REUSE_STEPS_FOR_LONG_CONTEXT
    x = reuse_steps
    block_online = float(sub.loc["blockwise", "online_index_cuda_ms_mean"])
    block_summary = float(sub.loc["blockwise", "summary_build_cuda_ms_mean"])
    block_amortized = [block_online + block_summary / reuse for reuse in reuse_steps]
    snap = float(sub.loc["snapkv", "online_index_cuda_ms_mean"])
    chunk = float(sub.loc["chunkkv", "online_index_cuda_ms_mean"])

    plot_line(ax, x, block_amortized, "blockwise_amortized", linestyle="--")
    ax.axhline(
        block_online,
        color=METHOD_COLORS["blockwise_online"],
        linewidth=1.1,
        linestyle="-",
        label=METHOD_LABELS["blockwise_online"],
    )
    ax.axhline(
        snap,
        color=METHOD_COLORS["snapkv"],
        linewidth=1.0,
        linestyle=":",
        label="SnapKV",
    )
    ax.axhline(
        chunk,
        color=METHOD_COLORS["chunkkv"],
        linewidth=1.0,
        linestyle=":",
        label="ChunkKV",
    )
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in reuse_steps])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Reuse steps")
    add_panel_label(ax, "(c)")
    style_axis(ax)


def ensure_index_entry(index_path: Path, entry: str) -> None:
    text = index_path.read_text() if index_path.exists() else ""
    if entry in text:
        return
    if text and not text.endswith("\n"):
        text += "\n"
    index_path.write_text(text + entry + "\n")


def write_readme(outputs: list[Path]) -> None:
    output_lines = "\n".join(f"- `{path.name}`" for path in outputs)
    FIGURE_README.write_text(
        f"""# {EXPERIMENT_NAME}

## Purpose

Paper-style single-column figure for sparse-index construction overhead.
The figure uses the 32-layer averaged Llama-3.1-8B-Instruct results on L40S.

## Source Data

- `evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/sparse_index_overhead_summary.csv`
- `evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/sparse_index_overhead_layers.csv`
- `evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/metadata.json`

## Figures

{output_lines}

## Notes

Panel (a) shows request-length scaling, panel (b) shows batch-size scaling, and panel (c) shows BlockWise summary amortization for `B=1,L=4096,ratio=0.5`.
BlockWisePress does not build `mean_values` or `multi_rep_keys` summaries in this benchmark.
All plotted values are CUDA-time repeat means averaged over 32 attention layers.
"""
    )


def main() -> int:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(SUMMARY_CSV)
    setup_paper_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(6.2, 1.9), sharey=False)
    plot_length_panel(axes[0], df)
    plot_batch_panel(axes[1], df)
    plot_reuse_panel(axes[2], df)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    ordered_labels = ["SnapKV", "ChunkKV", "KVCore online", "KVCore amort"]
    handle_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for label, handle in zip(labels, handles):
            handle_by_label.setdefault(label, handle)
    handles = [handle_by_label[label] for label in ordered_labels if label in handle_by_label]
    labels = [label for label in ordered_labels if label in handle_by_label]
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        handlelength=1.45,
        columnspacing=0.75,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.24, top=0.80, wspace=0.2)

    pdf_path = FIGURE_ROOT / "sparse_index_overhead_paper_acm_wide.pdf"
    png_path = FIGURE_ROOT / "sparse_index_overhead_paper_acm_wide.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    outputs = [pdf_path, png_path]
    write_readme(outputs)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: paper-style sparse-index overhead figure for SnapKV, ChunkKV, and KVCore.",
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
