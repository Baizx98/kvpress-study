from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENT_NAME = "end2end_serving_paper_draft_modelaware_predicted_20260610"
SOURCE_CSV = (
    REPO_ROOT
    / "figure"
    / "experiments"
    / SOURCE_EXPERIMENT_NAME
    / "paper_draft_end2end_modelaware_metrics_table.csv"
)
EXPERIMENT_NAME = "end2end_serving_paper_draft_modelaware_comparison_singlecol_20260610"
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"

MODEL_ORDER = [
    "llama31_8b_instruct",
    "mistral_7b_instruct_v03",
    "qwen3_8b",
]
MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1",
    "mistral_7b_instruct_v03": "Mistral",
    "qwen3_8b": "Qwen3",
}
MODEL_COLORS = {
    "llama31_8b_instruct": "#0072B2",
    "mistral_7b_instruct_v03": "#D55E00",
    "qwen3_8b": "#009E73",
}
SYSTEM_ORDER = ["vLLM", "InfiniGen", "KVCore (pred.)"]
BATCH_ORDER = [1, 8, 16]
OUTPUT_ORDER = [1024, 2048, 6144]
OUTPUT_LABELS = {1024: "1k", 2048: "2k", 6144: "6k"}
BS_REGION_COLORS = {
    1: "#F8FAFC",
    8: "#EFF6FF",
    16: "#F0FDF4",
}
METRICS_TO_PLOT = [
    ("decode_throughput_tok_s", "Decode throughput (tok/s)", "modelaware_singlecol_throughput"),
    ("median_ttft_s", "Median TTFT (s)", "modelaware_singlecol_ttft"),
    ("p99_e2e_s", "P99 E2E latency (s)", "modelaware_singlecol_p99_e2e"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.9,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_box_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, color="#D8DEE9", alpha=0.70)
    ax.tick_params(axis="both", which="major", direction="in", length=2.5, width=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(0.72)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return [pdf_path, png_path]


def x_layout() -> tuple[dict[tuple[int, int], float], list[float], list[str], dict[int, tuple[float, float, float]]]:
    positions: dict[tuple[int, int], float] = {}
    xticks: list[float] = []
    xticklabels: list[str] = []
    regions: dict[int, tuple[float, float, float]] = {}
    group_gap = 1.0
    block_gap = 0.95
    cursor = 0.0
    for batch_size in BATCH_ORDER:
        centers: list[float] = []
        for output_len in OUTPUT_ORDER:
            positions[(batch_size, output_len)] = cursor
            centers.append(cursor)
            xticks.append(cursor)
            xticklabels.append(OUTPUT_LABELS[output_len])
            cursor += group_gap
        regions[batch_size] = (centers[0] - 0.52, centers[-1] + 0.52, float(np.mean(centers)))
        cursor += block_gap
    return positions, xticks, xticklabels, regions


def draw_bs_regions(ax: plt.Axes, regions: dict[int, tuple[float, float, float]]) -> None:
    for batch_size, (left, right, center) in regions.items():
        ax.axvspan(left, right, color=BS_REGION_COLORS[batch_size], alpha=0.62, zorder=0)
        ax.text(
            center,
            0.955,
            f"BS={batch_size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.8,
            fontweight="bold",
            color="#374151",
        )
    values_regions = list(regions.values())
    for (_, right, _), next_region in zip(values_regions[:-1], values_regions[1:]):
        separator = (right + next_region[0]) / 2.0
        ax.axvline(separator, color="#6B7280", linewidth=0.55, linestyle=":", alpha=0.7, zorder=1)


def plot_metric(metrics: pd.DataFrame, metric: str, ylabel: str, stem: str) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(3.33, 3.05), sharex=True, sharey=True)
    positions, xticks, xticklabels, regions = x_layout()
    width = 0.22
    offsets = [-width, 0.0, width]
    ymax = float(metrics[metric].max())

    for ax, system in zip(axes, SYSTEM_ORDER):
        draw_bs_regions(ax, regions)
        sub = metrics[metrics["system"] == system]
        for idx, model_key in enumerate(MODEL_ORDER):
            xs: list[float] = []
            values: list[float] = []
            for batch_size in BATCH_ORDER:
                for output_len in OUTPUT_ORDER:
                    row = sub[
                        (sub["model_key"] == model_key)
                        & (sub["batch_size"] == batch_size)
                        & (sub["output_len_bucket"] == output_len)
                    ]
                    xs.append(positions[(batch_size, output_len)] + offsets[idx])
                    values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
            ax.bar(
                xs,
                values,
                width=width,
                label=MODEL_LABELS[model_key],
                color=MODEL_COLORS[model_key],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        ax.text(
            0.012,
            1.02,
            system,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            clip_on=False,
        )
        ax.set_xlim(min(xticks) - 0.72, max(xticks) + 0.72)
        ax.set_ylim(0, ymax * 1.16 if ymax > 0 else 1.0)
        style_box_axis(ax)

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels)
    axes[-1].set_xlabel("Output length")
    fig.supylabel(ylabel, x=0.015, fontsize=7.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.015),
        columnspacing=0.8,
        handlelength=1.05,
    )
    fig.subplots_adjust(left=0.18, right=0.995, bottom=0.11, top=0.90, hspace=0.17)
    return save_figure(fig, stem)


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

Single-column model-aware comparison figures for paper-draft end-to-end serving results.

## Source

- Metrics table: `figure/experiments/{SOURCE_EXPERIMENT_NAME}/paper_draft_end2end_modelaware_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_modelaware_comparison_singlecol.py`

## Figures

{output_lines}

## Layout

Each metric is one single-column figure. The three vertical panels are vLLM, InfiniGen, and KVCore. Within each panel, bars compare Llama-3.1, Mistral, and Qwen3 at the same batch size and output length. Batch-size regions use the merged-BS shaded layout.
"""
    )


def main() -> int:
    setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SOURCE_CSV)
    outputs: list[Path] = []
    for metric, ylabel, stem in METRICS_TO_PLOT:
        outputs.extend(plot_metric(metrics, metric, ylabel, stem))
    write_readme(outputs)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: single-column model-aware serving comparison figures.",
    )
    print(f"Read metrics: {SOURCE_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
