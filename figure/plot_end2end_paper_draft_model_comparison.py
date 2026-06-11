from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENT_NAME = "end2end_serving_paper_draft_predicted_20260610"
SOURCE_CSV = (
    REPO_ROOT
    / "figure"
    / "experiments"
    / SOURCE_EXPERIMENT_NAME
    / "paper_draft_end2end_metrics_table.csv"
)
EXPERIMENT_NAME = "end2end_serving_paper_draft_model_comparison_20260610"
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
    ("decode_throughput_tok_s", "Decode throughput (tok/s)", "model_compare_throughput"),
    ("median_ttft_s", "Median TTFT (s)", "model_compare_ttft"),
    ("p99_e2e_s", "P99 E2E latency (s)", "model_compare_p99_e2e"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_box_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, color="#D8DEE9", alpha=0.72)
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.72)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(0.82)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return [pdf_path, png_path]


def x_layout() -> tuple[dict[tuple[int, int], float], list[float], list[str], dict[int, tuple[float, float, float]]]:
    positions: dict[tuple[int, int], float] = {}
    xticks: list[float] = []
    xticklabels: list[str] = []
    regions: dict[int, tuple[float, float, float]] = {}
    group_gap = 1.0
    block_gap = 1.05
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


def plot_metric(metrics: pd.DataFrame, metric: str, ylabel: str, stem: str) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.15), sharey=True)
    positions, xticks, xticklabels, regions = x_layout()
    width = 0.22
    offsets = [-width, 0.0, width]
    ymax = float(metrics[metric].max())

    for ax, system in zip(axes, SYSTEM_ORDER):
        for batch_size, (left, right, center) in regions.items():
            ax.axvspan(left, right, color=BS_REGION_COLORS[batch_size], alpha=0.62, zorder=0)
            ax.text(
                center,
                0.975,
                f"BS={batch_size}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.7,
                fontweight="bold",
                color="#374151",
            )
        values_regions = list(regions.values())
        for (_, right, _), next_region in zip(values_regions[:-1], values_regions[1:]):
            separator = (right + next_region[0]) / 2.0
            ax.axvline(separator, color="#6B7280", linewidth=0.65, linestyle=":", alpha=0.7, zorder=1)

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
                linewidth=0.5,
                zorder=3,
            )

        ax.text(
            0.015,
            1.035,
            system,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            clip_on=False,
        )
        ax.set_xlim(min(xticks) - 0.72, max(xticks) + 0.72)
        ax.set_ylim(0, ymax * 1.16 if ymax > 0 else 1.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Output length")
        style_box_axis(ax)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.04),
        columnspacing=1.25,
        handlelength=1.25,
    )
    fig.subplots_adjust(left=0.08, right=0.997, bottom=0.22, top=0.74, wspace=0.08)
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

Diagnostic three-model comparison figures for the paper-draft end-to-end serving table.
These figures are meant to make model-level differences visible, because the previous per-model plots used the same relative prediction rule for KVCore and for missing InfiniGen rows.

## Source

- Metrics table: `figure/experiments/{SOURCE_EXPERIMENT_NAME}/paper_draft_end2end_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_model_comparison.py`

## Figures

{output_lines}

## Layout

Each metric is one figure with three panels: vLLM, InfiniGen, and KVCore. Within each panel, bars compare the three models at the same batch size and output length. Batch-size regions use the same merged-BS shaded layout as the previous figures.
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
        f"- `{EXPERIMENT_NAME}`: diagnostic model-comparison figures for paper-draft serving predictions.",
    )
    print(f"Read metrics: {SOURCE_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
