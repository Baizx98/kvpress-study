from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "kvcore_lifecycle_decode_longbench16_2pct_seed43_top_p095_skip2"
RESULT_ROOT = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
SUMMARY_CSV = RESULT_ROOT / "summary.csv"
NOTE_PATH = ROOT / "note" / f"{EXPERIMENT_NAME}_results_zh.md"
FIGURE_ROOT = ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_INDEX = ROOT / "figure" / "EXPERIMENT_INDEX.md"

OUTPUT_STEM = FIGURE_ROOT / "longbench_task_group_absolute_score_singlecol"
PLOT_CSV = FIGURE_ROOT / "longbench_task_group_absolute_score_singlecol.csv"
SUMMARY_JSON = FIGURE_ROOT / "summary.json"
README = FIGURE_ROOT / "README.md"

CATEGORY_ORDER = [
    "multi_doc_qa",
    "single_doc_qa",
    "summarization",
    "code",
    "synthetic",
    "few_shot",
]
CATEGORY_LABELS = {
    "multi_doc_qa": "Multi\nQA",
    "single_doc_qa": "Single\nQA",
    "summarization": "Summ.",
    "code": "Code",
    "synthetic": "Synth.",
    "few_shot": "Few",
}
METHODS = [
    ("full_kv", "Full KV", "#3F6FB5"),
    ("decode_qaware_blockwise_top_p", "KVCore", "#E0569B"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.0,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def ensure_index_entry() -> None:
    entry = (
        f"- [{EXPERIMENT_NAME}](/home10T/bzx/workspace/kvpress-study/"
        f"figure/experiments/{EXPERIMENT_NAME}/README.md): task-group average LongBench score bars."
    )
    text = FIGURE_INDEX.read_text() if FIGURE_INDEX.exists() else ""
    if entry in text:
        return
    if text and not text.endswith("\n"):
        text += "\n"
    FIGURE_INDEX.write_text(text + entry + "\n")


def aggregate() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(SUMMARY_CSV)
    df = pd.read_csv(SUMMARY_CSV)
    grouped = (
        df.groupby("category", as_index=False)
        .agg(
            full_kv=("full_kv", "mean"),
            decode_qaware_blockwise_top_p=("decode_qaware_blockwise_top_p", "mean"),
            delta=("delta", "mean"),
            num_tasks=("task", "count"),
            full_kv_samples=("full_kv_samples", "sum"),
            decode_qaware_blockwise_samples=("decode_qaware_blockwise_samples", "sum"),
        )
        .set_index("category")
        .reindex(CATEGORY_ORDER)
        .dropna(subset=["full_kv", "decode_qaware_blockwise_top_p"], how="all")
        .reset_index()
    )
    grouped["category_label"] = grouped["category"].map(CATEGORY_LABELS).fillna(grouped["category"])
    return grouped


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.65, linestyle="-", zorder=0)
    ax.tick_params(axis="both", direction="in", length=2.4, width=0.7)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.75)
        ax.spines[side].set_color("0.25")


def plot(grouped: pd.DataFrame) -> list[Path]:
    configure_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(grouped), dtype=float)
    width = 0.34
    offsets = [-width / 2, width / 2]

    fig, ax = plt.subplots(figsize=(3.33, 1.45))
    for (key, label, color), offset in zip(METHODS, offsets):
        ax.bar(
            x + offset,
            grouped[key],
            width=width,
            label=label,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )

    ax.set_ylabel("LongBench score")
    ax.set_xlabel("Task group")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["category_label"])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    style_axis(ax)
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        handlelength=1.0,
        borderpad=0.15,
        labelspacing=0.22,
    )

    fig.subplots_adjust(left=0.15, right=0.995, bottom=0.25, top=0.96)
    pdf = OUTPUT_STEM.with_suffix(".pdf")
    png = OUTPUT_STEM.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return [pdf, png]


def write_metadata(grouped: pd.DataFrame, outputs: list[Path]) -> None:
    grouped.to_csv(PLOT_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "experiment_name": EXPERIMENT_NAME,
                "source_summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
                "aggregation": "mean full_kv and decode_qaware_blockwise_top_p scores within each LongBench task category",
                "figure_pdf": str(outputs[0].relative_to(ROOT)),
                "figure_png": str(outputs[1].relative_to(ROOT)),
                "plot_csv": str(PLOT_CSV.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    README.write_text(
        f"""# {EXPERIMENT_NAME}

## Purpose

Single-column grouped bar chart showing average absolute LongBench score by task category.

## Source Data

- `{SUMMARY_CSV.relative_to(ROOT)}`
- `{NOTE_PATH.relative_to(ROOT)}`

## Methods

- `Full KV`
- `KVCore`: decode query-aware BlockWise active set, top-p `p=0.95`, first 2 layers skipped, LongBench fraction `0.02`, seed `43`.

## Aggregation

Scores are averaged within each LongBench task category. These are absolute LongBench scores, not deltas.

## Outputs

- `{outputs[0].relative_to(ROOT)}`
- `{outputs[1].relative_to(ROOT)}`
- `{PLOT_CSV.relative_to(ROOT)}`
- `{SUMMARY_JSON.relative_to(ROOT)}`
"""
    )
    ensure_index_entry()


def main() -> int:
    grouped = aggregate()
    outputs = plot(grouped)
    write_metadata(grouped, outputs)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
