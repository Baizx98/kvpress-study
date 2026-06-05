from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_token_level_temporal_similarity"
ARTIFACT_DIR = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
AGG_CSV = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_aggregate.csv"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#E6E6E6"
plt.rcParams["grid.linewidth"] = 0.7
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 8.5

RATIO_LABELS = {
    0.3: "Keep 70%",
    0.5: "Keep 50%",
    0.7: "Keep 30%",
}
RATIO_COLORS = {
    0.3: "#009E73",
    0.5: "#0072B2",
    0.7: "#D55E00",
}
RATIO_MARKERS = {
    0.3: "o",
    0.5: "s",
    0.7: "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot token-level temporal KV similarity.")
    parser.add_argument("--run-tag", default="full")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--head-agg", default="mean")
    return parser.parse_args()


def configure_paths(run_tag: str) -> None:
    global ARTIFACT_DIR, AGG_CSV, OUTDIR

    ARTIFACT_DIR = (
        ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
        if run_tag == "full"
        else ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts" / run_tag
    )
    AGG_CSV = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_aggregate.csv"
    OUTDIR = (
        ROOT / "figure" / "experiments" / EXPERIMENT_NAME
        if run_tag == "full"
        else ROOT / "figure" / "experiments" / EXPERIMENT_NAME / run_tag
    )


def load_data(head_agg: str, context_length: int | None) -> pd.DataFrame:
    if not AGG_CSV.exists():
        raise FileNotFoundError(f"Missing aggregate CSV: {AGG_CSV}")
    df = pd.read_csv(AGG_CSV)
    df = df[df["head_agg"] == head_agg].copy()
    if context_length is not None:
        df = df[df["context_length"] == context_length].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering aggregate CSV.")
    return df


def summarize_lag(df: pd.DataFrame) -> pd.DataFrame:
    lag = df[df["metric_group"] == "lag"].copy()
    lag["lag"] = lag["lag"].astype(int)
    grouped = (
        lag.groupby(["context_length", "compression_ratio", "lag"], as_index=False)
        .agg(
            overlap_mean=("overlap", "mean"),
            overlap_std=("overlap", "std"),
            jaccard_mean=("jaccard", "mean"),
            sample_count=("overlap", "count"),
        )
        .sort_values(["context_length", "compression_ratio", "lag"])
    )
    return grouped


def summarize_reuse(df: pd.DataFrame) -> pd.DataFrame:
    reuse = df[df["metric_group"] == "reuse"].copy()
    reuse["reuse_interval"] = reuse["reuse_interval"].astype(int)
    grouped = (
        reuse.groupby(["context_length", "compression_ratio", "reuse_interval"], as_index=False)
        .agg(
            reuse_recall_mean=("reuse_recall", "mean"),
            reuse_recall_std=("reuse_recall", "std"),
            reuse_jaccard_mean=("reuse_jaccard", "mean"),
            refresh_reduction=("refresh_reduction", "mean"),
            sample_count=("reuse_recall", "count"),
        )
        .sort_values(["context_length", "compression_ratio", "reuse_interval"])
    )
    return grouped


def plot_main(lag_df: pd.DataFrame, reuse_df: pd.DataFrame, run_tag: str, head_agg: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.85), constrained_layout=True)
    ax_lag, ax_reuse = axes

    for ratio in sorted(lag_df["compression_ratio"].unique()):
        subset = lag_df[lag_df["compression_ratio"] == ratio]
        curve = subset.groupby("lag", as_index=False).agg(
            overlap_mean=("overlap_mean", "mean"),
            overlap_std=("overlap_mean", "std"),
        )
        color = RATIO_COLORS.get(float(ratio), "#4D4D4D")
        label = RATIO_LABELS.get(float(ratio), f"Compression {ratio:g}")
        ax_lag.plot(
            curve["lag"],
            curve["overlap_mean"],
            marker=RATIO_MARKERS.get(float(ratio), "o"),
            linewidth=1.9,
            markersize=4.8,
            color=color,
            label=label,
        )

    ax_lag.set_xscale("log", base=2)
    ax_lag.set_ylim(0.0, 1.02)
    ax_lag.set_xlabel(r"Step distance $\Delta$")
    ax_lag.set_ylabel("Token-set overlap")
    ax_lag.grid(True, axis="both", alpha=0.25)
    ax_lag.set_title("(a) Future-token oracle overlap", loc="left", fontsize=10)

    for ratio in sorted(reuse_df["compression_ratio"].unique()):
        subset = reuse_df[reuse_df["compression_ratio"] == ratio]
        curve = subset.groupby("reuse_interval", as_index=False).agg(
            reuse_recall_mean=("reuse_recall_mean", "mean"),
            refresh_reduction=("refresh_reduction", "mean"),
        )
        color = RATIO_COLORS.get(float(ratio), "#4D4D4D")
        label = RATIO_LABELS.get(float(ratio), f"Compression {ratio:g}")
        ax_reuse.plot(
            curve["reuse_interval"],
            curve["reuse_recall_mean"],
            marker=RATIO_MARKERS.get(float(ratio), "o"),
            linewidth=1.9,
            markersize=4.8,
            color=color,
            label=label,
        )

    ax_reuse.set_xscale("log", base=2)
    ax_reuse.set_ylim(0.0, 1.02)
    ax_reuse.set_xlabel("Refresh interval R")
    ax_reuse.set_ylabel("Reuse recall vs. every-step oracle")
    ax_reuse.grid(True, axis="both", alpha=0.25)
    ax_reuse.set_title("(b) Fixed-refresh approximation", loc="left", fontsize=10)

    handles, labels = ax_lag.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Token-level temporal KV stability ({head_agg} over heads)", y=1.18, fontsize=11)

    out = OUTDIR / "figure6_token_level_temporal_similarity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    configure_paths(args.run_tag)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_data(args.head_agg, args.context_length)
    lag_df = summarize_lag(df)
    reuse_df = summarize_reuse(df)
    lag_csv = OUTDIR / "figure6_token_level_overlap_curve.csv"
    reuse_csv = OUTDIR / "figure6_token_level_reuse_curve.csv"
    lag_df.to_csv(lag_csv, index=False)
    reuse_df.to_csv(reuse_csv, index=False)
    figure_pdf = plot_main(lag_df, reuse_df, args.run_tag, args.head_agg)

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "run_tag": args.run_tag,
        "source": str(AGG_CSV.relative_to(ROOT)),
        "head_agg": args.head_agg,
        "context_length_filter": args.context_length,
        "outputs": {
            "figure_pdf": str(figure_pdf.relative_to(ROOT)),
            "figure_png": str(figure_pdf.with_suffix(".png").relative_to(ROOT)),
            "overlap_csv": str(lag_csv.relative_to(ROOT)),
            "reuse_csv": str(reuse_csv.relative_to(ROOT)),
        },
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUTDIR / "README.md").write_text(
        "# ATC26 Token-Level Temporal KV Similarity\n\n"
        f"Source: `{AGG_CSV.relative_to(ROOT)}`\n\n"
        "This figure compares top-k historical KV token sets between decode steps.\n"
        "The current self token is excluded before top-k selection.\n\n"
        "Primary outputs:\n"
        f"- `{figure_pdf.relative_to(ROOT)}`\n"
        f"- `{figure_pdf.with_suffix('.png').relative_to(ROOT)}`\n"
        f"- `{lag_csv.relative_to(ROOT)}`\n"
        f"- `{reuse_csv.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )
    print(figure_pdf)


if __name__ == "__main__":
    main()
