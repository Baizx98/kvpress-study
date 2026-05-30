from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_blockwise_temporal_index_similarity"
ARTIFACT_DIR = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
AGG_CSV = ARTIFACT_DIR / "ATC26_temporal_similarity_aggregate.csv"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME

MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B",
    "mistral_7b_instruct_v03": "Mistral-7B-v0.3",
    "qwen3_8b": "Qwen3-8B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ATC26 BlockWise temporal index similarity.")
    parser.add_argument("--run-tag", default="full")
    return parser.parse_args()


def configure_paths(run_tag: str) -> None:
    global ARTIFACT_DIR, AGG_CSV, OUTDIR

    ARTIFACT_DIR = (
        ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
        if run_tag == "full"
        else ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts" / run_tag
    )
    AGG_CSV = ARTIFACT_DIR / "ATC26_temporal_similarity_aggregate.csv"
    OUTDIR = (
        ROOT / "figure" / "experiments" / EXPERIMENT_NAME
        if run_tag == "full"
        else ROOT / "figure" / "experiments" / EXPERIMENT_NAME / run_tag
    )


def require_csv() -> pd.DataFrame:
    if not AGG_CSV.exists():
        raise FileNotFoundError(f"Missing aggregate CSV: {AGG_CSV}")
    df = pd.read_csv(AGG_CSV)
    if df.empty:
        raise RuntimeError(f"Aggregate CSV is empty: {AGG_CSV}")
    return df


def aggregate_lag(df: pd.DataFrame) -> pd.DataFrame:
    lag = df[df["metric_group"] == "lag"].copy()
    lag = lag.dropna(subset=["lag"])
    metrics = ["jaccard", "overlap", "score_cosine"]
    return (
        lag.groupby(["model_key", "context_length", "mode", "layer", "lag"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["model_key", "context_length", "mode", "layer", "lag"])
    )


def aggregate_reuse(df: pd.DataFrame) -> pd.DataFrame:
    reuse = df[df["metric_group"] == "reuse"].copy()
    reuse = reuse.dropna(subset=["reuse_interval"])
    metrics = ["reuse_jaccard", "reuse_recall", "refresh_reduction"]
    return (
        reuse.groupby(["model_key", "context_length", "mode", "reuse_interval"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["model_key", "context_length", "mode", "reuse_interval"])
    )


def plot_lag_heatmap(lag_df: pd.DataFrame, model_key: str, context_length: int, mode: str, metric: str) -> Path:
    subset = lag_df[
        (lag_df["model_key"] == model_key)
        & (lag_df["context_length"] == context_length)
        & (lag_df["mode"] == mode)
    ]
    pivot = subset.pivot(index="layer", columns="lag", values=metric).sort_index().sort_index(axis=1)
    arr = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=45, ha="right")
    tick_step = max(1, len(pivot.index) // 8)
    y_ticks = np.arange(0, len(pivot.index), tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(pivot.index[i])) for i in y_ticks])
    ax.set_xlabel("Step distance / lag")
    ax.set_ylabel("Layer")
    label = MODEL_LABELS.get(model_key, model_key)
    ax.set_title(f"{label}, ctx={context_length}, {mode}, {metric}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = OUTDIR / f"{model_key}__ctx{context_length}__{mode}__lag_{metric}_heatmap.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lag_curve(lag_df: pd.DataFrame, model_key: str, context_length: int, mode: str) -> Path:
    subset = lag_df[
        (lag_df["model_key"] == model_key)
        & (lag_df["context_length"] == context_length)
        & (lag_df["mode"] == mode)
    ]
    curve = subset.groupby("lag", as_index=False)[["jaccard", "overlap", "score_cosine"]].mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for metric, label in [
        ("overlap", "Top-k overlap"),
        ("jaccard", "Jaccard"),
        ("score_cosine", "Score cosine"),
    ]:
        ax.plot(curve["lag"], curve[metric], marker="o", linewidth=1.8, label=label)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Step distance / lag")
    ax.set_ylabel("Similarity")
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(frameon=False)
    label = MODEL_LABELS.get(model_key, model_key)
    ax.set_title(f"{label}, ctx={context_length}, {mode}: temporal similarity")

    out = OUTDIR / f"{model_key}__ctx{context_length}__{mode}__lag_curve.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_reuse_curve(reuse_df: pd.DataFrame, model_key: str, context_length: int, mode: str) -> Path:
    subset = reuse_df[
        (reuse_df["model_key"] == model_key)
        & (reuse_df["context_length"] == context_length)
        & (reuse_df["mode"] == mode)
    ].sort_values("reuse_interval")

    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    ax.plot(subset["reuse_interval"], subset["reuse_recall"], marker="o", linewidth=1.8, label="Reuse recall")
    ax.plot(subset["reuse_interval"], subset["reuse_jaccard"], marker="s", linewidth=1.8, label="Reuse Jaccard")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Refresh interval R")
    ax.set_ylabel("Similarity to every-step oracle")
    ax.grid(True, axis="both", alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        subset["reuse_interval"],
        subset["refresh_reduction"],
        color="#666666",
        linestyle="--",
        linewidth=1.5,
        label="Refresh reduction",
    )
    ax2.set_ylim(0.0, 1.02)
    ax2.set_ylabel("Refresh reduction")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="lower left")

    label = MODEL_LABELS.get(model_key, model_key)
    ax.set_title(f"{label}, ctx={context_length}, {mode}: fixed-step reuse")
    out = OUTDIR / f"{model_key}__ctx{context_length}__{mode}__reuse_curve.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    configure_paths(args.run_tag)
    df = require_csv()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lag_df = aggregate_lag(df)
    reuse_df = aggregate_reuse(df)

    figures: list[str] = []
    keys = lag_df[["model_key", "context_length", "mode"]].drop_duplicates()
    for row in keys.itertuples(index=False):
        for metric in ["overlap", "jaccard", "score_cosine"]:
            figures.append(str(plot_lag_heatmap(lag_df, row.model_key, int(row.context_length), row.mode, metric)))
        figures.append(str(plot_lag_curve(lag_df, row.model_key, int(row.context_length), row.mode)))
        figures.append(str(plot_reuse_curve(reuse_df, row.model_key, int(row.context_length), row.mode)))

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "source": str(AGG_CSV.relative_to(ROOT)),
        "figures": [str(Path(path).relative_to(ROOT)) for path in figures],
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUTDIR / "README.md").write_text(
        "# ATC26 BlockWise Temporal Index Similarity Figures\n\n"
        f"Source: `{AGG_CSV.relative_to(ROOT)}`\n\n"
        "Main views:\n"
        "- `*_lag_overlap_heatmap.*`: layer-by-lag top-k overlap.\n"
        "- `*_lag_curve.*`: layer-averaged temporal similarity over lag.\n"
        "- `*_reuse_curve.*`: fixed refresh interval versus every-step oracle.\n",
        encoding="utf-8",
    )
    print(OUTDIR)


if __name__ == "__main__":
    main()
