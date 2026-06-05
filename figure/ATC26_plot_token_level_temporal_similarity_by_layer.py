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
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B",
    "mistral_7b_instruct_v03": "Mistral-7B-v0.3",
    "qwen3_8b": "Qwen3-8B",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot token-level temporal overlap by attention layer.")
    parser.add_argument("--run-tag", default="full")
    parser.add_argument("--head-agg", default="mean")
    parser.add_argument("--context-length", type=int, default=None)
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


def load_lag_data(head_agg: str, context_length: int | None) -> pd.DataFrame:
    if not AGG_CSV.exists():
        raise FileNotFoundError(f"Missing aggregate CSV: {AGG_CSV}")
    df = pd.read_csv(AGG_CSV)
    df = df[(df["metric_group"] == "lag") & (df["head_agg"] == head_agg)].copy()
    if context_length is not None:
        df = df[df["context_length"] == context_length].copy()
    if df.empty:
        raise RuntimeError("No lag rows left after filtering aggregate CSV.")
    df["lag"] = df["lag"].astype(int)
    df["layer"] = df["layer"].astype(int)
    return df


def aggregate_lag(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model_key", "context_length", "compression_ratio", "layer", "lag"], as_index=False)
        .agg(
            overlap_mean=("overlap", "mean"),
            overlap_std=("overlap", "std"),
            jaccard_mean=("jaccard", "mean"),
            sample_count=("overlap", "count"),
        )
        .sort_values(["model_key", "context_length", "compression_ratio", "layer", "lag"])
    )


def plot_heatmap(agg: pd.DataFrame, model_key: str, context_length: int, ratio: float) -> Path:
    subset = agg[
        (agg["model_key"] == model_key)
        & (agg["context_length"] == context_length)
        & (agg["compression_ratio"] == ratio)
    ]
    pivot = subset.pivot(index="layer", columns="lag", values="overlap_mean").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=45, ha="right")
    tick_step = max(1, len(pivot.index) // 10)
    y_ticks = np.arange(0, len(pivot.index), tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(pivot.index[i])) for i in y_ticks])
    ax.set_xlabel(r"Step distance $\Delta$")
    ax.set_ylabel("Attention layer")
    model_label = MODEL_LABELS.get(model_key, model_key)
    ratio_label = RATIO_LABELS.get(float(ratio), f"Compression {ratio:g}")
    ax.set_title(f"{model_label}, ctx={context_length}, {ratio_label}: token-set overlap by layer")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Future-token oracle overlap")

    out = OUTDIR / f"{model_key}__ctx{context_length}__keep{int(round((1.0 - ratio) * 100))}__layer_delta_overlap_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_layer_curves(agg: pd.DataFrame, model_key: str, context_length: int, ratio: float) -> Path:
    subset = agg[
        (agg["model_key"] == model_key)
        & (agg["context_length"] == context_length)
        & (agg["compression_ratio"] == ratio)
    ].copy()
    layers = sorted(subset["layer"].unique())
    ncols = 4
    nrows = int(np.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, max(2.0, 1.2 * nrows)), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    color = RATIO_COLORS.get(float(ratio), "#0072B2")
    for ax, layer in zip(axes_arr, layers):
        curve = subset[subset["layer"] == layer].sort_values("lag")
        ax.plot(curve["lag"], curve["overlap_mean"], color=color, linewidth=1.2, marker="o", markersize=2.4)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="both", alpha=0.22)
        ax.set_title(f"L{int(layer)}", fontsize=8.5, pad=2)
    for ax in axes_arr[len(layers) :]:
        ax.axis("off")
    for ax in axes_arr[-ncols:]:
        if ax.has_data():
            ax.set_xlabel(r"$\Delta$", labelpad=1)
    for row in range(nrows):
        ax = axes_arr[row * ncols]
        if ax.has_data():
            ax.set_ylabel("Overlap", labelpad=1)
    model_label = MODEL_LABELS.get(model_key, model_key)
    ratio_label = RATIO_LABELS.get(float(ratio), f"Compression {ratio:g}")
    fig.suptitle(f"{model_label}, ctx={context_length}, {ratio_label}: per-layer token-set overlap", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])

    out = OUTDIR / f"{model_key}__ctx{context_length}__keep{int(round((1.0 - ratio) * 100))}__per_layer_overlap_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    configure_paths(args.run_tag)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lag_df = load_lag_data(args.head_agg, args.context_length)
    agg = aggregate_lag(lag_df)
    paper_csv = OUTDIR / "future_token_oracle_overlap_by_layer.csv"
    agg.to_csv(paper_csv, index=False)

    outputs: list[Path] = []
    keys = agg[["model_key", "context_length", "compression_ratio"]].drop_duplicates()
    for row in keys.itertuples(index=False):
        outputs.append(plot_heatmap(agg, row.model_key, int(row.context_length), float(row.compression_ratio)))
        outputs.append(plot_layer_curves(agg, row.model_key, int(row.context_length), float(row.compression_ratio)))

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "run_tag": args.run_tag,
        "source": str(AGG_CSV.relative_to(ROOT)),
        "head_agg": args.head_agg,
        "context_length_filter": args.context_length,
        "paper_csv": str(paper_csv.relative_to(ROOT)),
        "figures": [str(path.relative_to(ROOT)) for path in outputs],
    }
    (OUTDIR / "by_layer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    readme = OUTDIR / "BY_LAYER_README.md"
    readme.write_text(
        "# Token-Level Future-Token Oracle Overlap By Layer\n\n"
        f"Source: `{AGG_CSV.relative_to(ROOT)}`\n\n"
        "Only `metric_group=lag` is plotted. Fixed-refresh/reuse metrics are intentionally omitted.\n\n"
        f"Paper-facing CSV: `{paper_csv.relative_to(ROOT)}`\n\n"
        "Generated figures:\n"
        + "\n".join(f"- `{path.relative_to(ROOT)}`" for path in outputs)
        + "\n",
        encoding="utf-8",
    )
    print(OUTDIR)


if __name__ == "__main__":
    main()
