from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_token_level_temporal_similarity"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME / "single_column_three_models"
PLOT_LEFT = 0.14
PLOT_RIGHT = 0.995
PLOT_CENTER_X = (PLOT_LEFT + PLOT_RIGHT) / 2

INPUTS = [
    ROOT
    / "evaluation"
    / "results"
    / "experiments"
    / EXPERIMENT_NAME
    / "artifacts"
    / "llama_delta1024"
    / "ATC26_token_level_temporal_similarity_aggregate.csv",
    ROOT
    / "evaluation"
    / "results"
    / "experiments"
    / EXPERIMENT_NAME
    / "artifacts"
    / "cross_model_delta1024"
    / "ATC26_token_level_temporal_similarity_aggregate.csv",
]

MODEL_ORDER = ["llama31_8b_instruct", "mistral_7b_instruct_v03", "qwen3_8b"]
MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B Inst.",
    "mistral_7b_instruct_v03": "Mistral-7B",
    "qwen3_8b": "Qwen3-8B",
}
RATIO_ORDER = [0.3, 0.5, 0.7]
RATIO_LABELS = {
    0.3: "Keep 70%",
    0.5: "Keep 50%",
    0.7: "Keep 30%",
}
RATIO_COLORS = {
    0.3: "#2F80ED",  # blue
    0.5: "#2CA25F",  # green
    0.7: "#D95F8D",  # pink
}
RATIO_MARKERS = {
    0.3: "o",
    0.5: "s",
    0.7: "^",
}


def configure_style() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams["xtick.labelsize"] = 6.6
    plt.rcParams["ytick.labelsize"] = 6.8
    plt.rcParams["legend.fontsize"] = 6.9
    plt.rcParams["axes.linewidth"] = 0.8


def load_data() -> pd.DataFrame:
    frames = []
    for path in INPUTS:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    lag = df[(df["metric_group"] == "lag") & (df["head_agg"] == "mean")].copy()
    lag["lag"] = lag["lag"].astype(int)
    lag = lag[lag["model_key"].isin(MODEL_ORDER)]

    # Use only lags present for all three models so the panels are directly comparable.
    common_lags = set(lag.groupby("model_key")["lag"].apply(set).loc[MODEL_ORDER[0]])
    for model_key in MODEL_ORDER[1:]:
        common_lags &= set(lag.groupby("model_key")["lag"].apply(set).loc[model_key])
    lag = lag[lag["lag"].isin(sorted(common_lags))]
    return lag


def aggregate(lag: pd.DataFrame) -> pd.DataFrame:
    return (
        lag.groupby(["model_key", "compression_ratio", "lag"], as_index=False)
        .agg(
            overlap_mean=("overlap", "mean"),
            overlap_std=("overlap", "std"),
            layer_sample_count=("overlap", "count"),
        )
        .sort_values(["model_key", "compression_ratio", "lag"])
    )


def plot(agg: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(3.42, 1.62),
        sharey=True,
        gridspec_kw={"wspace": 0.20},
    )

    legend_handles = []
    legend_labels = []
    for ax, model_key in zip(axes, MODEL_ORDER):
        model_df = agg[agg["model_key"] == model_key]
        for ratio in RATIO_ORDER:
            curve = model_df[model_df["compression_ratio"] == ratio].sort_values("lag")
            line = ax.plot(
                curve["lag"],
                curve["overlap_mean"],
                color=RATIO_COLORS[ratio],
                marker=RATIO_MARKERS[ratio],
                markersize=3.2,
                linewidth=1.45,
                label=RATIO_LABELS[ratio],
            )[0]
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(RATIO_LABELS[ratio])

        ax.set_title(MODEL_LABELS[model_key], fontsize=7.4, pad=2.0)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.72, 1450)
        ax.set_ylim(0.35, 1.0)
        ax.set_xticks([1, 8, 64, 1024])
        ax.set_xticklabels(["1", "8", "64", "1024"])
        ax.set_yticks([1.0, 0.8, 0.6, 0.4])
        ax.set_yticklabels(["1.0", "0.8", "0.6", "0.4"])
        ax.tick_params(axis="both", which="both", direction="in", top=False, right=False, length=2.5, width=0.75)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("KV set overlap")
    fig.supxlabel(r"Step distance $\Delta$", fontsize=8.0, x=PLOT_CENTER_X, y=0.018)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(PLOT_CENTER_X, 1.025),
        columnspacing=1.05,
        handlelength=1.55,
    )
    fig.subplots_adjust(left=PLOT_LEFT, right=PLOT_RIGHT, bottom=0.26, top=0.78, wspace=0.20)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "figure6_token_level_three_model_single_column.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    configure_style()
    lag = load_data()
    agg = aggregate(lag)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / "figure6_token_level_three_model_single_column.csv"
    agg.to_csv(csv_path, index=False)
    figure_path = plot(agg)
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "inputs": [str(path.relative_to(ROOT)) for path in INPUTS],
        "common_lag_max": int(agg["lag"].max()),
        "output_pdf": str(figure_path.relative_to(ROOT)),
        "output_png": str(figure_path.with_suffix(".png").relative_to(ROOT)),
        "paper_csv": str(csv_path.relative_to(ROOT)),
        "note": "Uses common lags across Llama, Mistral, and Qwen after rerunning Llama with decode_steps=2048 and Delta=1024.",
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTDIR / "README.md").write_text(
        "# Token-Level Three-Model Single-Column Figure\n\n"
        "Single-column paper figure for future-token oracle overlap.\n\n"
        "Inputs:\n"
        + "\n".join(f"- `{path.relative_to(ROOT)}`" for path in INPUTS)
        + "\n\n"
        "The plot uses the common lag range across all three models, now including `Delta=1024` after rerunning Llama with `decode_steps=2048`.\n\n"
        f"- PDF: `{figure_path.relative_to(ROOT)}`\n"
        f"- PNG: `{figure_path.with_suffix('.png').relative_to(ROOT)}`\n"
        f"- CSV: `{csv_path.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )
    print(figure_path)


if __name__ == "__main__":
    main()
