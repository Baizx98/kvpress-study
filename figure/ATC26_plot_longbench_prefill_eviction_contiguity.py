from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_longbench_prefill_eviction_contiguity"
ARTIFACT_DIR = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
SUMMARY_JSON = ARTIFACT_DIR / "ATC26_eviction_contiguity_summary.json"
SUMMARY_CSV = ARTIFACT_DIR / "ATC26_eviction_contiguity_summary.csv"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME
# The bundled style file documents the target style, but older matplotlib
# versions in this environment reject its cycler syntax. Apply the needed
# paper-style parameters explicitly to keep the script warning-free.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.2",
        "axes.labelcolor": "0.13",
        "xtick.labelsize": 8,
        "xtick.color": "0.13",
        "ytick.labelsize": 8,
        "ytick.color": "0.13",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "kept": "#0072B2",
    "evicted": "#D9F0E8",
    "protected": "#CC79A7",
    "attention": "#0072B2",
    "random": "#009E73",
    "mismatch": "#D55E00",
    "ratio03": "#2A9D8F",
    "ratio05": "#0072B2",
    "ratio07": "#CC79A7",
}
DATASET_SHORT = {
    "hotpotqa": "H",
    "multifieldqa_en": "M",
    "qasper": "Q",
    "gov_report": "G",
}


def load_runs() -> list[dict]:
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(SUMMARY_JSON)
    return json.loads(SUMMARY_JSON.read_text())["runs"]


def load_mask(run: dict) -> np.ndarray:
    payload = np.load(ROOT / run["score_arrays"])
    kept = payload["kept_mask"].astype(bool)
    protected = payload["protected_mask"].astype(bool)
    row = np.zeros_like(kept, dtype=np.uint8)
    row[kept] = 1
    row[protected] = 2
    return row


def downsample_mask(row: np.ndarray, bins: int = 1024) -> np.ndarray:
    if row.size <= bins:
        return row
    edges = np.linspace(0, row.size, bins + 1).astype(int)
    out = np.zeros(bins, dtype=np.uint8)
    for idx in range(bins):
        segment = row[edges[idx] : edges[idx + 1]]
        if segment.size == 0:
            continue
        counts = np.bincount(segment, minlength=3)
        out[idx] = int(np.argmax(counts))
    return out


def save_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    outputs = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    for output in outputs:
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return outputs


def save_fixed_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    outputs = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    for output in outputs:
        fig.savefig(output, dpi=300)
    return outputs


def plot_mask_heatmap(runs: list[dict]) -> list[Path]:
    main_runs = [run for run in runs if float(run["compression_ratio"]) == 0.5]
    main_runs = sorted(
        main_runs,
        key=lambda item: (item["sample"]["dataset_name"], item["sample"]["dataset_row_index"], item["model_key"]),
    )
    rows = [downsample_mask(load_mask(run)) for run in main_runs]
    width = max(row.size for row in rows)
    matrix = np.full((len(rows), width), 2, dtype=np.uint8)
    for idx, row in enumerate(rows):
        matrix[idx, : row.size] = row

    labels = []
    dataset_counts: dict[str, int] = {}
    for run in main_runs:
        dataset = run["sample"]["dataset_name"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        labels.append(f"{DATASET_SHORT.get(dataset, dataset[:1].upper())}{dataset_counts[dataset]}")

    cmap = ListedColormap([COLORS["evicted"], COLORS["kept"], COLORS["protected"]])
    fig, ax = plt.subplots(figsize=(2.17, 1.8), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.30, top=0.76)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalized token position")
    ax.set_xticks(np.linspace(0, width - 1, 5))
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("Req.")
    ax.tick_params(axis="both", which="both", length=0, top=False, right=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    legend_handles = [
        Patch(facecolor=COLORS["evicted"], edgecolor="#7CB7A6", label="Evict"),
        Patch(facecolor=COLORS["kept"], edgecolor="none", label="Keep"),
        Patch(facecolor=COLORS["protected"], edgecolor="none", label="Prot."),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.42),
        ncol=3,
        handlelength=0.75,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    outputs = save_fixed_figure(fig, OUTDIR / "ATC26_eviction_mask_heatmap_main")
    plt.close(fig)
    return outputs


def plot_run_length(df: pd.DataFrame) -> list[Path]:
    run_df = df.drop_duplicates("run_id").copy()
    grouped = (
        run_df.groupby("compression_ratio", as_index=False)
        .agg(attention_run=("mean_evicted_run_length", "mean"), random_run=("random_mean_evicted_run_length_mean", "mean"))
        .sort_values("compression_ratio")
    )
    mismatch_grouped = (
        df[df["block_size"].eq(16)]
        .groupby("compression_ratio", as_index=False)
        .agg(mismatch=("token_decision_mismatch_rate", "mean"))
        .sort_values("compression_ratio")
    )
    grouped = grouped.merge(mismatch_grouped, on="compression_ratio", how="inner")
    x = np.arange(len(grouped), dtype=float)
    width = 0.28
    fig, ax = plt.subplots(figsize=(1.9, 1.8), constrained_layout=False)
    fig.subplots_adjust(left=0.25, right=0.76, bottom=0.30, top=0.76)
    attn_bars = ax.bar(
        x - width / 2,
        grouped["attention_run"],
        width=width,
        label="Token",
        color=COLORS["attention"],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        grouped["random_run"],
        width=width,
        label="Random",
        color=COLORS["random"],
        edgecolor="white",
        linewidth=0.6,
    )
    for idx, (bar, gain) in enumerate(zip(attn_bars, grouped["attention_run"] / grouped["random_run"])):
        x_pos = bar.get_x() + bar.get_width() / 2
        if idx == 0:
            x_pos += 0.04
        ax.text(
            x_pos,
            bar.get_height() + 0.65,
            f"{gain:.1f}x",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_ylabel("Mean evict-run len.")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in grouped["compression_ratio"]])
    ax.set_xlabel("Evict ratio")
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 15, 30])
    ax.grid(axis="y")
    style_boxed_axis(ax)
    ax.tick_params(axis="x", top=False)

    ax_mis = ax.twinx()
    ax_mis.plot(
        x,
        100.0 * grouped["mismatch"],
        color=COLORS["mismatch"],
        marker="o",
        markersize=3.2,
        linewidth=1.35,
        label="Mismatch",
    )
    ax_mis.set_ylabel("Mismatch (%)", color=COLORS["mismatch"], labelpad=1)
    ax_mis.tick_params(axis="y", labelcolor=COLORS["mismatch"])
    ax_mis.set_ylim(0, 20)
    ax_mis.set_yticks([0, 10, 20])
    style_boxed_axis(ax_mis)
    ax_mis.tick_params(axis="x", top=False)
    ax_mis.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_mis.get_legend_handles_labels()
    ax.legend(
        handles + handles_r,
        labels + labels_r,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.42),
        ncol=3,
        handlelength=0.8,
        columnspacing=0.45,
        borderaxespad=0.0,
    )
    outputs = save_fixed_figure(fig, OUTDIR / "ATC26_evicted_run_length_vs_random")
    plt.close(fig)
    return outputs


def plot_block_mismatch(df: pd.DataFrame) -> list[Path]:
    metrics = [
        ("token_decision_mismatch_rate", "Mismatch"),
        ("false_eviction_rate", "False evict"),
        ("false_keep_rate", "False keep"),
    ]
    grouped = (
        df.groupby(["compression_ratio", "block_size"], as_index=False)
        .agg(**{key: (key, "mean") for key, _ in metrics})
        .sort_values(["compression_ratio", "block_size"])
    )
    block_sizes = sorted(grouped["block_size"].unique())
    ratios = sorted(grouped["compression_ratio"].unique())
    x = np.arange(len(block_sizes), dtype=float)
    width = 0.23
    offsets = np.linspace(-width, width, len(ratios))
    ratio_colors = {
        0.3: COLORS["ratio03"],
        0.5: COLORS["ratio05"],
        0.7: COLORS["ratio07"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.85), constrained_layout=True, sharey=True)
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for offset, ratio in zip(offsets, ratios):
            ratio_df = grouped[grouped["compression_ratio"] == ratio].set_index("block_size")
            values = [100.0 * float(ratio_df.loc[block_size, metric_key]) for block_size in block_sizes]
            ax.bar(
                x + offset,
                values,
                width=width,
                label=f"{int(ratio * 100)}%",
                color=ratio_colors.get(float(ratio), "#0072B2"),
                edgecolor="white",
                linewidth=0.6,
            )
        ax.text(
            0.02,
            0.95,
            metric_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in block_sizes])
        ax.set_xlabel("Block size")
        ax.set_ylim(0, 42)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.grid(axis="y")
    axes[0].set_ylabel("Rate (%)")
    axes[0].legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.22),
        handlelength=0.9,
        columnspacing=0.8,
    )
    outputs = save_figure(fig, OUTDIR / "ATC26_block_projection_mismatch")
    plt.close(fig)
    return outputs


def style_boxed_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("0.2")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, length=3)


def build_heatmap_matrix(runs: list[dict]) -> tuple[np.ndarray, list[str], int]:
    main_runs = [run for run in runs if float(run["compression_ratio"]) == 0.5]
    main_runs = sorted(
        main_runs,
        key=lambda item: (item["sample"]["dataset_name"], item["sample"]["dataset_row_index"], item["model_key"]),
    )
    rows = [downsample_mask(load_mask(run), bins=512) for run in main_runs]
    width = max(row.size for row in rows)
    matrix = np.full((len(rows), width), 2, dtype=np.uint8)
    for idx, row in enumerate(rows):
        matrix[idx, : row.size] = row

    labels = []
    dataset_counts: dict[str, int] = {}
    for run in main_runs:
        dataset = run["sample"]["dataset_name"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        labels.append(f"{DATASET_SHORT.get(dataset, dataset[:1].upper())}{dataset_counts[dataset]}")
    return matrix, labels, width


def plot_two_panel_summary(runs: list[dict], df: pd.DataFrame) -> list[Path]:
    matrix, labels, width = build_heatmap_matrix(runs)
    run_df = df.drop_duplicates("run_id").copy()
    run_grouped = (
        run_df.groupby("compression_ratio", as_index=False)
        .agg(run_len=("mean_evicted_run_length", "mean"))
        .sort_values("compression_ratio")
    )
    mismatch_grouped = (
        df[df["block_size"].eq(16)]
        .groupby("compression_ratio", as_index=False)
        .agg(mismatch=("token_decision_mismatch_rate", "mean"))
        .sort_values("compression_ratio")
    )
    merged = run_grouped.merge(mismatch_grouped, on="compression_ratio", how="inner")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.35, 1.42),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.32},
    )
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.25, top=0.84, wspace=0.36)

    ax_heat = axes[0]
    cmap = ListedColormap([COLORS["evicted"], COLORS["kept"], COLORS["protected"]])
    ax_heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_xticks(np.linspace(0, width - 1, 3))
    ax_heat.set_xticklabels(["0", "50", "100"])
    ax_heat.set_xlabel("Token position (%)", labelpad=1)
    ax_heat.set_ylabel("Req.", labelpad=1)
    ax_heat.grid(False)
    style_boxed_axis(ax_heat)
    legend_handles = [
        Patch(facecolor=COLORS["evicted"], edgecolor="#B0A86E", label="Evict"),
        Patch(facecolor=COLORS["kept"], edgecolor="none", label="Keep"),
        Patch(facecolor=COLORS["protected"], edgecolor="none", label="Prot."),
    ]
    ax_heat.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.32),
        ncol=3,
        handlelength=0.8,
        columnspacing=0.55,
        borderaxespad=0.0,
    )

    ax_run = axes[1]
    x = np.arange(len(merged), dtype=float)
    ax_run.bar(
        x,
        merged["run_len"],
        width=0.58,
        color=COLORS["attention"],
        edgecolor="white",
        linewidth=0.6,
        label="Run len.",
    )
    ax_run.set_xticks(x)
    ax_run.set_xticklabels([f"{int(r * 100)}%" for r in merged["compression_ratio"]])
    ax_run.set_xlabel("Evict ratio", labelpad=1)
    ax_run.set_ylabel("Run len.", color=COLORS["attention"], labelpad=1)
    ax_run.tick_params(axis="y", labelcolor=COLORS["attention"])
    ax_run.set_ylim(0, 30)
    ax_run.set_yticks([0, 10, 20, 30])
    ax_run.grid(axis="y")
    style_boxed_axis(ax_run)

    ax_mis = ax_run.twinx()
    ax_mis.plot(
        x,
        100.0 * merged["mismatch"],
        color=COLORS["mismatch"],
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label="Mismatch",
    )
    ax_mis.set_ylabel("Mismatch (%)", color=COLORS["mismatch"], labelpad=1)
    ax_mis.tick_params(axis="y", labelcolor=COLORS["mismatch"])
    ax_mis.set_ylim(0, 20)
    ax_mis.set_yticks([0, 10, 20])
    style_boxed_axis(ax_mis)
    ax_mis.grid(False)

    handles, labels_left = ax_run.get_legend_handles_labels()
    handles_r, labels_r = ax_mis.get_legend_handles_labels()
    ax_run.legend(
        handles + handles_r,
        labels_left + labels_r,
        loc="upper left",
        bbox_to_anchor=(-0.06, 1.32),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.6,
        borderaxespad=0.0,
    )

    outputs = save_figure(fig, OUTDIR / "ATC26_eviction_contiguity_two_panel")
    plt.close(fig)
    return outputs


def write_readme(figures: list[Path]) -> None:
    lines = "\n".join(f"- `{path.name}`" for path in figures)
    (OUTDIR / "README.md").write_text(
        f"# {EXPERIMENT_NAME}\n\nFigures generated from `{SUMMARY_JSON.relative_to(ROOT)}`.\n\n{lines}\n"
    )


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    df = pd.read_csv(SUMMARY_CSV)
    figures = []
    figures.extend(plot_mask_heatmap(runs))
    figures.extend(plot_run_length(df))
    figures.extend(plot_block_mismatch(df))
    figures.extend(plot_two_panel_summary(runs, df))
    write_readme(figures)
    print(OUTDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
