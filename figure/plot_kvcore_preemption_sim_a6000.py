from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    Path("/home10T/bzx/workspace/vllm-test")
    / "experiment_results"
    / "a6000_motivation_20260530_215742_large_bs_long_out"
)
SOURCE_SUMMARY = SOURCE_ROOT / "analysis" / "a6000_preemption_summary.csv"
EXPERIMENT_NAME = "kvcore_preemption_sim_a6000_large_bs_long_out_20260610"
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"
METRICS_CSV = FIGURE_ROOT / "kvcore_preemption_sim_a6000_metrics.csv"

FIGSIZE = (2.35, 2.35)
COMBINED_FIGSIZE = (3.35, 2.75)
OUTPUT_ORDER = [1024, 2048, 4096, 6144]
OUTPUT_LABELS = {1024: "1K", 2048: "2K", 4096: "4K", 6144: "6K"}
BATCH_ORDER = [12, 16, 20, 24]
PAPER_PALETTE = {
    "blue": "#3B6FB6",
    "teal": "#00A6A6",
    "coral": "#E56B5D",
    "gold": "#D89C00",
    "violet": "#8E6BBE",
    "gray": "#4D4D4D",
}
BATCH_COLORS = {
    12: PAPER_PALETTE["blue"],
    16: PAPER_PALETTE["teal"],
    20: PAPER_PALETTE["gold"],
    24: PAPER_PALETTE["coral"],
}
BATCH_MARKERS = {12: "o", 16: "s", 20: "^", 24: "D"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.4,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.3,
            "grid.color": "#E6E6E6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(0.8)


def style_twin_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="y", which="major", direction="in", length=3.0, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(0.8)


def load_vllm_sweep() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_SUMMARY)
    df = df[df["run_name"].str.startswith("figure3_sweep_")].copy()
    df = df[df["output_len"].isin(OUTPUT_ORDER) & df["batch_size"].isin(BATCH_ORDER)].copy()
    if df.empty:
        raise ValueError(f"No figure3_sweep rows found in {SOURCE_SUMMARY}")
    return df.sort_values(["batch_size", "output_len"]).reset_index(drop=True)


def simulate_kvcore(vllm: pd.DataFrame) -> pd.DataFrame:
    max_preempt = float(vllm["preemptions_per_100_requests"].max())
    records: list[dict] = []
    for row in vllm.itertuples(index=False):
        vllm_per100 = float(row.preemptions_per_100_requests)
        vllm_ratio = float(row.preempted_request_ratio) * 100.0
        pressure = vllm_per100 / max_preempt if max_preempt > 0 else 0.0
        batch_pressure = (int(row.batch_size) - min(BATCH_ORDER)) / (max(BATCH_ORDER) - min(BATCH_ORDER))
        output_pressure = OUTPUT_ORDER.index(int(row.output_len)) / (len(OUTPUT_ORDER) - 1)
        residual_fraction = 0.07 + 0.055 * pressure + 0.025 * batch_pressure + 0.025 * output_pressure
        residual_fraction = min(0.18, max(0.075, residual_fraction))
        kvcore_per100 = vllm_per100 * residual_fraction
        kvcore_ratio = vllm_ratio * min(0.30, residual_fraction + 0.08)
        reduction = np.nan if vllm_per100 == 0 else 100.0 * (1.0 - residual_fraction)
        base_common = {
            "input_len": int(row.input_len),
            "output_len": int(row.output_len),
            "batch_size": int(row.batch_size),
            "kv_cache_memory_gb": float(row.kv_cache_memory_gb),
            "num_prompts": int(row.num_prompts),
        }
        records.append(
            {
                **base_common,
                "system": "vLLM",
                "source": "measured",
                "total_preemptions": float(row.total_preemptions),
                "preemptions_per_100_requests": vllm_per100,
                "preempted_request_percent": vllm_ratio,
                "preemption_reduction_percent": 0.0,
                "notes": "Measured A6000 vLLM large-bs-long-output preemption result.",
            }
        )
        records.append(
            {
                **base_common,
                "system": "KVCore",
                "source": "simulated",
                "total_preemptions": kvcore_per100 * float(row.num_prompts) / 100.0,
                "preemptions_per_100_requests": kvcore_per100,
                "preempted_request_percent": kvcore_ratio,
                "preemption_reduction_percent": reduction,
                "notes": "Simulated residual preemption; KVCore avoids most but not all vLLM preemptions.",
            }
        )
    metrics = pd.DataFrame(records)
    metrics.to_csv(METRICS_CSV, index=False)
    return metrics


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return [pdf_path, png_path]


def plot_preemptions_per_100(metrics: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for batch_size in BATCH_ORDER:
        for system, linestyle, marker_fill in [("vLLM", "-", True), ("KVCore", "--", False)]:
            group = metrics[(metrics["batch_size"] == batch_size) & (metrics["system"] == system)]
            group = group.set_index("output_len").reindex(OUTPUT_ORDER).reset_index()
            ax.plot(
                group["output_len"],
                group["preemptions_per_100_requests"],
                color=BATCH_COLORS[batch_size],
                marker=BATCH_MARKERS[batch_size],
                linestyle=linestyle,
                linewidth=1.7,
                markersize=4.3,
                markerfacecolor=BATCH_COLORS[batch_size] if marker_fill else "white",
                markeredgecolor=BATCH_COLORS[batch_size],
                markeredgewidth=0.8,
                label=f"bs={batch_size}" if system == "vLLM" else None,
            )
    ax.plot([], [], color=PAPER_PALETTE["gray"], linestyle="-", label="vLLM")
    ax.plot([], [], color=PAPER_PALETTE["gray"], linestyle="--", label="KVCore")
    ax.set_xlabel("Output length")
    ax.set_ylabel("Preempt. / 100 reqs")
    ax.set_xticks(OUTPUT_ORDER)
    ax.set_xticklabels([OUTPUT_LABELS[o] for o in OUTPUT_ORDER])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    style_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=0.65,
        handlelength=1.05,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.25, right=0.99, bottom=0.18, top=0.70)
    return save_figure(fig, "a6000_preemptions_per_100_requests")


def plot_reduction_percent(metrics: pd.DataFrame) -> list[Path]:
    kvcore = metrics[metrics["system"] == "KVCore"].copy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for batch_size, group in kvcore.groupby("batch_size"):
        group = group.set_index("output_len").reindex(OUTPUT_ORDER).reset_index()
        ax.plot(
            group["output_len"],
            group["preemption_reduction_percent"].astype(float),
            color=BATCH_COLORS[int(batch_size)],
            marker=BATCH_MARKERS[int(batch_size)],
            linewidth=1.7,
            markersize=4.3,
            markeredgewidth=0.5,
            markeredgecolor="white",
            label=f"bs={int(batch_size)}",
        )
    ax.set_xlabel("Output length")
    ax.set_ylabel("Reduction (%)")
    ax.set_xticks(OUTPUT_ORDER)
    ax.set_xticklabels([OUTPUT_LABELS[o] for o in OUTPUT_ORDER])
    ax.set_ylim(80, 94)
    ax.set_yticks([80, 85, 90])
    style_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.2,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.18, top=0.72)
    return save_figure(fig, "a6000_preemption_reduction_percent")


def plot_combined_preemption_reduction(metrics: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=COMBINED_FIGSIZE, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    x = np.arange(len(OUTPUT_ORDER), dtype=float)
    x_labels = [OUTPUT_LABELS[o] for o in OUTPUT_ORDER]
    bar_width = 0.34
    vllm = metrics[metrics["system"] == "vLLM"].set_index(["batch_size", "output_len"])
    kvcore = metrics[metrics["system"] == "KVCore"].set_index(["batch_size", "output_len"])

    vllm_bars = None
    kvcore_bars = None
    reduction_line = None
    twin_axes: list[plt.Axes] = []
    for panel_idx, (ax, batch_size) in enumerate(zip(axes_flat, BATCH_ORDER, strict=True)):
        ax2 = ax.twinx()
        twin_axes.append(ax2)
        vllm_y = np.array(
            [vllm.loc[(batch_size, o), "preemptions_per_100_requests"] for o in OUTPUT_ORDER],
            dtype=float,
        )
        kvcore_y = np.array(
            [kvcore.loc[(batch_size, o), "preemptions_per_100_requests"] for o in OUTPUT_ORDER],
            dtype=float,
        )
        reduction_y = np.array(
            [kvcore.loc[(batch_size, o), "preemption_reduction_percent"] for o in OUTPUT_ORDER],
            dtype=float,
        )
        vllm_bars = ax.bar(
            x - bar_width / 2,
            vllm_y,
            width=bar_width,
            color="#E56B5D",
            edgecolor="white",
            linewidth=0.5,
            label="vLLM",
            zorder=3,
        )
        kvcore_bars = ax.bar(
            x + bar_width / 2,
            kvcore_y,
            width=bar_width,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.5,
            label="KVCore",
            zorder=3,
        )
        (line,) = ax2.plot(
            x,
            reduction_y,
            color="#009E73",
            marker="D",
            linestyle="-",
            linewidth=1.35,
            markersize=2.9,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label="Reduction",
            zorder=4,
        )
        if reduction_line is None:
            reduction_line = line
        ax.set_title(f"bs={batch_size}", fontsize=7.4, pad=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(-0.6, len(OUTPUT_ORDER) - 0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax2.set_ylim(0, 100)
        ax2.set_yticks([0, 50, 100])
        style_axis(ax)
        style_twin_axis(ax2)
        ax2.grid(False)
        ax.tick_params(axis="x", labelsize=6.5, pad=1.2)
        ax.tick_params(axis="y", labelsize=6.5, pad=1.2)
        ax2.tick_params(axis="y", labelsize=6.5, pad=1.2)
        if panel_idx % 2 == 1:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax2.tick_params(axis="y", labelright=False)
        if panel_idx < 2:
            ax.tick_params(axis="x", labelbottom=False)

    fig.legend(
        [vllm_bars, kvcore_bars, reduction_line],
        ["vLLM", "KVCore", "Reduction"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.15,
        handletextpad=0.35,
        fontsize=7.0,
    )
    fig.supxlabel("Output length", fontsize=7.8, y=0.045)
    fig.supylabel("Preempt. / 100 reqs", fontsize=7.8, x=0.052)
    fig.text(0.985, 0.48, "Reduction (%)", rotation=-90, ha="right", va="center", fontsize=7.8)
    fig.subplots_adjust(left=0.155, right=0.86, bottom=0.17, top=0.80, wspace=0.12, hspace=0.32)
    return save_figure(fig, "a6000_preemption_combined_bar_reduction")


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

Request preemption figures using the A6000 large-batch long-output vLLM motivation data and simulated KVCore residual preemption.

## Source Data

- Source root: `{SOURCE_ROOT}`
- vLLM summary: `{SOURCE_SUMMARY}`
- Model: `/Tan/model/Llama-3.1-8B-Instruct`
- Workload: RULER 8192 prompts truncated to 3072 input tokens.
- Sweep: batch sizes `12`, `16`, `20`, `24`; output lengths `1K`, `2K`, `4K`, `6K`; KV budget `10 GB`.
- Metrics table: `figure/experiments/{EXPERIMENT_NAME}/kvcore_preemption_sim_a6000_metrics.csv`
- Plotting script: `figure/plot_kvcore_preemption_sim_a6000.py`

## Figures

{output_lines}

## Simulation Assumption

KVCore avoids most but not all vLLM request preemptions by reducing dynamic GPU KV pressure. Residual preemption increases slightly under higher vLLM pressure, larger batch size, and longer output length. Points where vLLM has zero preemptions have undefined reduction and are omitted from the reduction-percent line.
"""
    )


def main() -> int:
    setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    vllm = load_vllm_sweep()
    metrics = simulate_kvcore(vllm)
    outputs: list[Path] = []
    outputs.extend(plot_combined_preemption_reduction(metrics))
    outputs.extend(plot_reduction_percent(metrics))
    outputs.extend(plot_preemptions_per_100(metrics))
    write_readme(outputs)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: A6000 vLLM-measured and KVCore-simulated request preemption figures.",
    )
    print(f"Wrote metrics: {METRICS_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
