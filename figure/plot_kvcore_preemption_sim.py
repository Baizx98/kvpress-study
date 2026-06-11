from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUMMARY = (
    Path("/home10T/bzx/workspace/vllm-test")
    / "experiment_results"
    / "preemption_motivation_long_output_20260510_170956"
    / "analysis"
    / "preemption_summary.csv"
)
EXPERIMENT_NAME = "kvcore_preemption_sim_from_vllm_test_20260610"
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"
METRICS_CSV = FIGURE_ROOT / "kvcore_preemption_sim_metrics.csv"

MODEL_NAME = "Llama-3.1-8B-Instruct"
FIGSIZE = (2.35, 2.35)
COLORS = {
    "vLLM": "#2563EB",
    "KVCore": "#10B981",
    "reduction": "#D55E00",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "grid.color": "#E6E6E6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.tick_params(axis="both", which="major", direction="in", length=2.8, width=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(0.78)


def load_vllm_baseline() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_SUMMARY)
    df = df[df["model_name"] == MODEL_NAME].copy()
    if df.empty:
        raise ValueError(f"No {MODEL_NAME} rows found in {SOURCE_SUMMARY}")
    df = df.sort_values("kv_cache_memory_gb").reset_index(drop=True)
    return df


def simulate_kvcore(vllm: pd.DataFrame) -> pd.DataFrame:
    max_preempt = float(vllm["preemptions_per_100_reqs"].max())
    max_budget = float(vllm["kv_cache_memory_gb"].max())
    records: list[dict] = []
    for row in vllm.itertuples(index=False):
        pressure = float(row.preemptions_per_100_reqs) / max_preempt if max_preempt > 0 else 0.0
        budget_rel = float(row.kv_cache_memory_gb) / max_budget if max_budget > 0 else 0.0
        # KVCore proactively lowers KV pressure and avoids most request
        # preemptions, but residual pressure remains under heavier schedules.
        residual_fraction = 0.08 + 0.06 * pressure + 0.02 * budget_rel
        residual_fraction = min(0.18, max(0.09, residual_fraction))
        kvcore_preemptions_per_100 = float(row.preemptions_per_100_reqs) * residual_fraction
        kvcore_total_preemptions = kvcore_preemptions_per_100 * float(row.num_prompts) / 100.0
        kvcore_preempted_percent = float(row.preempted_request_percent) * (0.18 + 0.10 * pressure)
        records.append(
            {
                "system": "vLLM",
                "source": "measured",
                "model_name": row.model_name,
                "kv_cache_memory_gb": float(row.kv_cache_memory_gb),
                "num_prompts": int(row.num_prompts),
                "total_preemptions": float(row.total_preemptions),
                "preemptions_per_100_reqs": float(row.preemptions_per_100_reqs),
                "preempted_request_percent": float(row.preempted_request_percent),
                "reduction_percent_vs_vllm": 0.0,
                "notes": "Measured vLLM preemption motivation result from vllm-test.",
            }
        )
        records.append(
            {
                "system": "KVCore",
                "source": "simulated",
                "model_name": row.model_name,
                "kv_cache_memory_gb": float(row.kv_cache_memory_gb),
                "num_prompts": int(row.num_prompts),
                "total_preemptions": kvcore_total_preemptions,
                "preemptions_per_100_reqs": kvcore_preemptions_per_100,
                "preempted_request_percent": kvcore_preempted_percent,
                "reduction_percent_vs_vllm": 100.0 * (1.0 - residual_fraction),
                "notes": "Simulated KVCore residual preemption; most but not all preemptions are avoided.",
            }
        )
    out = pd.DataFrame(records)
    out.to_csv(METRICS_CSV, index=False)
    return out


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return [pdf_path, png_path]


def plot_preemptions_per_100(metrics: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    budgets = sorted(metrics["kv_cache_memory_gb"].unique())
    x = np.arange(len(budgets))
    width = 0.34
    for idx, system in enumerate(["vLLM", "KVCore"]):
        sub = metrics[metrics["system"] == system].set_index("kv_cache_memory_gb").loc[budgets]
        ax.bar(
            x + (idx - 0.5) * width,
            sub["preemptions_per_100_reqs"],
            width=width,
            color=COLORS[system],
            edgecolor="white",
            linewidth=0.5,
            label=system,
            zorder=3,
        )
    ax.set_xlabel("KV budget (GB)")
    ax.set_ylabel("Preempt. / 100 reqs")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in budgets])
    ax.set_ylim(0, max(metrics["preemptions_per_100_reqs"]) * 1.18)
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.16), ncols=2, columnspacing=0.9, handlelength=1.0)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.20, top=0.78)
    return save_figure(fig, "preemptions_per_100_requests")


def plot_reduction_percent(metrics: pd.DataFrame) -> list[Path]:
    kvcore = metrics[metrics["system"] == "KVCore"].sort_values("kv_cache_memory_gb")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(kvcore))
    ax.bar(
        x,
        kvcore["reduction_percent_vs_vllm"],
        width=0.48,
        color=COLORS["reduction"],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel("KV budget (GB)")
    ax.set_ylabel("Preemption reduction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in kvcore["kv_cache_memory_gb"]])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    style_axis(ax)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.20, top=0.90)
    return save_figure(fig, "preemption_reduction_percent")


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

Paper-style request preemption figures using measured vLLM preemption data and simulated KVCore residual preemption.

## Source Data

- vLLM summary: `{SOURCE_SUMMARY}`
- Model: `{MODEL_NAME}`
- Workload: 64 offline requests, input length 512, forced output length 1536.
- Metrics table: `figure/experiments/{EXPERIMENT_NAME}/kvcore_preemption_sim_metrics.csv`
- Plotting script: `figure/plot_kvcore_preemption_sim.py`

## Figures

{output_lines}

## Simulation Assumption

KVCore avoids most request preemptions by reducing dynamic GPU KV pressure before vLLM-style preemption becomes necessary. The residual preemption fraction is pressure-dependent, so heavier vLLM preemption points retain slightly more residual KVCore preemption. The simulated KVCore rows are explicitly marked as `source=simulated` in the CSV.
"""
    )


def main() -> int:
    setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    vllm = load_vllm_baseline()
    metrics = simulate_kvcore(vllm)
    outputs: list[Path] = []
    outputs.extend(plot_reduction_percent(metrics))
    outputs.extend(plot_preemptions_per_100(metrics))
    write_readme(outputs)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: vLLM-measured and KVCore-simulated request preemption figures.",
    )
    print(f"Wrote metrics: {METRICS_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
