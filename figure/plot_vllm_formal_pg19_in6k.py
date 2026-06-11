from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "vllm_formal_pg19_in6k"
SOURCE_EXPERIMENT = "end2end_serving_kvcore_vllm_infinigen_longreq"
SOURCE_RAW_DIR = (
    REPO_ROOT
    / "evaluation"
    / "results"
    / "experiments"
    / SOURCE_EXPERIMENT
    / "artifacts"
    / "raw"
    / "vllm"
)
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"
METRICS_CSV = FIGURE_ROOT / "vllm_formal_pg19_in6k_metrics.csv"

MODEL_ORDER = [
    "llama31_8b_instruct",
    "qwen3_8b",
    "mistral_7b_instruct_v03",
]
MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B",
    "qwen3_8b": "Qwen3-8B",
    "mistral_7b_instruct_v03": "Mistral-7B",
}
OUTPUT_ORDER = [1024, 2048, 6144]
OUTPUT_LABELS = {
    1024: "out1k",
    2048: "out2k",
    6144: "out6k",
}
OUTPUT_COLORS = {
    1024: "#0072B2",
    2048: "#D55E00",
    6144: "#009E73",
}
OUTPUT_MARKERS = {
    1024: "o",
    2048: "s",
    6144: "^",
}
BATCH_ORDER = [1, 8, 16, 24]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.65, color="#E6E6E6")
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [pdf_path, png_path]


def read_vllm_raw() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(SOURCE_RAW_DIR.glob("vllm__*__pg19__in6k_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                obj["source_file"] = path.name
                rows.append(obj)
    if not rows:
        raise FileNotFoundError(f"No vLLM raw rows found under {SOURCE_RAW_DIR}")
    return pd.DataFrame(rows)


def percentile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean.to_numpy(dtype=float), q))


def aggregate_vllm(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    group_cols = ["model_key", "output_len_bucket", "batch_size"]
    for (model_key, output_len, batch_size), sub_all in df.groupby(group_cols, sort=False):
        sub = sub_all[sub_all["status"] == "completed"].copy()
        duration = float(sub["finish_time_s"].max() - sub["submit_time_s"].min()) if not sub.empty else float("nan")
        decode_throughput = float(sub["actual_output_len"].sum() / duration) if duration > 0 else float("nan")
        request_throughput = float(len(sub) / duration) if duration > 0 else float("nan")
        records.append(
            {
                "model_key": model_key,
                "model_label": MODEL_LABELS.get(model_key, model_key),
                "output_len_bucket": int(output_len),
                "output_label": OUTPUT_LABELS.get(int(output_len), str(output_len)),
                "batch_size": int(batch_size),
                "num_requests": int(len(sub_all)),
                "num_completed": int(len(sub)),
                "num_failed": int((sub_all["status"] != "completed").sum()),
                "status": "ok" if len(sub) > 0 and (sub_all["status"] != "completed").sum() == 0 else "oom",
                "median_ttft_s": percentile(sub["ttft_s"], 50),
                "p95_ttft_s": percentile(sub["ttft_s"], 95),
                "median_e2e_s": percentile(sub["e2e_latency_s"], 50),
                "p95_e2e_s": percentile(sub["e2e_latency_s"], 95),
                "median_tpot_ms": percentile(sub["tpot_ms"], 50),
                "p95_tpot_ms": percentile(sub["tpot_ms"], 95),
                "gpu_peak_memory_gb": percentile(sub["gpu_peak_memory_gb"], 50),
                "decode_throughput_tok_s": decode_throughput,
                "request_throughput_req_s": request_throughput,
            }
        )
    metrics = pd.DataFrame(records)
    metrics = metrics.sort_values(["model_key", "output_len_bucket", "batch_size"]).reset_index(drop=True)
    metrics.to_csv(METRICS_CSV, index=False)
    return metrics


def plot_missing_points(ax: plt.Axes, missing_batches: list[int]) -> None:
    if not missing_batches:
        return
    ymin, ymax = ax.get_ylim()
    y = ymin + (ymax - ymin) * 0.04
    for batch_size in missing_batches:
        ax.scatter(
            [batch_size],
            [y],
            marker="x",
            s=48,
            linewidths=1.6,
            color="#8C2D04",
            zorder=5,
        )


def plot_metric_panel(ax: plt.Axes, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    for output_len in OUTPUT_ORDER:
        sub = data[data["output_len_bucket"] == output_len].set_index("batch_size").reindex(BATCH_ORDER)
        ok_mask = sub["status"].eq("ok")
        y = sub[metric].astype(float).where(ok_mask).to_numpy()
        ax.plot(
            BATCH_ORDER,
            y,
            color=OUTPUT_COLORS[output_len],
            marker=OUTPUT_MARKERS[output_len],
            linewidth=1.9,
            markersize=5.0,
            label=OUTPUT_LABELS[output_len],
        )
    ax.set_xticks(BATCH_ORDER)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    style_axis(ax)
    missing = sorted(data.loc[data["status"] != "ok", "batch_size"].unique().tolist())
    plot_missing_points(ax, missing)


def plot_model_summary(metrics: pd.DataFrame, model_key: str) -> list[Path]:
    data = metrics[metrics["model_key"] == model_key]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    panels = [
        (axes[0, 0], "median_ttft_s", "Median TTFT (s)"),
        (axes[0, 1], "median_e2e_s", "Median E2E latency (s)"),
        (axes[1, 0], "median_tpot_ms", "Median TPOT (ms/token)"),
        (axes[1, 1], "decode_throughput_tok_s", "Decode throughput (tok/s)"),
    ]
    for ax, metric, ylabel in panels:
        plot_metric_panel(ax, data, metric, ylabel)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=OUTPUT_COLORS[o],
            marker=OUTPUT_MARKERS[o],
            linewidth=1.9,
            markersize=5.0,
            label=OUTPUT_LABELS[o],
        )
        for o in OUTPUT_ORDER
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#8C2D04",
            marker="x",
            linestyle="None",
            linewidth=0,
            markersize=6.0,
            label="OOM / failed",
        )
    )
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
    )
    fig.suptitle(f"{MODEL_LABELS[model_key]}: vLLM PG19 in6k", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.79, wspace=0.30, hspace=0.38)
    return save_figure(fig, f"vllm_pg19_in6k_{model_key}_by_bs_output")


def plot_model_comparison_heatmap(metrics: pd.DataFrame) -> list[Path]:
    labels = [f"{OUTPUT_LABELS[o]}\nbs{b}" for o in OUTPUT_ORDER for b in BATCH_ORDER]
    matrix = []
    annotations: list[list[str]] = []
    for model_key in MODEL_ORDER:
        row = []
        ann_row = []
        for output_len in OUTPUT_ORDER:
            for batch_size in BATCH_ORDER:
                match = metrics[
                    (metrics["model_key"] == model_key)
                    & (metrics["output_len_bucket"] == output_len)
                    & (metrics["batch_size"] == batch_size)
                ]
                if match.empty or match.iloc[0]["status"] != "ok":
                    row.append(np.nan)
                    ann_row.append("OOM")
                else:
                    row.append(float(match.iloc[0]["median_tpot_ms"]))
                    ann_row.append("")
        matrix.append(row)
        annotations.append(ann_row)

    fig, ax = plt.subplots(figsize=(7.8, 2.65))
    arr = np.array(matrix, dtype=float)
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_title("Median TPOT by model, output length, and batch size")
    for y, row in enumerate(annotations):
        for x, text in enumerate(row):
            if text:
                ax.text(x, y, text, ha="center", va="center", color="#8C2D04", fontsize=8, fontweight="bold")
    for x in np.arange(-0.5, len(labels), 1):
        ax.axvline(x, color="white", linewidth=0.8)
    for y in np.arange(-0.5, len(MODEL_ORDER), 1):
        ax.axhline(y, color="white", linewidth=0.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Median TPOT (ms/token)")
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.24, top=0.84)
    return save_figure(fig, "vllm_pg19_in6k_all_models_tpot_heatmap")


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

Measured vLLM-only figures for the formal PG19 in6k serving sweep.
The three models are plotted separately. Within each model figure, batch size is the x-axis and output length is encoded by color and marker.

## Source Data

- Raw JSONL: `evaluation/results/experiments/{SOURCE_EXPERIMENT}/artifacts/raw/vllm/`
- Metrics CSV: `figure/experiments/{EXPERIMENT_NAME}/vllm_formal_pg19_in6k_metrics.csv`
- Plotting script: `figure/plot_vllm_formal_pg19_in6k.py`

## Figures

{output_lines}

## Metric Definitions

- Median TTFT: median `ttft_s` over completed requests.
- Median E2E latency: median `e2e_latency_s` over completed requests.
- Median TPOT: median `tpot_ms` over completed requests.
- Decode throughput: `sum(actual_output_len) / (max(finish_time_s) - min(submit_time_s))` for completed requests in the point.

## Notes

- Warmup requests are excluded by the runner and therefore excluded from these plots.
- Failed points are shown as `OOM / failed` markers instead of being interpolated.
- The Llama out2k bs8 point uses the successful rerun raw file from 2026-06-09.
"""
    )


def main() -> int:
    setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = aggregate_vllm(read_vllm_raw())
    outputs: list[Path] = []
    for model_key in MODEL_ORDER:
        outputs.extend(plot_model_summary(metrics, model_key))
    outputs.extend(plot_model_comparison_heatmap(metrics))
    write_readme(outputs)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: measured vLLM PG19 in6k figures split by model, batch size, and output length.",
    )
    print(f"Wrote metrics: {METRICS_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
