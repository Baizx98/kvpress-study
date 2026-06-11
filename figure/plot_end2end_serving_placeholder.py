from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "end2end_serving_placeholder"
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
METRICS_CSV = FIGURE_ROOT / "placeholder_metrics.csv"

METHOD_ORDER = ["vLLM", "InfiniGen (sim.)", "KVCore (sim.)"]
METHOD_COLORS = {
    "vLLM": "#4D4D4D",
    "InfiniGen (sim.)": "#D55E00",
    "KVCore (sim.)": "#0072B2",
}
METHOD_MARKERS = {
    "vLLM": "o",
    "InfiniGen (sim.)": "s",
    "KVCore (sim.)": "^",
}
MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B",
    "qwen3_8b": "Qwen3-8B",
    "mistral_7b_instruct_v03": "Mistral-7B",
}
OUTPUT_LABELS = {
    1024: "out1k",
    2048: "out2k",
    6144: "out6k",
}
MODEL_ORDER = ["llama31_8b_instruct", "qwen3_8b", "mistral_7b_instruct_v03"]
OUTPUT_ORDER = [1024, 2048, 6144]
BATCH_ORDER = [1, 8, 16, 24]
LLAMA_KEY = "llama31_8b_instruct"


def setup_style() -> None:
    # Keep the local research-figure style guide's palette and typography while
    # avoiding dependence on an external mplstyle parser version.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, alpha=0.32)
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
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
                "method": "vLLM",
                "source": "measured",
                "model_key": model_key,
                "output_len_bucket": int(output_len),
                "batch_size": int(batch_size),
                "num_requests": int(len(sub_all)),
                "num_completed": int(len(sub)),
                "num_failed": int((sub_all["status"] != "completed").sum()),
                "status": "ok" if len(sub) > 0 and (sub_all["status"] != "completed").sum() == 0 else "oom",
                "median_ttft_s": percentile(sub["ttft_s"], 50),
                "p95_ttft_s": percentile(sub["ttft_s"], 95),
                "p99_ttft_s": percentile(sub["ttft_s"], 99),
                "median_e2e_s": percentile(sub["e2e_latency_s"], 50),
                "p95_e2e_s": percentile(sub["e2e_latency_s"], 95),
                "p99_e2e_s": percentile(sub["e2e_latency_s"], 99),
                "median_tpot_ms": percentile(sub["tpot_ms"], 50),
                "p95_tpot_ms": percentile(sub["tpot_ms"], 95),
                "p99_tpot_ms": percentile(sub["tpot_ms"], 99),
                "gpu_peak_memory_gb": percentile(sub["gpu_peak_memory_gb"], 50),
                "decode_throughput_tok_s": decode_throughput,
                "request_throughput_req_s": request_throughput,
            }
        )
    return pd.DataFrame(records)


def reference_value(measured: pd.DataFrame, model_key: str, output_len: int, batch_size: int, metric: str) -> float:
    match = measured[
        (measured["model_key"] == model_key)
        & (measured["output_len_bucket"] == output_len)
        & (measured["batch_size"] == batch_size)
    ]
    if not match.empty:
        value = float(match.iloc[0][metric])
        if np.isfinite(value):
            return value

    same_output = measured[
        (measured["model_key"] == model_key)
        & (measured["output_len_bucket"] == output_len)
        & np.isfinite(measured[metric])
    ].sort_values("batch_size")
    if same_output.empty:
        return float("nan")
    lower = same_output[same_output["batch_size"] < batch_size]
    base = lower.iloc[-1] if not lower.empty else same_output.iloc[-1]
    base_bs = max(float(base["batch_size"]), 1.0)
    base_value = float(base[metric])
    if "throughput" in metric:
        return base_value * min(1.18, (batch_size / base_bs) ** 0.18)
    if metric == "gpu_peak_memory_gb":
        return min(46.5, base_value * 1.08)
    return base_value * min(1.45, (batch_size / base_bs) ** 0.30)


def simulate_methods(measured: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    keys = sorted(measured["model_key"].unique())
    for model_key in keys:
        for output_len in OUTPUT_ORDER:
            for batch_size in BATCH_ORDER:
                for method in ["InfiniGen (sim.)", "KVCore (sim.)"]:
                    stress = 1.0 + 0.08 * BATCH_ORDER.index(batch_size) + 0.10 * OUTPUT_ORDER.index(output_len)
                    if method == "KVCore (sim.)":
                        throughput_gain = 1.18 + 0.11 * BATCH_ORDER.index(batch_size) + 0.15 * OUTPUT_ORDER.index(output_len)
                        e2e_scale = max(0.50, 0.78 - 0.06 * stress)
                        ttft_scale = max(0.70, 0.92 - 0.03 * stress)
                        tpot_scale = max(0.46, 0.72 - 0.08 * stress)
                        mem_scale = max(0.55, 0.78 - 0.04 * stress)
                    else:
                        throughput_gain = 1.07 + 0.06 * BATCH_ORDER.index(batch_size) + 0.07 * OUTPUT_ORDER.index(output_len)
                        e2e_scale = max(0.66, 0.90 - 0.04 * stress)
                        ttft_scale = max(0.82, 0.98 - 0.02 * stress)
                        tpot_scale = max(0.62, 0.88 - 0.05 * stress)
                        mem_scale = max(0.66, 0.86 - 0.04 * stress)

                    rec = {
                        "method": method,
                        "source": "simulated_placeholder",
                        "model_key": model_key,
                        "output_len_bucket": output_len,
                        "batch_size": batch_size,
                        "num_requests": int(48 if batch_size == 24 else 32),
                        "num_completed": int(48 if batch_size == 24 else 32),
                        "num_failed": 0,
                        "status": "ok",
                    }
                    for metric in ["decode_throughput_tok_s", "request_throughput_req_s"]:
                        rec[metric] = reference_value(measured, model_key, output_len, batch_size, metric) * throughput_gain
                    for metric in ["median_e2e_s", "p95_e2e_s", "p99_e2e_s"]:
                        rec[metric] = reference_value(measured, model_key, output_len, batch_size, metric) * e2e_scale
                    for metric in ["median_ttft_s", "p95_ttft_s", "p99_ttft_s"]:
                        rec[metric] = reference_value(measured, model_key, output_len, batch_size, metric) * ttft_scale
                    for metric in ["median_tpot_ms", "p95_tpot_ms", "p99_tpot_ms"]:
                        rec[metric] = reference_value(measured, model_key, output_len, batch_size, metric) * tpot_scale
                    rec["gpu_peak_memory_gb"] = reference_value(measured, model_key, output_len, batch_size, "gpu_peak_memory_gb") * mem_scale
                    records.append(rec)
    return pd.DataFrame(records)


def build_metrics() -> pd.DataFrame:
    raw = read_vllm_raw()
    measured = aggregate_vllm(raw)
    simulated = simulate_methods(measured)
    metrics = pd.concat([measured, simulated], ignore_index=True)
    metrics["output_label"] = metrics["output_len_bucket"].map(OUTPUT_LABELS)
    metrics["model_label"] = metrics["model_key"].map(MODEL_LABELS)
    metrics.to_csv(METRICS_CSV, index=False)
    return metrics


def add_placeholder_note(fig: plt.Figure, y: float = 0.01) -> None:
    fig.text(
        0.5,
        y,
        "Measured: vLLM. Simulated placeholders: KVCore, InfiniGen.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )


def plot_throughput_vs_batch(metrics: pd.DataFrame) -> list[Path]:
    data = metrics[metrics["model_key"] == LLAMA_KEY]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.55), sharey=True)
    for ax, output_len in zip(axes, OUTPUT_ORDER):
        sub = data[data["output_len_bucket"] == output_len]
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("batch_size")
            y = m["decode_throughput_tok_s"].to_numpy(dtype=float)
            if method == "vLLM":
                y = np.where(m["status"].to_numpy() == "ok", y, np.nan)
            ax.plot(
                m["batch_size"],
                y,
                label=method,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.8,
                markersize=4.2,
            )
        oom = sub[(sub["method"] == "vLLM") & (sub["status"] != "ok")]
        for _, row in oom.iterrows():
            ax.text(row["batch_size"], ax.get_ylim()[0], "OOM", ha="center", va="bottom", fontsize=8, color="#8C2D04")
        ax.set_title(OUTPUT_LABELS[output_len])
        ax.set_xlabel("Batch size")
        ax.set_xticks(BATCH_ORDER)
        style_axis(ax)
    axes[0].set_ylabel("Decode throughput (tok/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    add_placeholder_note(fig, y=-0.035)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.78, wspace=0.12)
    return save_figure(fig, "throughput_vs_batch_llama31")


def plot_metric_grid_by_model(
    metrics: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    stem: str,
    higher_is_better: bool,
) -> list[Path]:
    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(OUTPUT_ORDER),
        figsize=(8.6, 6.6),
        sharex=True,
        sharey="row",
    )
    for row_idx, model_key in enumerate(MODEL_ORDER):
        for col_idx, output_len in enumerate(OUTPUT_ORDER):
            ax = axes[row_idx, col_idx]
            sub = metrics[(metrics["model_key"] == model_key) & (metrics["output_len_bucket"] == output_len)]
            for method in METHOD_ORDER:
                m = sub[sub["method"] == method].sort_values("batch_size")
                y = m[metric].to_numpy(dtype=float)
                if method == "vLLM":
                    y = np.where(m["status"].to_numpy() == "ok", y, np.nan)
                ax.plot(
                    m["batch_size"],
                    y,
                    label=method,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=1.55,
                    markersize=3.6,
                )
            oom = sub[(sub["method"] == "vLLM") & (sub["status"] != "ok")]
            for _, oom_row in oom.iterrows():
                y_top = ax.get_ylim()[1]
                ax.text(
                    oom_row["batch_size"],
                    y_top * 0.97,
                    "OOM",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="#8C2D04",
                )
            if row_idx == 0:
                ax.set_title(OUTPUT_LABELS[output_len])
            if col_idx == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model_key]}\n{ylabel}")
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Batch size")
            ax.set_xticks(BATCH_ORDER)
            style_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    direction = "higher is better" if higher_is_better else "lower is better"
    fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.035, f"{direction}. Measured: vLLM. Simulated placeholders: KVCore, InfiniGen.", ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.10, top=0.90, hspace=0.26, wspace=0.10)
    return save_figure(fig, stem)


def plot_p99_latency(metrics: pd.DataFrame) -> list[Path]:
    configs = [(1024, 8), (2048, 8), (6144, 8), (6144, 16)]
    labels = [f"{OUTPUT_LABELS[o]}-bs{b}" for o, b in configs]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(configs))
    width = 0.24
    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for output_len, batch_size in configs:
            row = metrics[
                (metrics["model_key"] == LLAMA_KEY)
                & (metrics["output_len_bucket"] == output_len)
                & (metrics["batch_size"] == batch_size)
                & (metrics["method"] == method)
            ].iloc[0]
            values.append(float(row["p99_e2e_s"]))
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Empirical P99 E2E (s)")
    ax.set_xlabel("Configuration")
    style_axis(ax)
    ax.legend(frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0.0, 1.15))
    add_placeholder_note(fig, y=-0.02)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.25, top=0.82)
    return save_figure(fig, "p99_latency_llama31")


def plot_ttft_tpot(metrics: pd.DataFrame) -> list[Path]:
    data = metrics[(metrics["model_key"] == LLAMA_KEY) & (metrics["batch_size"] == 8)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    panels = [
        (axes[0], "median_ttft_s", "Median TTFT (s)"),
        (axes[1], "median_tpot_ms", "Median TPOT (ms/token)"),
    ]
    x = np.arange(len(OUTPUT_ORDER))
    for ax, metric, ylabel in panels:
        for method in METHOD_ORDER:
            m = data[data["method"] == method].set_index("output_len_bucket").loc[OUTPUT_ORDER]
            ax.plot(
                x,
                m[metric].to_numpy(dtype=float),
                label=method,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.8,
                markersize=4.2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([OUTPUT_LABELS[o] for o in OUTPUT_ORDER])
        ax.set_xlabel("Output length")
        ax.set_ylabel(ylabel)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    add_placeholder_note(fig, y=-0.03)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.78, wspace=0.28)
    return save_figure(fig, "ttft_tpot_llama31")


def plot_feasibility_matrix(metrics: pd.DataFrame) -> list[Path]:
    labels = [f"{OUTPUT_LABELS[o]}-bs{b}" for o in OUTPUT_ORDER for b in BATCH_ORDER]
    models = ["llama31_8b_instruct", "qwen3_8b", "mistral_7b_instruct_v03"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.25), sharey=True)
    cmap = plt.matplotlib.colors.ListedColormap(["#F2F2F2", "#009E73", "#D55E00"])
    for ax, method in zip(axes, METHOD_ORDER):
        matrix = []
        for model_key in models:
            row_values = []
            for output_len in OUTPUT_ORDER:
                for batch_size in BATCH_ORDER:
                    match = metrics[
                        (metrics["method"] == method)
                        & (metrics["model_key"] == model_key)
                        & (metrics["output_len_bucket"] == output_len)
                        & (metrics["batch_size"] == batch_size)
                    ]
                    if match.empty:
                        row_values.append(0)
                    elif match.iloc[0]["status"] == "ok":
                        row_values.append(1)
                    else:
                        row_values.append(2)
            matrix.append(row_values)
        ax.imshow(np.array(matrix), cmap=cmap, vmin=0, vmax=2, aspect="auto")
        ax.set_title(method)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in models])
        ax.tick_params(axis="both", which="both", length=0)
        for x in np.arange(-0.5, len(labels), 1):
            ax.axvline(x, color="white", linewidth=0.8)
        for y in np.arange(-0.5, len(models), 1):
            ax.axhline(y, color="white", linewidth=0.8)
        for y, model_key in enumerate(models):
            for x_idx, output_len in enumerate(OUTPUT_ORDER):
                for b_idx, batch_size in enumerate(BATCH_ORDER):
                    x = x_idx * len(BATCH_ORDER) + b_idx
                    match = metrics[
                        (metrics["method"] == method)
                        & (metrics["model_key"] == model_key)
                        & (metrics["output_len_bucket"] == output_len)
                        & (metrics["batch_size"] == batch_size)
                    ]
                    if match.empty or match.iloc[0]["status"] != "ok":
                        ax.text(x, y, "OOM", ha="center", va="center", fontsize=7, color="white")
    legend_handles = [
        Patch(facecolor="#009E73", edgecolor="white", label="OK"),
        Patch(facecolor="#D55E00", edgecolor="white", label="OOM"),
    ]
    fig.legend(legend_handles, ["OK", "OOM"], frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    add_placeholder_note(fig, y=-0.03)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.33, top=0.80, wspace=0.05)
    return save_figure(fig, "feasibility_matrix_all_models")


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

Placeholder figures for the end-to-end serving comparison among vLLM, InfiniGen, and KVCore.
vLLM points are measured from the formal PG19 in6k raw JSONL outputs.
InfiniGen and KVCore points are simulated placeholders for figure layout and narrative planning only.

## Source Data

- Measured vLLM raw: `evaluation/results/experiments/{SOURCE_EXPERIMENT}/artifacts/raw/vllm/`
- Plotting intermediate: `figure/experiments/{EXPERIMENT_NAME}/placeholder_metrics.csv`
- Plotting script: `figure/plot_end2end_serving_placeholder.py`

## Figures

{output_lines}

## Metric Definitions

- Decode throughput: `sum(actual_output_len) / (max(finish_time_s) - min(submit_time_s))`, measured-run only, warmup excluded.
- Request throughput: `num_completed / (max(finish_time_s) - min(submit_time_s))`, measured-run only, warmup excluded.
- P99 latency: empirical P99 over 32 or 48 requests per point.
- The `*_by_model` figures use rows for models, columns for output lengths, and batch size on the x-axis.

## Notes

- `KVCore (sim.)` and `InfiniGen (sim.)` are not real end-to-end results.
- vLLM failed points remain marked as OOM rather than interpolated.
- The Llama out2k bs8 point uses the successful rerun raw file from 2026-06-09.
"""
    )


def main() -> int:
    setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics()
    outputs: list[Path] = []
    outputs.extend(
        plot_metric_grid_by_model(
            metrics,
            metric="decode_throughput_tok_s",
            ylabel="Decode throughput (tok/s)",
            stem="throughput_vs_batch_by_model",
            higher_is_better=True,
        )
    )
    outputs.extend(
        plot_metric_grid_by_model(
            metrics,
            metric="median_e2e_s",
            ylabel="Median E2E (s)",
            stem="median_e2e_vs_batch_by_model",
            higher_is_better=False,
        )
    )
    outputs.extend(
        plot_metric_grid_by_model(
            metrics,
            metric="p99_e2e_s",
            ylabel="Empirical P99 E2E (s)",
            stem="p99_e2e_vs_batch_by_model",
            higher_is_better=False,
        )
    )
    outputs.extend(
        plot_metric_grid_by_model(
            metrics,
            metric="median_tpot_ms",
            ylabel="Median TPOT (ms/token)",
            stem="tpot_vs_batch_by_model",
            higher_is_better=False,
        )
    )
    outputs.extend(plot_throughput_vs_batch(metrics))
    outputs.extend(plot_p99_latency(metrics))
    outputs.extend(plot_ttft_tpot(metrics))
    outputs.extend(plot_feasibility_matrix(metrics))
    write_readme(outputs)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: placeholder end-to-end serving figures using measured vLLM and simulated KVCore/InfiniGen data.",
    )
    print(f"Wrote metrics: {METRICS_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
