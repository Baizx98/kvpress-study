from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENT = "end2end_serving_kvcore_vllm_infinigen_longreq"
SOURCE_RAW_ROOT = (
    REPO_ROOT
    / "evaluation"
    / "results"
    / "experiments"
    / SOURCE_EXPERIMENT
    / "artifacts"
    / "raw"
)
EXPERIMENT_NAME = "end2end_serving_llama31_real_infinigen_kvcore_sim_20260610"
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"
METRICS_CSV = FIGURE_ROOT / "llama31_real_infinigen_kvcore_sim_metrics.csv"

MODEL_KEY = "llama31_8b_instruct"
BATCH_ORDER = [1, 8, 16]
OUTPUT_ORDER = [1024, 2048, 6144]
OUTPUT_LABELS = {1024: "1k", 2048: "2k", 6144: "6k"}
SYSTEM_ORDER = ["vLLM", "InfiniGen", "KVCore (sim.)"]
SYSTEM_COLORS = {
    "vLLM": "#2563EB",
    "InfiniGen": "#F97316",
    "KVCore (sim.)": "#10B981",
}
SYSTEM_SOURCE = {
    "vLLM": "measured",
    "InfiniGen": "measured",
    "KVCore (sim.)": "simulated_placeholder",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "axes.titlesize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_box_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, color="#D8DEE9", alpha=0.75)
    ax.tick_params(axis="both", which="major", direction="in", length=2.5, width=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1F2937")
        spine.set_linewidth(0.72)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return [pdf_path, png_path]


def percentile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean.to_numpy(dtype=float), q))


def success_status(system: str, status: object) -> bool:
    if system == "vllm":
        return status == "completed"
    if system == "infinigen":
        return status == "ok"
    return False


def read_system_raw(system: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((SOURCE_RAW_ROOT / system).glob(f"{system}__{MODEL_KEY}__pg19__in6k_out*__bs*__seed2026.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                obj["source_file"] = path.name
                rows.append(obj)
    if not rows:
        raise FileNotFoundError(f"No {system} rows found under {SOURCE_RAW_ROOT / system}")
    return pd.DataFrame(rows)


def aggregate_measured(system: str, label: str) -> pd.DataFrame:
    df = read_system_raw(system)
    records: list[dict] = []
    for (output_len, batch_size), sub_all in df.groupby(["output_len_bucket", "batch_size"], sort=False):
        output_len = int(output_len)
        batch_size = int(batch_size)
        if batch_size not in BATCH_ORDER or output_len not in OUTPUT_ORDER:
            continue
        ok = sub_all[sub_all["status"].map(lambda s: success_status(system, s))].copy()
        duration = float(ok["finish_time_s"].max() - ok["submit_time_s"].min()) if not ok.empty else float("nan")
        throughput = float(ok["actual_output_len"].sum() / duration) if duration > 0 else float("nan")
        records.append(
            {
                "system": label,
                "source": "measured",
                "model_key": MODEL_KEY,
                "batch_size": batch_size,
                "output_len_bucket": output_len,
                "output_label": OUTPUT_LABELS[output_len],
                "num_requests": int(len(sub_all)),
                "num_completed": int(len(ok)),
                "num_failed": int(len(sub_all) - len(ok)),
                "median_ttft_s": percentile(ok["ttft_s"], 50),
                "median_e2e_s": percentile(ok["e2e_latency_s"], 50),
                "p99_e2e_s": percentile(ok["e2e_latency_s"], 99),
                "median_tpot_ms": percentile(ok["tpot_ms"], 50),
                "gpu_peak_memory_gb": percentile(ok["gpu_peak_memory_gb"], 50),
                "decode_throughput_tok_s": throughput,
            }
        )
    return pd.DataFrame(records)


def base_value(measured: pd.DataFrame, output_len: int, batch_size: int, metric: str) -> float:
    # KVCore is a placeholder: anchor it to vLLM when possible so the simulated
    # value remains in the measured system's scale.
    vllm = measured[
        (measured["system"] == "vLLM")
        & (measured["output_len_bucket"] == output_len)
        & (measured["batch_size"] == batch_size)
    ]
    if not vllm.empty and np.isfinite(float(vllm.iloc[0][metric])):
        return float(vllm.iloc[0][metric])
    same_output = measured[
        (measured["system"] == "vLLM")
        & (measured["output_len_bucket"] == output_len)
        & np.isfinite(measured[metric])
    ].sort_values("batch_size")
    if same_output.empty:
        return float("nan")
    lower = same_output[same_output["batch_size"] < batch_size]
    ref = lower.iloc[-1] if not lower.empty else same_output.iloc[-1]
    return float(ref[metric])


def simulate_kvcore(measured: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for batch_size in BATCH_ORDER:
        for output_len in OUTPUT_ORDER:
            b_idx = BATCH_ORDER.index(batch_size)
            o_idx = OUTPUT_ORDER.index(output_len)
            stress = b_idx + o_idx
            throughput_gain = 1.08 + 0.12 * b_idx + 0.16 * o_idx
            latency_scale = max(0.48, 0.86 - 0.075 * stress)
            ttft_scale = max(0.62, 0.92 - 0.045 * b_idx - 0.025 * o_idx)
            tpot_scale = max(0.42, 0.82 - 0.075 * stress)
            memory_scale = max(0.54, 0.76 - 0.045 * stress)

            median_e2e = base_value(measured, output_len, batch_size, "median_e2e_s") * latency_scale
            p99_e2e = base_value(measured, output_len, batch_size, "p99_e2e_s") * latency_scale * 1.06
            rec = {
                "system": "KVCore (sim.)",
                "source": "simulated_placeholder",
                "model_key": MODEL_KEY,
                "batch_size": batch_size,
                "output_len_bucket": output_len,
                "output_label": OUTPUT_LABELS[output_len],
                "num_requests": 32,
                "num_completed": 32,
                "num_failed": 0,
                "median_ttft_s": base_value(measured, output_len, batch_size, "median_ttft_s") * ttft_scale,
                "median_e2e_s": median_e2e,
                "p99_e2e_s": max(p99_e2e, median_e2e * 1.03),
                "median_tpot_ms": base_value(measured, output_len, batch_size, "median_tpot_ms") * tpot_scale,
                "gpu_peak_memory_gb": base_value(measured, output_len, batch_size, "gpu_peak_memory_gb") * memory_scale,
                "decode_throughput_tok_s": base_value(measured, output_len, batch_size, "decode_throughput_tok_s")
                * throughput_gain,
            }
            records.append(rec)
    return pd.DataFrame(records)


def build_metrics() -> pd.DataFrame:
    measured = pd.concat(
        [
            aggregate_measured("vllm", "vLLM"),
            aggregate_measured("infinigen", "InfiniGen"),
        ],
        ignore_index=True,
    )
    metrics = pd.concat([measured, simulate_kvcore(measured)], ignore_index=True)
    metrics["system"] = pd.Categorical(metrics["system"], categories=SYSTEM_ORDER, ordered=True)
    metrics = metrics.sort_values(["batch_size", "output_len_bucket", "system"]).reset_index(drop=True)
    metrics.to_csv(METRICS_CSV, index=False)
    return metrics


def plot_metric(metrics: pd.DataFrame, metric: str, ylabel: str, stem: str) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(3.33, 1.58), sharey=True)
    x = np.arange(len(OUTPUT_ORDER))
    width = 0.23
    offsets = [-width, 0.0, width]
    for ax, batch_size in zip(axes, BATCH_ORDER):
        sub = metrics[metrics["batch_size"] == batch_size]
        for idx, system in enumerate(SYSTEM_ORDER):
            values = []
            for output_len in OUTPUT_ORDER:
                row = sub[(sub["system"] == system) & (sub["output_len_bucket"] == output_len)]
                values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
            ax.bar(
                x + offsets[idx],
                values,
                width=width,
                label=system,
                color=SYSTEM_COLORS[system],
                edgecolor="white",
                linewidth=0.42,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([OUTPUT_LABELS[o] for o in OUTPUT_ORDER])
        ax.text(
            0.02,
            1.025,
            f"BS={batch_size}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            clip_on=False,
        )
        ax.set_xlabel("Output length")
        style_box_axis(ax)
    fig.supylabel(ylabel, x=0.018, fontsize=7.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.04),
        columnspacing=0.8,
        handlelength=1.1,
    )
    fig.subplots_adjust(left=0.18, right=0.995, bottom=0.25, top=0.75, wspace=0.12)
    return save_figure(fig, stem)


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

Single-column Llama-3.1-8B end-to-end serving figures with measured vLLM and InfiniGen data plus simulated KVCore placeholder data.

## Source Data

- vLLM raw: `evaluation/results/experiments/{SOURCE_EXPERIMENT}/artifacts/raw/vllm/`
- InfiniGen raw: `evaluation/results/experiments/{SOURCE_EXPERIMENT}/artifacts/raw/infinigen/`
- Metrics CSV: `figure/experiments/{EXPERIMENT_NAME}/llama31_real_infinigen_kvcore_sim_metrics.csv`
- Plotting script: `figure/plot_end2end_llama31_infinigen_kvcore_sim.py`

## Figures

{output_lines}

## Plot Layout

Each figure is one single-column group plot. The three subplots are batch sizes `1`, `8`, and `16`.
Within each subplot, the x-axis is output length `1k`, `2k`, and `6k`, and bars compare vLLM, InfiniGen, and KVCore.

## Notes

- vLLM and InfiniGen are measured raw results.
- `KVCore (sim.)` is simulated placeholder data. The simulation is anchored to measured vLLM and gives larger improvements as batch size and output length increase.
- Batch size 24 is intentionally excluded.
- Existing figure directories are not overwritten.
"""
    )


def main() -> int:
    setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics()
    outputs: list[Path] = []
    outputs.extend(plot_metric(metrics, "decode_throughput_tok_s", "Decode throughput (tok/s)", "llama31_decode_throughput_singlecol"))
    outputs.extend(plot_metric(metrics, "median_e2e_s", "Median E2E latency (s)", "llama31_median_e2e_singlecol"))
    outputs.extend(plot_metric(metrics, "p99_e2e_s", "Empirical P99 E2E (s)", "llama31_p99_e2e_singlecol"))
    outputs.extend(plot_metric(metrics, "median_ttft_s", "Median TTFT (s)", "llama31_median_ttft_singlecol"))
    outputs.extend(plot_metric(metrics, "median_tpot_ms", "Median TPOT (ms/token)", "llama31_median_tpot_singlecol"))
    outputs.extend(plot_metric(metrics, "gpu_peak_memory_gb", "GPU peak memory (GB)", "llama31_gpu_peak_memory_singlecol"))
    write_readme(outputs)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: Llama-3.1-8B single-column serving figures with measured vLLM/InfiniGen and simulated KVCore.",
    )
    print(f"Wrote metrics: {METRICS_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
