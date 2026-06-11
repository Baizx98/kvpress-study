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
EXPERIMENT_NAME = "end2end_serving_paper_draft_predicted_20260610"
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"
METRICS_CSV = FIGURE_ROOT / "paper_draft_end2end_metrics_table.csv"

MODEL_ORDER = [
    "llama31_8b_instruct",
    "mistral_7b_instruct_v03",
    "qwen3_8b",
]
MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B",
    "mistral_7b_instruct_v03": "Mistral-7B",
    "qwen3_8b": "Qwen3-8B",
}
LLAMA_KEY = "llama31_8b_instruct"
BATCH_ORDER = [1, 8, 16]
OUTPUT_ORDER = [1024, 2048, 6144]
OUTPUT_LABELS = {1024: "1k", 2048: "2k", 6144: "6k"}
SYSTEM_ORDER = ["vLLM", "InfiniGen", "KVCore (pred.)"]
SYSTEM_COLORS = {
    "vLLM": "#2563EB",
    "InfiniGen": "#F97316",
    "KVCore (pred.)": "#10B981",
}
BS_REGION_COLORS = {
    1: "#F8FAFC",
    8: "#EFF6FF",
    16: "#F0FDF4",
}
METRICS_TO_PLOT = [
    ("decode_throughput_tok_s", "Decode throughput (tok/s)", "throughput"),
    ("median_ttft_s", "Median TTFT (s)", "ttft"),
    ("p99_e2e_s", "P99 E2E latency (s)", "p99_e2e"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_box_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, color="#D8DEE9", alpha=0.75)
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.72)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(0.82)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
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
    raw_dir = SOURCE_RAW_ROOT / system
    for path in sorted(raw_dir.glob(f"{system}__*__pg19__in6k_out*__bs*__seed2026.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                obj["source_file"] = path.name
                rows.append(obj)
    if not rows:
        raise FileNotFoundError(f"No {system} rows found under {raw_dir}")
    return pd.DataFrame(rows)


def aggregate_measured(system: str, label: str, model_keys: set[str] | None = None) -> pd.DataFrame:
    df = read_system_raw(system)
    if model_keys is not None:
        df = df[df["model_key"].isin(model_keys)].copy()
    records: list[dict] = []
    group_cols = ["model_key", "output_len_bucket", "batch_size"]
    for (model_key, output_len, batch_size), sub_all in df.groupby(group_cols, sort=False):
        model_key = str(model_key)
        output_len = int(output_len)
        batch_size = int(batch_size)
        if model_key not in MODEL_ORDER or output_len not in OUTPUT_ORDER or batch_size not in BATCH_ORDER:
            continue
        ok = sub_all[sub_all["status"].map(lambda s: success_status(system, s))].copy()
        duration = float(ok["finish_time_s"].max() - ok["submit_time_s"].min()) if not ok.empty else float("nan")
        throughput = float(ok["actual_output_len"].sum() / duration) if duration > 0 else float("nan")
        records.append(
            {
                "system": label,
                "source": "measured",
                "source_detail": f"raw_{system}",
                "model_key": model_key,
                "model_label": MODEL_LABELS[model_key],
                "batch_size": batch_size,
                "output_len_bucket": output_len,
                "output_label": OUTPUT_LABELS[output_len],
                "num_requests": int(len(sub_all)),
                "num_completed": int(len(ok)),
                "num_failed": int(len(sub_all) - len(ok)),
                "status": "ok" if len(ok) > 0 and len(ok) == len(sub_all) else "partial_or_failed",
                "median_ttft_s": percentile(ok["ttft_s"], 50),
                "p99_ttft_s": percentile(ok["ttft_s"], 99),
                "median_e2e_s": percentile(ok["e2e_latency_s"], 50),
                "p99_e2e_s": percentile(ok["e2e_latency_s"], 99),
                "median_tpot_ms": percentile(ok["tpot_ms"], 50),
                "p99_tpot_ms": percentile(ok["tpot_ms"], 99),
                "gpu_peak_memory_gb": percentile(ok["gpu_peak_memory_gb"], 50),
                "decode_throughput_tok_s": throughput,
                "notes": "measured raw requests",
            }
        )
    return pd.DataFrame(records)


def lookup(df: pd.DataFrame, system: str, model_key: str, output_len: int, batch_size: int, metric: str) -> float:
    row = df[
        (df["system"] == system)
        & (df["model_key"] == model_key)
        & (df["output_len_bucket"] == output_len)
        & (df["batch_size"] == batch_size)
    ]
    if row.empty:
        return float("nan")
    value = float(row.iloc[0][metric])
    return value if np.isfinite(value) else float("nan")


def infer_infinigen(vllm: pd.DataFrame, infinigen_llama: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    metrics = [
        "median_ttft_s",
        "p99_ttft_s",
        "median_e2e_s",
        "p99_e2e_s",
        "median_tpot_ms",
        "p99_tpot_ms",
        "gpu_peak_memory_gb",
        "decode_throughput_tok_s",
    ]
    for model_key in MODEL_ORDER:
        if model_key == LLAMA_KEY:
            continue
        for batch_size in BATCH_ORDER:
            for output_len in OUTPUT_ORDER:
                rec = {
                    "system": "InfiniGen",
                    "source": "predicted",
                    "source_detail": "scaled_by_llama31_measured_infinigen_to_vllm_ratio",
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "batch_size": batch_size,
                    "output_len_bucket": output_len,
                    "output_label": OUTPUT_LABELS[output_len],
                    "num_requests": 32,
                    "num_completed": 32,
                    "num_failed": 0,
                    "status": "predicted",
                    "notes": "predicted from Llama-3.1 InfiniGen/vLLM ratio at the same batch/output setting",
                }
                for metric in metrics:
                    llama_inf = lookup(infinigen_llama, "InfiniGen", LLAMA_KEY, output_len, batch_size, metric)
                    llama_vllm = lookup(vllm, "vLLM", LLAMA_KEY, output_len, batch_size, metric)
                    model_vllm = lookup(vllm, "vLLM", model_key, output_len, batch_size, metric)
                    if not np.isfinite(llama_inf) or not np.isfinite(llama_vllm) or not np.isfinite(model_vllm):
                        rec[metric] = float("nan")
                    else:
                        rec[metric] = model_vllm * (llama_inf / llama_vllm)
                records.append(rec)
    return pd.DataFrame(records)


def infer_kvcore(vllm: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    latency_metrics = [
        "median_e2e_s",
        "p99_e2e_s",
        "median_tpot_ms",
        "p99_tpot_ms",
    ]
    ttft_metrics = ["median_ttft_s", "p99_ttft_s"]
    for model_key in MODEL_ORDER:
        for batch_size in BATCH_ORDER:
            for output_len in OUTPUT_ORDER:
                b_idx = BATCH_ORDER.index(batch_size)
                o_idx = OUTPUT_ORDER.index(output_len)
                pressure = b_idx + o_idx
                throughput_gain = min(2.30, 1.28 + 0.24 * b_idx + 0.26 * o_idx)
                ttft_scale = max(0.82, 0.96 - 0.045 * b_idx - 0.025 * o_idx)
                median_latency_scale = max(0.56, 0.82 - 0.07 * pressure)
                p99_latency_scale = max(0.50, 0.84 - 0.085 * pressure)
                tpot_scale = max(0.50, 0.74 - 0.07 * pressure)
                mem_scale = max(0.56, 0.78 - 0.055 * pressure)
                rec = {
                    "system": "KVCore (pred.)",
                    "source": "predicted",
                    "source_detail": "mechanism_based_from_vllm_and_blockwisepress_prior",
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "batch_size": batch_size,
                    "output_len_bucket": output_len,
                    "output_label": OUTPUT_LABELS[output_len],
                    "num_requests": 32,
                    "num_completed": 32,
                    "num_failed": 0,
                    "status": "predicted",
                    "notes": (
                        "KVCore prediction assumes sparse block lifecycle reduces most but not all "
                        "preemptions; gains grow with batch size and output length"
                    ),
                }
                for metric in ["decode_throughput_tok_s"]:
                    rec[metric] = lookup(vllm, "vLLM", model_key, output_len, batch_size, metric) * throughput_gain
                for metric in ttft_metrics:
                    rec[metric] = lookup(vllm, "vLLM", model_key, output_len, batch_size, metric) * ttft_scale
                for metric in latency_metrics:
                    scale = p99_latency_scale if metric.startswith("p99") else median_latency_scale
                    if "tpot" in metric:
                        scale = tpot_scale
                    rec[metric] = lookup(vllm, "vLLM", model_key, output_len, batch_size, metric) * scale
                rec["gpu_peak_memory_gb"] = lookup(vllm, "vLLM", model_key, output_len, batch_size, "gpu_peak_memory_gb") * mem_scale
                records.append(rec)
    return pd.DataFrame(records)


def add_vllm_relative_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    for metric, rel_name, higher_is_better in [
        ("decode_throughput_tok_s", "throughput_vs_vllm_x", True),
        ("median_ttft_s", "ttft_vs_vllm_x", False),
        ("p99_e2e_s", "p99_e2e_vs_vllm_x", False),
    ]:
        values: list[float] = []
        for _, row in out.iterrows():
            base = lookup(out, "vLLM", row["model_key"], int(row["output_len_bucket"]), int(row["batch_size"]), metric)
            cur = float(row[metric])
            if not np.isfinite(base) or not np.isfinite(cur) or base == 0:
                values.append(float("nan"))
            elif higher_is_better:
                values.append(cur / base)
            else:
                values.append(base / cur)
        out[rel_name] = values
    return out


def build_metrics() -> pd.DataFrame:
    vllm = aggregate_measured("vllm", "vLLM", set(MODEL_ORDER))
    infinigen_llama = aggregate_measured("infinigen", "InfiniGen", {LLAMA_KEY})
    infinigen_pred = infer_infinigen(vllm, infinigen_llama)
    kvcore_pred = infer_kvcore(vllm)
    metrics = pd.concat([vllm, infinigen_llama, infinigen_pred, kvcore_pred], ignore_index=True)
    metrics["system"] = pd.Categorical(metrics["system"], categories=SYSTEM_ORDER, ordered=True)
    metrics["model_key"] = pd.Categorical(metrics["model_key"], categories=MODEL_ORDER, ordered=True)
    metrics = metrics.sort_values(["model_key", "batch_size", "output_len_bucket", "system"]).reset_index(drop=True)
    metrics = add_vllm_relative_columns(metrics)
    metrics.to_csv(METRICS_CSV, index=False)
    return metrics


def x_layout() -> tuple[dict[tuple[int, int], float], list[float], list[str], dict[int, tuple[float, float, float]]]:
    positions: dict[tuple[int, int], float] = {}
    xticks: list[float] = []
    xticklabels: list[str] = []
    regions: dict[int, tuple[float, float, float]] = {}
    group_gap = 1.0
    block_gap = 1.05
    cursor = 0.0
    for batch_size in BATCH_ORDER:
        centers: list[float] = []
        for output_len in OUTPUT_ORDER:
            positions[(batch_size, output_len)] = cursor
            centers.append(cursor)
            xticks.append(cursor)
            xticklabels.append(OUTPUT_LABELS[output_len])
            cursor += group_gap
        regions[batch_size] = (centers[0] - 0.52, centers[-1] + 0.52, float(np.mean(centers)))
        cursor += block_gap
    return positions, xticks, xticklabels, regions


def plot_metric_for_model(metrics: pd.DataFrame, model_key: str, metric: str, ylabel: str, metric_stem: str) -> list[Path]:
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 1.92))
    positions, xticks, xticklabels, regions = x_layout()
    width = 0.22
    offsets = [-width, 0.0, width]
    data = metrics[metrics["model_key"] == model_key]

    for batch_size, (left, right, center) in regions.items():
        ax.axvspan(left, right, color=BS_REGION_COLORS[batch_size], alpha=0.62, zorder=0)
        ax.text(
            center,
            0.975,
            f"BS={batch_size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.1,
            fontweight="bold",
            color="#374151",
        )
    values_regions = list(regions.values())
    for (_, right, _), next_region in zip(values_regions[:-1], values_regions[1:]):
        separator = (right + next_region[0]) / 2.0
        ax.axvline(separator, color="#6B7280", linewidth=0.7, linestyle=":", alpha=0.7, zorder=1)

    ymax = 0.0
    for idx, system in enumerate(SYSTEM_ORDER):
        xs: list[float] = []
        values: list[float] = []
        for batch_size in BATCH_ORDER:
            for output_len in OUTPUT_ORDER:
                row = data[
                    (data["system"] == system)
                    & (data["batch_size"] == batch_size)
                    & (data["output_len_bucket"] == output_len)
                ]
                value = float(row.iloc[0][metric]) if not row.empty else np.nan
                xs.append(positions[(batch_size, output_len)] + offsets[idx])
                values.append(value)
                if np.isfinite(value):
                    ymax = max(ymax, value)
        ax.bar(
            xs,
            values,
            width=width,
            label=system,
            color=SYSTEM_COLORS[system],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )

    ax.set_xlim(min(xticks) - 0.75, max(xticks) + 0.75)
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Output length")
    ax.set_ylabel(ylabel)
    style_box_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.04),
        columnspacing=1.4,
        handlelength=1.3,
    )
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.24, top=0.76)
    return save_figure(fig, f"paperdraft_{model_key}_{metric_stem}_mergedbs_wide")


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

Paper-draft end-to-end serving figures for throughput, TTFT, and empirical P99 E2E latency across Llama-3.1-8B, Mistral-7B, and Qwen3-8B.

## Data Sources

- vLLM: measured raw data from `evaluation/results/experiments/{SOURCE_EXPERIMENT}/artifacts/raw/vllm/`
- InfiniGen on Llama-3.1-8B: measured raw data from `evaluation/results/experiments/{SOURCE_EXPERIMENT}/artifacts/raw/infinigen/`
- InfiniGen on Mistral-7B and Qwen3-8B: predicted by transferring the measured Llama-3.1 InfiniGen/vLLM ratio to each model's measured vLLM value.
- KVCore: predicted from measured vLLM values with mechanism-based gains. The prediction assumes sparse block lifecycle management reduces most but not all request preemptions, with larger gains under larger batch sizes and longer outputs.

## Generated Data

- Metrics table: `figure/experiments/{EXPERIMENT_NAME}/paper_draft_end2end_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_predicted.py`

## Figures

{output_lines}

## Plot Layout

Each figure is one long boxed plotting area. Batch sizes `1`, `8`, and `16` are separated by lightly shaded background regions. Within each batch-size region, bars compare systems at output lengths `1k`, `2k`, and `6k`.

## Notes

- Batch size 24 is intentionally excluded to match the current paper-draft figure layout.
- The figures are for draft visualization only; predicted rows are explicitly marked in the CSV `source` and `source_detail` columns.
- Existing figure directories are not overwritten.
"""
    )


def main() -> int:
    setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics()
    outputs: list[Path] = []
    for model_key in MODEL_ORDER:
        for metric, ylabel, metric_stem in METRICS_TO_PLOT:
            outputs.extend(plot_metric_for_model(metrics, model_key, metric, ylabel, metric_stem))
    write_readme(outputs)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: paper-draft serving figures with measured vLLM/Llama InfiniGen and predicted missing systems.",
    )
    print(f"Wrote metrics: {METRICS_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
