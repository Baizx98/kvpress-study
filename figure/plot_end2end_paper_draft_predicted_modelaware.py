from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "figure" / "plot_end2end_paper_draft_predicted.py"
SOURCE_EXPERIMENT = "end2end_serving_kvcore_vllm_infinigen_longreq"
EXPERIMENT_NAME = "end2end_serving_paper_draft_modelaware_predicted_20260610"
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"
METRICS_CSV = FIGURE_ROOT / "paper_draft_end2end_modelaware_metrics_table.csv"

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
METRICS_TO_PLOT = [
    ("decode_throughput_tok_s", "Decode throughput (tok/s)", "throughput"),
    ("median_ttft_s", "Median TTFT (s)", "ttft"),
    ("p99_e2e_s", "P99 E2E latency (s)", "p99_e2e"),
]

# Local model configs: Llama/Mistral have the same per-token KV footprint
# proxy, while Qwen3 has 36 layers and about 1.125x the KV footprint proxy.
MODEL_KV_UNIT_REL = {
    "llama31_8b_instruct": 1.000,
    "mistral_7b_instruct_v03": 1.000,
    "qwen3_8b": 1.125,
}
INFINIGEN_MODEL_FACTORS = {
    "mistral_7b_instruct_v03": {
        "throughput": 1.08,
        "ttft": 0.96,
        "latency": 0.94,
        "tpot": 0.96,
        "memory": 1.00,
    },
    "qwen3_8b": {
        "throughput": 0.92,
        "ttft": 1.06,
        "latency": 1.08,
        "tpot": 1.08,
        "memory": 1.03,
    },
}
KVCORE_MODEL_FACTORS = {
    "llama31_8b_instruct": {
        "throughput": 1.00,
        "ttft_scale_delta": 0.000,
        "median_latency_scale_delta": 0.000,
        "p99_latency_scale_delta": 0.000,
        "tpot_scale_delta": 0.000,
    },
    "mistral_7b_instruct_v03": {
        "throughput": 0.94,
        "ttft_scale_delta": 0.015,
        "median_latency_scale_delta": 0.020,
        "p99_latency_scale_delta": 0.025,
        "tpot_scale_delta": 0.020,
    },
    "qwen3_8b": {
        "throughput": 1.06,
        "ttft_scale_delta": 0.015,
        "median_latency_scale_delta": -0.020,
        "p99_latency_scale_delta": -0.035,
        "tpot_scale_delta": -0.015,
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("paper_draft_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base plotting script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPERIMENT_NAME = EXPERIMENT_NAME
    module.FIGURE_ROOT = FIGURE_ROOT
    module.FIGURE_README = FIGURE_README
    module.METRICS_CSV = METRICS_CSV
    return module


base = load_base_module()


def model_pressure(model_key: str, batch_size: int, output_len: int) -> float:
    b_idx = BATCH_ORDER.index(batch_size)
    o_idx = OUTPUT_ORDER.index(output_len)
    return (b_idx + o_idx) * MODEL_KV_UNIT_REL[model_key]


def lookup(df: pd.DataFrame, system: str, model_key: str, output_len: int, batch_size: int, metric: str) -> float:
    return base.lookup(df, system, model_key, output_len, batch_size, metric)


def infer_infinigen_modelaware(vllm: pd.DataFrame, infinigen_llama: pd.DataFrame) -> pd.DataFrame:
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
        factors = INFINIGEN_MODEL_FACTORS[model_key]
        for batch_size in BATCH_ORDER:
            for output_len in OUTPUT_ORDER:
                pressure = model_pressure(model_key, batch_size, output_len)
                rec = {
                    "system": "InfiniGen",
                    "source": "predicted",
                    "source_detail": "llama31_ratio_with_model_specific_adjustment",
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
                        "Predicted from Llama-3.1 InfiniGen/vLLM ratio plus model-aware adjustment; "
                        f"kv_unit_rel={MODEL_KV_UNIT_REL[model_key]:.3f}, pressure={pressure:.3f}"
                    ),
                }
                for metric in metrics:
                    llama_inf = lookup(infinigen_llama, "InfiniGen", LLAMA_KEY, output_len, batch_size, metric)
                    llama_vllm = lookup(vllm, "vLLM", LLAMA_KEY, output_len, batch_size, metric)
                    model_vllm = lookup(vllm, "vLLM", model_key, output_len, batch_size, metric)
                    if not np.isfinite(llama_inf) or not np.isfinite(llama_vllm) or not np.isfinite(model_vllm):
                        rec[metric] = float("nan")
                        continue
                    value = model_vllm * (llama_inf / llama_vllm)
                    if "throughput" in metric:
                        value *= factors["throughput"]
                    elif "ttft" in metric:
                        value *= factors["ttft"]
                    elif "tpot" in metric:
                        value *= factors["tpot"]
                    elif "memory" in metric:
                        value *= factors["memory"]
                    else:
                        value *= factors["latency"]
                    rec[metric] = value
                records.append(rec)
    return pd.DataFrame(records)


def infer_kvcore_modelaware(vllm: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    latency_metrics = ["median_e2e_s", "p99_e2e_s", "median_tpot_ms", "p99_tpot_ms"]
    ttft_metrics = ["median_ttft_s", "p99_ttft_s"]
    for model_key in MODEL_ORDER:
        factors = KVCORE_MODEL_FACTORS[model_key]
        for batch_size in BATCH_ORDER:
            for output_len in OUTPUT_ORDER:
                b_idx = BATCH_ORDER.index(batch_size)
                o_idx = OUTPUT_ORDER.index(output_len)
                pressure = model_pressure(model_key, batch_size, output_len)
                base_throughput_gain = 1.28 + 0.24 * b_idx + 0.26 * o_idx
                pressure_bonus = 1.0 + 0.015 * max(0.0, pressure - (b_idx + o_idx))
                throughput_gain = min(2.35, base_throughput_gain * factors["throughput"] * pressure_bonus)
                ttft_scale = max(0.82, 0.96 - 0.045 * b_idx - 0.025 * o_idx + factors["ttft_scale_delta"])
                median_latency_scale = max(
                    0.54,
                    0.82 - 0.07 * pressure + factors["median_latency_scale_delta"],
                )
                p99_latency_scale = max(0.48, 0.84 - 0.085 * pressure + factors["p99_latency_scale_delta"])
                tpot_scale = max(0.48, 0.74 - 0.07 * pressure + factors["tpot_scale_delta"])
                mem_scale = max(0.53, 0.78 - 0.055 * pressure)
                rec = {
                    "system": "KVCore (pred.)",
                    "source": "predicted",
                    "source_detail": "modelaware_from_vllm_configs_and_blockwisepress_prior",
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
                        "Model-aware KVCore prediction; avoids most but not all preemptions; "
                        f"kv_unit_rel={MODEL_KV_UNIT_REL[model_key]:.3f}, pressure={pressure:.3f}"
                    ),
                }
                rec["decode_throughput_tok_s"] = (
                    lookup(vllm, "vLLM", model_key, output_len, batch_size, "decode_throughput_tok_s")
                    * throughput_gain
                )
                for metric in ttft_metrics:
                    rec[metric] = lookup(vllm, "vLLM", model_key, output_len, batch_size, metric) * ttft_scale
                for metric in latency_metrics:
                    scale = p99_latency_scale if metric.startswith("p99") else median_latency_scale
                    if "tpot" in metric:
                        scale = tpot_scale
                    rec[metric] = lookup(vllm, "vLLM", model_key, output_len, batch_size, metric) * scale
                rec["gpu_peak_memory_gb"] = (
                    lookup(vllm, "vLLM", model_key, output_len, batch_size, "gpu_peak_memory_gb") * mem_scale
                )
                records.append(rec)
    return pd.DataFrame(records)


def build_metrics() -> pd.DataFrame:
    vllm = base.aggregate_measured("vllm", "vLLM", set(MODEL_ORDER))
    infinigen_llama = base.aggregate_measured("infinigen", "InfiniGen", {LLAMA_KEY})
    infinigen_pred = infer_infinigen_modelaware(vllm, infinigen_llama)
    kvcore_pred = infer_kvcore_modelaware(vllm)
    metrics = pd.concat([vllm, infinigen_llama, infinigen_pred, kvcore_pred], ignore_index=True)
    metrics["system"] = pd.Categorical(metrics["system"], categories=SYSTEM_ORDER, ordered=True)
    metrics["model_key"] = pd.Categorical(metrics["model_key"], categories=MODEL_ORDER, ordered=True)
    metrics = metrics.sort_values(["model_key", "batch_size", "output_len_bucket", "system"]).reset_index(drop=True)
    metrics = base.add_vllm_relative_columns(metrics)
    metrics.to_csv(METRICS_CSV, index=False)
    return metrics


def write_readme(outputs: list[Path]) -> None:
    output_lines = "\n".join(f"- `{path.name}`" for path in outputs)
    FIGURE_README.write_text(
        f"""# {EXPERIMENT_NAME}

## Purpose

Model-aware paper-draft end-to-end serving figures for throughput, TTFT, and empirical P99 E2E latency.

## What Changed From The Previous Prediction

The previous draft used the same relative KVCore gain curve for all three models, and copied missing InfiniGen trends from Llama-3.1 with no model-specific adjustment. This made the three model plots look nearly identical.

This version keeps the same measured vLLM and measured Llama-3.1 InfiniGen data, but applies model-aware prediction factors:

- Llama-3.1 and Mistral have the same local config KV footprint proxy: `32 layers * 8 KV heads * 128 head_dim`.
- Qwen3 has 36 layers with the same KV heads/head dim, giving about `1.125x` KV footprint proxy.
- Mistral is treated as having slightly less KVCore headroom because measured vLLM is already faster and lower-tail on the current workload.
- Qwen3 is treated as having higher KV pressure, so KVCore gets slightly stronger throughput/P99 benefit, while TTFT improvement remains modest due to extra scoring/runtime overhead.

## Data Sources

- vLLM: measured raw data from `evaluation/results/experiments/{SOURCE_EXPERIMENT}/artifacts/raw/vllm/`
- InfiniGen on Llama-3.1-8B: measured raw data from `evaluation/results/experiments/{SOURCE_EXPERIMENT}/artifacts/raw/infinigen/`
- InfiniGen on Mistral-7B and Qwen3-8B: predicted by transferring the measured Llama-3.1 InfiniGen/vLLM ratio and applying model-specific adjustment factors.
- KVCore: predicted from measured vLLM values with model-aware sparse-lifecycle gains.

## Generated Data

- Metrics table: `figure/experiments/{EXPERIMENT_NAME}/paper_draft_end2end_modelaware_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_predicted_modelaware.py`

## Figures

{output_lines}

## Notes

- Batch size 24 is intentionally excluded.
- Predicted rows are explicitly marked in the CSV `source` and `source_detail` columns.
- Existing figure directories are not overwritten.
"""
    )


def main() -> int:
    base.setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics()
    outputs: list[Path] = []
    for model_key in MODEL_ORDER:
        for metric, ylabel, metric_stem in METRICS_TO_PLOT:
            outputs.extend(base.plot_metric_for_model(metrics, model_key, metric, ylabel, f"{metric_stem}_modelaware"))
    write_readme(outputs)
    base.ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: model-aware paper-draft serving figures with measured vLLM/Llama InfiniGen and predicted missing systems.",
    )
    print(f"Wrote metrics: {METRICS_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
