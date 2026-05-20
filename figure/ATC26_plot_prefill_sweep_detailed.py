from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19"
ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
METRICS_FULL_LONG = ARTIFACTS_DIR / "ATC26_metrics_full_long.csv"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME

LONGBENCH_SUBDATASET_LONG = ARTIFACTS_DIR / "ATC26_longbench_subdataset_long.csv"
LONGBENCH_SUBDATASET_WIDE = ARTIFACTS_DIR / "ATC26_longbench_subdataset_wide.csv"
NEEDLE_DEPTH_LONG = ARTIFACTS_DIR / "ATC26_needle_depth_long.csv"
NEEDLE_DEPTH_WIDE = ARTIFACTS_DIR / "ATC26_needle_depth_wide.csv"

METHOD_ORDER = ["blockwise", "snapkv", "chunkkv"]
METHOD_LABELS = {
    "blockwise": "BlockWise",
    "snapkv": "SnapKV",
    "chunkkv": "ChunkKV",
}
METHOD_COLORS = {
    "blockwise": "#1f77b4",
    "snapkv": "#2ca02c",
    "chunkkv": "#ff7f0e",
}
MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral_7b_instruct_v03": "Mistral-7B-Instruct-v0.3",
    "qwen3_8b": "Qwen3-8B",
}
LONGBENCH_ORDER = [
    "longbench:qasper",
    "longbench:multifieldqa_en",
    "longbench:hotpotqa",
    "longbench:2wikimqa",
    "longbench:musique",
    "longbench:triviaqa",
]


def ratio_label(value: float) -> str:
    return f"r{value:.1f}"


def extract_needle_rouge_l(metrics: Any) -> list[float]:
    if not isinstance(metrics, list):
        return []
    values: list[float] = []
    for item in metrics:
        score = item.get("rouge-l") or item.get("rouge_l")
        if isinstance(score, dict) and "f" in score:
            values.append(float(score["f"]) * 100.0)
    return values


def build_longbench_tables(df: pd.DataFrame) -> pd.DataFrame:
    longbench = df[df["dataset"].eq("longbench")].copy()
    longbench["model_label"] = longbench["model"].map(MODEL_LABELS).fillna(longbench["model"])
    longbench["method_label"] = longbench["method"].map(METHOD_LABELS).fillna(longbench["method"])
    longbench = longbench.sort_values(["model", "dataset_key", "method", "compression_ratio"])
    longbench.to_csv(LONGBENCH_SUBDATASET_LONG, index=False)

    wide = longbench.pivot_table(
        index=["model", "dataset_key", "method"],
        columns="compression_ratio",
        values="score",
        aggfunc="first",
    )
    wide = wide.rename(columns={ratio: ratio_label(float(ratio)) for ratio in wide.columns})
    wide.reset_index().to_csv(LONGBENCH_SUBDATASET_WIDE, index=False)
    return longbench


def build_needle_depth_tables(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    needle = df[df["dataset"].eq("needle_in_haystack")].copy()
    for row in needle.itertuples(index=False):
        metrics_path = REPO_ROOT / str(row.metrics_path)
        predictions_path = REPO_ROOT / str(row.predictions_path)
        if not metrics_path.exists() or not predictions_path.exists():
            continue
        scores = extract_needle_rouge_l(json.loads(metrics_path.read_text()))
        predictions = pd.read_csv(predictions_path)
        for idx, pred in predictions.iterrows():
            score = scores[idx] if idx < len(scores) else None
            records.append(
                {
                    "model": row.model,
                    "model_label": MODEL_LABELS.get(row.model, row.model),
                    "method": row.method,
                    "method_label": METHOD_LABELS.get(row.method, row.method),
                    "compression_ratio": float(row.compression_ratio),
                    "dataset_key": row.dataset_key,
                    "needle_depth": int(pred["needle_depth"]),
                    "rouge_l_f": score,
                    "predicted_answer": pred.get("predicted_answer", ""),
                }
            )
    depth_df = pd.DataFrame.from_records(records)
    depth_df = depth_df.sort_values(["model", "needle_depth", "method", "compression_ratio"])
    depth_df.to_csv(NEEDLE_DEPTH_LONG, index=False)

    wide = depth_df.pivot_table(
        index=["model", "needle_depth", "method"],
        columns="compression_ratio",
        values="rouge_l_f",
        aggfunc="first",
    )
    wide = wide.rename(columns={ratio: ratio_label(float(ratio)) for ratio in wide.columns})
    wide.reset_index().to_csv(NEEDLE_DEPTH_WIDE, index=False)
    return depth_df


def plot_longbench_subdatasets(longbench: pd.DataFrame) -> Path:
    models = list(MODEL_LABELS)
    datasets = LONGBENCH_ORDER
    fig, axes = plt.subplots(
        len(models),
        len(datasets),
        figsize=(3.3 * len(datasets), 3.0 * len(models)),
        sharex=True,
        squeeze=False,
    )
    for row_idx, model in enumerate(models):
        for col_idx, dataset_key in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            subset = longbench[(longbench["model"].eq(model)) & (longbench["dataset_key"].eq(dataset_key))]
            for method in METHOD_ORDER:
                method_df = subset[subset["method"].eq(method)].sort_values("compression_ratio")
                if method_df.empty:
                    continue
                ax.plot(
                    method_df["compression_ratio"],
                    method_df["score"],
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            if row_idx == 0:
                ax.set_title(dataset_key.replace("longbench:", ""), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(MODEL_LABELS.get(model, model), fontsize=9)
            if row_idx == len(models) - 1:
                ax.set_xlabel("Compression ratio")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHOD_ORDER), frameon=False)
    fig.suptitle("ATC26 LongBench Subdataset Score vs Compression", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output = FIGURE_DIR / "ATC26_longbench_subdataset_quality_grid.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_needle_depths(depth_df: pd.DataFrame) -> Path:
    models = list(MODEL_LABELS)
    depths = sorted(depth_df["needle_depth"].dropna().unique())
    fig, axes = plt.subplots(
        len(models),
        len(depths),
        figsize=(3.0 * len(depths), 3.0 * len(models)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_idx, model in enumerate(models):
        for col_idx, depth in enumerate(depths):
            ax = axes[row_idx][col_idx]
            subset = depth_df[(depth_df["model"].eq(model)) & (depth_df["needle_depth"].eq(depth))]
            for method in METHOD_ORDER:
                method_df = subset[subset["method"].eq(method)].sort_values("compression_ratio")
                if method_df.empty:
                    continue
                ax.plot(
                    method_df["compression_ratio"],
                    method_df["rouge_l_f"],
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            if row_idx == 0:
                ax.set_title(f"depth={depth}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(MODEL_LABELS.get(model, model), fontsize=9)
            if row_idx == len(models) - 1:
                ax.set_xlabel("Compression ratio")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHOD_ORDER), frameon=False)
    fig.suptitle("ATC26 Needle ROUGE-L F1 by Depth vs Compression", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output = FIGURE_DIR / "ATC26_needle_depth_quality_grid.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_FULL_LONG)
    longbench = build_longbench_tables(df)
    needle_depth = build_needle_depth_tables(df)
    outputs = [
        plot_longbench_subdatasets(longbench),
        plot_needle_depths(needle_depth),
    ]
    for output in outputs:
        print(output)
    print(LONGBENCH_SUBDATASET_LONG)
    print(LONGBENCH_SUBDATASET_WIDE)
    print(NEEDLE_DEPTH_LONG)
    print(NEEDLE_DEPTH_WIDE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
