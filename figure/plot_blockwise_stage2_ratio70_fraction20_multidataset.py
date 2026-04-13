from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "blockwise_stage2_ratio70_fraction20_multidataset"
ARTIFACTS_DIR = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME
SUMMARY_JSON = OUTDIR / "summary.json"
FIG_ALL = OUTDIR / "stage2_multidataset_grouped_bar.png"
FIG_BLOCKWISE = OUTDIR / "stage2_blockwise_family_bar.png"
FIG_DELTA = OUTDIR / "stage2_blockwise_vs_chunkkv_delta.png"

DATASET_ORDER = [
    "ruler:4096",
    "longbench:qasper",
    "longbench:multifieldqa_en",
    "longbench:hotpotqa",
    "longbench:2wikimqa",
    "longbench:musique",
    "longbench:triviaqa",
    "needle_in_haystack:16384",
]

DATASET_LABELS = {
    "ruler:4096": "RULER\n4096",
    "longbench:qasper": "qasper",
    "longbench:multifieldqa_en": "multifieldqa_en",
    "longbench:hotpotqa": "hotpotqa",
    "longbench:2wikimqa": "2wikimqa",
    "longbench:musique": "musique",
    "longbench:triviaqa": "triviaqa",
    "needle_in_haystack:16384": "Needle\n16k",
}

METHOD_ORDER = [
    "blockwise_main",
    "blockwise_norm_topk",
    "blockwise_multi_rep",
    "blockwise_tail_query_special",
    "chunkkv_prefill",
]

METHOD_LABELS = {
    "blockwise_main": "BW-main",
    "blockwise_norm_topk": "BW-norm-topk",
    "blockwise_multi_rep": "BW-multi-rep",
    "blockwise_tail_query_special": "BW-tail-query",
    "chunkkv_prefill": "ChunkKV",
}

METHOD_COLORS = {
    "blockwise_main": "#1f77b4",
    "blockwise_norm_topk": "#4e9cd6",
    "blockwise_multi_rep": "#17becf",
    "blockwise_tail_query_special": "#2ca02c",
    "chunkkv_prefill": "#ff7f0e",
}


@dataclass
class Record:
    dataset_key: str
    method_key: str
    score: float
    detail: str
    path: Path


def classify_dataset(cfg: dict) -> str:
    dataset = cfg.get("dataset")
    data_dir = cfg.get("data_dir")
    if dataset == "ruler":
        return f"ruler:{data_dir}"
    if dataset == "longbench":
        return f"longbench:{data_dir}"
    if dataset == "needle_in_haystack":
        return f"needle_in_haystack:{cfg.get('max_context_length')}"
    return f"{dataset}:{data_dir}"


def classify_method(cfg: dict) -> str:
    press = cfg.get("press_name")
    if press == "chunkkv_prefill_per_layer":
        return "chunkkv_prefill"

    summary = cfg.get("summary_mode")
    rep = cfg.get("representative_mode")
    qagg = cfg.get("query_agg_mode")
    hagg = cfg.get("head_agg_mode")

    if rep == "tail_query_relevance":
        return "blockwise_tail_query_special"
    if summary == "norm_topk_mean_only":
        return "blockwise_norm_topk"
    if summary == "multi_rep_max":
        return "blockwise_multi_rep"
    if (
        summary == "mean_plus_norm_topk_mean"
        and rep == "key_norm"
        and qagg == "max"
        and hagg == "uniform_mean"
    ):
        return "blockwise_main"
    return "unknown"


def parse_metrics(dataset_key: str, metrics_path: Path) -> tuple[float, str]:
    value = json.loads(metrics_path.read_text())

    if isinstance(value, (int, float)):
        score = float(value)
        return score, f"{score:.2f}"

    if isinstance(value, dict):
        task_scores = []
        parts = []
        for task_name in sorted(value):
            task_metrics = value[task_name]
            if isinstance(task_metrics, dict) and "string_match" in task_metrics:
                task_score = float(task_metrics["string_match"])
                task_scores.append(task_score)
                parts.append(f"{task_name}={task_score:.2f}")
        if task_scores:
            avg_score = sum(task_scores) / len(task_scores)
            return avg_score, f"avg={avg_score:.2f}; " + ", ".join(parts)

    if isinstance(value, list) and dataset_key.startswith("needle_in_haystack:"):
        rouge_l_scores = []
        for item in value:
            rouge_l = item.get("rouge-l") if isinstance(item, dict) else None
            if isinstance(rouge_l, dict) and "f" in rouge_l:
                rouge_l_scores.append(float(rouge_l["f"]) * 100.0)
        if rouge_l_scores:
            avg_score = sum(rouge_l_scores) / len(rouge_l_scores)
            return avg_score, f"avg_rouge_l_f={avg_score:.2f}"

    raise ValueError(f"Unsupported metrics format: {metrics_path}")


def load_best_records() -> dict[str, dict[str, Record]]:
    best: dict[str, dict[str, Record]] = {dataset: {} for dataset in DATASET_ORDER}
    for cfg_path in ARTIFACTS_DIR.rglob("config.yaml"):
        metrics_path = cfg_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        dataset_key = classify_dataset(cfg)
        method_key = classify_method(cfg)
        if dataset_key not in DATASET_ORDER or method_key not in METHOD_ORDER:
            continue
        score, detail = parse_metrics(dataset_key, metrics_path)
        record = Record(dataset_key=dataset_key, method_key=method_key, score=score, detail=detail, path=cfg_path.parent)
        current = best[dataset_key].get(method_key)
        if current is None or len(record.path.parts) < len(current.path.parts) or record.path.stat().st_mtime_ns >= current.path.stat().st_mtime_ns:
            best[dataset_key][method_key] = record
    return best


def dump_summary(best: dict[str, dict[str, Record]]) -> None:
    data = {}
    for dataset_key in DATASET_ORDER:
        data[dataset_key] = {}
        for method_key in METHOD_ORDER:
            record = best[dataset_key].get(method_key)
            if record is None:
                continue
            data[dataset_key][method_key] = {
                "score": record.score,
                "detail": record.detail,
                "path": str(record.path),
            }
    SUMMARY_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def plot_grouped_bar(best: dict[str, dict[str, Record]]) -> None:
    x = np.arange(len(DATASET_ORDER))
    width = 0.15

    fig, ax = plt.subplots(figsize=(18, 6.5), constrained_layout=True)
    for idx, method_key in enumerate(METHOD_ORDER):
        scores = []
        for dataset_key in DATASET_ORDER:
            record = best[dataset_key].get(method_key)
            scores.append(np.nan if record is None else record.score)
        offset = (idx - 2) * width
        bars = ax.bar(
            x + offset,
            scores,
            width=width,
            label=METHOD_LABELS[method_key],
            color=METHOD_COLORS[method_key],
            alpha=0.95,
        )
        for bar, score in zip(bars, scores):
            if np.isnan(score):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, score + 0.45, f"{score:.1f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[key] for key in DATASET_ORDER], rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("Stage2: Multi-dataset Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=5, frameon=False, fontsize=9)
    fig.savefig(FIG_ALL, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_blockwise_family(best: dict[str, dict[str, Record]]) -> None:
    methods = [
        "blockwise_main",
        "blockwise_norm_topk",
        "blockwise_multi_rep",
        "blockwise_tail_query_special",
    ]
    x = np.arange(len(DATASET_ORDER))
    width = 0.18

    fig, ax = plt.subplots(figsize=(18, 6.5), constrained_layout=True)
    for idx, method_key in enumerate(methods):
        scores = []
        for dataset_key in DATASET_ORDER:
            record = best[dataset_key].get(method_key)
            scores.append(np.nan if record is None else record.score)
        offset = (idx - 1.5) * width
        ax.bar(
            x + offset,
            scores,
            width=width,
            label=METHOD_LABELS[method_key],
            color=METHOD_COLORS[method_key],
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[key] for key in DATASET_ORDER], rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("Stage2: Blockwise Family Only")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, frameon=False, fontsize=9)
    fig.savefig(FIG_BLOCKWISE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta_vs_chunkkv(best: dict[str, dict[str, Record]]) -> None:
    methods = ["blockwise_main", "blockwise_norm_topk", "blockwise_multi_rep", "blockwise_tail_query_special"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()

    for ax, method_key in zip(axes, methods):
        labels = []
        deltas = []
        colors = []
        for dataset_key in DATASET_ORDER:
            block_record = best[dataset_key].get(method_key)
            chunk_record = best[dataset_key].get("chunkkv_prefill")
            if block_record is None or chunk_record is None:
                continue
            delta = block_record.score - chunk_record.score
            labels.append(DATASET_LABELS[dataset_key])
            deltas.append(delta)
            colors.append("#2ca02c" if delta >= 0 else "#d62728")

        ypos = np.arange(len(labels))
        ax.barh(ypos, deltas, color=colors, alpha=0.9)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.set_title(f"{METHOD_LABELS[method_key]} - ChunkKV")
        ax.set_xlabel("Score Delta")
        ax.grid(axis="x", alpha=0.25)
        for y, delta in zip(ypos, deltas):
            ax.text(delta + (0.2 if delta >= 0 else -0.2), y, f"{delta:.2f}", va="center", ha="left" if delta >= 0 else "right", fontsize=8)

    fig.suptitle("Stage2: Blockwise Variants vs ChunkKV")
    fig.savefig(FIG_DELTA, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    best = load_best_records()
    dump_summary(best)
    plot_grouped_bar(best)
    plot_blockwise_family(best)
    plot_delta_vs_chunkkv(best)
    print(FIG_ALL)
    print(FIG_BLOCKWISE)
    print(FIG_DELTA)
    print(SUMMARY_JSON)


if __name__ == "__main__":
    main()
