from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figure" / "experiments" / "ruler_cross_layer_residual_50pct"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_ruler_metric_map(path: Path) -> dict[str, float]:
    metrics = json.loads(path.read_text())
    return {task: float(item["string_match"]) for task, item in metrics.items()}


def plot_group(metric_maps: dict[str, dict[str, float]], metric_names: list[str], output_path: Path, title: str):
    x = np.arange(len(metric_names))
    width = 0.2
    order = [
        "block_wise_old",
        "block_wise_token_correction",
        "block_wise_cross_layer_residual",
        "chunkkv",
    ]
    colors = {
        "block_wise_old": "#4C78A8",
        "block_wise_token_correction": "#F58518",
        "block_wise_cross_layer_residual": "#E45756",
        "chunkkv": "#54A24B",
    }
    labels = {
        "block_wise_old": "BlockWise old",
        "block_wise_token_correction": "BlockWise + token correction",
        "block_wise_cross_layer_residual": "BlockWise + cross-layer residual",
        "chunkkv": "ChunkKV",
    }

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(metric_names)), 5.8))
    for idx, name in enumerate(order):
        values = [metric_maps[name][metric] for metric in metric_names]
        ax.bar(x + (idx - 1.5) * width, values, width=width, color=colors[name], label=labels[name])

    ax.set_ylabel("String Match")
    ax.set_xlabel("RULER Subtasks")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


def main():
    metric_paths = {
        "block_wise_old": ROOT
        / "evaluation/results/experiments/prefill_compare_50pct_blockwise_chunkkv/artifacts/"
        / "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__block_wise__0.70__fraction0.500__query_aware/metrics.json",
        "block_wise_token_correction": ROOT
        / "evaluation/results/experiments/ruler_token_correction_50pct/artifacts/"
        / "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__block_wise__0.70__fraction0.500__query_aware/metrics.json",
        "block_wise_cross_layer_residual": ROOT
        / "evaluation/results/experiments/ruler_cross_layer_residual_50pct/artifacts/"
        / "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__block_wise__0.70__fraction0.500__query_aware__layerresw0.20/metrics.json",
        "chunkkv": ROOT
        / "evaluation/results/experiments/prefill_compare_50pct_blockwise_chunkkv/artifacts/"
        / "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__chunkkv__0.70__fraction0.500__query_aware/metrics.json",
    }
    metric_maps = {name: load_ruler_metric_map(path) for name, path in metric_paths.items()}

    all_metrics = list(metric_maps["block_wise_old"].keys())
    focus_metrics = ["niah_multikey_2", "niah_multikey_3", "niah_single_3", "qa_1", "qa_2"]

    plot_group(
        metric_maps,
        all_metrics,
        FIGURE_DIR / "ruler_cross_layer_residual_vs_baselines.png",
        "RULER 0.7: Cross-layer residual vs baselines",
    )
    plot_group(
        metric_maps,
        focus_metrics,
        FIGURE_DIR / "ruler_cross_layer_residual_vs_baselines_focus.png",
        "RULER 0.7 Focused Comparison",
    )


if __name__ == "__main__":
    main()
