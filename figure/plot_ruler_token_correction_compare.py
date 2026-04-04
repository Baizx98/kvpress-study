from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figure" / "experiments" / "ruler_token_correction_50pct"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_ruler_metric_map(path: Path) -> dict[str, float]:
    metrics = json.loads(path.read_text())
    return {task: float(item["string_match"]) for task, item in metrics.items()}


def main():
    metric_paths = {
        "block_wise_old": ROOT
        / "evaluation/results/experiments/prefill_compare_50pct_blockwise_chunkkv/artifacts/"
        / "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__block_wise__0.70__fraction0.500__query_aware/metrics.json",
        "block_wise_token_correction": ROOT
        / "evaluation/results/experiments/ruler_token_correction_50pct/artifacts/"
        / "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__block_wise__0.70__fraction0.500__query_aware/metrics.json",
        "chunkkv": ROOT
        / "evaluation/results/experiments/prefill_compare_50pct_blockwise_chunkkv/artifacts/"
        / "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__chunkkv__0.70__fraction0.500__query_aware/metrics.json",
    }

    metric_maps = {name: load_ruler_metric_map(path) for name, path in metric_paths.items()}
    metric_names = list(metric_maps["block_wise_old"].keys())

    x = np.arange(len(metric_names))
    width = 0.25
    colors = {
        "block_wise_old": "#4C78A8",
        "block_wise_token_correction": "#F58518",
        "chunkkv": "#54A24B",
    }
    labels = {
        "block_wise_old": "BlockWise old",
        "block_wise_token_correction": "BlockWise + token correction",
        "chunkkv": "ChunkKV",
    }

    fig, ax = plt.subplots(figsize=(17, 5.8))
    for idx, name in enumerate(["block_wise_old", "block_wise_token_correction", "chunkkv"]):
        values = [metric_maps[name][metric] for metric in metric_names]
        ax.bar(x + (idx - 1) * width, values, width=width, color=colors[name], label=labels[name])

    ax.set_ylabel("String Match")
    ax.set_xlabel("RULER Subtasks")
    ax.set_title("RULER 0.7: BlockWise token correction vs old BlockWise vs ChunkKV")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / "ruler_token_correction_vs_baselines.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)

    focus_metrics = ["niah_multikey_2", "niah_multikey_3", "niah_single_3", "qa_1", "qa_2"]
    x_focus = np.arange(len(focus_metrics))
    fig2, ax2 = plt.subplots(figsize=(10.5, 5))
    for idx, name in enumerate(["block_wise_old", "block_wise_token_correction", "chunkkv"]):
        values = [metric_maps[name][metric] for metric in focus_metrics]
        ax2.bar(x_focus + (idx - 1) * width, values, width=width, color=colors[name], label=labels[name])

    ax2.set_ylabel("String Match")
    ax2.set_xlabel("Key RULER Subtasks")
    ax2.set_title("RULER 0.7 Focused Comparison")
    ax2.set_xticks(x_focus)
    ax2.set_xticklabels(focus_metrics, rotation=25, ha="right")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    focus_output_path = FIGURE_DIR / "ruler_token_correction_vs_baselines_focus.png"
    fig2.savefig(focus_output_path, dpi=220, bbox_inches="tight")
    print(focus_output_path)


if __name__ == "__main__":
    main()
