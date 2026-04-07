from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "evaluation" / "results" / "experiments" / "batch_main_compare_ratio05" / "artifacts"
OUTDIR = ROOT / "figure" / "experiments" / "batch_main_compare_ratio05"


def load_scalar_metric(path: Path) -> float:
    return float(path.read_text().strip())


def load_json_metric(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    longbench_tasks = ["hotpotqa", "multifieldqa_en", "triviaqa"]
    methods = ["block_wise", "chunkkv"]
    labels = {"block_wise": "BlockWise", "chunkkv": "ChunkKV"}
    colors = {"block_wise": "#1f77b4", "chunkkv": "#ff7f0e"}

    longbench_scores: dict[str, dict[str, float]] = {m: {} for m in methods}
    for task in longbench_tasks:
        for method in methods:
            path = (
                ARTIFACTS
                / f"longbench__{task}__--Tan--model--Llama-3.1-8B-Instruct__{method}__0.50__query_aware"
                / "metrics.json"
            )
            longbench_scores[method][task] = load_scalar_metric(path)

    lb2 = {}
    for method in methods:
        path = (
            ARTIFACTS
            / f"longbench-v2__0shot__--Tan--model--Llama-3.1-8B-Instruct__{method}__0.50__max_context32768__query_aware"
            / "metrics.json"
        )
        lb2[method] = load_json_metric(path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    x = list(range(len(longbench_tasks)))
    width = 0.36
    for i, method in enumerate(methods):
        vals = [longbench_scores[method][task] for task in longbench_tasks]
        offset = -width / 2 if i == 0 else width / 2
        bars = ax.bar(
            [xi + offset for xi in x],
            vals,
            width=width,
            label=labels[method],
            color=colors[method],
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(longbench_tasks, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("LongBench")
    ax.legend(frameon=False)

    ax = axes[1]
    lb2_keys = ["average", "easy", "hard", "short", "medium", "long"]
    x = list(range(len(lb2_keys)))
    for i, method in enumerate(methods):
        vals = [100.0 * lb2[method][k] for k in lb2_keys]
        offset = -width / 2 if i == 0 else width / 2
        bars = ax.bar(
            [xi + offset for xi in x],
            vals,
            width=width,
            label=labels[method],
            color=colors[method],
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(lb2_keys, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("LongBench-v2 / 0shot (max_context=32768)")
    ax.legend(frameon=False)

    fig.suptitle("BlockWise vs ChunkKV at Compression Ratio 0.5\nCompleted Batch-Oriented Benchmarks", fontsize=14)
    out = OUTDIR / "batch_main_compare_ratio05_partial.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
