from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "experiments" / "ruler_needle_prefill_layer_ratio_compare_15pct" / "artifacts"
OUTDIR = ROOT / "figure" / "experiments" / "ruler_needle_prefill_layer_ratio_compare_15pct"
SUMMARY_JSON = OUTDIR / "summary.json"
FIG_RULER = OUTDIR / "ruler_compare.png"
FIG_NEEDLE = OUTDIR / "needle_compare.png"

RULER_PATTERN = re.compile(
    r"ruler__4096__.*__(block_wise_prefill_per_layer|chunkkv_prefill_per_layer)__([0-9.]+)__fraction0\.150__query_aware__skipfirst(\d+)__.*"
)
NEEDLE_PATTERN = re.compile(
    r"needle_in_haystack__.*__(block_wise_prefill_per_layer|chunkkv_prefill_per_layer)__([0-9.]+)__fraction0\.150__max_context16384__query_aware__skipfirst(\d+)__needle_depth50"
)


def _method_label(method: str) -> str:
    return "BlockWise" if method.startswith("block_wise") else "ChunkKV"


def load_data():
    ruler_rows = []
    needle_rows = []
    for metrics_path in RESULT_ROOT.rglob("metrics.json"):
        rel = str(metrics_path.parent.relative_to(RESULT_ROOT))
        content = json.loads(metrics_path.read_text())

        mr = RULER_PATTERN.match(rel)
        if mr:
            method, ratio, skip = mr.groups()
            tasks = {
                task_name: float(metric["string_match"])
                for task_name, metric in content.items()
                if isinstance(metric, dict) and "string_match" in metric
            }
            macro = float(np.mean(list(tasks.values()))) if tasks else float("nan")
            ruler_rows.append(
                {
                    "method": method,
                    "ratio": float(ratio),
                    "skip": int(skip),
                    "macro": macro,
                    "tasks": tasks,
                }
            )
            continue

        mn = NEEDLE_PATTERN.match(rel)
        if mn:
            method, ratio, skip = mn.groups()
            rouge_l_f = float(content[0]["rouge-l"]["f"])
            needle_rows.append(
                {
                    "method": method,
                    "ratio": float(ratio),
                    "skip": int(skip),
                    "rouge_l_f": rouge_l_f,
                }
            )
    return ruler_rows, needle_rows


def plot_ruler(ruler_rows):
    ratios = [0.3, 0.5, 0.7]
    skips = [0, 1, 2]
    methods = ["block_wise_prefill_per_layer", "chunkkv_prefill_per_layer"]

    macro = {}
    task_scores = defaultdict(dict)
    for row in ruler_rows:
        key = (row["method"], row["ratio"], row["skip"])
        macro[key] = row["macro"]
        for task_name, score in row["tasks"].items():
            task_scores[task_name][key] = score

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Left: macro grouped bars by ratio/skip
    x = np.arange(len(ratios))
    width = 0.12
    color_map = {
        ("block_wise_prefill_per_layer", 0): "#1f77b4",
        ("block_wise_prefill_per_layer", 1): "#4e9cd6",
        ("block_wise_prefill_per_layer", 2): "#8fc2f0",
        ("chunkkv_prefill_per_layer", 0): "#ff7f0e",
        ("chunkkv_prefill_per_layer", 1): "#ffad5a",
        ("chunkkv_prefill_per_layer", 2): "#ffd199",
    }
    offset_idx = 0
    for method in methods:
        for skip in skips:
            ys = [macro.get((method, ratio, skip), np.nan) for ratio in ratios]
            offset = (offset_idx - 2.5) * width
            axes[0].bar(
                x + offset,
                ys,
                width=width,
                color=color_map[(method, skip)],
                label=f"{_method_label(method)}-s{skip}",
            )
            offset_idx += 1
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"r={r}" for r in ratios])
    axes[0].set_ylim(0, 105)
    axes[0].set_title("RULER (macro over available subtasks)")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncols=2, fontsize=8)

    # Right: task-level for skip=0 only (more readable)
    task_names = sorted(task_scores.keys())
    x2 = np.arange(len(task_names))
    width2 = 0.12
    offset_idx = 0
    for method in methods:
        for ratio in ratios:
            ys = [task_scores[t].get((method, ratio, 0), np.nan) for t in task_names]
            offset = (offset_idx - 2.5) * width2
            label = f"{_method_label(method)} r={ratio}"
            color = "#1f77b4" if method.startswith("block") else "#ff7f0e"
            alpha = 0.5 + 0.2 * ratios.index(ratio)
            axes[1].bar(x2 + offset, ys, width=width2, color=color, alpha=alpha, label=label)
            offset_idx += 1
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(task_names, rotation=20, ha="right")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("RULER task detail (skip_first=0)")
    axes[1].set_ylabel("String Match")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(ncols=2, fontsize=8)

    fig.suptitle("RULER: Prefill Layer Skip Comparison")
    fig.savefig(FIG_RULER, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_needle(needle_rows):
    ratios = [0.3, 0.5, 0.7]
    skips = [0, 1, 2]
    methods = ["block_wise_prefill_per_layer", "chunkkv_prefill_per_layer"]

    score = {}
    for row in needle_rows:
        score[(row["method"], row["ratio"], row["skip"])] = row["rouge_l_f"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Left: grouped bars
    x = np.arange(len(ratios))
    width = 0.12
    color_map = {
        ("block_wise_prefill_per_layer", 0): "#1f77b4",
        ("block_wise_prefill_per_layer", 1): "#4e9cd6",
        ("block_wise_prefill_per_layer", 2): "#8fc2f0",
        ("chunkkv_prefill_per_layer", 0): "#ff7f0e",
        ("chunkkv_prefill_per_layer", 1): "#ffad5a",
        ("chunkkv_prefill_per_layer", 2): "#ffd199",
    }
    offset_idx = 0
    for method in methods:
        for skip in skips:
            ys = [score.get((method, ratio, skip), np.nan) for ratio in ratios]
            offset = (offset_idx - 2.5) * width
            axes[0].bar(
                x + offset,
                ys,
                width=width,
                color=color_map[(method, skip)],
                label=f"{_method_label(method)}-s{skip}",
            )
            offset_idx += 1
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"r={r}" for r in ratios])
    axes[0].set_ylim(0.65, 0.75)
    axes[0].set_title("Needle (ROUGE-L F)")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncols=2, fontsize=8)

    # Right: delta from skip=0 (same method/ratio)
    x2 = np.arange(len(ratios))
    width2 = 0.18
    for i, method in enumerate(methods):
        delta_s1 = [score[(method, r, 1)] - score[(method, r, 0)] for r in ratios]
        delta_s2 = [score[(method, r, 2)] - score[(method, r, 0)] for r in ratios]
        base = -0.25 if i == 0 else 0.25
        axes[1].bar(x2 + base - width2 / 2, delta_s1, width=width2, label=f"{_method_label(method)} s1-s0")
        axes[1].bar(x2 + base + width2 / 2, delta_s2, width=width2, label=f"{_method_label(method)} s2-s0")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([f"r={r}" for r in ratios])
    axes[1].set_title("Needle delta vs skip_first=0")
    axes[1].set_ylabel("Delta ROUGE-L F")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Needle in a Haystack: Prefill Layer Skip Comparison")
    fig.savefig(FIG_NEEDLE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary(ruler_rows, needle_rows):
    summary = {
        "ruler": ruler_rows,
        "needle": needle_rows,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ruler_rows, needle_rows = load_data()
    if len(ruler_rows) != 18 or len(needle_rows) != 18:
        raise RuntimeError(
            f"Expected 18 rows for each dataset; got ruler={len(ruler_rows)} needle={len(needle_rows)}"
        )
    save_summary(ruler_rows, needle_rows)
    plot_ruler(ruler_rows)
    plot_needle(needle_rows)
    print(FIG_RULER)
    print(FIG_NEEDLE)
    print(SUMMARY_JSON)


if __name__ == "__main__":
    main()
