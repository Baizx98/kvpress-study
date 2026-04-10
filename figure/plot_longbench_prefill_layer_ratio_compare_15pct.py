from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "experiments" / "longbench_prefill_layer_ratio_compare_15pct" / "artifacts"
OUTDIR = ROOT / "figure" / "experiments" / "longbench_prefill_layer_ratio_compare_15pct"
SUMMARY_JSON = OUTDIR / "summary.json"
FIG_BAR = OUTDIR / "dataset_grouped_bar.png"
FIG_LINE = OUTDIR / "ratio_trend_by_dataset.png"

PATTERN = re.compile(
    r"longbench__(.*?)__.*__(block_wise_prefill_per_layer|chunkkv_prefill_per_layer)__([0-9.]+)__fraction0\.150__query_aware__skipfirst(\d+)(?:/(\d+))?$"
)


def load_best_results() -> dict[tuple[str, str, float, int], dict]:
    rows: list[dict] = []
    for metrics_path in RESULT_ROOT.rglob("metrics.json"):
        try:
            score = float(json.loads(metrics_path.read_text()))
        except Exception:
            continue
        rel = str(metrics_path.parent.relative_to(RESULT_ROOT))
        match = PATTERN.match(rel)
        if not match:
            continue
        dataset, press, ratio, skip_first, rerun = match.groups()
        rows.append(
            {
                "dataset": dataset,
                "press": press,
                "ratio": float(ratio),
                "skip_first": int(skip_first),
                "score": score,
                "rel": rel,
                "rerun": int(rerun) if rerun is not None else 0,
            }
        )

    best: dict[tuple[str, str, float, int], dict] = {}
    for row in rows:
        key = (row["dataset"], row["press"], row["ratio"], row["skip_first"])
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue
        # Prefer explicit rerun subdir with larger id, then lexicographically newer path.
        if (row["rerun"], row["rel"]) > (prev["rerun"], prev["rel"]):
            best[key] = row
    return best


def plot_grouped_bar(best: dict[tuple[str, str, float, int], dict]) -> None:
    datasets = sorted({k[0] for k in best.keys()})
    ratios = [0.3, 0.5, 0.7]
    methods = ["block_wise_prefill_per_layer", "chunkkv_prefill_per_layer"]
    skips = [0, 1, 2]

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    color_map = {
        ("block_wise_prefill_per_layer", 0): "#1f77b4",
        ("block_wise_prefill_per_layer", 1): "#4e9cd6",
        ("block_wise_prefill_per_layer", 2): "#8fc2f0",
        ("chunkkv_prefill_per_layer", 0): "#ff7f0e",
        ("chunkkv_prefill_per_layer", 1): "#ffad5a",
        ("chunkkv_prefill_per_layer", 2): "#ffd199",
    }

    for ax, dataset in zip(axes, datasets):
        bar_labels = [f"r={ratio}" for ratio in ratios]
        x = np.arange(len(ratios))
        width = 0.12
        offset_idx = 0
        for method in methods:
            for skip in skips:
                y = []
                for ratio in ratios:
                    row = best.get((dataset, method, ratio, skip))
                    y.append(np.nan if row is None else row["score"])
                offset = (offset_idx - 2.5) * width
                label = f"{'BW' if method.startswith('block') else 'CK'}-s{skip}"
                ax.bar(x + offset, y, width=width, label=label, color=color_map[(method, skip)], alpha=0.95)
                offset_idx += 1

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels)
        ax.set_title(dataset)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncols=2, fontsize=8)

    fig.suptitle("LongBench 15%: Prefill Per-Layer Ratio (BW vs ChunkKV)")
    fig.savefig(FIG_BAR, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_trend(best: dict[tuple[str, str, float, int], dict]) -> None:
    datasets = sorted({k[0] for k in best.keys()})
    ratios = [0.3, 0.5, 0.7]
    methods = ["block_wise_prefill_per_layer", "chunkkv_prefill_per_layer"]
    skips = [0, 1, 2]
    method_name = {
        "block_wise_prefill_per_layer": "BlockWise",
        "chunkkv_prefill_per_layer": "ChunkKV",
    }

    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 4 * len(datasets)), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    style = {
        ("block_wise_prefill_per_layer", 0): ("#1f77b4", "-"),
        ("block_wise_prefill_per_layer", 1): ("#1f77b4", "--"),
        ("block_wise_prefill_per_layer", 2): ("#1f77b4", ":"),
        ("chunkkv_prefill_per_layer", 0): ("#ff7f0e", "-"),
        ("chunkkv_prefill_per_layer", 1): ("#ff7f0e", "--"),
        ("chunkkv_prefill_per_layer", 2): ("#ff7f0e", ":"),
    }

    for ax, dataset in zip(axes, datasets):
        for method in methods:
            for skip in skips:
                ys = []
                for ratio in ratios:
                    row = best.get((dataset, method, ratio, skip))
                    ys.append(np.nan if row is None else row["score"])
                color, ls = style[(method, skip)]
                ax.plot(
                    ratios,
                    ys,
                    marker="o",
                    linestyle=ls,
                    color=color,
                    linewidth=1.8,
                    label=f"{method_name[method]} skip{skip}",
                )
        ax.set_xlim(0.28, 0.72)
        ax.set_xticks(ratios)
        ax.set_title(dataset)
        ax.set_ylabel("Score")
        ax.grid(alpha=0.25)
        ax.legend(ncols=3, fontsize=8)

    axes[-1].set_xlabel("Compression Ratio")
    fig.suptitle("Trend by Ratio and Skip-First-Layers")
    fig.savefig(FIG_LINE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def dump_summary(best: dict[tuple[str, str, float, int], dict]) -> None:
    serializable = {}
    for key, row in sorted(best.items()):
        dataset, press, ratio, skip = key
        serializable.setdefault(dataset, {}).setdefault(press, {}).setdefault(str(ratio), {})[str(skip)] = row["score"]
    SUMMARY_JSON.write_text(json.dumps(serializable, indent=2))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    best = load_best_results()
    if not best:
        raise RuntimeError(f"No valid metrics found under {RESULT_ROOT}")
    dump_summary(best)
    plot_grouped_bar(best)
    plot_ratio_trend(best)
    print(FIG_BAR)
    print(FIG_LINE)
    print(SUMMARY_JSON)


if __name__ == "__main__":
    main()
