from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results" / "experiments" / "ruler_residual_ablation_fast" / "artifacts"
FIGURE_DIR = ROOT / "figure" / "experiments" / "ruler_residual_ablation_fast"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_records():
    records = {}
    for metrics_path in sorted(RESULTS_DIR.glob("**/metrics.json")):
        match = re.search(r"layerresw([0-9.]+)", str(metrics_path))
        if not match:
            continue
        weight = float(match.group(1))
        records[weight] = json.loads(metrics_path.read_text())
    return dict(sorted(records.items()))


def main():
    records = load_records()
    if not records:
        raise RuntimeError("No residual ablation results found.")

    tasks = list(next(iter(records.values())).keys())
    weights = list(records.keys())

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.2), squeeze=False)
    flat_axes = axes.flatten()

    for axis, task in zip(flat_axes, tasks):
        values = [records[weight][task]["string_match"] for weight in weights]
        axis.plot(weights, values, marker="o", linewidth=2, color="#E45756")
        axis.set_title(task)
        axis.set_xlabel("Residual Weight")
        axis.set_ylabel("String Match")
        axis.grid(True, alpha=0.3)
        axis.set_xticks(weights)

    fig.suptitle("RULER fast residual ablation on key subtasks", y=1.02, fontsize=14)
    fig.tight_layout()
    output_path = FIGURE_DIR / "ruler_residual_ablation_fast.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
