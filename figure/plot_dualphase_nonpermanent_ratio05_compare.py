from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PERM_DIR = ROOT / "evaluation" / "results" / "experiments" / "batch_main_compare_ratio05" / "artifacts"
NONPERM_DIR = ROOT / "evaluation" / "results" / "experiments" / "dualphase_nonpermanent_ratio05" / "artifacts"
OUTDIR = ROOT / "figure" / "experiments" / "dualphase_nonpermanent_ratio05"

METHODS = ["block_wise", "dual_phase_per_layer", "chunkkv"]
LABELS = {
    "block_wise": "BlockWise Permanent",
    "dual_phase_per_layer": "DualPhase Non-Permanent",
    "chunkkv": "ChunkKV",
}
COLORS = {
    "block_wise": "#1f77b4",
    "dual_phase_per_layer": "#2ca02c",
    "chunkkv": "#ff7f0e",
}


def _metric_path(base: Path, dataset: str, data_dir: str, method: str) -> Path:
    pattern = f"{dataset}__{data_dir}__*__{method}__0.50__*query_aware/metrics.json"
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def load_scalar_metric(base: Path, dataset: str, data_dir: str, method: str) -> float:
    path = _metric_path(base, dataset, data_dir, method)
    return float(path.read_text().strip())


def load_json_metric(base: Path, dataset: str, data_dir: str, method: str) -> dict:
    path = _metric_path(base, dataset, data_dir, method)
    return json.loads(path.read_text())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    longbench_tasks = ["hotpotqa", "multifieldqa_en", "triviaqa"]
    lb_scores: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    for task in longbench_tasks:
        for method in METHODS:
            base = NONPERM_DIR if method == "dual_phase_per_layer" else PERM_DIR
            lb_scores[method][task] = load_scalar_metric(base, "longbench", task, method)

    lb2_scores = {}
    for method in METHODS:
        base = NONPERM_DIR if method == "dual_phase_per_layer" else PERM_DIR
        lb2_scores[method] = load_json_metric(base, "longbench-v2", "0shot", method)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)

    width = 0.24
    x = list(range(len(longbench_tasks)))
    ax = axes[0]
    for i, method in enumerate(METHODS):
        vals = [lb_scores[method][task] for task in longbench_tasks]
        offs = (i - 1) * width
        bars = ax.bar(
            [xi + offs for xi in x],
            vals,
            width=width,
            label=LABELS[method],
            color=COLORS[method],
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.45, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(longbench_tasks, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("LongBench")
    ax.legend(frameon=False, fontsize=9)

    lb2_keys = ["average", "easy", "hard", "short", "medium", "long"]
    x = list(range(len(lb2_keys)))
    ax = axes[1]
    for i, method in enumerate(METHODS):
        vals = [100.0 * lb2_scores[method][k] for k in lb2_keys]
        offs = (i - 1) * width
        bars = ax.bar(
            [xi + offs for xi in x],
            vals,
            width=width,
            label=LABELS[method],
            color=COLORS[method],
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(lb2_keys, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("LongBench-v2 / 0shot (max_context=32768)")

    fig.suptitle(
        "Compression Ratio 0.5: Permanent vs Non-Permanent BlockWise vs ChunkKV",
        fontsize=14,
    )
    out = OUTDIR / "dualphase_nonpermanent_ratio05_compare.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
