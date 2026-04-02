from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figure"
FIGURE_DIR.mkdir(exist_ok=True)

ABLATION_PARAMS = {
    "q_window": ("q_window_size", [32, 64, 96]),
    "summary_topk": ("summary_topk_keys", [1, 2, 4]),
    "protected_recent": ("protected_recent_blocks", [2, 4, 8]),
    "mean_key_weight": ("mean_key_weight", [0.25, 0.5, 0.75]),
}


def load_records(results_dir: Path):
    records = []
    for config_path in results_dir.rglob("config.yaml"):
        metrics_path = config_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text()) or {}
        if config.get("dataset") != "ruler" or config.get("press_name") != "block_wise":
            continue
        metrics = json.loads(metrics_path.read_text())
        record = {
            "ratio": float(config["compression_ratio"]),
            "q_window_size": int(config.get("q_window_size", 64)),
            "summary_topk_keys": int(config.get("summary_topk_keys", 2)),
            "protected_recent_blocks": int(config.get("protected_recent_blocks", 4)),
            "mean_key_weight": float(config.get("mean_key_weight", 0.5)),
            "metrics": {k: float(v["string_match"]) for k, v in metrics.items() if "string_match" in v},
        }
        records.append(record)
    return records


def filter_records(records, ablation_name: str):
    param_name, _ = ABLATION_PARAMS[ablation_name]
    defaults = {
        "q_window_size": 64,
        "summary_topk_keys": 2,
        "protected_recent_blocks": 4,
        "mean_key_weight": 0.5,
    }
    filtered = []
    for record in records:
        keep = True
        for other_name, default_value in defaults.items():
            if other_name == param_name:
                continue
            if record[other_name] != default_value:
                keep = False
                break
        if keep:
            filtered.append(record)
    return filtered


def plot_ablation(records, ablation_name: str, output_path: Path):
    param_name, param_values = ABLATION_PARAMS[ablation_name]
    filtered = filter_records(records, ablation_name)
    metric_names = sorted({metric for record in filtered for metric in record["metrics"]})
    if not filtered or not metric_names:
        raise RuntimeError(f"No usable records found for ablation={ablation_name}")

    n_metrics = len(metric_names)
    ncols = 4
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for axis, metric_name in zip(flat_axes, metric_names):
        for ratio in [0.5, 0.7]:
            ratio_records = sorted(
                [record for record in filtered if record["ratio"] == ratio],
                key=lambda item: param_values.index(item[param_name]),
            )
            if not ratio_records:
                continue
            axis.plot(
                [record[param_name] for record in ratio_records],
                [record["metrics"][metric_name] for record in ratio_records],
                marker="o",
                linewidth=2,
                label=f"ratio={ratio}",
            )

        axis.set_title(metric_name)
        axis.set_xlabel(param_name)
        axis.set_ylabel("string_match")
        axis.grid(True, alpha=0.3)
        axis.set_xticks(param_values)

    for axis in flat_axes[n_metrics:]:
        axis.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle(f"RULER Ablation: {ablation_name}", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=list(ABLATION_PARAMS.keys()),
        choices=list(ABLATION_PARAMS.keys()),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ruler_ablation_10pct",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.results_dir)
    for ablation_name in args.ablations:
        output_path = FIGURE_DIR / f"{args.prefix}_{ablation_name}.png"
        plot_ablation(records, ablation_name, output_path)


if __name__ == "__main__":
    main()
