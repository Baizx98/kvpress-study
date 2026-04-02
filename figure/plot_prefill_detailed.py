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


def load_records(results_dir: Path):
    records_by_key = {}
    for config_path in results_dir.rglob("config.yaml"):
        metrics_path = config_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue

        config = yaml.safe_load(config_path.read_text()) or {}
        dataset = config.get("dataset")
        press = config.get("press_name")
        ratio = config.get("compression_ratio")
        if dataset is None or press is None or ratio is None:
            continue

        key = (dataset, press, float(ratio))
        candidate = {
            "dataset": dataset,
            "press": press,
            "ratio": float(ratio),
            "metrics": json.loads(metrics_path.read_text()),
            "needle_depth": config.get("needle_depth"),
            "_path_depth": len(config_path.relative_to(results_dir).parts),
        }
        current = records_by_key.get(key)
        if current is None or candidate["_path_depth"] < current["_path_depth"]:
            records_by_key[key] = candidate

    records = []
    for record in records_by_key.values():
        record.pop("_path_depth", None)
        records.append(record)
    return records


def build_dataset_metric_map(records):
    dataset_metric_map = {}
    for record in records:
        dataset = record["dataset"]
        metrics = record["metrics"]
        metric_map: dict[str, float] = {}

        if dataset == "ruler":
            for task_name, task_metrics in metrics.items():
                if isinstance(task_metrics, dict) and "string_match" in task_metrics:
                    metric_map[task_name] = float(task_metrics["string_match"])
        elif dataset == "needle_in_haystack":
            metric_name = "rouge-l_f"
            for depth, metric_item in zip(record.get("needle_depth") or [], metrics):
                metric_map[f"depth_{depth}_{metric_name}"] = float(metric_item["rouge-l"]["f"])
        elif isinstance(metrics, (int, float)):
            metric_map["score"] = float(metrics)
        else:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    if "accuracy" in metric_value:
                        metric_map[metric_name] = float(metric_value["accuracy"])
                    elif "string_match" in metric_value:
                        metric_map[metric_name] = float(metric_value["string_match"])

        record["metric_map"] = metric_map
        dataset_metric_map.setdefault(dataset, set()).update(metric_map.keys())

    return {k: sorted(v) for k, v in dataset_metric_map.items()}


def plot_dataset(records, dataset: str, metric_names: list[str], output_path: Path):
    dataset_records = [record for record in records if record["dataset"] == dataset]
    if not dataset_records or not metric_names:
        raise RuntimeError(f"No records or metrics found for dataset={dataset}")

    presses = sorted({record["press"] for record in dataset_records})
    n_metrics = len(metric_names)
    ncols = min(4, n_metrics)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for axis, metric_name in zip(flat_axes, metric_names):
        for press in presses:
            press_records = sorted(
                [record for record in dataset_records if record["press"] == press and metric_name in record["metric_map"]],
                key=lambda item: item["ratio"],
            )
            if not press_records:
                continue

            axis.plot(
                [record["ratio"] for record in press_records],
                [record["metric_map"][metric_name] for record in press_records],
                marker="o",
                linewidth=2,
                label=press,
            )

        axis.set_title(metric_name)
        axis.set_xlabel("Compression Ratio")
        axis.set_ylabel("Score")
        axis.grid(True, alpha=0.3)
        axis.set_xticks([0.3, 0.5, 0.7])

    for axis in flat_axes[n_metrics:]:
        axis.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle(dataset, y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ruler", "longbench", "needle_in_haystack"],
        help="Datasets to plot.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="prefill_compare_15pct_detailed",
        help="Output filename prefix under ./figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.results_dir)
    dataset_metric_map = build_dataset_metric_map(records)

    for dataset in args.datasets:
        metric_names = dataset_metric_map.get(dataset, [])
        output_path = FIGURE_DIR / f"{args.prefix}_{dataset}.png"
        plot_dataset(records, dataset, metric_names, output_path)


if __name__ == "__main__":
    main()
