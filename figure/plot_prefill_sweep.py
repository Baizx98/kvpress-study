from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "evaluation" / "results_prefill_sweep_10pct"
FIGURE_DIR = ROOT / "figure"
FIGURE_DIR.mkdir(exist_ok=True)


def score_metrics(dataset: str, metrics: dict) -> float:
    if dataset == "loogle":
        task = next(iter(metrics))
        return float(metrics[task]["bert"])

    values = []
    for value in metrics.values():
        if isinstance(value, dict):
            if "string_match" in value:
                values.append(float(value["string_match"]))
            elif "accuracy" in value:
                values.append(float(value["accuracy"]))
    return sum(values) / len(values) if values else 0.0


def parse_result_dirs(results_dir: Path):
    records = []
    for metrics_path in results_dir.rglob("metrics.json"):
        name = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text())
        parts = name.split("__")
        if len(parts) < 5:
            continue

        dataset = parts[0]
        press = parts[3]
        ratio = float(parts[4])
        records.append(
            {
                "dataset": dataset,
                "press": press,
                "ratio": ratio,
                "score": score_metrics(dataset, metrics),
            }
        )
    return records


def plot(records, results_dir: Path, output_name: str):
    if not records:
        raise RuntimeError(f"No metrics found in {results_dir}")

    datasets = sorted({record["dataset"] for record in records})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)

    for axis, dataset in zip(axes[0], datasets):
        for press in sorted({record["press"] for record in records if record["dataset"] == dataset}):
            press_records = sorted(
                [record for record in records if record["dataset"] == dataset and record["press"] == press],
                key=lambda item: item["ratio"],
            )
            axis.plot(
                [record["ratio"] for record in press_records],
                [record["score"] for record in press_records],
                marker="o",
                label=press,
            )

        axis.set_title(dataset)
        axis.set_xlabel("Compression Ratio")
        axis.set_ylabel("Aggregate Score")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    output_path = FIGURE_DIR / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing evaluation run subdirectories with metrics.json files.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="prefill_blockwise_vs_snapkv_10pct.png",
        help="Output filename under ./figure.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(parse_result_dirs(args.results_dir), args.results_dir, args.output_name)
