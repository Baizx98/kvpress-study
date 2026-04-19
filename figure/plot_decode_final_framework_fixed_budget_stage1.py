from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "decode_final_framework_fixed_budget_stage1"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
SUMMARY_PATH = FIGURE_DIR / "summary.json"

LONG_BENCH_DATASETS = ["gov_report", "qmsum", "multi_news"]
RULER_TASKS = ["niah_single_3", "niah_multikey_2", "niah_multikey_3", "qa_2"]
BUDGETS = [96, 128, 160]
RULER_BUDGETS = [128, 160]
ROUTE_ORDER = ["dense+perm", "dense+cold", "blockwise+perm", "blockwise+cold"]
ROUTE_LABELS = {
    "dense+perm": "Dense + Perm",
    "dense+cold": "Dense + Cold",
    "blockwise+perm": "Blockwise + Perm",
    "blockwise+cold": "Blockwise + Cold",
}
ROUTE_COLORS = {
    "dense+perm": "#1b5e20",
    "dense+cold": "#2e7d32",
    "blockwise+perm": "#0d47a1",
    "blockwise+cold": "#1976d2",
}


@dataclass
class RunRecord:
    dataset: str
    data_dir: str
    route: str
    budget: int
    score: float
    raw_metrics: dict
    path: str
    mtime_ns: int


def classify_route(cfg: dict) -> str:
    prefill = "blockwise" if float(cfg["compression_ratio"]) > 0 else "dense"
    decode = "perm" if cfg.get("dual_phase_mode") == "permanent_fixed_budget" else "cold"
    return f"{prefill}+{decode}"


def classify_budget(cfg: dict) -> int:
    if cfg.get("decode_block_budget") is not None:
        return int(cfg["decode_block_budget"])
    return int(cfg["decode_cold_block_budget"])


def parse_score(cfg: dict, metrics: dict) -> float:
    if cfg["dataset"] == "longbench":
        return float(metrics)
    if cfg["dataset"] == "ruler":
        vals = []
        for task in RULER_TASKS:
            task_result = metrics.get(task, {})
            if isinstance(task_result, dict) and "string_match" in task_result:
                vals.append(float(task_result["string_match"]))
        return sum(vals) / len(vals)
    raise ValueError(f"Unsupported dataset: {cfg['dataset']}")


def load_latest_records() -> list[RunRecord]:
    latest: dict[tuple, RunRecord] = {}
    for config_path in RESULT_ROOT.rglob("config.yaml"):
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        record = RunRecord(
            dataset=str(cfg["dataset"]),
            data_dir=str(cfg["data_dir"]),
            route=classify_route(cfg),
            budget=classify_budget(cfg),
            score=parse_score(cfg, metrics),
            raw_metrics=metrics,
            path=str(config_path.parent),
            mtime_ns=config_path.parent.stat().st_mtime_ns,
        )
        key = (record.dataset, record.data_dir, record.route, record.budget)
        if key not in latest or latest[key].mtime_ns < record.mtime_ns:
            latest[key] = record
    return sorted(latest.values(), key=lambda r: (r.dataset, r.data_dir, r.route, r.budget))


def build_summary(records: list[RunRecord]) -> dict:
    longbench = defaultdict(dict)
    ruler = defaultdict(dict)
    for record in records:
        if record.dataset == "longbench":
            longbench[record.data_dir][f"{record.route}@{record.budget}"] = record.score
        elif record.dataset == "ruler":
            ruler[f"{record.route}@{record.budget}"] = record.raw_metrics

    macro = {}
    for route in ROUTE_ORDER:
        for budget in BUDGETS:
            key = f"{route}@{budget}"
            vals = [longbench[dataset][key] for dataset in LONG_BENCH_DATASETS]
            macro[key] = sum(vals) / len(vals)

    return {
        "experiment_name": EXPERIMENT_NAME,
        "budgets": BUDGETS,
        "longbench_datasets": LONG_BENCH_DATASETS,
        "ruler_tasks": RULER_TASKS,
        "longbench": longbench,
        "ruler": ruler,
        "longbench_macro": macro,
        "record_count": len(records),
    }


def plot_longbench_lines(summary: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, dataset in zip(axes, LONG_BENCH_DATASETS):
        for route in ROUTE_ORDER:
            ys = [summary["longbench"][dataset][f"{route}@{budget}"] for budget in BUDGETS]
            ax.plot(BUDGETS, ys, marker="o", linewidth=2.2, color=ROUTE_COLORS[route], label=ROUTE_LABELS[route])
        ax.set_title(dataset)
        ax.set_xlabel("Decode Budget (blocks)")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.3)
        ax.set_xticks(BUDGETS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("LongBench Long-Output: Fixed-Budget Decode Frameworks", y=1.03)
    fig.tight_layout()
    out = FIGURE_DIR / "longbench_fixed_budget_lines.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_longbench_macro(summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for route in ROUTE_ORDER:
        ys = [summary["longbench_macro"][f"{route}@{budget}"] for budget in BUDGETS]
        ax.plot(BUDGETS, ys, marker="o", linewidth=2.4, color=ROUTE_COLORS[route], label=ROUTE_LABELS[route])
    ax.set_title("LongBench Macro Average")
    ax.set_xlabel("Decode Budget (blocks)")
    ax.set_ylabel("Average Score")
    ax.set_xticks(BUDGETS)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGURE_DIR / "longbench_fixed_budget_macro.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ruler_grouped(summary: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    width = 0.32
    x = list(range(len(RULER_TASKS)))
    route_pairs = [("dense+perm", 128), ("dense+cold", 128), ("dense+perm", 160), ("dense+cold", 160)]
    colors = ["#1b5e20", "#2e7d32", "#0d47a1", "#1976d2"]
    labels = ["Perm @128", "Cold @128", "Perm @160", "Cold @160"]

    # Plot 128/160 together on one axis, but split by route family for readability
    for idx, ax in enumerate(axes):
        sub_pairs = route_pairs[idx * 2 : idx * 2 + 2]
        sub_colors = colors[idx * 2 : idx * 2 + 2]
        sub_labels = labels[idx * 2 : idx * 2 + 2]
        offsets = [-width / 2, width / 2]
        for (route, budget), color, label, offset in zip(sub_pairs, sub_colors, sub_labels, offsets):
            ys = [summary["ruler"][f"{route}@{budget}"][task]["string_match"] for task in RULER_TASKS]
            ax.bar([i + offset for i in x], ys, width=width, color=color, label=label)
        ax.set_title(f"RULER Decode-Only @ {RULER_BUDGETS[idx]} blocks")
        ax.set_xticks(x)
        ax.set_xticklabels(RULER_TASKS, rotation=20)
        ax.set_ylabel("String Match")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGURE_DIR / "ruler_fixed_budget_grouped.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    records = load_latest_records()
    summary = build_summary(records)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    generated = [
        plot_longbench_lines(summary),
        plot_longbench_macro(summary),
        plot_ruler_grouped(summary),
    ]
    print(json.dumps({"summary": str(SUMMARY_PATH), "figures": [str(p) for p in generated]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
