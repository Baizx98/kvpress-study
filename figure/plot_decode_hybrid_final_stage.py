from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "decode_hybrid_final_stage"
ARTIFACT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
FIG_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
SUMMARY_PATH = FIG_ROOT / "summary.json"


def load_rows():
    rows: list[dict] = []
    for config_path in ARTIFACT_ROOT.rglob("config.yaml"):
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue

        cfg = yaml.safe_load(config_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        mode = cfg["dual_phase_mode"]
        decode_budget = cfg.get("decode_block_budget")
        cold_budget = cfg.get("decode_cold_block_budget")

        if mode == "permanent_fixed_budget":
            route = "permanent"
            route_label = f"Permanent {decode_budget}"
            total_budget = decode_budget
            active_budget = decode_budget
        elif mode == "compute_cold_fixed_budget":
            route = "compute_cold"
            route_label = f"Compute-Cold {cold_budget}"
            total_budget = cold_budget
            active_budget = cold_budget
        elif mode == "hybrid_fixed_budget":
            route = "hybrid"
            route_label = f"Hybrid {decode_budget}/{cold_budget}"
            total_budget = decode_budget
            active_budget = cold_budget
        else:
            continue

        row = {
            "dataset": cfg["dataset"],
            "data_dir": cfg["data_dir"],
            "route": route,
            "route_label": route_label,
            "mode": mode,
            "total_budget": total_budget,
            "active_budget": active_budget,
            "metrics": metrics,
            "path": str(config_path.parent.relative_to(REPO_ROOT)),
        }
        rows.append(row)
    return rows


def build_summary(rows: list[dict]) -> dict:
    longbench_rows = [row for row in rows if row["dataset"] == "longbench"]
    ruler_rows = [row for row in rows if row["dataset"] == "ruler"]

    longbench_scores: dict[str, dict[str, float]] = {}
    for row in sorted(longbench_rows, key=lambda item: (item["data_dir"], item["route"], item["total_budget"])):
        task = str(row["data_dir"])
        longbench_scores.setdefault(task, {})[row["route_label"]] = float(row["metrics"])

    longbench_macro = {
        route_label: round(mean(scores[route_label] for scores in longbench_scores.values()), 2)
        for route_label in sorted({label for scores in longbench_scores.values() for label in scores})
    }

    ruler_scores: dict[str, dict[str, float]] = {}
    ruler_task_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for row in sorted(ruler_rows, key=lambda item: (item["route"], item["total_budget"])):
        route_label = row["route_label"]
        task_metrics = {
            task: float(values["string_match"]) for task, values in row["metrics"].items()
        }
        ruler_task_metrics[route_label] = task_metrics
        ruler_scores[route_label] = {"macro_string_match": round(mean(task_metrics.values()), 2)}

    false_failures = []
    failed_final = ARTIFACT_ROOT / "failed_jobs_final.jsonl"
    if failed_final.exists():
        for line in failed_final.read_text().splitlines():
            if line.strip():
                false_failures.append(json.loads(line))

    return {
        "experiment": EXPERIMENT_NAME,
        "longbench": {
            "tasks": longbench_scores,
            "macro_avg": longbench_macro,
        },
        "ruler": {
            "route_macro": ruler_scores,
            "route_task_scores": ruler_task_metrics,
        },
        "false_failures": false_failures,
    }


def plot_longbench_lines(summary: dict) -> None:
    task_scores = summary["longbench"]["tasks"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    budget_positions = [128, 160]
    colors = {
        "Permanent": "#c0392b",
        "Compute-Cold": "#1f77b4",
        "Hybrid": "#2ca02c",
    }

    for ax, task in zip(axes, ["gov_report", "qmsum", "multi_news"]):
        scores = task_scores[task]
        permanent = [scores[f"Permanent {budget}"] for budget in budget_positions]
        cold = [scores[f"Compute-Cold {budget}"] for budget in budget_positions]
        hybrid = [scores["Hybrid 128/96"], scores["Hybrid 160/128"]]

        ax.plot(budget_positions, permanent, marker="o", linewidth=2.2, color=colors["Permanent"], label="Permanent")
        ax.plot(budget_positions, cold, marker="s", linewidth=2.2, color=colors["Compute-Cold"], label="Compute-Cold")
        ax.plot(budget_positions, hybrid, marker="^", linewidth=2.2, color=colors["Hybrid"], label="Hybrid")
        ax.set_title(task)
        ax.set_xlabel("Total Budget (blocks)")
        ax.set_xticks(budget_positions)
        ax.grid(alpha=0.25, linestyle="--")
        if task == "gov_report":
            ax.set_ylabel("LongBench Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Decode Hybrid Final Stage: LongBench Task Scores", y=1.12)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "longbench_hybrid_budget_lines.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_longbench_macro(summary: dict) -> None:
    macro = summary["longbench"]["macro_avg"]
    order = [
        "Permanent 128",
        "Permanent 160",
        "Compute-Cold 128",
        "Compute-Cold 160",
        "Hybrid 128/96",
        "Hybrid 160/128",
    ]
    values = [macro[label] for label in order]
    colors = ["#c0392b", "#c0392b", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(order, values, color=colors)
    ax.set_ylabel("LongBench Macro Avg")
    ax.set_title("Decode Hybrid Final Stage: LongBench Macro Average")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(min(values) - 0.8, max(values) + 0.8)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "longbench_hybrid_macro.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ruler_grouped(summary: dict) -> None:
    route_task_scores = summary["ruler"]["route_task_scores"]
    tasks = ["niah_single_3", "niah_multikey_2", "niah_multikey_3", "qa_2"]
    order = [
        "Permanent 128",
        "Permanent 160",
        "Compute-Cold 128",
        "Compute-Cold 160",
        "Hybrid 128/96",
        "Hybrid 160/128",
    ]
    colors = ["#c0392b", "#c0392b", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = list(range(len(tasks)))
    width = 0.12
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    for offset, label, color in zip(offsets, order, colors):
        vals = [route_task_scores[label][task] for task in tasks]
        ax.bar([item + offset * width for item in x], vals, width=width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("String Match")
    ax.set_title("Decode Hybrid Final Stage: RULER Supplement")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "ruler_hybrid_grouped.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = build_summary(rows)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    plot_longbench_lines(summary)
    plot_longbench_macro(summary)
    plot_ruler_grouped(summary)


if __name__ == "__main__":
    main()
