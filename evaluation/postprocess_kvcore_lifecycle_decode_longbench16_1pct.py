from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "kvcore_lifecycle_decode_longbench16_1pct")
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
SUMMARY_CSV = RESULT_ROOT / "summary.csv"
SUMMARY_JSON = RESULT_ROOT / "summary.json"
NOTE_PATH = REPO_ROOT / "note" / f"{EXPERIMENT_NAME}_results_zh.md"

TASK_CATEGORIES = {
    "narrativeqa": "single_doc_qa",
    "qasper": "single_doc_qa",
    "multifieldqa_en": "single_doc_qa",
    "hotpotqa": "multi_doc_qa",
    "2wikimqa": "multi_doc_qa",
    "musique": "multi_doc_qa",
    "triviaqa": "multi_doc_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "samsum": "few_shot",
    "trec": "few_shot",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}


def read_metric(path: Path) -> float:
    raw = path.read_text().strip()
    value = json.loads(raw)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected scalar LongBench metric in {path}, got {type(value).__name__}")


def method_key(cfg: dict[str, Any]) -> str | None:
    if cfg.get("press_name") == "no_press":
        return "full_kv"
    if (
        cfg.get("press_name") == "dual_phase_per_layer"
        and cfg.get("dual_phase_mode") == "compute_cold_fixed_budget"
        and cfg.get("decode_top_p_threshold") is not None
    ):
        return "decode_qaware_blockwise_top_p"
    return None


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for metrics_path in ARTIFACTS_DIR.rglob("metrics.json"):
        config_path = metrics_path.with_name("config.yaml")
        predictions_path = metrics_path.with_name("predictions.csv")
        if not config_path.exists() or not predictions_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text()) or {}
        if cfg.get("dataset") != "longbench":
            continue
        method = method_key(cfg)
        if method is None:
            continue
        prediction_rows = 0
        try:
            with predictions_path.open(newline="") as f:
                prediction_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
        except Exception:
            prediction_rows = 0
        task = str(cfg.get("data_dir"))
        rows.append(
            {
                "task": task,
                "category": TASK_CATEGORIES.get(task, "unknown"),
                "method": method,
                "score": read_metric(metrics_path),
                "num_samples": prediction_rows,
                "model": cfg.get("model"),
                "fraction": cfg.get("fraction"),
                "seed": cfg.get("seed"),
                "max_new_tokens": cfg.get("max_new_tokens"),
                "decode_top_p_threshold": cfg.get("decode_top_p_threshold"),
                "decode_skip_first_layers": cfg.get("decode_skip_first_layers"),
                "path": str(metrics_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise SystemExit(f"No completed metrics found under {ARTIFACTS_DIR}")

    raw = raw.sort_values(["task", "method", "path"]).drop_duplicates(["task", "method"], keep="last")
    pivot = raw.pivot(index=["task", "category"], columns="method", values="score").reset_index()
    sample_counts = raw.pivot(index=["task", "category"], columns="method", values="num_samples").reset_index()
    sample_counts = sample_counts.rename(
        columns={
            "full_kv": "full_kv_samples",
            "decode_qaware_blockwise_top_p": "decode_qaware_blockwise_samples",
        }
    )
    summary = pivot.merge(sample_counts, on=["task", "category"], how="left")

    if "full_kv" in summary.columns and "decode_qaware_blockwise_top_p" in summary.columns:
        summary["delta"] = summary["decode_qaware_blockwise_top_p"] - summary["full_kv"]
        summary["relative_delta_pct"] = summary["delta"] / summary["full_kv"].replace(0, pd.NA) * 100.0

    method_cols = [col for col in ["full_kv", "decode_qaware_blockwise_top_p", "delta", "relative_delta_pct"] if col in summary]
    macro = {col: float(summary[col].dropna().mean()) for col in method_cols}
    by_category = (
        summary.groupby("category", as_index=False)[method_cols]
        .mean(numeric_only=True)
        .sort_values("category")
        .to_dict(orient="records")
    )
    metadata = {
        "models": sorted(str(v) for v in raw["model"].dropna().unique()),
        "fractions": sorted(float(v) for v in raw["fraction"].dropna().unique()),
        "seeds": sorted(int(v) for v in raw["seed"].dropna().unique()),
        "decode_top_p_thresholds": sorted(
            float(v) for v in raw["decode_top_p_threshold"].dropna().unique()
        ),
        "decode_skip_first_layers": sorted(
            int(v) for v in raw["decode_skip_first_layers"].dropna().unique()
        ),
    }
    payload = {
        "experiment": EXPERIMENT_NAME,
        "completed_tasks": int(summary["task"].nunique()),
        "metadata": metadata,
        "macro": macro,
        "by_category": by_category,
        "rows": summary.to_dict(orient="records"),
    }
    return summary, payload


def write_note(summary: pd.DataFrame, payload: dict[str, Any]) -> None:
    macro = payload["macro"]
    metadata = payload.get("metadata", {})
    models = metadata.get("models") or ["/Tan/model/Llama-3.1-8B-Instruct"]
    fractions = metadata.get("fractions") or []
    seeds = metadata.get("seeds") or []
    top_ps = metadata.get("decode_top_p_thresholds") or []
    skipped_layers = metadata.get("decode_skip_first_layers") or []

    fraction_text = ", ".join(f"{value:g}" for value in fractions) if fractions else "n/a"
    seed_text = ", ".join(str(value) for value in seeds) if seeds else "n/a"
    top_p_text = ", ".join(f"{value:g}" for value in top_ps) if top_ps else "n/a"
    skipped_layer_text = ", ".join(str(value) for value in skipped_layers) if skipped_layers else "0"
    lines = [
        f"# KVCore Lifecycle Decode LongBench16 准确率消融结果：`{EXPERIMENT_NAME}`",
        "",
        "## 实验设置",
        "",
        f"- 模型：`{models[0]}`",
        "- 数据集：LongBench 16 个英文子数据集",
        f"- 采样：每个子数据集 `fraction={fraction_text}`，`seed={seed_text}`",
        "- 对比：`full_kv` vs `decode_qaware_blockwise_top_p`",
        f"- decode active block budget：block score softmax 后 top-p，`p={top_p_text}`",
        f"- decode 前 `{skipped_layer_text}` 层不压缩",
        "- 运行设备：NVIDIA L40S",
        "",
        "## 结论摘要",
        "",
    ]
    if "delta" in macro:
        lines.append(f"- 16-task macro delta：`{macro['delta']:.4f}` 分。")
    if "full_kv" in macro:
        lines.append(f"- `full_kv` macro：`{macro['full_kv']:.4f}`。")
    if "decode_qaware_blockwise_top_p" in macro:
        lines.append(f"- `decode_qaware_blockwise_top_p` macro：`{macro['decode_qaware_blockwise_top_p']:.4f}`。")
    lines.extend(
        [
            "",
            "注意：这个实验测的是 decode 阶段 query-aware sparse active set 的质量影响；真实 offload/prefetch 如果能在 attention 前恢复所需 KV，数学上应与 full KV 一致。",
            "",
            "## Per-task 结果",
            "",
            "| Task | Category | Samples | Full KV | Decode top-p | Delta | Rel. delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.sort_values("task").to_dict(orient="records"):
        samples = row.get("full_kv_samples") or row.get("decode_qaware_blockwise_samples") or ""
        rel = row.get("relative_delta_pct")
        rel_text = "n/a" if rel is None or pd.isna(rel) else f"{rel:.2f}%"
        lines.append(
            f"| `{row['task']}` | `{row['category']}` | {samples} | "
            f"{row.get('full_kv', float('nan')):.2f} | "
            f"{row.get('decode_qaware_blockwise_top_p', float('nan')):.2f} | "
            f"{row.get('delta', float('nan')):.2f} | "
            f"{rel_text} |"
        )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- 汇总 CSV：`evaluation/results/experiments/{EXPERIMENT_NAME}/summary.csv`",
            f"- 汇总 JSON：`evaluation/results/experiments/{EXPERIMENT_NAME}/summary.json`",
            f"- 原始结果：`evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/`",
            "",
        ]
    )
    NOTE_PATH.write_text("\n".join(lines))


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    summary, payload = build_summary(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_note(summary, payload)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {NOTE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
