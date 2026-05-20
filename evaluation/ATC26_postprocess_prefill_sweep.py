from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
METRICS_LONG = ARTIFACTS_DIR / "ATC26_metrics_long.csv"
METRICS_WIDE = ARTIFACTS_DIR / "ATC26_metrics_wide.csv"
METRICS_FULL_LONG = ARTIFACTS_DIR / "ATC26_metrics_full_long.csv"
METRICS_FULL_WIDE = ARTIFACTS_DIR / "ATC26_metrics_full_wide.csv"
JOB_STATUS = ARTIFACTS_DIR / "ATC26_job_status.csv"
RESULT_README = RESULT_ROOT / "README.md"
EVAL_INDEX = REPO_ROOT / "evaluation" / "results" / "EXPERIMENT_INDEX.md"


@dataclass
class Record:
    model: str
    dataset: str
    data_dir: str
    dataset_key: str
    method: str
    compression_ratio: float
    fraction: float
    score_name: str
    score: float
    lower_is_better: bool
    metrics_path: Path
    config_path: Path
    predictions_path: Path | None
    status: str


def model_key(model_path: str) -> str:
    name = Path(model_path).name
    known = {
        "Llama-3.1-8B-Instruct": "llama31_8b_instruct",
        "Mistral-7B-Instruct-v0.3": "mistral_7b_instruct_v03",
        "Qwen3-8B": "qwen3_8b",
    }
    if name in known:
        return known[name]
    return name.lower().replace("-", "_").replace(".", "p")


def dataset_key(cfg: dict[str, Any]) -> str:
    dataset = cfg.get("dataset")
    if dataset == "longbench":
        return f"longbench:{cfg.get('data_dir')}"
    if dataset == "needle_in_haystack":
        return f"needle_in_haystack:{cfg.get('max_context_length')}"
    if dataset == "pg19":
        return "pg19:test"
    return f"{dataset}:{cfg.get('data_dir') or 'default'}"


def method_key(cfg: dict[str, Any]) -> str:
    press_name = cfg.get("press_name")
    if press_name == "block_wise":
        return "blockwise"
    if press_name == "snapkv":
        return "snapkv"
    if press_name == "chunkkv":
        return "chunkkv"
    return str(press_name)


def parse_score(dataset: str, metrics: Any) -> tuple[str, float, bool]:
    if dataset == "pg19:test":
        return "subword_perplexity", float(metrics["subword_perplexity"]), True
    if dataset.startswith("needle_in_haystack:") and isinstance(metrics, list):
        rouge_l = []
        for item in metrics:
            score = item.get("rouge-l") or item.get("rouge_l")
            if isinstance(score, dict) and "f" in score:
                rouge_l.append(float(score["f"]) * 100.0)
        value = sum(rouge_l) / len(rouge_l) if rouge_l else 0.0
        return "avg_rouge_l_f", value, False
    return "score", float(metrics), False


def find_prediction(config_path: Path) -> Path | None:
    for name in ("ATC26_predictions.csv", "predictions.csv"):
        candidate = config_path.with_name(name)
        if candidate.exists():
            return candidate
    return None


def load_records() -> list[Record]:
    records: list[Record] = []
    for config_path in RAW_DIR.rglob("ATC26_config.yaml"):
        metrics_path = config_path.with_name("ATC26_metrics.json")
        if not metrics_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text()) or {}
        metrics = json.loads(metrics_path.read_text())
        dkey = dataset_key(cfg)
        score_name, score, lower_is_better = parse_score(dkey, metrics)
        records.append(
            Record(
                model=model_key(str(cfg.get("model"))),
                dataset=str(cfg.get("dataset")),
                data_dir=str(cfg.get("data_dir") or ""),
                dataset_key=dkey,
                method=method_key(cfg),
                compression_ratio=float(cfg.get("compression_ratio", 0.0)),
                fraction=float(cfg.get("fraction", 1.0)),
                score_name=score_name,
                score=score,
                lower_is_better=lower_is_better,
                metrics_path=metrics_path,
                config_path=config_path,
                predictions_path=find_prediction(config_path),
                status="success",
            )
        )
    return records


def dedupe(records: list[Record]) -> list[Record]:
    latest: dict[tuple[str, str, str, float, float], Record] = {}
    for record in sorted(records, key=lambda item: item.metrics_path.stat().st_mtime_ns):
        key = (record.model, record.dataset_key, record.method, record.compression_ratio, record.fraction)
        latest[key] = record
    return list(latest.values())


def write_tables(records: list[Record]) -> None:
    rows = []
    for record in sorted(
        records,
        key=lambda item: (item.model, item.dataset_key, item.method, item.compression_ratio),
    ):
        rows.append(
            {
                "model": record.model,
                "dataset": record.dataset,
                "data_dir": record.data_dir,
                "dataset_key": record.dataset_key,
                "method": record.method,
                "compression_ratio": record.compression_ratio,
                "fraction": record.fraction,
                "score_name": record.score_name,
                "score": record.score,
                "lower_is_better": record.lower_is_better,
                "metrics_path": str(record.metrics_path.relative_to(REPO_ROOT)),
                "config_path": str(record.config_path.relative_to(REPO_ROOT)),
                "predictions_path": str(record.predictions_path.relative_to(REPO_ROOT)) if record.predictions_path else "",
                "status": record.status,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_LONG, index=False)
    if not df.empty:
        wide = df.pivot_table(
            index=["model", "dataset_key", "compression_ratio", "fraction"],
            columns="method",
            values="score",
            aggfunc="first",
        ).reset_index()
        wide.to_csv(METRICS_WIDE, index=False)
    else:
        pd.DataFrame().to_csv(METRICS_WIDE, index=False)
    full_df = df[df["fraction"] == 1.0].copy() if not df.empty else df
    full_df.to_csv(METRICS_FULL_LONG, index=False)
    if not full_df.empty:
        full_wide = full_df.pivot_table(
            index=["model", "dataset_key", "compression_ratio", "fraction"],
            columns="method",
            values="score",
            aggfunc="first",
        ).reset_index()
        full_wide.to_csv(METRICS_FULL_WIDE, index=False)
    else:
        pd.DataFrame().to_csv(METRICS_FULL_WIDE, index=False)
    df.to_csv(JOB_STATUS, index=False)


def ensure_index_entry(index_path: Path, entry: str) -> None:
    text = index_path.read_text() if index_path.exists() else ""
    if entry in text:
        return
    if text and not text.endswith("\n"):
        text += "\n"
    index_path.write_text(text + entry + "\n")


def write_readme(records: list[Record]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    models = sorted({record.model for record in records})
    datasets = sorted({record.dataset_key for record in records})
    methods = sorted({record.method for record in records})
    ratios = sorted({record.compression_ratio for record in records})
    text = f"""# {EXPERIMENT_NAME}

## 实验目的

为 ATC26 论文补充 prefill-only KVCache 压缩实验，比较 BlockWise、SnapKV、ChunkKV 在 LongBench、needle_in_haystack、PG19 上的质量变化。

## 运行脚本

- `evaluation/ATC26_run_prefill_sweep.py`
- `evaluation/ATC26_postprocess_prefill_sweep.py`
- `figure/ATC26_plot_prefill_sweep.py`

## 数据集

{chr(10).join(f"- `{item}`" for item in datasets) if datasets else "- 尚无成功结果"}

## 方法

{chr(10).join(f"- `{item}`" for item in methods) if methods else "- 尚无成功结果"}

## 模型

{chr(10).join(f"- `{item}`" for item in models) if models else "- 尚无成功结果"}

## 压缩率

{", ".join(f"`{item:.1f}`" for item in ratios) if ratios else "尚无成功结果"}

## 产物位置

- 原始结果：`evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/raw/`
- 长表：`evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/ATC26_metrics_long.csv`
- 宽表：`evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/ATC26_metrics_wide.csv`
- full-only 长表：`evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/ATC26_metrics_full_long.csv`
- full-only 宽表：`evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/ATC26_metrics_full_wide.csv`
- 进度日志：`evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/ATC26_progress.md`
"""
    RESULT_README.write_text(text)
    ensure_index_entry(
        EVAL_INDEX,
        f"- `{EXPERIMENT_NAME}`: ATC26 prefill-only sweep for BlockWise, SnapKV, and ChunkKV.",
    )


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    records = dedupe(load_records())
    write_tables(records)
    write_readme(records)
    print(f"Wrote {len(records)} records to {METRICS_LONG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
