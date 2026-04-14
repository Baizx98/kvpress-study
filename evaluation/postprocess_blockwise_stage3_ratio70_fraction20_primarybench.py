from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "blockwise_stage3_ratio70_fraction20_primarybench"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
NOTE_PATH = REPO_ROOT / "note" / f"{EXPERIMENT_NAME}_analysis_zh.md"
RESULT_README = RESULT_ROOT / "README.md"
FIGURE_README = FIGURE_DIR / "README.md"
EVAL_INDEX = REPO_ROOT / "evaluation" / "results" / "EXPERIMENT_INDEX.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"
RUN_SCRIPT = REPO_ROOT / "evaluation" / "run_blockwise_stage3_ratio70_fraction20_primarybench.py"

DATASET_ORDER = [
    "longbench:qasper",
    "longbench:multifieldqa_en",
    "longbench:hotpotqa",
    "longbench:2wikimqa",
    "longbench:musique",
    "longbench:triviaqa",
    "needle_in_haystack:16384",
    "pg19:test",
]
METHOD_ORDER = [
    "blockwise_main",
    "blockwise_multi_rep",
    "blockwise_adaptive_fusion_v1",
    "blockwise_multi_rep_diverse_v1",
    "chunkkv_prefill",
]
LOWER_IS_BETTER_DATASETS = {"pg19:test"}
METHOD_LABELS = {
    "blockwise_main": "`blockwise_main`",
    "blockwise_multi_rep": "`blockwise_multi_rep`",
    "blockwise_adaptive_fusion_v1": "`adaptive_fusion_v1`",
    "blockwise_multi_rep_diverse_v1": "`multi_rep_diverse_v1`",
    "chunkkv_prefill": "`chunkkv_prefill_per_layer`",
}


@dataclass
class RunRecord:
    dataset_key: str
    method_key: str
    score: float
    score_detail: str
    path: Path


def classify_dataset(cfg: dict) -> str:
    dataset = cfg.get("dataset")
    data_dir = cfg.get("data_dir")
    if dataset == "longbench":
        return f"longbench:{data_dir}"
    if dataset == "needle_in_haystack":
        return f"needle_in_haystack:{cfg.get('max_context_length')}"
    if dataset == "pg19":
        return "pg19:test"
    return f"{dataset}:{data_dir}"


def classify_method(cfg: dict) -> str:
    if cfg.get("press_name") == "chunkkv_prefill_per_layer":
        return "chunkkv_prefill"
    if cfg.get("representative_mode") == "key_norm_diverse":
        return "blockwise_multi_rep_diverse_v1"
    if cfg.get("summary_mode") == "adaptive_fusion_v1":
        return "blockwise_adaptive_fusion_v1"
    if cfg.get("summary_mode") == "multi_rep_max":
        return "blockwise_multi_rep"
    return "blockwise_main"


def parse_metrics(dataset_key: str, metrics_path: Path) -> tuple[float, str]:
    value = json.loads(metrics_path.read_text())
    if dataset_key == "pg19:test":
        score = float(value["subword_perplexity"])
        return score, f"subword_ppl={score:.4f}; word_ppl={float(value['word_perplexity']):.4f}"
    if isinstance(value, (int, float)):
        return float(value), f"{float(value):.2f}"
    if isinstance(value, list) and dataset_key.startswith("needle_in_haystack:"):
        rouge_l_scores = []
        for item in value:
            rouge_l = item.get("rouge-l") or item.get("rouge_l")
            if isinstance(rouge_l, dict) and "f" in rouge_l:
                rouge_l_scores.append(float(rouge_l["f"]) * 100.0)
        score = sum(rouge_l_scores) / len(rouge_l_scores)
        return score, f"avg_rouge_l_f={score:.2f}"
    return float(value), f"{float(value):.2f}"


def load_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    for config_path in ARTIFACTS_DIR.rglob("config.yaml"):
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text())
        dataset_key = classify_dataset(cfg)
        method_key = classify_method(cfg)
        if dataset_key not in DATASET_ORDER or method_key not in METHOD_ORDER:
            continue
        score, detail = parse_metrics(dataset_key, metrics_path)
        records.append(RunRecord(dataset_key, method_key, score, detail, config_path.parent))
    return records


def dedupe_records(records: Iterable[RunRecord]) -> dict[str, dict[str, RunRecord]]:
    deduped = {dataset: {} for dataset in DATASET_ORDER}
    for record in sorted(records, key=lambda item: item.path.stat().st_mtime_ns):
        deduped[record.dataset_key][record.method_key] = record
    return deduped


def choose_best(dataset_key: str, records: dict[str, RunRecord]) -> tuple[str, RunRecord] | None:
    if not records:
        return None
    items = list(records.items())
    reverse = dataset_key not in LOWER_IS_BETTER_DATASETS
    items.sort(key=lambda item: item[1].score, reverse=reverse)
    return items[0]


def summarize_failures(deduped: dict[str, dict[str, RunRecord]]) -> list[str]:
    lines: list[str] = []
    final_path = ARTIFACTS_DIR / "failed_jobs_final.jsonl"
    if not final_path.exists():
        return lines
    for raw in final_path.read_text().splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        dataset_key = item.get("dataset")
        method_key = item.get("method")
        if dataset_key in deduped and method_key in deduped[dataset_key]:
            continue
        lines.append(f"- `{item['job_id']}`: attempts={item.get('attempts', '?')}, reason={item.get('last_reason', 'unknown')}")
    return lines


def ensure_index_entry(index_path: Path, entry: str) -> None:
    text = index_path.read_text() if index_path.exists() else ""
    if entry in text:
        return
    if text and not text.endswith("\n"):
        text += "\n"
    index_path.write_text(text + entry + "\n")


def write_result_readme(deduped: dict[str, dict[str, RunRecord]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    completeness = [f"- `{dataset}`：{len(deduped[dataset])}/{len(METHOD_ORDER)}" for dataset in DATASET_ORDER]
    failures = summarize_failures(deduped) or ["- 无最终失败项"]
    RESULT_README.write_text(
        f"""# {EXPERIMENT_NAME}

## 实验目的

验证 stage3 第一批候选方法在当前主数据集上的表现，并与 `chunkkv_prefill_per_layer` 做直接比较。

## 运行脚本

- [{RUN_SCRIPT.name}]({RUN_SCRIPT})

## 数据集

- `LongBench / qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, triviaqa`
- `needle_in_haystack / 16384`
- `PG19 / test`

## 方法

- `blockwise_main`
- `blockwise_multi_rep`
- `blockwise_adaptive_fusion_v1`
- `blockwise_multi_rep_diverse_v1`
- `chunkkv_prefill_per_layer`

## 关键配置

- `compression_ratio=0.7`
- `fraction=0.2`
- `device=cuda:0`

## 当前完整性

{chr(10).join(completeness)}

## 产物位置

- [artifacts]({ARTIFACTS_DIR})
- [run.log]({ARTIFACTS_DIR / 'run.log'})
- [failed_jobs.jsonl]({ARTIFACTS_DIR / 'failed_jobs.jsonl'})
- [failed_jobs_final.jsonl]({ARTIFACTS_DIR / 'failed_jobs_final.jsonl'})
- [分析文档]({NOTE_PATH})

## 最终失败项

{chr(10).join(failures)}
"""
    )


def write_figure_readme() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_README.write_text(
        f"""# {EXPERIMENT_NAME}

当前阶段尚未生成专门图表。后续如补绘图，请统一放在该目录，并在此 README 中补充说明。
"""
    )


def write_note(deduped: dict[str, dict[str, RunRecord]]) -> None:
    lines = [
        f"# {EXPERIMENT_NAME} 分析",
        "",
        "## 完整性",
        "",
    ]
    for dataset in DATASET_ORDER:
        lines.append(f"- `{dataset}`：{len(deduped[dataset])}/{len(METHOD_ORDER)}")
    lines.extend(["", "## 当前最优", ""])
    for dataset in DATASET_ORDER:
        best = choose_best(dataset, deduped[dataset])
        if best is None:
            lines.append(f"- `{dataset}`：暂无结果")
        else:
            method_key, record = best
            better = "越低越好" if dataset in LOWER_IS_BETTER_DATASETS else "越高越好"
            lines.append(f"- `{dataset}`：{method_key} = {record.score:.4f}（{better}）")
    lines.extend(["", "## 各数据集结果", ""])
    for dataset in DATASET_ORDER:
        lines.append(f"### {dataset}")
        records = deduped[dataset]
        if not records:
            lines.append("- 暂无结果")
            lines.append("")
            continue
        for method in METHOD_ORDER:
            record = records.get(method)
            if record is None:
                lines.append(f"- `{method}`：缺失")
            else:
                lines.append(f"- `{method}`：{record.score_detail}")
        lines.append("")
    failures = summarize_failures(deduped)
    lines.extend(["## 最终失败项", ""])
    lines.extend(failures or ["- 无最终失败项"])
    NOTE_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    deduped = dedupe_records(load_records())
    write_result_readme(deduped)
    write_figure_readme()
    write_note(deduped)
    ensure_index_entry(EVAL_INDEX, f"- [{EXPERIMENT_NAME}]({RESULT_ROOT / 'README.md'})")
    ensure_index_entry(FIGURE_INDEX, f"- [{EXPERIMENT_NAME}]({FIGURE_DIR / 'README.md'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
