from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "blockwise_stage2_ratio70_fraction20_multidataset"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
NOTE_PATH = REPO_ROOT / "note" / f"{EXPERIMENT_NAME}_analysis_zh.md"
RESULT_README = RESULT_ROOT / "README.md"
FIGURE_README = FIGURE_DIR / "README.md"
EVAL_INDEX = REPO_ROOT / "evaluation" / "results" / "EXPERIMENT_INDEX.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"
RUN_SCRIPT = REPO_ROOT / "evaluation" / "run_blockwise_stage2_ratio70_fraction20_multidataset.py"

DATASET_ORDER = [
    "ruler:4096",
    "longbench:qasper",
    "longbench:multifieldqa_en",
    "longbench:hotpotqa",
    "longbench:2wikimqa",
    "longbench:musique",
    "longbench:triviaqa",
    "needle_in_haystack:16384",
]

METHOD_ORDER = [
    "blockwise_main",
    "blockwise_norm_topk",
    "blockwise_multi_rep",
    "blockwise_tail_query_special",
    "chunkkv_prefill",
]

METHOD_LABELS = {
    "blockwise_main": "`mean_plus_norm_topk_mean + key_norm + max + uniform_mean`",
    "blockwise_norm_topk": "`norm_topk_mean_only + key_norm + max + uniform_mean`",
    "blockwise_multi_rep": "`multi_rep_max + key_norm + max + uniform_mean`",
    "blockwise_tail_query_special": "`mean_plus_norm_topk_mean + tail_query_relevance + mean + uniform_mean`",
    "chunkkv_prefill": "`chunkkv_prefill_per_layer`",
}


@dataclass
class RunRecord:
    dataset_key: str
    method_key: str
    score: float
    score_detail: str
    path: Path
    cfg: dict


def classify_dataset(cfg: dict) -> str:
    dataset = cfg.get("dataset")
    data_dir = cfg.get("data_dir")
    if dataset == "ruler":
        return f"ruler:{data_dir}"
    if dataset == "longbench":
        return f"longbench:{data_dir}"
    if dataset == "needle_in_haystack":
        return f"needle_in_haystack:{cfg.get('max_context_length')}"
    return f"{dataset}:{data_dir}"


def classify_method(cfg: dict) -> str:
    press = cfg.get("press_name")
    if press == "chunkkv_prefill_per_layer":
        return "chunkkv_prefill"

    summary = cfg.get("summary_mode")
    rep = cfg.get("representative_mode")
    qagg = cfg.get("query_agg_mode")
    hagg = cfg.get("head_agg_mode")

    if rep == "tail_query_relevance":
        return "blockwise_tail_query_special"
    if summary == "norm_topk_mean_only":
        return "blockwise_norm_topk"
    if summary == "multi_rep_max":
        return "blockwise_multi_rep"
    if (
        summary == "mean_plus_norm_topk_mean"
        and rep == "key_norm"
        and qagg == "max"
        and hagg == "uniform_mean"
    ):
        return "blockwise_main"
    return "unknown"


def parse_metrics(dataset_key: str, metrics_path: Path) -> tuple[float, str]:
    value = json.loads(metrics_path.read_text())

    if isinstance(value, (int, float)):
        score = float(value)
        return score, f"{score:.2f}"

    if isinstance(value, dict):
        task_scores: list[float] = []
        parts: list[str] = []
        for task_name in sorted(value):
            task_metrics = value[task_name]
            if isinstance(task_metrics, dict) and "string_match" in task_metrics:
                task_score = float(task_metrics["string_match"])
                task_scores.append(task_score)
                parts.append(f"{task_name}={task_score:.2f}")
        if task_scores:
            avg_score = sum(task_scores) / len(task_scores)
            return avg_score, f"avg={avg_score:.2f}; " + ", ".join(parts)

    if isinstance(value, list) and dataset_key.startswith("needle_in_haystack:"):
        rouge_l_scores = []
        for item in value:
            if isinstance(item, dict):
                rouge_l = item.get("rouge-l") or item.get("rouge_l")
                if isinstance(rouge_l, dict) and "f" in rouge_l:
                    rouge_l_scores.append(float(rouge_l["f"]) * 100.0)
        if rouge_l_scores:
            avg_score = sum(rouge_l_scores) / len(rouge_l_scores)
            return avg_score, f"avg_rouge_l_f={avg_score:.2f}"

    raise ValueError(f"Unsupported metrics format: {metrics_path}")


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
        records.append(
            RunRecord(
                dataset_key=dataset_key,
                method_key=method_key,
                score=score,
                score_detail=detail,
                path=config_path.parent,
                cfg=cfg,
            )
        )
    return records


def dedupe_records(records: Iterable[RunRecord]) -> dict[str, dict[str, RunRecord]]:
    deduped: dict[str, dict[str, RunRecord]] = {dataset: {} for dataset in DATASET_ORDER}
    for record in sorted(records, key=lambda item: item.path.stat().st_mtime_ns):
        deduped[record.dataset_key][record.method_key] = record
    return deduped


def format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def ensure_index_entry(index_path: Path, entry: str) -> None:
    text = index_path.read_text()
    if entry in text:
        return
    lines = text.splitlines()
    insert_at = len(lines)
    for idx, line in enumerate(lines):
        if line.startswith("每个实验子目录包含") or line.startswith("每组实验目录下统一包含"):
            insert_at = idx
            break
    lines.insert(insert_at, entry)
    index_path.write_text("\n".join(lines) + "\n")


def summarize_failures() -> list[str]:
    failure_lines: list[str] = []
    final_path = ARTIFACTS_DIR / "failed_jobs_final.jsonl"
    if final_path.exists():
        for raw in final_path.read_text().splitlines():
            if not raw.strip():
                continue
            item = json.loads(raw)
            failure_lines.append(
                f"- `{item['job_id']}`: attempts={item.get('attempts', '?')}, reason={item.get('last_reason', 'unknown')}"
            )
    return failure_lines


def write_result_readme(deduped: dict[str, dict[str, RunRecord]]) -> None:
    completeness_lines = []
    for dataset_key in DATASET_ORDER:
        actual = len(deduped[dataset_key])
        expected = 5 if dataset_key.startswith("longbench:") else 4
        completeness_lines.append(
            f"- `{dataset_key}`：{actual}/{expected}"
        )

    failure_lines = summarize_failures()
    failure_block = "\n".join(failure_lines) if failure_lines else "- 无最终失败项"
    content = f"""# {EXPERIMENT_NAME}

## 实验目的

基于 stage2 设计报告，验证 blockwise 主线推荐矩阵在多数据集上的稳定性，并加入 `chunkkv` 作为额外对照方法。

## 运行脚本

- 主总控脚本：
  [{RUN_SCRIPT.name}]({RUN_SCRIPT})

## 数据集

- `RULER / 4096 / niah_single_3, niah_multikey_3, qa_2`
- `LongBench / qasper`
- `LongBench / multifieldqa_en`
- `LongBench / hotpotqa`
- `LongBench / 2wikimqa`
- `LongBench / musique`
- `LongBench / triviaqa`
- `Needle in a Haystack / max_context_length=16384 / needle_depth=[0,25,50,75,100]`

## 方法

- `blockwise_main`
- `blockwise_norm_topk`
- `blockwise_multi_rep`
- `blockwise_tail_query_special`（仅 LongBench）
- `chunkkv_prefill_per_layer`

## 关键配置

- `compression_ratio=0.7`
- `fraction=0.2`
- `query_aware=true`
- 不设置 `prefill_skip_first_layers`

## 产物位置

- 原始结果：
  [artifacts]({ARTIFACTS_DIR})
- 主运行日志：
  [run.log]({ARTIFACTS_DIR / 'run.log'})
- 失败记录：
  - [failed_jobs.jsonl]({ARTIFACTS_DIR / 'failed_jobs.jsonl'})
  - [failed_jobs_final.jsonl]({ARTIFACTS_DIR / 'failed_jobs_final.jsonl'})

## 当前完整性

{chr(10).join(completeness_lines)}

## 最终失败项

{failure_block}

## 推荐优先查看

- 中文分析：
  [{NOTE_PATH.name}]({NOTE_PATH})
"""
    RESULT_README.write_text(content + "\n")


def write_figure_readme() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    content = f"""# {EXPERIMENT_NAME}

## 图像说明

本轮实验当前尚未生成正式图像文件。

## 对应结果目录

- [{RESULT_ROOT}]({RESULT_ROOT})

## 推荐优先查看

- 中文分析：
  [{NOTE_PATH.name}]({NOTE_PATH})
"""
    FIGURE_README.write_text(content + "\n")


def build_dataset_section(dataset_key: str, records: dict[str, RunRecord]) -> str:
    lines = [
        f"## `{dataset_key}`",
        "",
        "| 方法 | 分数 | 说明 |",
        "|---|---:|---|",
    ]
    available = []
    for method_key in METHOD_ORDER:
        record = records.get(method_key)
        if record is None:
            continue
        available.append((method_key, record.score))
        lines.append(
            f"| {METHOD_LABELS[method_key]} | {record.score:.2f} | {record.score_detail} |"
        )
    missing = [method for method in METHOD_ORDER if method not in records]
    if missing:
        lines.append("")
        lines.append("缺失方法：")
        for method in missing:
            lines.append(f"- `{method}`")
    if available:
        best_method, best_score = max(available, key=lambda item: item[1])
        lines.append("")
        lines.append(
            f"最佳方法：{METHOD_LABELS[best_method]}，分数 `{best_score:.2f}`"
        )
    return "\n".join(lines)


def write_note(deduped: dict[str, dict[str, RunRecord]]) -> None:
    completeness_lines = []
    headline_lines = []
    chunkkv_lines = []
    for dataset_key in DATASET_ORDER:
        actual = len(deduped[dataset_key])
        expected = 5 if dataset_key.startswith("longbench:") else 4
        completeness_lines.append(f"- `{dataset_key}`：{actual}/{expected}")

        dataset_records = deduped[dataset_key]
        best_record = None
        for method_key in METHOD_ORDER:
            record = dataset_records.get(method_key)
            if record is None:
                continue
            if best_record is None or record.score > best_record.score:
                best_record = record
        if best_record is not None:
            headline_lines.append(
                f"- `{dataset_key}`：最佳方法为 {METHOD_LABELS[best_record.method_key]}，分数 `{best_record.score:.2f}`"
            )

        main_record = dataset_records.get("blockwise_main")
        chunk_record = dataset_records.get("chunkkv_prefill")
        if main_record is not None and chunk_record is not None:
            delta = main_record.score - chunk_record.score
            chunkkv_lines.append(
                f"- `{dataset_key}`：blockwise_main 相对 chunkkv 变化 `{delta:.2f}`（blockwise=`{main_record.score:.2f}`，chunkkv=`{chunk_record.score:.2f}`）"
            )

    failure_lines = summarize_failures()
    failure_block = "\n".join(failure_lines) if failure_lines else "- 无最终失败项"
    dataset_sections = [build_dataset_section(dataset_key, deduped[dataset_key]) for dataset_key in DATASET_ORDER]

    note = f"""# Blockwise Stage2 多数据集实验分析（ratio=0.7, fraction=0.2）

## 实验设置

- 运行脚本：
  - [{RUN_SCRIPT.name}]({RUN_SCRIPT})
- 结果目录：
  - [artifacts]({ARTIFACTS_DIR})
  - [run.log]({ARTIFACTS_DIR / 'run.log'})
- 模型：
  - `/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：
  - `RULER / 4096 / niah_single_3, niah_multikey_3, qa_2`
  - `LongBench / qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, triviaqa`
  - `Needle in a Haystack / 16384 / [0,25,50,75,100]`
- 方法：
  - `blockwise_main`
  - `blockwise_norm_topk`
  - `blockwise_multi_rep`
  - `blockwise_tail_query_special`
  - `chunkkv_prefill_per_layer`

## 完整性说明

{chr(10).join(completeness_lines)}

## 主要观察

{chr(10).join(headline_lines) if headline_lines else '- 暂无可用结果'}

## Blockwise vs ChunkKV

{chr(10).join(chunkkv_lines) if chunkkv_lines else '- 当前还没有足够结果比较 blockwise 与 chunkkv'}

## 最终失败项

{failure_block}

## 数据集明细

{chr(10).join(dataset_sections)}
"""
    NOTE_PATH.write_text(note + "\n")


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    deduped = dedupe_records(records)
    write_result_readme(deduped)
    write_figure_readme()
    write_note(deduped)
    ensure_index_entry(EVAL_INDEX, f"- `experiments/{EXPERIMENT_NAME}`")
    ensure_index_entry(FIGURE_INDEX, f"- `experiments/{EXPERIMENT_NAME}`")
    print(f"Postprocessed {EXPERIMENT_NAME} from {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
