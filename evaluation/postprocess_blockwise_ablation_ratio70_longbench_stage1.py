from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "blockwise_ablation_ratio70_longbench_stage1"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
NOTE_PATH = REPO_ROOT / "note" / f"{EXPERIMENT_NAME}_analysis_zh.md"
RESULT_README = RESULT_ROOT / "README.md"
FIGURE_README = FIGURE_DIR / "README.md"
EVAL_INDEX = REPO_ROOT / "evaluation" / "results" / "EXPERIMENT_INDEX.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"

RUN_SCRIPT = REPO_ROOT / "evaluation" / "run_blockwise_ablation_ratio70_longbench_stage1.sh"
RERUN_SCRIPT = (
    REPO_ROOT / "evaluation" / "rerun_blockwise_ablation_ratio70_longbench_stage1_triviaqa_missing.sh"
)

DATASETS = ["hotpotqa", "multifieldqa_en", "triviaqa"]
EXPECTED_TAGS = [
    "baseline",
    "A_mean_only",
    "A_norm_topk_mean_only",
    "A_multi_rep_max",
    "B_tail_query_relevance",
    "B_random_topk_seed42",
    "B_random_topk_seed43",
    "B_random_topk_seed44",
    "C_max",
    "C_topr_mean",
    "D_strength_weighted",
    "D_top_head_only",
    "quest_prefill",
]

TAG_DESCRIPTIONS = {
    "baseline": "`mean_plus_norm_topk_mean` + `key_norm` + `mean` + `uniform_mean`",
    "A_mean_only": "`mean_only` + `key_norm` + `mean` + `uniform_mean`",
    "A_norm_topk_mean_only": "`norm_topk_mean_only` + `key_norm` + `mean` + `uniform_mean`",
    "A_multi_rep_max": "`multi_rep_max` + `key_norm` + `mean` + `uniform_mean`",
    "B_tail_query_relevance": "`mean_plus_norm_topk_mean` + `tail_query_relevance` + `mean` + `uniform_mean`",
    "B_random_topk_seed42": "`random_topk(seed=42)` + baseline",
    "B_random_topk_seed43": "`random_topk(seed=43)` + baseline",
    "B_random_topk_seed44": "`random_topk(seed=44)` + baseline",
    "C_max": "`mean_plus_norm_topk_mean` + `key_norm` + `max` + `uniform_mean`",
    "C_topr_mean": "`mean_plus_norm_topk_mean` + `key_norm` + `topr_mean` + `uniform_mean`",
    "D_strength_weighted": "`mean_plus_norm_topk_mean` + `key_norm` + `mean` + `strength_weighted`",
    "D_top_head_only": "`mean_plus_norm_topk_mean` + `key_norm` + `mean` + `top_head_only`",
    "quest_prefill": "`Quest-prefill (minmax)`",
}


@dataclass
class RunRecord:
    dataset: str
    tag: str
    score: float
    path: Path
    cfg: dict


def score_from_metrics(path: Path) -> float:
    value = json.loads(path.read_text())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "accuracy", "metric", "value"):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
    raise ValueError(f"Unsupported metrics format: {path}")


def classify_tag(cfg: dict) -> str:
    press = cfg.get("press_name")
    if press == "quest_blockwise_prefill_per_layer":
        return "quest_prefill"

    summary = cfg.get("summary_mode")
    rep = cfg.get("representative_mode")
    qagg = cfg.get("query_agg_mode")
    hagg = cfg.get("head_agg_mode")
    seed = cfg.get("random_seed")

    if summary == "mean_only":
        return "A_mean_only"
    if summary == "norm_topk_mean_only":
        return "A_norm_topk_mean_only"
    if summary == "multi_rep_max":
        return "A_multi_rep_max"
    if rep == "tail_query_relevance":
        return "B_tail_query_relevance"
    if rep == "random_topk":
        return f"B_random_topk_seed{seed}"
    if qagg == "max":
        return "C_max"
    if qagg == "topr_mean":
        return "C_topr_mean"
    if hagg == "strength_weighted":
        return "D_strength_weighted"
    if hagg == "top_head_only":
        return "D_top_head_only"
    return "baseline"


def load_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    for cfg_path in ARTIFACTS_DIR.rglob("config.yaml"):
        metrics_path = cfg_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        dataset = cfg.get("data_dir")
        if dataset not in DATASETS:
            continue
        tag = classify_tag(cfg)
        score = score_from_metrics(metrics_path)
        records.append(RunRecord(dataset=dataset, tag=tag, score=score, path=cfg_path.parent, cfg=cfg))
    return records


def dedupe_records(records: Iterable[RunRecord]) -> dict[str, dict[str, RunRecord]]:
    deduped: dict[str, dict[str, RunRecord]] = {dataset: {} for dataset in DATASETS}
    for record in sorted(records, key=lambda item: item.path.stat().st_mtime_ns):
        deduped[record.dataset][record.tag] = record
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


def write_result_readme(deduped: dict[str, dict[str, RunRecord]]) -> None:
    completeness_lines = []
    for dataset in DATASETS:
        done = len(deduped[dataset])
        status = "完整" if done == len(EXPECTED_TAGS) else f"缺失 {len(EXPECTED_TAGS) - done} 组"
        completeness_lines.append(f"- `{dataset}`：{done}/13，{status}")

    complete_datasets = [dataset for dataset in DATASETS if len(deduped[dataset]) == len(EXPECTED_TAGS)]
    complete_desc = "、".join(f"`{dataset}`" for dataset in complete_datasets) if complete_datasets else "暂无"
    content = f"""# {EXPERIMENT_NAME}

## 实验目的

将 `RULER stage1` 的 blockwise 逐轴消融迁移到 `LongBench`，在高压缩率 `ratio=0.7` 下比较：

- A. `block summary form`
- B. `block-internal representative selection`
- C. `query window aggregation`
- D. `head aggregation`
- 以及 `Quest-style prefill block scorer`

## 运行脚本

- 主脚本：
  [{RUN_SCRIPT.name}]({RUN_SCRIPT})
- `triviaqa` 缺失补跑：
  [{RERUN_SCRIPT.name}]({RERUN_SCRIPT})

## 数据集

- `LongBench / hotpotqa`
- `LongBench / multifieldqa_en`
- `LongBench / triviaqa`

## 方法

- `block_wise_prefill_per_layer`
- `quest_blockwise_prefill_per_layer`

## 关键 sweep 维度

- `compression_ratio=0.7`
- `fraction=0.2`
- `block_size=16`
- `q_window_size=64`
- 不设置 `skip_first`

## 采样比例

- 不使用 `samples_per_task`
- 各任务直接按 `fraction=0.2` 采样

## 产物位置

- 原始结果：
  [artifacts]({ARTIFACTS_DIR})
- 主运行日志：
  [run.log]({ARTIFACTS_DIR / 'run.log'})
- 失败记录：
  - [failed_tasks.txt]({ARTIFACTS_DIR / 'failed_tasks.txt'})
  - [failed_tasks_rerun.txt]({ARTIFACTS_DIR / 'failed_tasks_rerun.txt'})

## 当前完整性

{chr(10).join(completeness_lines)}

当前可以做公平比较的数据集：

- {complete_desc}

## 推荐优先查看

- 中文分析：
  [{NOTE_PATH.name}]({NOTE_PATH})
"""
    RESULT_README.write_text(content + "\n")


def write_figure_readme(deduped: dict[str, dict[str, RunRecord]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for dataset in DATASETS:
        remain = len(EXPECTED_TAGS) - len(deduped[dataset])
        if remain > 0:
            missing.append(f"`{dataset}` 缺失 {remain} 组")
    coverage = "；".join(missing) if missing else "全部 3 个数据集已补齐。"
    content = f"""# {EXPERIMENT_NAME}

## 图像说明

本轮实验目前还没有正式生成图像文件。

当前目录用于为后续图像归档预留规范位置，并与结果目录保持一一对应。

## 对应结果目录

- [{RESULT_ROOT}]({RESULT_ROOT})

## 推荐优先查看

- 中文分析：
  [{NOTE_PATH.name}]({NOTE_PATH})

## 覆盖说明

- {coverage}
"""
    FIGURE_README.write_text(content + "\n")


def write_note(deduped: dict[str, dict[str, RunRecord]]) -> None:
    completeness_lines = []
    for dataset in DATASETS:
        done = len(deduped[dataset])
        missing = len(EXPECTED_TAGS) - done
        if missing == 0:
            completeness_lines.append(f"- `{dataset}`：13/13 完整")
        else:
            completeness_lines.append(f"- `{dataset}`：{done}/13，仍缺失 {missing} 组")

    per_dataset_sections = []
    best_candidates: list[tuple[str, str, float]] = []
    baseline_delta_lines = []
    quest_delta_lines = []
    max_delta_lines = []

    for dataset in DATASETS:
        dataset_records = deduped[dataset]
        rows = []
        for tag in EXPECTED_TAGS:
            record = dataset_records.get(tag)
            score = record.score if record else None
            rows.append((tag, score))
        available = [(tag, score) for tag, score in rows if score is not None]
        available.sort(key=lambda item: item[1], reverse=True)
        if available:
            best_tag, best_score = available[0]
            best_candidates.append((dataset, best_tag, best_score))
        baseline = dataset_records.get("baseline")
        qmax = dataset_records.get("C_max")
        quest = dataset_records.get("quest_prefill")
        if baseline and qmax:
            max_delta_lines.append(
                f"- `{dataset}`：`query_agg=max` 相对 baseline 提升 `{qmax.score - baseline.score:.2f}`"
            )
        if baseline and quest:
            quest_delta_lines.append(
                f"- `{dataset}`：Quest 相对 baseline 变化 `{quest.score - baseline.score:.2f}`"
            )
        if baseline:
            baseline_delta_lines.append(f"- `{dataset}` baseline：`{baseline.score:.2f}`")

        table_lines = [
            "| 配置 | 分数 |",
            "|---|---:|",
        ]
        for tag, score in available:
            table_lines.append(f"| {TAG_DESCRIPTIONS[tag]} | {format_score(score)} |")
        missing_tags = [tag for tag in EXPECTED_TAGS if tag not in dataset_records]
        if missing_tags:
            table_lines.append("")
            table_lines.append("缺失配置：")
            for tag in missing_tags:
                table_lines.append(f"- `{tag}`")
        per_dataset_sections.append(
            f"""## `{dataset}`

{chr(10).join(table_lines)}"""
        )

    best_lines = [
        f"- `{dataset}`：最佳配置为 {TAG_DESCRIPTIONS[tag]}，分数 `{score:.2f}`"
        for dataset, tag, score in best_candidates
    ]

    note = f"""# BlockWise LongBench Stage1 消融实验分析（ratio=0.7, fraction=0.2）

## 实验设置

- 运行脚本：
  - [{RUN_SCRIPT.name}]({RUN_SCRIPT})
  - `triviaqa` 补跑：
    [{RERUN_SCRIPT.name}]({RERUN_SCRIPT})
- 结果目录：
  - [artifacts]({ARTIFACTS_DIR})
  - [run.log]({ARTIFACTS_DIR / 'run.log'})
- 模型：
  - `/Tan/model/Llama-3.1-8B-Instruct`
- 数据集：
  - `LongBench / hotpotqa`
  - `LongBench / multifieldqa_en`
  - `LongBench / triviaqa`
- 压缩设置：
  - `compression_ratio=0.7`
  - `block_size=16`
  - `q_window_size=64`
  - `summary_topk_keys=4`
  - `mean_key_weight=0.75`
  - `representative_k=4`
  - `multi_rep_k=4`
  - `query_topr=16`
  - `head_topk=1`
  - `query_aware=true`
- 采样设置：
  - 不使用 `samples_per_task`
  - 各任务直接按 `fraction=0.2` 采样

## 完整性说明

{chr(10).join(completeness_lines)}

## 总体观察

{chr(10).join(best_lines) if best_lines else '- 暂无可用结果'}

{chr(10).join(max_delta_lines) if max_delta_lines else '- `query_agg=max` 仍缺少可比较对照'}

{chr(10).join(quest_delta_lines) if quest_delta_lines else '- Quest 仍缺少可比较对照'}

{chr(10).join(baseline_delta_lines)}

## 数据集明细

{chr(10).join(per_dataset_sections)}

## 阶段性结论

- `query_agg=max` 仍然是最值得优先关注的候选项；如果三个数据集都可比较，它通常会是最稳的增强方向。
- `Quest-prefill` 目前主要还是对照组，是否能追平 summary-based blockwise，要看 `triviaqa` 补齐后的完整对比。
- 如果本轮补跑后 `triviaqa` 也补齐，这份文档就可以直接作为 LongBench stage1 的正式归档说明；否则仍应把它视为“部分完成”的阶段性记录。
"""
    NOTE_PATH.write_text(note + "\n")


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    deduped = dedupe_records(records)
    write_result_readme(deduped)
    write_figure_readme(deduped)
    write_note(deduped)
    ensure_index_entry(EVAL_INDEX, f"- `experiments/{EXPERIMENT_NAME}`")
    ensure_index_entry(FIGURE_INDEX, f"- `experiments/{EXPERIMENT_NAME}`")
    print(f"Postprocessed {EXPERIMENT_NAME} from {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
