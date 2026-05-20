from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "evaluation" / "ATC26_run_prefill_sweep.py"
EXPERIMENT_NAME = "ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv"

LONGBENCH16_MAX_NEW_TOKENS = {
    "narrativeqa": "148",
    "qasper": "148",
    "multifieldqa_en": "84",
    "hotpotqa": "52",
    "2wikimqa": "52",
    "musique": "52",
    "triviaqa": "52",
    "gov_report": "532",
    "qmsum": "532",
    "multi_news": "532",
    "samsum": "148",
    "trec": "84",
    "passage_count": "52",
    "passage_retrieval_en": "52",
    "lcc": "84",
    "repobench-p": "84",
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("atc26_base_runner", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def longbench_args(data_dir: str) -> list[str]:
    return [
        "--dataset",
        "longbench",
        "--data_dir",
        data_dir,
        "--max_new_tokens",
        LONGBENCH16_MAX_NEW_TOKENS[data_dir],
    ]


def dataset_specs() -> list[tuple[str, list[str]]]:
    only_tasks = {
        task.strip()
        for task in os.environ.get("ATC26_ONLY_LONGBENCH_TASKS", "").split(",")
        if task.strip()
    }
    skip_tasks = {
        task.strip()
        for task in os.environ.get("ATC26_SKIP_LONGBENCH_TASKS", "").split(",")
        if task.strip()
    }
    return [
        (f"longbench:{task}", longbench_args(task))
        for task in LONGBENCH16_MAX_NEW_TOKENS
        if not only_tasks or task in only_tasks
        if task not in skip_tasks
    ]


def method_key_from_config(cfg: dict[str, Any]) -> str:
    press_name = cfg.get("press_name")
    if press_name == "block_wise":
        return "blockwise"
    if press_name == "snapkv":
        return "snapkv"
    if press_name == "chunkkv":
        return "chunkkv"
    return str(press_name)


def patch_base_module(base: Any) -> None:
    result_root = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
    artifacts_dir = result_root / "artifacts"

    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.RESULT_ROOT = result_root
    base.ARTIFACTS_DIR = artifacts_dir
    base.RAW_DIR = artifacts_dir / "raw"
    base.LOG_DIR = artifacts_dir / "logs"
    base.RUN_LOG = artifacts_dir / "ATC26_run.log"
    base.PROGRESS_JSONL = artifacts_dir / "ATC26_progress.jsonl"
    base.PROGRESS_MD = artifacts_dir / "ATC26_progress.md"
    base.MANIFEST_JSONL = artifacts_dir / "ATC26_manifest.jsonl"
    base.FAILED_JOBS = artifacts_dir / "ATC26_failed_jobs.jsonl"
    base.FAILED_FINAL = artifacts_dir / "ATC26_failed_jobs_final.jsonl"
    base.POSTPROCESS_SCRIPT = REPO_ROOT / "evaluation" / "ATC26_postprocess_longbench16_prefill_sweep.py"

    base.FULL_COMPRESSION_RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7]
    base.SMOKE_COMPRESSION_RATIOS = [0.5]
    base.METHODS = [
        (
            "blockwise",
            [
                "--press_name",
                "block_wise",
                "--block_size",
                "16",
                "--q_window_size",
                "64",
                "--summary_topk_keys",
                "4",
                "--mean_key_weight",
                "0.75",
                "--representative_k",
                "4",
                "--multi_rep_k",
                "4",
                "--query_topr",
                "16",
                "--head_topk",
                "1",
                "--summary_mode",
                "mean_plus_norm_topk_mean",
                "--representative_mode",
                "key_norm",
                "--query_agg_mode",
                "max",
                "--head_agg_mode",
                "uniform_mean",
                "--prefill_skip_first_layers",
                "2",
            ],
        ),
        ("snapkv", ["--press_name", "snapkv"]),
        ("chunkkv", ["--press_name", "chunkkv", "--block_size", "16"]),
    ]
    original_classify_failure = base.classify_failure
    base.dataset_specs = dataset_specs
    base.has_completed_results = has_completed_results_factory(base)
    base.classify_failure = classify_failure_factory(original_classify_failure)


def classify_failure_factory(original_classify_failure: Any):
    def classify_failure(text: str, return_code: int) -> str:
        lowered = text.lower()
        if "proxyerror" in lowered or "remotedisconnected" in lowered:
            return "network"
        return original_classify_failure(text, return_code)

    return classify_failure


def has_completed_results_factory(base: Any):
    def has_completed_results(job: Any) -> bool:
        output_dir = Path(job.output_dir)
        expected_data_dir = job.dataset_key.split(":", 1)[1]
        for metrics_path in output_dir.rglob("ATC26_metrics.json"):
            config_path = metrics_path.with_name("ATC26_config.yaml")
            if not config_path.exists():
                continue
            try:
                cfg = yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                continue
            if str(cfg.get("dataset")) != "longbench":
                continue
            if str(cfg.get("data_dir")) != expected_data_dir:
                continue
            if str(cfg.get("model")) != job.model_path:
                continue
            if method_key_from_config(cfg) != job.method_key:
                continue
            if float(cfg.get("compression_ratio", -1.0)) != job.compression_ratio:
                continue
            if float(cfg.get("fraction", -1.0)) != job.fraction:
                continue
            if job.method_key == "blockwise" and int(cfg.get("prefill_skip_first_layers", -1)) != 2:
                continue
            return True
        return False

    return has_completed_results


def main() -> int:
    base = load_base_module()
    patch_base_module(base)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
