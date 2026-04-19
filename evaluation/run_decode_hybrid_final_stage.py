from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "decode_hybrid_final_stage"
OUTPUT_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
RUN_LOG = OUTPUT_DIR / "run.log"
FAILED_JOBS = OUTPUT_DIR / "failed_jobs.jsonl"
FAILED_FINAL = OUTPUT_DIR / "failed_jobs_final.jsonl"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"

MODEL = os.environ.get("MODEL", "/Tan/model/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda:0")
GPU_INDEX = int(os.environ.get("GPU_INDEX", "1"))
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "36000"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "60"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

BLOCK_SIZE = 16
PROTECTED_RECENT_BLOCKS = 2
LONG_BENCH_TASKS = ["gov_report", "qmsum", "multi_news"]
LONG_BENCH_FILTER_ARGS = [
    "--min_answer_tokens",
    "64",
    "--min_context_tokens",
    "4000",
    "--max_filtered_samples",
    "20",
]
RULER_TASK_FILTER = ["niah_single_3", "niah_multikey_2", "niah_multikey_3", "qa_2"]
RULER_BUDGETS = [128, 160]
ROUTES = [
    ("dense_prefill__hybrid", {"total_budget": 128, "active_budget": 96}),
    ("dense_prefill__hybrid", {"total_budget": 160, "active_budget": 128}),
    ("dense_prefill__permanent", {"total_budget": 128}),
    ("dense_prefill__permanent", {"total_budget": 160}),
    ("dense_prefill__compute_cold", {"active_budget": 128}),
    ("dense_prefill__compute_cold", {"active_budget": 160}),
]


@dataclass
class Job:
    job_id: str
    cli_args: list[str]
    match_fields: dict[str, Any]


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with RUN_LOG.open("a") as f:
        f.write(line + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def wait_for_gpu() -> None:
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        free_mb = int(output[GPU_INDEX].strip())
        if free_mb >= MIN_FREE_MB:
            log(f"CUDA:{GPU_INDEX} free memory {free_mb}MB >= {MIN_FREE_MB}MB, continue.")
            return
        log(f"CUDA:{GPU_INDEX} free memory {free_mb}MB < {MIN_FREE_MB}MB, waiting {POLL_SECONDS}s.")
        time.sleep(POLL_SECONDS)


def classify_failure(text: str, return_code: int) -> str:
    lowered = text.lower()
    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        return "oom"
    if "ssl" in lowered or "connection error" in lowered or "readtimeout" in lowered or "connection reset" in lowered:
        return "network"
    if "traceback" in lowered or "runtimeerror:" in lowered or "assertionerror:" in lowered:
        return "traceback"
    if return_code == -9:
        return "killed"
    if return_code == 0:
        return "missing_metrics"
    return "unknown"


def _normalize_match_value(value: Any) -> Any:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts if len(parts) > 1 else value
    if isinstance(value, tuple):
        return [_normalize_match_value(item) for item in value]
    return value


def _match_config(cfg_value: Any, expected_value: Any) -> bool:
    cfg_norm = _normalize_match_value(cfg_value)
    exp_norm = _normalize_match_value(expected_value)
    if isinstance(cfg_norm, list) and isinstance(exp_norm, list):
        return list(cfg_norm) == list(exp_norm)
    return cfg_norm == exp_norm


def has_completed_results(match_fields: dict[str, Any]) -> bool:
    for config_path in OUTPUT_DIR.rglob("config.yaml"):
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text())
        if all(_match_config(cfg.get(key), value) for key, value in match_fields.items()):
            return True
    return False


def run_job(job: Job) -> bool:
    if has_completed_results(job.match_fields):
        log(f"Skipping completed job={job.job_id}")
        return True

    reason = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        wait_for_gpu()
        log(f"Running job={job.job_id} attempt={attempt}")
        cmd = [str(PYTHON_BIN), "evaluation/evaluate.py", *job.cli_args]
        env = os.environ.copy()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        tail_buffer: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip("\n")
            tail_buffer.append(stripped)
            if len(tail_buffer) > 300:
                tail_buffer = tail_buffer[-300:]
            with RUN_LOG.open("a") as f:
                f.write(stripped + "\n")
        return_code = process.wait()

        if has_completed_results(job.match_fields):
            log(f"Completed job={job.job_id}")
            return True

        reason = classify_failure("\n".join(tail_buffer), return_code)
        append_jsonl(
            FAILED_JOBS,
            {
                "job_id": job.job_id,
                "attempt": attempt,
                "return_code": return_code,
                "reason": reason,
            },
        )
        log(f"Failed job={job.job_id} attempt={attempt} reason={reason} return_code={return_code}")
        if reason in {"oom", "killed"}:
            time.sleep(POLL_SECONDS)
        elif reason == "network":
            time.sleep(20)
        else:
            time.sleep(5)

    append_jsonl(
        FAILED_FINAL,
        {
            "job_id": job.job_id,
            "attempts": MAX_RETRIES,
            "last_reason": reason,
        },
    )
    return False


def common_args() -> list[str]:
    return [
        "--model",
        MODEL,
        "--device",
        DEVICE,
        "--press_name",
        "dual_phase_per_layer",
        "--compression_ratio",
        "0.0",
        "--block_size",
        str(BLOCK_SIZE),
        "--q_window_size",
        str(BLOCK_SIZE),
        "--query_agg_mode",
        "max",
        "--summary_mode",
        "mean_plus_norm_topk_mean",
        "--representative_mode",
        "key_norm",
        "--head_agg_mode",
        "uniform_mean",
        "--protected_recent_blocks",
        str(PROTECTED_RECENT_BLOCKS),
        "--compression_interval",
        str(BLOCK_SIZE),
        "--output_dir",
        str(OUTPUT_DIR),
    ]


def route_args(route: str, budgets: dict[str, int]) -> tuple[list[str], dict[str, Any]]:
    total_budget = budgets.get("total_budget")
    active_budget = budgets.get("active_budget")
    if route == "dense_prefill__permanent":
        assert total_budget is not None
        return [
            "--dual_phase_mode",
            "permanent_fixed_budget",
            "--decode_block_budget",
            str(total_budget),
        ], {
            "press_name": "dual_phase_per_layer",
            "compression_ratio": 0.0,
            "dual_phase_mode": "permanent_fixed_budget",
            "decode_block_budget": total_budget,
        }
    if route == "dense_prefill__compute_cold":
        assert active_budget is not None
        return [
            "--dual_phase_mode",
            "compute_cold_fixed_budget",
            "--decode_cold_block_budget",
            str(active_budget),
        ], {
            "press_name": "dual_phase_per_layer",
            "compression_ratio": 0.0,
            "dual_phase_mode": "compute_cold_fixed_budget",
            "decode_cold_block_budget": active_budget,
        }
    if route == "dense_prefill__hybrid":
        assert total_budget is not None and active_budget is not None
        return [
            "--dual_phase_mode",
            "hybrid_fixed_budget",
            "--decode_block_budget",
            str(total_budget),
            "--decode_cold_block_budget",
            str(active_budget),
        ], {
            "press_name": "dual_phase_per_layer",
            "compression_ratio": 0.0,
            "dual_phase_mode": "hybrid_fixed_budget",
            "decode_block_budget": total_budget,
            "decode_cold_block_budget": active_budget,
        }
    raise ValueError(f"Unknown route: {route}")


def make_jobs() -> list[Job]:
    jobs: list[Job] = []

    for task in LONG_BENCH_TASKS:
        for route, budgets in ROUTES:
            extra_args, match = route_args(route, budgets)
            budget_tag = (
                f"total{budgets['total_budget']}_active{budgets['active_budget']}"
                if route == "dense_prefill__hybrid"
                else f"budget{budgets.get('total_budget', budgets.get('active_budget'))}"
            )
            args = common_args() + [
                "--dataset",
                "longbench",
                "--data_dir",
                task,
                *LONG_BENCH_FILTER_ARGS,
                *extra_args,
            ]
            jobs.append(
                Job(
                    job_id=f"longbench:{task}__{route}__{budget_tag}",
                    cli_args=args,
                    match_fields={
                        "dataset": "longbench",
                        "data_dir": task,
                        "min_answer_tokens": 64,
                        "min_context_tokens": 4000,
                        "max_filtered_samples": 20,
                        **match,
                    },
                )
            )

    ruler_routes = [
        ("dense_prefill__permanent", {"total_budget": budget}) for budget in RULER_BUDGETS
    ] + [
        ("dense_prefill__compute_cold", {"active_budget": budget}) for budget in RULER_BUDGETS
    ] + [
        ("dense_prefill__hybrid", {"total_budget": 128, "active_budget": 96}),
        ("dense_prefill__hybrid", {"total_budget": 160, "active_budget": 128}),
    ]
    for route, budgets in ruler_routes:
        extra_args, match = route_args(route, budgets)
        budget_tag = (
            f"total{budgets['total_budget']}_active{budgets['active_budget']}"
            if route == "dense_prefill__hybrid"
            else f"budget{budgets.get('total_budget', budgets.get('active_budget'))}"
        )
        args = common_args() + [
            "--dataset",
            "ruler",
            "--data_dir",
            "4096",
            "--task_filter",
            ",".join(RULER_TASK_FILTER),
            "--samples_per_task",
            "20",
            "--max_new_tokens",
            "128",
            *extra_args,
        ]
        jobs.append(
            Job(
                job_id=f"ruler:4096__{route}__{budget_tag}",
                cli_args=args,
                match_fields={
                    "dataset": "ruler",
                    "data_dir": "4096",
                    "task_filter": RULER_TASK_FILTER,
                    "samples_per_task": 20,
                    "max_new_tokens": 128,
                    **match,
                },
            )
        )

    return jobs


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Starting experiment={EXPERIMENT_NAME} on device={DEVICE}")
    jobs = make_jobs()
    completed = 0
    for job in jobs:
        completed += int(run_job(job))
    log(f"Finished experiment={EXPERIMENT_NAME}; completed_jobs={completed}/{len(jobs)}")
    return 0 if completed == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
