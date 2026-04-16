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
EXPERIMENT_NAME = "decode_long_output_longbench_stage1"
OUTPUT_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
RUN_LOG = OUTPUT_DIR / "run.log"
FAILED_JOBS = OUTPUT_DIR / "failed_jobs.jsonl"
FAILED_FINAL = OUTPUT_DIR / "failed_jobs_final.jsonl"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
MODEL = os.environ.get("MODEL", "/Tan/model/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda:0")
GPU_INDEX = int(os.environ.get("GPU_INDEX", "0"))
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "40000"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "60"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

TASKS = ["gov_report", "qmsum", "multi_news"]
BLOCK_SIZE = 16
MIN_ANSWER_TOKENS = 64
MIN_CONTEXT_TOKENS = 4000
MAX_FILTERED_SAMPLES = 20
PROTECTED_RECENT_BLOCKS = 2


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
    if "no samples remain after token-length filtering" in lowered:
        return "empty_filtered_dataset"
    if return_code == -9:
        return "killed"
    return "unknown"


def has_completed_results(match_fields: dict[str, Any]) -> bool:
    for config_path in OUTPUT_DIR.rglob("config.yaml"):
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text())
        if all(cfg.get(key) == value for key, value in match_fields.items()):
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
            if len(tail_buffer) > 200:
                tail_buffer = tail_buffer[-200:]
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


def base_common_args() -> list[str]:
    return [
        "--model",
        MODEL,
        "--device",
        DEVICE,
        "--dataset",
        "longbench",
        "--compression_ratio",
        "0.3",
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
        "--min_answer_tokens",
        str(MIN_ANSWER_TOKENS),
        "--min_context_tokens",
        str(MIN_CONTEXT_TOKENS),
        "--max_filtered_samples",
        str(MAX_FILTERED_SAMPLES),
        "--output_dir",
        str(OUTPUT_DIR),
    ]


def make_jobs() -> list[Job]:
    jobs: list[Job] = []
    for task in TASKS:
        common = base_common_args() + ["--data_dir", task]

        jobs.append(
            Job(
                job_id=f"longbench:{task}__prefill_only_no_decode_pruning",
                cli_args=common + ["--press_name", "block_wise_prefill_per_layer"],
                match_fields={
                    "dataset": "longbench",
                    "data_dir": task,
                    "press_name": "block_wise_prefill_per_layer",
                    "compression_ratio": 0.3,
                    "min_answer_tokens": MIN_ANSWER_TOKENS,
                    "min_context_tokens": MIN_CONTEXT_TOKENS,
                    "max_filtered_samples": MAX_FILTERED_SAMPLES,
                },
            )
        )

        jobs.append(
            Job(
                job_id=f"longbench:{task}__decode_permanent_eviction_fixed_budget",
                cli_args=common
                + [
                    "--press_name",
                    "dual_phase_per_layer",
                    "--dual_phase_mode",
                    "permanent_fixed_budget",
                    "--compression_interval",
                    str(BLOCK_SIZE),
                ],
                match_fields={
                    "dataset": "longbench",
                    "data_dir": task,
                    "press_name": "dual_phase_per_layer",
                    "dual_phase_mode": "permanent_fixed_budget",
                    "compression_ratio": 0.3,
                    "compression_interval": BLOCK_SIZE,
                    "min_answer_tokens": MIN_ANSWER_TOKENS,
                    "min_context_tokens": MIN_CONTEXT_TOKENS,
                    "max_filtered_samples": MAX_FILTERED_SAMPLES,
                },
            )
        )

        jobs.append(
            Job(
                job_id=f"longbench:{task}__decode_compute_cold_fixed_active_budget",
                cli_args=common
                + [
                    "--press_name",
                    "dual_phase_per_layer",
                    "--dual_phase_mode",
                    "compute_cold_fixed_budget",
                    "--compression_interval",
                    str(BLOCK_SIZE),
                ],
                match_fields={
                    "dataset": "longbench",
                    "data_dir": task,
                    "press_name": "dual_phase_per_layer",
                    "dual_phase_mode": "compute_cold_fixed_budget",
                    "compression_ratio": 0.3,
                    "compression_interval": BLOCK_SIZE,
                    "min_answer_tokens": MIN_ANSWER_TOKENS,
                    "min_context_tokens": MIN_CONTEXT_TOKENS,
                    "max_filtered_samples": MAX_FILTERED_SAMPLES,
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
