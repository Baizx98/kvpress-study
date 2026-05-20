from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_needle_heatmap_llama31_8b_ratio50"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
LOGS_DIR = ARTIFACTS_DIR / "logs"
MANIFEST_PATH = ARTIFACTS_DIR / "ATC26_needle_heatmap_manifest.jsonl"
PROGRESS_PATH = ARTIFACTS_DIR / "ATC26_needle_heatmap_progress.md"
STATUS_PATH = ARTIFACTS_DIR / "ATC26_needle_heatmap_status.json"

MODEL_PATH = "/Tan/model/Llama-3.1-8B-Instruct"
COMPRESSION_RATIO = 0.5
NEEDLE_DEPTHS_FULL = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
NEEDLE_DEPTHS_SMOKE = [0, 50, 100]
CONTEXT_LENGTHS_FULL = [4096, 8192, 16384, 32768, 65536]
CONTEXT_LENGTHS_SMOKE = [4096, 16384]
METHODS_FULL = ["block_wise", "snapkv", "chunkkv"]
METHODS_SMOKE = ["block_wise", "chunkkv"]

GPU_INDEX = os.environ.get("ATC26_GPU_INDEX", "1")
GPU_UUID = os.environ.get("ATC26_GPU_UUID", "GPU-4eac01c5-47d1-3958-95bb-98d357b8b9c3")


@dataclass(frozen=True)
class NeedleJob:
    job_id: str
    mode: str
    method: str
    context_length: int
    needle_depths: list[int]
    output_dir: str
    log_path: str


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_status() -> dict[str, dict]:
    if not STATUS_PATH.exists():
        return {}
    return json.loads(STATUS_PATH.read_text())


def _save_status(status: dict[str, dict]) -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def _write_progress(jobs: list[NeedleJob], status: dict[str, dict]) -> None:
    counts: dict[str, int] = {}
    for job in jobs:
        state = status.get(job.job_id, {}).get("state", "pending")
        counts[state] = counts.get(state, 0) + 1

    lines = [
        "# ATC26 Needle Heatmap Progress",
        "",
        f"- updated_at: `{_now()}`",
        f"- experiment: `{EXPERIMENT_NAME}`",
        f"- gpu: physical `{GPU_INDEX}`, uuid `{GPU_UUID}`",
        f"- total_jobs: `{len(jobs)}`",
        "",
        "| state | count |",
        "|---|---:|",
    ]
    for state in ["success", "running", "failed", "pending", "skipped"]:
        lines.append(f"| `{state}` | {counts.get(state, 0)} |")

    recent = sorted(
        (
            (item.get("finished_at") or item.get("started_at") or "", job_id, item)
            for job_id, item in status.items()
        ),
        reverse=True,
    )[:20]
    lines.extend(["", "## Recent Jobs", "", "| job | state | detail |", "|---|---|---|"])
    for _, job_id, item in recent:
        detail = item.get("detail", "")
        lines.append(f"| `{job_id}` | `{item.get('state')}` | `{detail}` |")

    PROGRESS_PATH.write_text("\n".join(lines) + "\n")


def _build_jobs(mode: str) -> list[NeedleJob]:
    if mode == "smoke":
        methods = METHODS_SMOKE
        context_lengths = CONTEXT_LENGTHS_SMOKE
        needle_depths = NEEDLE_DEPTHS_SMOKE
    else:
        methods = METHODS_FULL
        context_lengths = CONTEXT_LENGTHS_FULL
        needle_depths = NEEDLE_DEPTHS_FULL

    jobs = []
    for method in methods:
        for context_length in context_lengths:
            job_id = f"{mode}__{method}__ctx{context_length}"
            jobs.append(
                NeedleJob(
                    job_id=job_id,
                    mode=mode,
                    method=method,
                    context_length=context_length,
                    needle_depths=needle_depths,
                    output_dir=str(RAW_DIR),
                    log_path=str(LOGS_DIR / f"{job_id}.log"),
                )
            )
    return jobs


def _write_manifest(jobs: list[NeedleJob]) -> None:
    with MANIFEST_PATH.open("w") as f:
        for job in jobs:
            f.write(json.dumps(asdict(job), sort_keys=True) + "\n")


def _base_command(job: NeedleJob) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "evaluation" / "evaluate.py"),
        "--config_file",
        "/dev/null",
        "--dataset",
        "needle_in_haystack",
        "--model",
        MODEL_PATH,
        "--device",
        "cuda:0",
        "--press_name",
        job.method,
        "--compression_ratio",
        str(COMPRESSION_RATIO),
        "--max_context_length",
        str(job.context_length),
        "--needle_depth",
        json.dumps(job.needle_depths),
        "--max_new_tokens",
        "50",
        "--query_aware",
        "True",
        "--output_dir",
        job.output_dir,
        "--result_file_prefix",
        "ATC26",
        "--seed",
        "42",
        "--log_level",
        "INFO",
    ]
    if job.method == "block_wise":
        cmd.extend(
            [
                "--block_size",
                "16",
                "--q_window_size",
                "64",
                "--summary_topk_keys",
                "4",
                "--mean_key_weight",
                "0.75",
                "--summary_mode",
                "mean_plus_norm_topk_mean",
                "--representative_mode",
                "key_norm",
                "--query_agg_mode",
                "max",
                "--head_agg_mode",
                "uniform_mean",
                "--representative_k",
                "4",
                "--multi_rep_k",
                "4",
                "--query_topr",
                "16",
                "--head_topk",
                "1",
            ]
        )
    return cmd


def _run_job(job: NeedleJob, status: dict[str, dict]) -> None:
    status[job.job_id] = {
        "state": "running",
        "started_at": _now(),
        "method": job.method,
        "context_length": job.context_length,
    }
    _save_status(status)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    env.setdefault("HF_HOME", "/Tan/dataset/hf_home")
    env.setdefault("HF_DATASETS_CACHE", "/Tan/dataset/hf_home/datasets")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/Tan/dataset/hf_home/hub")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = _base_command(job)
    with open(job.log_path, "w") as log_f:
        log_f.write(f"# started_at={_now()}\n")
        log_f.write(" ".join(cmd) + "\n\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode == 0:
        status[job.job_id] = {
            **status[job.job_id],
            "state": "success",
            "finished_at": _now(),
            "detail": "ok",
        }
    else:
        status[job.job_id] = {
            **status[job.job_id],
            "state": "failed",
            "finished_at": _now(),
            "detail": f"returncode={proc.returncode}",
        }
    _save_status(status)


def _write_readme(mode: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    readme = RESULTS_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# ATC26 Needle Heatmap",
                "",
                f"- mode: `{mode}`",
                f"- model: `{MODEL_PATH}`",
                f"- methods: `{', '.join(METHODS_FULL)}`",
                f"- compression_ratio: `{COMPRESSION_RATIO}`",
                f"- haystack: `alessiodevoto/paul_graham_essays`",
                f"- gpu: physical `{GPU_INDEX}`, uuid `{GPU_UUID}`",
                f"- progress: `artifacts/ATC26_needle_heatmap_progress.md`",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in [ARTIFACTS_DIR, RAW_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(args.mode)
    _write_manifest(jobs)
    _write_readme(args.mode)
    status = _load_status() if args.resume else {}

    for job in jobs:
        if args.resume and status.get(job.job_id, {}).get("state") == "success":
            continue
        _write_progress(jobs, status)
        _run_job(job, status)
        _write_progress(jobs, status)

    _write_progress(jobs, status)


if __name__ == "__main__":
    main()
