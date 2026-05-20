from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
LOG_DIR = ARTIFACTS_DIR / "logs"
RUN_LOG = ARTIFACTS_DIR / "ATC26_run.log"
PROGRESS_JSONL = ARTIFACTS_DIR / "ATC26_progress.jsonl"
PROGRESS_MD = ARTIFACTS_DIR / "ATC26_progress.md"
MANIFEST_JSONL = ARTIFACTS_DIR / "ATC26_manifest.jsonl"
FAILED_JOBS = ARTIFACTS_DIR / "ATC26_failed_jobs.jsonl"
FAILED_FINAL = ARTIFACTS_DIR / "ATC26_failed_jobs_final.jsonl"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
POSTPROCESS_SCRIPT = REPO_ROOT / "evaluation" / "ATC26_postprocess_prefill_sweep.py"

MODELS = [
    ("llama31_8b_instruct", "/Tan/model/Llama-3.1-8B-Instruct"),
    ("mistral_7b_instruct_v03", "/Tan/model/Mistral-7B-Instruct-v0.3"),
    ("qwen3_8b", "/Tan/model/Qwen3-8B"),
]
FULL_COMPRESSION_RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SMOKE_COMPRESSION_RATIOS = [0.5]
METHODS = [
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
        ],
    ),
    ("snapkv", ["--press_name", "snapkv"]),
    ("chunkkv", ["--press_name", "chunkkv", "--block_size", "16"]),
]


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    physical_gpu: int
    gpu_uuid: str
    gpu_name: str
    min_free_mb: int


@dataclass(frozen=True)
class Job:
    job_id: str
    model_key: str
    model_path: str
    dataset_key: str
    method_key: str
    compression_ratio: float
    fraction: float
    cli_args: list[str]
    output_dir: str


class Progress:
    def __init__(self, jobs: list[Job]) -> None:
        self.lock = threading.Lock()
        self.status: dict[str, dict[str, Any]] = {
            job.job_id: {
                "status": "pending",
                "model": job.model_key,
                "dataset": job.dataset_key,
                "method": job.method_key,
                "compression_ratio": job.compression_ratio,
                "worker": None,
                "gpu": None,
                "attempt": 0,
                "started_at": None,
                "finished_at": None,
                "reason": None,
            }
            for job in jobs
        }
        self.recent: list[str] = []

    def update(self, job: Job, **fields: Any) -> None:
        with self.lock:
            self.status[job.job_id].update(fields)
            event = {"time": now(), "job_id": job.job_id, **self.status[job.job_id]}
            append_jsonl(PROGRESS_JSONL, event)
            if fields.get("status") in {"success", "failed", "skipped"}:
                self.recent.append(job.job_id)
                self.recent = self.recent[-20:]
            self.write_markdown()

    def write_markdown(self) -> None:
        total = len(self.status)
        counts = count_by(self.status.values(), "status")
        lines = [
            "# ATC26 Progress",
            "",
            f"- Updated: `{now()}`",
            f"- Total: `{total}`",
            f"- Pending: `{counts.get('pending', 0)}`",
            f"- Running: `{counts.get('running', 0)}`",
            f"- Success: `{counts.get('success', 0)}`",
            f"- Failed: `{counts.get('failed', 0)}`",
            f"- Skipped: `{counts.get('skipped', 0)}`",
            "",
            "## By Model",
            *format_group_counts(self.status.values(), "model"),
            "",
            "## By Dataset",
            *format_group_counts(self.status.values(), "dataset"),
            "",
            "## By Method",
            *format_group_counts(self.status.values(), "method"),
            "",
            "## By Worker",
            *format_group_counts(self.status.values(), "worker"),
            "",
            "## Running",
            *format_running(self.status),
            "",
            "## Recent Completed",
            *(f"- `{job_id}`: `{self.status[job_id]['status']}`" for job_id in reversed(self.recent)),
            "",
            "## Failed",
            *format_failed(self.status),
            "",
        ]
        PROGRESS_MD.write_text("\n".join(lines))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with RUN_LOG.open("a") as f:
        f.write(line + "\n")


def count_by(records, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "none")
        counts[value] = counts.get(value, 0) + 1
    return counts


def format_group_counts(records, group_key: str) -> list[str]:
    grouped: dict[str, dict[str, int]] = {}
    for record in records:
        group = str(record.get(group_key) or "none")
        grouped.setdefault(group, {})
        status = str(record.get("status"))
        grouped[group][status] = grouped[group].get(status, 0) + 1
    return [
        f"- `{group}`: "
        + ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        for group, counts in sorted(grouped.items())
    ]


def format_running(status: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for job_id, record in sorted(status.items()):
        if record["status"] == "running":
            lines.append(
                f"- `{job_id}` on worker `{record.get('worker')}` gpu `{record.get('gpu')}` attempt `{record.get('attempt')}`"
            )
    return lines or ["- none"]


def format_failed(status: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for job_id, record in sorted(status.items()):
        if record["status"] == "failed":
            lines.append(f"- `{job_id}`: `{record.get('reason')}`")
    return lines or ["- none"]


def safe_job_component(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace(":", "_")
        .replace(".", "p")
        .replace("-", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "_")
    )


def longbench_args(data_dir: str) -> list[str]:
    max_new_tokens_map = {
        "qasper": "148",
        "multifieldqa_en": "84",
        "hotpotqa": "52",
        "2wikimqa": "52",
        "musique": "52",
        "triviaqa": "52",
    }
    return ["--dataset", "longbench", "--data_dir", data_dir, "--max_new_tokens", max_new_tokens_map[data_dir]]


def dataset_specs() -> list[tuple[str, list[str]]]:
    return [
        ("longbench:qasper", longbench_args("qasper")),
        ("longbench:multifieldqa_en", longbench_args("multifieldqa_en")),
        ("longbench:hotpotqa", longbench_args("hotpotqa")),
        ("longbench:2wikimqa", longbench_args("2wikimqa")),
        ("longbench:musique", longbench_args("musique")),
        ("longbench:triviaqa", longbench_args("triviaqa")),
        (
            "needle_in_haystack:16384",
            [
                "--dataset",
                "needle_in_haystack",
                "--max_context_length",
                "16384",
                "--needle_depth",
                "[0,25,50,75,100]",
                "--max_new_tokens",
                "50",
            ],
        ),
        (
            "pg19:test",
            [
                "--dataset",
                "pg19",
                "--pg19_source_dataset",
                "/Tan/dataset/pg19-test",
                "--max_context_length",
                "4096",
                "--pg19_target_tokens",
                "256",
            ],
        ),
    ]


def build_jobs(mode: str) -> list[Job]:
    fraction = 0.01 if mode == "smoke" else 1.0
    ratios = SMOKE_COMPRESSION_RATIOS if mode == "smoke" else FULL_COMPRESSION_RATIOS
    jobs: list[Job] = []
    for model_key, model_path in MODELS:
        for dataset_key, dataset_args in dataset_specs():
            for method_key, method_args in METHODS:
                for ratio in ratios:
                    ratio_tag = f"r{ratio:.1f}".replace(".", "p")
                    job_id = "__".join(
                        [
                            safe_job_component(model_key),
                            safe_job_component(dataset_key),
                            safe_job_component(method_key),
                            ratio_tag,
                        ]
                    )
                    output_dir = RAW_DIR / job_id
                    cli_args = [
                        "--config_file",
                        "/dev/null",
                        "--model",
                        model_path,
                        "--device",
                        "cuda:0",
                        "--compression_ratio",
                        f"{ratio:.1f}",
                        "--fraction",
                        f"{fraction:.2f}",
                        "--query_aware",
                        "true",
                        "--output_dir",
                        str(output_dir),
                        "--result_file_prefix",
                        "ATC26",
                        *dataset_args,
                        *method_args,
                    ]
                    jobs.append(
                        Job(
                            job_id=job_id,
                            model_key=model_key,
                            model_path=model_path,
                            dataset_key=dataset_key,
                            method_key=method_key,
                            compression_ratio=ratio,
                            fraction=fraction,
                            cli_args=cli_args,
                            output_dir=str(output_dir),
                        )
                    )
    return jobs


def parse_gpu_list() -> list[int]:
    raw = os.environ.get("ATC26_GPUS", "0,2")
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_min_free_map() -> dict[int, int]:
    default = {0: 36000, 2: 40000}
    raw = os.environ.get("ATC26_MIN_FREE_MB")
    if not raw:
        return default
    result = default.copy()
    for item in raw.split(","):
        if not item.strip():
            continue
        gpu, value = item.split(":", 1)
        result[int(gpu)] = int(value)
    return result


def discover_workers() -> list[WorkerSpec]:
    gpus = parse_gpu_list()
    min_free = parse_min_free_map()
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    gpu_info = {}
    for line in output:
        idx, uuid, name = line.split(",", 2)
        gpu_info[int(idx.strip())] = (uuid.strip(), name.strip())
    workers = [
        WorkerSpec(
            worker_id=f"worker{pos}",
            physical_gpu=gpu,
            gpu_uuid=gpu_info.get(gpu, ("", "unknown"))[0],
            gpu_name=gpu_info.get(gpu, ("", "unknown"))[1],
            min_free_mb=min_free.get(gpu, 36000),
        )
        for pos, gpu in enumerate(gpus)
    ]
    return workers


def wait_for_gpu(worker: WorkerSpec, poll_seconds: int) -> None:
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        free_mb = int(output[worker.physical_gpu].strip())
        if free_mb >= worker.min_free_mb:
            log(
                f"{worker.worker_id} physical_gpu={worker.physical_gpu} free={free_mb}MB >= {worker.min_free_mb}MB"
            )
            return
        log(
            f"{worker.worker_id} physical_gpu={worker.physical_gpu} free={free_mb}MB < {worker.min_free_mb}MB, wait {poll_seconds}s"
        )
        time.sleep(poll_seconds)


def has_completed_results(job: Job) -> bool:
    output_dir = Path(job.output_dir)
    for metrics_path in output_dir.rglob("ATC26_metrics.json"):
        config_path = metrics_path.with_name("ATC26_config.yaml")
        if not config_path.exists():
            continue
        try:
            import yaml

            cfg = yaml.safe_load(config_path.read_text()) or {}
        except Exception:
            continue
        if (
            str(cfg.get("model")) == job.model_path
            and float(cfg.get("compression_ratio", -1.0)) == job.compression_ratio
            and float(cfg.get("fraction", -1.0)) == job.fraction
        ):
            return True
    return False


def classify_failure(text: str, return_code: int) -> str:
    lowered = text.lower()
    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        return "oom"
    if return_code == -9:
        return "killed"
    if "ssl" in lowered or "connection error" in lowered or "readtimeout" in lowered or "connection reset" in lowered:
        return "network"
    if "cache mismatch" in lowered or "couldn't find cache" in lowered:
        return "cache_mismatch"
    if "failed to load pg19 source dataset" in lowered:
        return "pg19_network"
    return "unknown"


def run_job(job: Job, worker: WorkerSpec, progress: Progress, max_retries: int, poll_seconds: int) -> bool:
    if has_completed_results(job):
        log(f"Skipping completed job={job.job_id}")
        progress.update(
            job,
            status="skipped",
            worker=worker.worker_id,
            gpu=worker.physical_gpu,
            finished_at=now(),
        )
        return True

    reason = "unknown"
    for attempt in range(1, max_retries + 1):
        wait_for_gpu(worker, poll_seconds)
        progress.update(
            job,
            status="running",
            worker=worker.worker_id,
            gpu=worker.physical_gpu,
            attempt=attempt,
            started_at=now(),
            reason=None,
        )
        job_log = LOG_DIR / f"ATC26_job_{job.job_id}__gpu{worker.physical_gpu}__attempt{attempt}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = worker.gpu_uuid or str(worker.physical_gpu)
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        cmd = [str(PYTHON_BIN), "evaluation/evaluate.py", *job.cli_args]
        log(f"Running job={job.job_id} worker={worker.worker_id} gpu={worker.physical_gpu} attempt={attempt}")
        with job_log.open("w") as f:
            f.write(" ".join(cmd) + "\n")
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            tail: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                f.write(line)
                tail.append(line.rstrip("\n"))
                tail = tail[-200:]
            return_code = process.wait()

        if has_completed_results(job):
            progress.update(job, status="success", finished_at=now(), reason=None)
            log(f"Completed job={job.job_id}")
            return True

        reason = classify_failure("\n".join(tail), return_code)
        append_jsonl(
            FAILED_JOBS,
            {
                "time": now(),
                "job_id": job.job_id,
                "worker": worker.worker_id,
                "gpu": worker.physical_gpu,
                "attempt": attempt,
                "return_code": return_code,
                "reason": reason,
                "job_log": str(job_log),
            },
        )
        progress.update(job, status="pending", finished_at=now(), reason=reason)
        log(f"Failed job={job.job_id} attempt={attempt} reason={reason} return_code={return_code}")
        if reason in {"oom", "killed"}:
            time.sleep(poll_seconds)
        elif reason in {"network", "cache_mismatch", "pg19_network"}:
            time.sleep(20)
        else:
            time.sleep(5)

    append_jsonl(
        FAILED_FINAL,
        {
            "time": now(),
            "job_id": job.job_id,
            "worker": worker.worker_id,
            "gpu": worker.physical_gpu,
            "attempts": max_retries,
            "last_reason": reason,
        },
    )
    progress.update(job, status="failed", finished_at=now(), reason=reason)
    return False


def worker_main(
    worker: WorkerSpec,
    job_queue: queue.Queue[Job],
    progress: Progress,
    max_retries: int,
    poll_seconds: int,
) -> None:
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(job, worker, progress, max_retries, poll_seconds)
        finally:
            job_queue.task_done()


def write_manifest(jobs: list[Job]) -> None:
    MANIFEST_JSONL.write_text("")
    for job in jobs:
        append_jsonl(MANIFEST_JSONL, asdict(job))


def validate_environment(workers: list[WorkerSpec]) -> None:
    missing = [path for _, path in MODELS if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model paths: {missing}")
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing python interpreter: {PYTHON_BIN}")
    for worker in workers:
        log(
            f"Configured {worker.worker_id}: physical_gpu={worker.physical_gpu}, uuid={worker.gpu_uuid}, name={worker.gpu_name}, min_free_mb={worker.min_free_mb}"
        )


def run_postprocess() -> None:
    if POSTPROCESS_SCRIPT.exists():
        subprocess.run([str(PYTHON_BIN), str(POSTPROCESS_SCRIPT)], cwd=REPO_ROOT, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ATC26 prefill compression sweep.")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.touch(exist_ok=True)
    PROGRESS_JSONL.touch(exist_ok=True)
    FAILED_JOBS.touch(exist_ok=True)
    FAILED_FINAL.touch(exist_ok=True)

    workers = discover_workers()
    validate_environment(workers)
    jobs = build_jobs(args.mode)
    write_manifest(jobs)
    log(f"Prepared {len(jobs)} jobs for mode={args.mode}, resume={args.resume}, dry_run={args.dry_run}")
    if args.dry_run:
        return 0

    progress = Progress(jobs)
    progress.write_markdown()
    job_queue: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    max_retries = int(os.environ.get("MAX_RETRIES", "3"))
    poll_seconds = int(os.environ.get("POLL_SECONDS", "60"))
    threads = [
        threading.Thread(
            target=worker_main,
            args=(worker, job_queue, progress, max_retries, poll_seconds),
            daemon=False,
        )
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    run_postprocess()
    log(f"ATC26 runner finished mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
