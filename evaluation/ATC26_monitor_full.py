from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19"
ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
MONITOR_LOG = ARTIFACTS_DIR / "ATC26_monitor.log"
FULL_LOG = ARTIFACTS_DIR / "ATC26_nohup_full.log"
PROGRESS_MD = ARTIFACTS_DIR / "ATC26_progress.md"
FAILED_FINAL = ARTIFACTS_DIR / "ATC26_failed_jobs_final.jsonl"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "evaluation" / "ATC26_run_prefill_sweep.py"

RUNNER_PATTERN = "evaluation/ATC26_run_prefill_sweep.py --mode full --resume"
EVALUATE_PATTERN = "evaluation/evaluate.py --config_file /dev/null"
GPU_ENV = {
    "ATC26_GPUS": "0,2",
    "ATC26_MIN_FREE_MB": "0:36000,2:24000",
    "MAX_RETRIES": "3",
    "POLL_SECONDS": "60",
}
CHECK_INTERVAL_SECONDS = int(os.environ.get("ATC26_MONITOR_INTERVAL", "300"))
STALL_SECONDS = int(os.environ.get("ATC26_MONITOR_STALL_SECONDS", "10800"))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with MONITOR_LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def pgrep(pattern: str) -> list[int]:
    proc = sh(["pgrep", "-f", pattern])
    if proc.returncode != 0:
        return []
    return [int(line) for line in proc.stdout.splitlines() if line.strip().isdigit()]


def gpu_snapshot() -> str:
    proc = sh(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    return proc.stdout.strip()


def compute_snapshot() -> str:
    proc = sh(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    return proc.stdout.strip()


def count_metrics() -> int:
    raw_dir = ARTIFACTS_DIR / "raw"
    if not raw_dir.exists():
        return 0
    return sum(1 for _ in raw_dir.rglob("ATC26_metrics.json"))


def failed_final_count() -> int:
    if not FAILED_FINAL.exists():
        return 0
    return sum(1 for line in FAILED_FINAL.read_text().splitlines() if line.strip())


def latest_activity_mtime() -> float:
    candidates = [PROGRESS_MD, FULL_LOG]
    logs_dir = ARTIFACTS_DIR / "logs"
    if logs_dir.exists():
        candidates.extend(logs_dir.glob("ATC26_job_*.log"))
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    return max(mtimes) if mtimes else 0.0


def start_runner() -> int:
    env = os.environ.copy()
    env.update(GPU_ENV)
    FULL_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = FULL_LOG.open("a")
    process = subprocess.Popen(
        [str(PYTHON_BIN), str(RUNNER), "--mode", "full", "--resume"],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    log(f"restarted runner pid={process.pid}")
    return process.pid


def stop_processes() -> None:
    pids = set(pgrep(RUNNER_PATTERN) + pgrep(EVALUATE_PATTERN))
    current = os.getpid()
    pids.discard(current)
    for pid in sorted(pids):
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    if pids:
        time.sleep(10)
    for pid in sorted(pids):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def write_status() -> None:
    status = {
        "time": now(),
        "runner_pids": pgrep(RUNNER_PATTERN),
        "evaluate_pids": pgrep(EVALUATE_PATTERN),
        "metrics_count": count_metrics(),
        "failed_final_count": failed_final_count(),
        "gpu": gpu_snapshot().splitlines(),
        "compute": compute_snapshot().splitlines(),
    }
    status_path = ARTIFACTS_DIR / "ATC26_monitor_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    log(
        "status "
        f"runner={status['runner_pids']} eval={status['evaluate_pids']} "
        f"metrics={status['metrics_count']} failed_final={status['failed_final_count']}"
    )


def monitor_once() -> None:
    runner_pids = pgrep(RUNNER_PATTERN)
    if not runner_pids:
        log("runner missing; restarting")
        start_runner()
        return

    age = time.time() - latest_activity_mtime()
    eval_pids = pgrep(EVALUATE_PATTERN)
    if age > STALL_SECONDS and not eval_pids:
        log(f"no activity for {int(age)}s and no eval children; restarting runner")
        stop_processes()
        start_runner()
        return

    write_status()


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log(
        f"monitor started interval={CHECK_INTERVAL_SECONDS}s stall={STALL_SECONDS}s "
        f"gpu_env={GPU_ENV}"
    )
    while True:
        try:
            monitor_once()
        except Exception as exc:  # keep monitor alive
            log(f"monitor error {type(exc).__name__}: {exc}")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
