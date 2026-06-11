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
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "kvcore_lifecycle_decode_longbench16_1pct")
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
OUTPUT_DIR = RESULT_ROOT / "artifacts"
RUN_LOG = OUTPUT_DIR / "run.log"
PROGRESS_JSONL = OUTPUT_DIR / "progress.jsonl"
MANIFEST_JSONL = OUTPUT_DIR / "manifest.jsonl"
FAILED_JOBS = OUTPUT_DIR / "failed_jobs.jsonl"
FAILED_FINAL = OUTPUT_DIR / "failed_jobs_final.jsonl"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
POSTPROCESS_SCRIPT = REPO_ROOT / "evaluation" / "postprocess_kvcore_lifecycle_decode_longbench16_1pct.py"

MODEL = os.environ.get("MODEL", "/Tan/model/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda:0")
PHYSICAL_GPU_INDEX = int(os.environ.get("GPU_INDEX", "0"))
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", str(PHYSICAL_GPU_INDEX))
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "38000"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "60"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
SMOKE = os.environ.get("SMOKE", "0") == "1"

FRACTION = float(os.environ.get("FRACTION", "0.01"))
SEED = int(os.environ.get("SEED", "42"))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "16"))
TOP_P = float(os.environ.get("DECODE_TOP_P", "0.9"))
COMPRESSION_INTERVAL = int(os.environ.get("COMPRESSION_INTERVAL", str(BLOCK_SIZE)))
PROTECTED_RECENT_BLOCKS = int(os.environ.get("PROTECTED_RECENT_BLOCKS", "2"))
DECODE_SKIP_FIRST_LAYERS = int(os.environ.get("DECODE_SKIP_FIRST_LAYERS", "0"))

LONGBENCH16_MAX_NEW_TOKENS = {
    "narrativeqa": 148,
    "qasper": 148,
    "multifieldqa_en": 84,
    "hotpotqa": 52,
    "2wikimqa": 52,
    "musique": 52,
    "triviaqa": 52,
    "gov_report": 532,
    "qmsum": 532,
    "multi_news": 532,
    "samsum": 148,
    "trec": 84,
    "passage_count": 52,
    "passage_retrieval_en": 52,
    "lcc": 84,
    "repobench-p": 84,
}


@dataclass(frozen=True)
class Job:
    job_id: str
    method_key: str
    task: str
    cli_args: list[str]
    match_fields: dict[str, Any]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with RUN_LOG.open("a") as f:
        f.write(line + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def selected_tasks() -> list[str]:
    only = [task.strip() for task in os.environ.get("ONLY_LONGBENCH_TASKS", "").split(",") if task.strip()]
    if SMOKE and not only:
        only = ["trec"]
    if only:
        unknown = sorted(set(only) - set(LONGBENCH16_MAX_NEW_TOKENS))
        if unknown:
            raise ValueError(f"Unknown LongBench tasks: {unknown}")
        return [task for task in LONGBENCH16_MAX_NEW_TOKENS if task in set(only)]
    return list(LONGBENCH16_MAX_NEW_TOKENS)


def wait_for_l40s() -> None:
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        fields = [part.strip() for part in output[PHYSICAL_GPU_INDEX].split(",")]
        name = fields[1]
        free_mb = int(fields[2])
        if "L40S" not in name:
            raise RuntimeError(f"GPU {PHYSICAL_GPU_INDEX} is {name}, expected an NVIDIA L40S")
        if free_mb >= MIN_FREE_MB:
            log(f"L40S physical GPU {PHYSICAL_GPU_INDEX} free memory {free_mb}MB >= {MIN_FREE_MB}MB, continue.")
            return
        log(
            f"L40S physical GPU {PHYSICAL_GPU_INDEX} free memory {free_mb}MB < {MIN_FREE_MB}MB, "
            f"waiting {POLL_SECONDS}s."
        )
        time.sleep(POLL_SECONDS)


def classify_failure(text: str, return_code: int) -> str:
    lowered = text.lower()
    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        return "oom"
    if return_code == -9:
        return "killed"
    if "proxyerror" in lowered or "connection error" in lowered or "readtimeout" in lowered:
        return "network"
    if "expected an nvidia l40s" in lowered:
        return "wrong_gpu"
    return "unknown"


def config_matches(cfg: dict[str, Any], fields: dict[str, Any]) -> bool:
    for key, expected in fields.items():
        actual = cfg.get(key)
        if isinstance(expected, float):
            try:
                if abs(float(actual) - expected) > 1e-9:
                    return False
            except Exception:
                return False
        else:
            if actual != expected:
                return False
    return True


def has_completed_results(match_fields: dict[str, Any]) -> bool:
    for config_path in OUTPUT_DIR.rglob("config.yaml"):
        metrics_path = config_path.with_name("metrics.json")
        predictions_path = config_path.with_name("predictions.csv")
        if not metrics_path.exists() or not predictions_path.exists():
            continue
        try:
            cfg = yaml.safe_load(config_path.read_text()) or {}
        except Exception:
            continue
        if config_matches(cfg, match_fields):
            return True
    return False


def common_args(task: str) -> list[str]:
    return [
        "--model",
        MODEL,
        "--device",
        DEVICE,
        "--dataset",
        "longbench",
        "--data_dir",
        task,
        "--fraction",
        str(FRACTION),
        "--seed",
        str(SEED),
        "--max_new_tokens",
        str(LONGBENCH16_MAX_NEW_TOKENS[task]),
        "--output_dir",
        str(OUTPUT_DIR),
    ]


def blockwise_decode_args() -> list[str]:
    return [
        "--press_name",
        "dual_phase_per_layer",
        "--dual_phase_mode",
        "compute_cold_fixed_budget",
        "--compression_ratio",
        "0.0",
        "--block_size",
        str(BLOCK_SIZE),
        "--q_window_size",
        str(BLOCK_SIZE),
        "--compression_interval",
        str(COMPRESSION_INTERVAL),
        "--decode_top_p_threshold",
        str(TOP_P),
        "--decode_skip_first_layers",
        str(DECODE_SKIP_FIRST_LAYERS),
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
        "--protected_recent_blocks",
        str(PROTECTED_RECENT_BLOCKS),
    ]


def make_jobs() -> list[Job]:
    jobs: list[Job] = []
    for task in selected_tasks():
        full_fields = {
            "dataset": "longbench",
            "data_dir": task,
            "model": MODEL,
            "press_name": "no_press",
            "compression_ratio": 0.0,
            "fraction": FRACTION,
            "seed": SEED,
            "max_new_tokens": LONGBENCH16_MAX_NEW_TOKENS[task],
        }
        jobs.append(
            Job(
                job_id=f"longbench:{task}__full_kv",
                method_key="full_kv",
                task=task,
                cli_args=common_args(task) + ["--press_name", "no_press", "--compression_ratio", "0.0"],
                match_fields=full_fields,
            )
        )

        blockwise_fields = {
            "dataset": "longbench",
            "data_dir": task,
            "model": MODEL,
            "press_name": "dual_phase_per_layer",
            "dual_phase_mode": "compute_cold_fixed_budget",
            "compression_ratio": 0.0,
            "fraction": FRACTION,
            "seed": SEED,
            "max_new_tokens": LONGBENCH16_MAX_NEW_TOKENS[task],
            "block_size": BLOCK_SIZE,
            "q_window_size": BLOCK_SIZE,
            "compression_interval": COMPRESSION_INTERVAL,
            "decode_top_p_threshold": TOP_P,
            "decode_skip_first_layers": DECODE_SKIP_FIRST_LAYERS,
        }
        jobs.append(
            Job(
                job_id=f"longbench:{task}__decode_qaware_blockwise_top_p{TOP_P:g}",
                method_key="decode_qaware_blockwise_top_p",
                task=task,
                cli_args=common_args(task) + blockwise_decode_args(),
                match_fields=blockwise_fields,
            )
        )
    return jobs


def write_readme() -> None:
    readme = RESULT_ROOT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# kvcore_lifecycle_decode_longbench16_1pct",
                "",
                "## 实验目的",
                "",
                "评估 KVCore lifecycle 语义下，decode query-aware block-wise active set 对 LongBench 准确率的影响。",
                "",
                "## 运行脚本",
                "",
                "- `evaluation/run_kvcore_lifecycle_decode_longbench16_1pct.py`",
                "- `evaluation/postprocess_kvcore_lifecycle_decode_longbench16_1pct.py`",
                "",
                "## 数据集",
                "",
                "- LongBench 16 个英文子数据集",
                f"- 采样比例：`{FRACTION}`",
                f"- 随机种子：`{SEED}`",
                "",
                "## 方法",
                "",
                "- `full_kv`: `press_name=no_press`",
                f"- `decode_qaware_blockwise_top_p`: `dual_phase_per_layer`, `compute_cold_fixed_budget`, decode score top-p `p={TOP_P}`, skip first `{DECODE_SKIP_FIRST_LAYERS}` layers",
                "",
                "## 模型与设备",
                "",
                f"- 模型：`{MODEL}`",
                f"- 目标设备：physical GPU `{PHYSICAL_GPU_INDEX}` / NVIDIA L40S, process-visible `{DEVICE}`",
                "",
                "## 产物",
                "",
                "- 原始产物：`artifacts/`",
                "- 汇总：`summary.csv`、`summary.json`",
                "- 分析文档：`note/kvcore_lifecycle_decode_longbench16_1pct_results_zh.md`",
                "",
            ]
        )
    )


def run_job(job: Job) -> bool:
    if has_completed_results(job.match_fields):
        log(f"Skipping completed job={job.job_id}")
        append_jsonl(PROGRESS_JSONL, {"time": now(), "job_id": job.job_id, "status": "skipped"})
        return True

    reason = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        wait_for_l40s()
        log(f"Running job={job.job_id} attempt={attempt}")
        append_jsonl(
            PROGRESS_JSONL,
            {"time": now(), "job_id": job.job_id, "status": "running", "attempt": attempt},
        )
        cmd = [str(PYTHON_BIN), "evaluation/evaluate.py", *job.cli_args]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
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
            tail_buffer = tail_buffer[-200:]
            with RUN_LOG.open("a") as f:
                f.write(stripped + "\n")
        return_code = process.wait()

        if has_completed_results(job.match_fields):
            log(f"Completed job={job.job_id}")
            append_jsonl(PROGRESS_JSONL, {"time": now(), "job_id": job.job_id, "status": "success"})
            return True

        reason = classify_failure("\n".join(tail_buffer), return_code)
        append_jsonl(
            FAILED_JOBS,
            {
                "time": now(),
                "job_id": job.job_id,
                "attempt": attempt,
                "return_code": return_code,
                "reason": reason,
            },
        )
        append_jsonl(
            PROGRESS_JSONL,
            {"time": now(), "job_id": job.job_id, "status": "failed_attempt", "attempt": attempt, "reason": reason},
        )
        log(f"Failed job={job.job_id} attempt={attempt} reason={reason} return_code={return_code}")
        time.sleep(20 if reason == "network" else POLL_SECONDS)

    append_jsonl(FAILED_FINAL, {"time": now(), "job_id": job.job_id, "attempts": MAX_RETRIES, "last_reason": reason})
    return False


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_readme()
    jobs = make_jobs()
    MANIFEST_JSONL.write_text("")
    for job in jobs:
        append_jsonl(MANIFEST_JSONL, {"job_id": job.job_id, "method": job.method_key, "task": job.task, "args": job.cli_args})

    log(
        f"Starting experiment={EXPERIMENT_NAME}; jobs={len(jobs)}; model={MODEL}; "
        f"physical_gpu={PHYSICAL_GPU_INDEX}; cuda_visible_devices={CUDA_VISIBLE_DEVICES}; "
        f"top_p={TOP_P}; decode_skip_first_layers={DECODE_SKIP_FIRST_LAYERS}; fraction={FRACTION}; seed={SEED}"
    )
    completed = 0
    for job in jobs:
        completed += int(run_job(job))

    log(f"Finished experiment={EXPERIMENT_NAME}; completed_jobs={completed}/{len(jobs)}")
    if completed == len(jobs):
        subprocess.run([str(PYTHON_BIN), str(POSTPROCESS_SCRIPT)], cwd=REPO_ROOT, check=False)
    return 0 if completed == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
