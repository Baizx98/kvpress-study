from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "blockwise_stage2_ratio70_fraction20_multidataset"
OUTPUT_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
RUN_LOG = OUTPUT_DIR / "run.log"
FAILED_JOBS = OUTPUT_DIR / "failed_jobs.jsonl"
FAILED_FINAL = OUTPUT_DIR / "failed_jobs_final.jsonl"
POSTPROCESS_SCRIPT = REPO_ROOT / "evaluation" / "postprocess_blockwise_stage2_ratio70_fraction20_multidataset.py"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
MODEL = os.environ.get("MODEL", "/Tan/model/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda:0")
GPU_INDEX = int(os.environ.get("GPU_INDEX", "0"))
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "40000"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "60"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))


def _compact_component(value: str, max_len: int = 48) -> str:
    import hashlib

    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    keep = max_len - len(digest) - 1
    return f"{value[:keep]}-{digest}"


def _compact_dir_name(components: list[str], max_len: int = 180) -> str:
    import hashlib

    dir_name = "__".join(_compact_component(str(component)) for component in components if component)
    if len(dir_name) <= max_len:
        return dir_name

    digest = hashlib.sha1(dir_name.encode("utf-8")).hexdigest()[:12]
    compact_components = [
        _compact_component(str(component), max_len=20)
        for component in components
        if component
    ]
    fallback = "__".join(compact_components[:8])
    fallback = f"{fallback}__{digest}"
    if len(fallback) <= max_len:
        return fallback
    return digest


def _normalize_task_filter(task_filter: Any) -> list[str]:
    if task_filter is None:
        return []
    if isinstance(task_filter, str):
        return [task.strip() for task in task_filter.split(",") if task.strip()]
    if isinstance(task_filter, (list, tuple)):
        normalized: list[str] = []
        for item in task_filter:
            normalized.extend(_normalize_task_filter(item))
        return normalized
    text = str(task_filter).strip()
    return [text] if text else []


@dataclass
class Job:
    job_id: str
    cli_args: list[str]
    dataset_key: str
    method_key: str


def build_results_base_dir(args_map: dict[str, Any]) -> Path:
    dataset = args_map["dataset"]
    data_dir = args_map.get("data_dir")
    compression_ratio = float(args_map["compression_ratio"])
    components = [
        dataset,
        str(data_dir) if data_dir else "",
        MODEL.replace("/", "--"),
        args_map["press_name"],
        f"{compression_ratio:.2f}",
    ]
    fraction = float(args_map.get("fraction", 1.0))
    if fraction < 1.0:
        components.append(f"fraction{fraction:.3f}")
    if args_map.get("max_context_length") is not None:
        components.append(f"max_context{args_map['max_context_length']}")
    if args_map.get("query_aware"):
        components.append("query_aware")
    if args_map.get("q_window_size") is not None:
        components.append(f"qwindow{args_map['q_window_size']}")
    if args_map.get("summary_topk_keys") is not None:
        components.append(f"topk{args_map['summary_topk_keys']}")
    if args_map.get("mean_key_weight") is not None:
        components.append(f"meankeyw{float(args_map['mean_key_weight']):.2f}")
    if args_map.get("summary_mode") is not None:
        components.append(f"summary{args_map['summary_mode']}")
    if args_map.get("representative_mode") is not None:
        components.append(f"rep{args_map['representative_mode']}")
    if args_map.get("query_agg_mode") is not None:
        components.append(f"qagg{args_map['query_agg_mode']}")
    if args_map.get("head_agg_mode") is not None:
        components.append(f"hagg{args_map['head_agg_mode']}")
    if args_map.get("representative_k") is not None:
        components.append(f"repk{args_map['representative_k']}")
    if args_map.get("multi_rep_k") is not None:
        components.append(f"multirep{args_map['multi_rep_k']}")
    if args_map.get("query_topr") is not None:
        components.append(f"qtopr{args_map['query_topr']}")
    if args_map.get("head_topk") is not None:
        components.append(f"htopk{args_map['head_topk']}")
    if args_map.get("prefill_skip_first_layers") is not None:
        components.append(f"skipfirst{args_map['prefill_skip_first_layers']}")
    task_filter = _normalize_task_filter(args_map.get("task_filter"))
    if task_filter:
        components.append(f"tasks{'-'.join(task_filter)}")
    if args_map.get("needle_depth") is not None and dataset == "needle_in_haystack":
        components.append(f"needle_depth{args_map['needle_depth']}")
    return OUTPUT_DIR / _compact_dir_name(components)


def has_completed_results(base_dir: Path) -> bool:
    return any(base_dir.rglob("metrics.json"))


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
            [
                "nvidia-smi",
                f"--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
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
    if "outofmemoryerror" in lowered or "cuda out of memory" in lowered:
        return "oom"
    if "connection error" in lowered or "readtimeout" in lowered or "http error" in lowered:
        return "network"
    if "dataset cache mismatch" in lowered:
        return "cache_mismatch"
    if return_code == -9:
        return "killed"
    return "unknown"


def run_job(job: Job) -> bool:
    args_map = {}
    it = iter(job.cli_args)
    for key in it:
        if not key.startswith("--"):
            continue
        args_map[key[2:]] = next(it)
    base_dir = build_results_base_dir(args_map)
    if has_completed_results(base_dir):
        log(f"Skipping completed job={job.job_id} base_dir={base_dir}")
        return True

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        wait_for_gpu()
        log(f"Running job={job.job_id} attempt={attempt} dataset={job.dataset_key} method={job.method_key}")
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
        tail_buffer: deque[str] = deque(maxlen=200)
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip("\n")
            tail_buffer.append(stripped)
            with RUN_LOG.open("a") as f:
                f.write(stripped + "\n")
        return_code = process.wait()

        if has_completed_results(base_dir):
            log(f"Completed job={job.job_id} base_dir={base_dir}")
            return True

        tail_text = "\n".join(tail_buffer)
        reason = classify_failure(tail_text, return_code)
        append_jsonl(
            FAILED_JOBS,
            {
                "job_id": job.job_id,
                "dataset": job.dataset_key,
                "method": job.method_key,
                "attempt": attempt,
                "return_code": return_code,
                "reason": reason,
            },
        )
        log(f"Failed job={job.job_id} attempt={attempt} reason={reason} return_code={return_code}")
        if reason in {"oom", "killed"}:
            time.sleep(POLL_SECONDS)
        elif reason in {"network", "cache_mismatch"}:
            time.sleep(15)
        else:
            time.sleep(5)

    append_jsonl(
        FAILED_FINAL,
        {
            "job_id": job.job_id,
            "dataset": job.dataset_key,
            "method": job.method_key,
            "attempts": MAX_RETRIES,
            "last_reason": reason,
        },
    )
    return False


def longbench_args(data_dir: str) -> list[str]:
    max_new_tokens_map = {
        "qasper": "148",
        "multifieldqa_en": "84",
        "hotpotqa": "52",
        "2wikimqa": "52",
        "musique": "52",
        "triviaqa": "52",
    }
    return [
        "--dataset", "longbench",
        "--data_dir", data_dir,
        "--max_new_tokens", max_new_tokens_map[data_dir],
    ]


def base_common_args() -> list[str]:
    return [
        "--model", MODEL,
        "--device", DEVICE,
        "--compression_ratio", "0.7",
        "--fraction", "0.2",
        "--query_aware", "true",
        "--output_dir", str(OUTPUT_DIR),
    ]


def blockwise_common_args(summary_mode: str, representative_mode: str, query_agg_mode: str, head_agg_mode: str) -> list[str]:
    return [
        "--press_name", "block_wise_prefill_per_layer",
        "--block_size", "16",
        "--q_window_size", "64",
        "--summary_topk_keys", "4",
        "--mean_key_weight", "0.75",
        "--representative_k", "4",
        "--multi_rep_k", "4",
        "--query_topr", "16",
        "--head_topk", "1",
        "--summary_mode", summary_mode,
        "--representative_mode", representative_mode,
        "--query_agg_mode", query_agg_mode,
        "--head_agg_mode", head_agg_mode,
    ]


def build_jobs() -> list[Job]:
    jobs: list[Job] = []

    dataset_specs = [
        (
            "ruler:4096",
            [
                "--dataset", "ruler",
                "--data_dir", "4096",
                "--max_new_tokens", "128",
                "--task_filter", "niah_single_3,niah_multikey_3,qa_2",
            ],
        ),
        (
            "needle_in_haystack:16384",
            [
                "--dataset", "needle_in_haystack",
                "--max_context_length", "16384",
                "--needle_depth", "[0,25,50,75,100]",
                "--max_new_tokens", "50",
            ],
        ),
    ]

    for longbench_task in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]:
        dataset_specs.append((f"longbench:{longbench_task}", longbench_args(longbench_task)))

    blockwise_methods = [
        (
            "blockwise_main",
            blockwise_common_args(
                "mean_plus_norm_topk_mean", "key_norm", "max", "uniform_mean"
            ),
        ),
        (
            "blockwise_norm_topk",
            blockwise_common_args(
                "norm_topk_mean_only", "key_norm", "max", "uniform_mean"
            ),
        ),
        (
            "blockwise_multi_rep",
            blockwise_common_args(
                "multi_rep_max", "key_norm", "max", "uniform_mean"
            ),
        ),
    ]

    for dataset_key, dataset_args in dataset_specs:
        for method_key, method_args in blockwise_methods:
            jobs.append(
                Job(
                    job_id=f"{dataset_key}__{method_key}",
                    cli_args=[*base_common_args(), *dataset_args, *method_args],
                    dataset_key=dataset_key,
                    method_key=method_key,
                )
            )

        if dataset_key.startswith("longbench:"):
            jobs.append(
                Job(
                    job_id=f"{dataset_key}__blockwise_tail_query_special",
                    cli_args=[
                        *base_common_args(),
                        *dataset_args,
                        *blockwise_common_args(
                            "mean_plus_norm_topk_mean",
                            "tail_query_relevance",
                            "mean",
                            "uniform_mean",
                        ),
                    ],
                    dataset_key=dataset_key,
                    method_key="blockwise_tail_query_special",
                )
            )

        jobs.append(
            Job(
                job_id=f"{dataset_key}__chunkkv_prefill",
                cli_args=[
                    *base_common_args(),
                    *dataset_args,
                    "--press_name", "chunkkv_prefill_per_layer",
                    "--block_size", "16",
                ],
                dataset_key=dataset_key,
                method_key="chunkkv_prefill",
            )
        )

    return jobs


def run_postprocess() -> None:
    log("Running stage2 postprocess")
    subprocess.run(
        [str(PYTHON_BIN), str(POSTPROCESS_SCRIPT)],
        cwd=REPO_ROOT,
        check=False,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.touch(exist_ok=True)
    FAILED_JOBS.touch(exist_ok=True)
    FAILED_FINAL.touch(exist_ok=True)
    jobs = build_jobs()
    log(f"Prepared {len(jobs)} jobs for {EXPERIMENT_NAME}")
    try:
        for job in jobs:
            run_job(job)
    finally:
        run_postprocess()
    log(f"Stage2 controller finished for {EXPERIMENT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
