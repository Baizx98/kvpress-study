from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

GPU_INDEX = os.environ.get("ATC26_GPU_INDEX", "1")
GPU_UUID = os.environ.get("ATC26_GPU_UUID", "GPU-4eac01c5-47d1-3958-95bb-98d357b8b9c3")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ.setdefault("HF_HOME", "/Tan/dataset/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/Tan/dataset/hf_home/datasets")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Tan/dataset/hf_home/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from evaluate import EvaluationConfig, EvaluationRunner  # noqa: E402


EXPERIMENT_NAME = "ATC26_needle_8k_token_length_depth_heatmap"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
LOGS_DIR = ARTIFACTS_DIR / "logs"
PREDICTIONS_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_predictions.csv"
STATUS_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_status.json"
PROGRESS_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_progress.md"
MANIFEST_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_manifest.jsonl"

MODEL_PATH = "/Tan/model/Llama-3.1-8B-Instruct"
DATASET_ID = "alessiodevoto/paul_graham_essays"
COMPRESSION_RATIO = 0.5
METHODS_FULL = ["block_wise", "snapkv", "chunkkv"]
METHODS_SMOKE = ["block_wise", "chunkkv"]
TOKEN_LENGTHS_FULL = list(range(256, 8192 + 1, 256))
TOKEN_LENGTHS_SMOKE = [256, 2048, 8192]
DEPTHS_FULL = list(range(0, 101, 10))
DEPTHS_SMOKE = [0, 50, 100]
SEEDS_FULL = [42, 43, 44]
SEEDS_SMOKE = [42]
CONTEXT_WRAPPER = "This is a very long story book: <book> {context} </book>."


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_status() -> dict[str, dict]:
    if not STATUS_PATH.exists():
        return {}
    return json.loads(STATUS_PATH.read_text())


def _save_status(status: dict[str, dict]) -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def _job_id(mode: str, method: str, token_length: int, depth: int, seed: int) -> str:
    return f"{mode}__{method}__tok{token_length}__depth{depth}__seed{seed}"


def _grid(mode: str) -> list[dict]:
    if mode == "smoke":
        methods = METHODS_SMOKE
        token_lengths = TOKEN_LENGTHS_SMOKE
        depths = DEPTHS_SMOKE
        seeds = SEEDS_SMOKE
    else:
        methods = METHODS_FULL
        token_lengths = TOKEN_LENGTHS_FULL
        depths = DEPTHS_FULL
        seeds = SEEDS_FULL

    jobs = []
    for method in methods:
        for token_length in token_lengths:
            for depth in depths:
                for seed in seeds:
                    jobs.append(
                        {
                            "job_id": _job_id(mode, method, token_length, depth, seed),
                            "mode": mode,
                            "method": method,
                            "token_length": token_length,
                            "needle_depth": depth,
                            "seed": seed,
                        }
                    )
    return jobs


def _write_manifest(jobs: list[dict]) -> None:
    with MANIFEST_PATH.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True) + "\n")


def _write_progress(jobs: list[dict], status: dict[str, dict]) -> None:
    counts: dict[str, int] = {}
    for job in jobs:
        state = status.get(job["job_id"], {}).get("state", "pending")
        counts[state] = counts.get(state, 0) + 1

    recent = sorted(
        (
            (item.get("finished_at") or item.get("started_at") or "", job_id, item)
            for job_id, item in status.items()
        ),
        reverse=True,
    )[:20]

    lines = [
        "# ATC26 Needle 8K Heatmap Progress",
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

    lines.extend(["", "## Recent Jobs", "", "| job | state | detail |", "|---|---|---|"])
    for _, job_id, item in recent:
        lines.append(f"| `{job_id}` | `{item.get('state')}` | `{item.get('detail', '')}` |")
    PROGRESS_PATH.write_text("\n".join(lines) + "\n")


def _append_prediction(row: dict) -> None:
    exists = PREDICTIONS_PATH.exists()
    with PREDICTIONS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_id",
                "mode",
                "method",
                "compression_ratio",
                "token_length",
                "needle_depth",
                "seed",
                "window_start_token",
                "window_token_budget",
                "needle",
                "question",
                "answer_prefix",
                "predicted_answer",
                "started_at",
                "finished_at",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _make_context(
    tokenizer,
    source_ids: list[int],
    needle: str,
    token_length: int,
    depth: int,
    seed: int,
) -> tuple[str, int, int]:
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    wrapper_overhead = len(tokenizer.encode(CONTEXT_WRAPPER.format(context=""), add_special_tokens=False))
    # Leave a small margin for tokenizer/template variation. The pipeline still
    # receives max_context_length=token_length and will enforce the hard cap.
    window_budget = max(32, token_length - len(needle_ids) - wrapper_overhead - 16)
    window_budget = min(window_budget, len(source_ids))
    max_start = max(0, len(source_ids) - window_budget)
    rng = random.Random(seed * 1_000_003 + token_length * 1009)
    window_start = rng.randint(0, max_start) if max_start > 0 else 0
    window_ids = source_ids[window_start : window_start + window_budget]
    insert_idx = int(len(window_ids) * depth / 100)
    context_ids = window_ids[:insert_idx] + needle_ids + window_ids[insert_idx:]
    return CONTEXT_WRAPPER.format(context=_decode(tokenizer, context_ids)), window_start, window_budget


def _build_config(method: str) -> EvaluationConfig:
    kwargs = {
        "dataset": "needle_in_haystack",
        "model": MODEL_PATH,
        "device": "cuda:0",
        "press_name": method,
        "compression_ratio": COMPRESSION_RATIO,
        "max_context_length": 8192,
        "needle_depth": 0,
        "query_aware": True,
        "output_dir": str(ARTIFACTS_DIR),
        "log_level": "INFO",
        "seed": 42,
    }
    if method == "block_wise":
        kwargs.update(
            {
                "block_size": 16,
                "q_window_size": 64,
                "summary_topk_keys": 4,
                "mean_key_weight": 0.75,
                "summary_mode": "mean_plus_norm_topk_mean",
                "representative_mode": "key_norm",
                "query_agg_mode": "max",
                "head_agg_mode": "uniform_mean",
                "representative_k": 4,
                "multi_rep_k": 4,
                "query_topr": 16,
                "head_topk": 1,
            }
        )
    return EvaluationConfig(**kwargs)


def _run_method(
    method: str,
    method_jobs: list[dict],
    all_jobs: list[dict],
    dataset_row: dict,
    source_ids: list[int],
    status: dict[str, dict],
) -> None:
    config = _build_config(method)
    runner = EvaluationRunner(config)
    runner._setup_press()
    runner._setup_model_pipeline()
    pipeline = runner.pipeline
    press = runner.press
    tokenizer = pipeline.tokenizer

    for job in method_jobs:
        job_id = job["job_id"]
        if status.get(job_id, {}).get("state") == "success":
            continue

        started_at = _now()
        status[job_id] = {**job, "state": "running", "started_at": started_at}
        _save_status(status)
        _write_progress(all_jobs, status)
        try:
            context, window_start, window_budget = _make_context(
                tokenizer=tokenizer,
                source_ids=source_ids,
                needle=dataset_row["needle"],
                token_length=job["token_length"],
                depth=job["needle_depth"],
                seed=job["seed"],
            )
            output = pipeline(
                context,
                question=dataset_row["question"],
                answer_prefix=dataset_row["answer_prefix"],
                press=press,
                max_new_tokens=int(dataset_row["max_new_tokens"]),
                max_context_length=job["token_length"],
            )
            predicted_answer = output["answer"]
            finished_at = _now()
            _append_prediction(
                {
                    **job,
                    "compression_ratio": COMPRESSION_RATIO,
                    "window_start_token": window_start,
                    "window_token_budget": window_budget,
                    "needle": dataset_row["needle"],
                    "question": dataset_row["question"],
                    "answer_prefix": dataset_row["answer_prefix"],
                    "predicted_answer": predicted_answer,
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
            )
            status[job_id] = {**job, "state": "success", "started_at": started_at, "finished_at": finished_at, "detail": "ok"}
        except Exception as exc:
            finished_at = _now()
            status[job_id] = {
                **job,
                "state": "failed",
                "started_at": started_at,
                "finished_at": finished_at,
                "detail": f"{type(exc).__name__}: {exc}",
            }
            _save_status(status)
            _write_progress(all_jobs, status)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _save_status(status)
        _write_progress(all_jobs, status)

    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in [ARTIFACTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    jobs = _grid(args.mode)
    _write_manifest(jobs)
    status = _load_status() if args.resume else {}
    _write_progress(jobs, status)

    dataset_row = load_dataset(DATASET_ID, split="test")[0]
    bootstrap_tokenizer = None
    # The tokenizer from the first method pipeline is reused for context ids, but
    # we need source_ids before grouping by method. Load a tokenizer through the
    # first runner and release it immediately after tokenizing.
    from transformers import AutoTokenizer

    bootstrap_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    source_ids = bootstrap_tokenizer.encode(dataset_row["context"], add_special_tokens=False)
    del bootstrap_tokenizer

    for method in sorted({job["method"] for job in jobs}, key=(METHODS_FULL + METHODS_SMOKE).index):
        method_jobs = [job for job in jobs if job["method"] == method]
        _run_method(method, method_jobs, jobs, dataset_row, source_ids, status)

    _write_progress(jobs, status)


if __name__ == "__main__":
    main()
