from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/Tan/dataset/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/Tan/dataset/hf_home/datasets")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Tan/dataset/hf_home/hub")

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from benchmarks.pg19.create_huggingface_dataset import load_pg19_source_dataframe  # noqa: E402


EXPERIMENT_NAME = "ATC26_token_level_temporal_similarity"
RESULT_ROOT = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACT_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACT_DIR / "raw"
LOG_DIR = ARTIFACT_DIR / "logs"
AGG_JSON = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_aggregate.json"
AGG_CSV = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_aggregate.csv"
RAW_JSONL = RAW_DIR / "ATC26_token_level_temporal_similarity_raw.jsonl"
MANIFEST_JSON = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_manifest.json"
CONFIG_JSON = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_config.json"
HEARTBEAT_JSON = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_heartbeat.json"

MODELS = {
    "llama31_8b_instruct": "/Tan/model/Llama-3.1-8B-Instruct",
    "mistral_7b_instruct_v03": "/Tan/model/Mistral-7B-Instruct-v0.3",
    "qwen3_8b": "/Tan/model/Qwen3-8B",
}


@dataclass(frozen=True)
class Job:
    model_key: str
    context_length: int
    sample_index: int
    book_id: str

    @property
    def job_id(self) -> str:
        return f"{self.model_key}__ctx{self.context_length}__sample{self.sample_index:03d}__book{self.book_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect token-level temporal top-k KV similarity traces.")
    parser.add_argument("--model-key", choices=sorted(MODELS), default="llama31_8b_instruct")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", choices=["pg19"], default="pg19")
    parser.add_argument("--pg19-source-dataset", default=os.environ.get("PG19_SOURCE_DATASET", "/Tan/dataset/pg19-test"))
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--samples-per-length", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=1024)
    parser.add_argument("--compression-ratios", type=float, nargs="+", default=[0.7, 0.5, 0.3])
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--reuse-intervals", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--head-agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-tag", default="full")
    parser.add_argument("--max-books-to-tokenize", type=int, default=None)
    return parser.parse_args()


def configure_output_paths(run_tag: str) -> None:
    global ARTIFACT_DIR, RAW_DIR, LOG_DIR, AGG_JSON, AGG_CSV, RAW_JSONL, MANIFEST_JSON, CONFIG_JSON, HEARTBEAT_JSON

    ARTIFACT_DIR = RESULT_ROOT / "artifacts" if run_tag == "full" else RESULT_ROOT / "artifacts" / run_tag
    RAW_DIR = ARTIFACT_DIR / "raw"
    LOG_DIR = ARTIFACT_DIR / "logs"
    AGG_JSON = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_aggregate.json"
    AGG_CSV = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_aggregate.csv"
    RAW_JSONL = RAW_DIR / "ATC26_token_level_temporal_similarity_raw.jsonl"
    MANIFEST_JSON = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_manifest.json"
    CONFIG_JSON = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_config.json"
    HEARTBEAT_JSON = ARTIFACT_DIR / "ATC26_token_level_temporal_similarity_heartbeat.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_heartbeat(payload: dict[str, Any]) -> None:
    write_json(HEARTBEAT_JSON, {"updated_at": utc_now(), **payload})


def completed_job_ids() -> set[str]:
    if not RAW_JSONL.exists():
        return set()
    done: set[str] = set()
    with RAW_JSONL.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["job_id"])
            except Exception:
                continue
    return done


def load_model_and_tokenizer(model_path: str, device: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if dtype == "auto":
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def set_attention_implementation(model, implementation: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(implementation)
        return
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = implementation


def tokenize_pg19_books(args: argparse.Namespace, tokenizer) -> list[dict[str, Any]]:
    min_tokens = max(args.context_lengths) + args.decode_steps + 1
    source_df = load_pg19_source_dataframe(dataset_id=args.pg19_source_dataset, split="test").reset_index(drop=True)
    if args.max_books_to_tokenize is not None:
        source_df = source_df.iloc[: args.max_books_to_tokenize]

    books: list[dict[str, Any]] = []
    for row_idx, row in tqdm(source_df.iterrows(), total=len(source_df), desc="Tokenizing PG19 books"):
        ids = tokenizer(
            str(row["text"]),
            add_special_tokens=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        if len(ids) <= min_tokens:
            continue
        books.append(
            {
                "book_id": str(row_idx),
                "short_book_title": str(row.get("short_book_title", row_idx)),
                "source_token_count": len(ids),
                "input_ids": ids,
            }
        )
    if not books:
        raise RuntimeError(f"No PG19 books have enough tokens for max context/decode length: {min_tokens}")
    return books


def build_jobs(args: argparse.Namespace, books: list[dict[str, Any]]) -> list[Job]:
    jobs: list[Job] = []
    for context_length in args.context_lengths:
        eligible = [book for book in books if len(book["input_ids"]) > context_length + args.decode_steps]
        if len(eligible) < args.samples_per_length:
            raise RuntimeError(
                f"Only {len(eligible)} books are long enough for context_length={context_length}; "
                f"need {args.samples_per_length}."
            )
        for sample_index, book in enumerate(eligible[: args.samples_per_length]):
            jobs.append(
                Job(
                    model_key=args.model_key,
                    context_length=context_length,
                    sample_index=sample_index,
                    book_id=str(book["book_id"]),
                )
            )
    return jobs


def mean_or_nan(values: list[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return float(np.mean(finite)) if finite else float("nan")


def build_membership(topk_rows: list[np.ndarray], lengths: np.ndarray, hist_len_max: int) -> np.ndarray:
    membership = np.zeros((len(topk_rows), hist_len_max), dtype=bool)
    for step, indices in enumerate(topk_rows):
        if indices.size == 0:
            continue
        membership[step, indices] = True
    return membership


def lag_metrics_from_membership(membership: np.ndarray, lengths: np.ndarray, lag: int) -> dict[str, float]:
    if lag >= membership.shape[0]:
        return {"jaccard": float("nan"), "overlap": float("nan")}
    left = membership[:-lag]
    right = membership[lag:]
    inter = np.logical_and(left, right).sum(axis=1).astype(np.float32)
    len_left = lengths[:-lag].astype(np.float32)
    len_right = lengths[lag:].astype(np.float32)
    union = len_left + len_right - inter
    jaccard_vals = np.divide(inter, union, out=np.full_like(inter, np.nan), where=union > 0)
    overlap_vals = np.divide(inter, len_right, out=np.full_like(inter, np.nan), where=len_right > 0)
    return {
        "jaccard": float(np.nanmean(jaccard_vals)),
        "overlap": float(np.nanmean(overlap_vals)),
    }


def reuse_metrics_from_membership(membership: np.ndarray, lengths: np.ndarray, interval: int) -> dict[str, float]:
    if interval > membership.shape[0]:
        return {"reuse_jaccard": float("nan"), "reuse_recall": float("nan")}
    anchors = (np.arange(membership.shape[0]) // interval) * interval
    anchor_membership = membership[anchors]
    inter = np.logical_and(anchor_membership, membership).sum(axis=1).astype(np.float32)
    len_anchor = lengths[anchors].astype(np.float32)
    len_current = lengths.astype(np.float32)
    union = len_anchor + len_current - inter
    jaccard_vals = np.divide(inter, union, out=np.full_like(inter, np.nan), where=union > 0)
    recall_vals = np.divide(inter, len_current, out=np.full_like(inter, np.nan), where=len_current > 0)
    return {
        "reuse_jaccard": float(np.nanmean(jaccard_vals)),
        "reuse_recall": float(np.nanmean(recall_vals)),
    }


def attention_token_scores(attention: torch.Tensor, hist_len: int, head_agg: str) -> torch.Tensor:
    scores_by_head = attention[0, :, -1, :hist_len].detach().float()
    if head_agg == "max":
        return scores_by_head.max(dim=0).values
    return scores_by_head.mean(dim=0)


@torch.inference_mode()
def run_job(
    job: Job,
    book: dict[str, Any],
    model,
    args: argparse.Namespace,
    job_index: int,
    total_jobs: int,
) -> dict[str, Any]:
    input_ids = book["input_ids"]
    context_ids = torch.tensor([input_ids[: job.context_length]], dtype=torch.long, device=args.device)
    continuation = input_ids[job.context_length : job.context_length + args.decode_steps]

    write_heartbeat(
        {
            "status": "prefill",
            "job_id": job.job_id,
            "job_index": job_index,
            "total_jobs": total_jobs,
            "context_length": job.context_length,
        }
    )
    set_attention_implementation(model, "sdpa")
    prefill = model(context_ids, use_cache=True, output_attentions=False)
    past_key_values = prefill.past_key_values
    set_attention_implementation(model, "eager")

    num_layers = len(model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else prefill.attentions or [])
    if num_layers == 0:
        num_layers = int(getattr(model.config, "num_hidden_layers"))

    topk_rows: dict[tuple[float, int], list[np.ndarray]] = {
        (ratio, layer): [] for ratio in args.compression_ratios for layer in range(num_layers)
    }
    lengths: dict[tuple[float, int], list[int]] = {
        (ratio, layer): [] for ratio in args.compression_ratios for layer in range(num_layers)
    }
    hist_len_max = job.context_length + args.decode_steps

    for step, token_id in enumerate(tqdm(continuation, desc=job.job_id)):
        token = torch.tensor([[token_id]], dtype=torch.long, device=args.device)
        outputs = model(
            token,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
        )
        past_key_values = outputs.past_key_values
        attentions = outputs.attentions
        hist_len = int(attentions[0].shape[-1]) - 1
        for layer, attention in enumerate(attentions):
            token_scores = attention_token_scores(attention, hist_len=hist_len, head_agg=args.head_agg)
            for ratio in args.compression_ratios:
                keep_budget = min(hist_len, max(1, int(math.ceil(hist_len * (1.0 - ratio)))))
                indices = torch.topk(token_scores, k=keep_budget, largest=True, sorted=False).indices
                indices_np = indices.detach().cpu().numpy().astype(np.int32)
                topk_rows[(ratio, layer)].append(indices_np)
                lengths[(ratio, layer)].append(keep_budget)
        if step % 16 == 0 or step == args.decode_steps - 1:
            write_heartbeat(
                {
                    "status": "decode",
                    "job_id": job.job_id,
                    "job_index": job_index,
                    "total_jobs": total_jobs,
                    "current_step": step,
                    "decode_steps": args.decode_steps,
                    "context_length": job.context_length,
                }
            )
        del outputs, attentions

    summary: dict[str, Any] = {
        "job_id": job.job_id,
        "model_key": job.model_key,
        "dataset": args.dataset,
        "context_length": job.context_length,
        "sample_index": job.sample_index,
        "book_id": job.book_id,
        "decode_steps": args.decode_steps,
        "compression_ratios": args.compression_ratios,
        "head_agg": args.head_agg,
        "lags": args.lags,
        "reuse_intervals": args.reuse_intervals,
        "metric_scope": "historical_tokens_excluding_current_self_token",
        "lag_summary": [],
        "reuse_summary": [],
    }

    write_heartbeat(
        {
            "status": "summarize",
            "job_id": job.job_id,
            "job_index": job_index,
            "total_jobs": total_jobs,
        }
    )
    for ratio in args.compression_ratios:
        for layer in range(num_layers):
            key = (ratio, layer)
            length_arr = np.asarray(lengths[key], dtype=np.int32)
            membership = build_membership(topk_rows[key], length_arr, hist_len_max)
            for lag in args.lags:
                metrics = lag_metrics_from_membership(membership, length_arr, lag)
                summary["lag_summary"].append(
                    {
                        "layer": layer,
                        "compression_ratio": ratio,
                        "lag": lag,
                        **metrics,
                    }
                )
            for interval in args.reuse_intervals:
                metrics = reuse_metrics_from_membership(membership, length_arr, interval)
                summary["reuse_summary"].append(
                    {
                        "layer": layer,
                        "compression_ratio": ratio,
                        "reuse_interval": interval,
                        "refresh_reduction": 1.0 - (1.0 / interval),
                        **metrics,
                    }
                )
            del membership

    return summary


def write_aggregate(summaries: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        base = {
            "job_id": summary["job_id"],
            "model_key": summary["model_key"],
            "dataset": summary["dataset"],
            "context_length": summary["context_length"],
            "sample_index": summary["sample_index"],
            "book_id": summary["book_id"],
            "decode_steps": summary["decode_steps"],
            "head_agg": summary["head_agg"],
            "metric_scope": summary["metric_scope"],
        }
        for item in summary["lag_summary"]:
            rows.append(
                {
                    **base,
                    "metric_group": "lag",
                    "layer": item["layer"],
                    "compression_ratio": item["compression_ratio"],
                    "lag": item["lag"],
                    "overlap": item["overlap"],
                    "jaccard": item["jaccard"],
                    "reuse_interval": "",
                    "reuse_recall": "",
                    "reuse_jaccard": "",
                    "refresh_reduction": "",
                }
            )
        for item in summary["reuse_summary"]:
            rows.append(
                {
                    **base,
                    "metric_group": "reuse",
                    "layer": item["layer"],
                    "compression_ratio": item["compression_ratio"],
                    "lag": "",
                    "overlap": "",
                    "jaccard": "",
                    "reuse_interval": item["reuse_interval"],
                    "reuse_recall": item["reuse_recall"],
                    "reuse_jaccard": item["reuse_jaccard"],
                    "refresh_reduction": item["refresh_reduction"],
                }
            )

    AGG_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "model_key",
        "dataset",
        "context_length",
        "sample_index",
        "book_id",
        "decode_steps",
        "head_agg",
        "metric_scope",
        "metric_group",
        "layer",
        "compression_ratio",
        "lag",
        "overlap",
        "jaccard",
        "reuse_interval",
        "reuse_recall",
        "reuse_jaccard",
        "refresh_reduction",
    ]
    with AGG_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        AGG_JSON,
        {
            "experiment_name": EXPERIMENT_NAME,
            "generated_at": utc_now(),
            "aggregate_csv": str(AGG_CSV.relative_to(ROOT)),
            "num_rows": len(rows),
            "jobs": [summary["job_id"] for summary in summaries],
        },
    )


def read_completed_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not RAW_JSONL.exists():
        return summaries
    with RAW_JSONL.open() as f:
        for line in f:
            if not line.strip():
                continue
            summaries.append(json.loads(line))
    return summaries


def main() -> None:
    args = parse_args()
    configure_output_paths(args.run_tag)
    seed_everything(args.seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model_path = args.model or MODELS[args.model_key]
    write_json(CONFIG_JSON, vars(args) | {"model_path": model_path, "experiment_name": EXPERIMENT_NAME})
    model, tokenizer = load_model_and_tokenizer(model_path, args.device, args.dtype)
    books = tokenize_pg19_books(args, tokenizer)
    jobs = build_jobs(args, books)
    book_by_id = {str(book["book_id"]): book for book in books}
    done = completed_job_ids() if args.resume else set()
    summaries = read_completed_summaries() if args.resume else []

    write_json(
        MANIFEST_JSON,
        {
            "experiment_name": EXPERIMENT_NAME,
            "created_at": utc_now(),
            "jobs": [asdict(job) | {"job_id": job.job_id} for job in jobs],
            "completed_jobs": sorted(done),
            "config": vars(args) | {"model_path": model_path},
        },
    )

    for idx, job in enumerate(jobs):
        if job.job_id in done:
            continue
        summary = run_job(job, book_by_id[job.book_id], model, args, idx, len(jobs))
        append_jsonl(RAW_JSONL, summary)
        summaries.append(summary)
        done.add(job.job_id)
        write_aggregate(summaries)
        write_heartbeat(
            {
                "status": "job_complete",
                "job_id": job.job_id,
                "completed_jobs": len(done),
                "total_jobs": len(jobs),
            }
        )

    write_aggregate(summaries)
    write_heartbeat({"status": "complete", "completed_jobs": len(done), "total_jobs": len(jobs)})
    print(AGG_CSV)


if __name__ == "__main__":
    main()
