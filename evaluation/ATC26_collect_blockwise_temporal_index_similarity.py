from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import signal
import sys
import time
from collections import defaultdict
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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from benchmarks.pg19.create_huggingface_dataset import load_pg19_source_dataframe  # noqa: E402
from kvpress import BlockWisePress  # noqa: E402
from kvpress.utils import extract_keys_and_values  # noqa: E402


EXPERIMENT_NAME = "ATC26_blockwise_temporal_index_similarity"
RESULT_ROOT = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACT_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACT_DIR / "raw"
SCORE_DIR = ARTIFACT_DIR / "scores"
LOG_DIR = ARTIFACT_DIR / "logs"
AGG_JSON = ARTIFACT_DIR / "ATC26_temporal_similarity_aggregate.json"
AGG_CSV = ARTIFACT_DIR / "ATC26_temporal_similarity_aggregate.csv"
RAW_JSONL = RAW_DIR / "ATC26_temporal_similarity_raw.jsonl"
MANIFEST_JSON = ARTIFACT_DIR / "ATC26_temporal_similarity_manifest.json"
CONFIG_JSON = ARTIFACT_DIR / "ATC26_temporal_similarity_config.json"
HEARTBEAT_JSON = ARTIFACT_DIR / "ATC26_temporal_similarity_heartbeat.json"

MODELS = {
    "llama31_8b_instruct": "/Tan/model/Llama-3.1-8B-Instruct",
    "mistral_7b_instruct_v03": "/Tan/model/Mistral-7B-Instruct-v0.3",
    "qwen3_8b": "/Tan/model/Qwen3-8B",
}

BLOCKWISE_CONFIG = {
    "block_size": 16,
    "q_window_size": 64,
    "summary_topk_keys": 4,
    "mean_key_weight": 0.75,
    "representative_k": 4,
    "multi_rep_k": 4,
    "query_topr": 16,
    "head_topk": 1,
    "summary_mode": "mean_plus_norm_topk_mean",
    "representative_mode": "key_norm",
    "query_agg_mode": "max",
    "head_agg_mode": "uniform_mean",
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
    parser = argparse.ArgumentParser(description="Collect BlockWise kept-block temporal similarity traces.")
    parser.add_argument("--model-key", choices=sorted(MODELS), default="llama31_8b_instruct")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", choices=["pg19"], default="pg19")
    parser.add_argument("--pg19-source-dataset", default=os.environ.get("PG19_SOURCE_DATASET", "/Tan/dataset/pg19-test"))
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--samples-per-length", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--window-query-size", type=int, default=16)
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--reuse-intervals",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-tag", default="full")
    parser.add_argument("--save-full-scores", action="store_true")
    parser.add_argument("--max-books-to-tokenize", type=int, default=None)
    return parser.parse_args()


def configure_output_paths(run_tag: str) -> None:
    global ARTIFACT_DIR, RAW_DIR, SCORE_DIR, LOG_DIR
    global AGG_JSON, AGG_CSV, RAW_JSONL, MANIFEST_JSON, CONFIG_JSON, HEARTBEAT_JSON

    ARTIFACT_DIR = RESULT_ROOT / "artifacts" if run_tag == "full" else RESULT_ROOT / "artifacts" / run_tag
    RAW_DIR = ARTIFACT_DIR / "raw"
    SCORE_DIR = ARTIFACT_DIR / "scores"
    LOG_DIR = ARTIFACT_DIR / "logs"
    AGG_JSON = ARTIFACT_DIR / "ATC26_temporal_similarity_aggregate.json"
    AGG_CSV = ARTIFACT_DIR / "ATC26_temporal_similarity_aggregate.csv"
    RAW_JSONL = RAW_DIR / "ATC26_temporal_similarity_raw.jsonl"
    MANIFEST_JSON = ARTIFACT_DIR / "ATC26_temporal_similarity_manifest.json"
    CONFIG_JSON = ARTIFACT_DIR / "ATC26_temporal_similarity_config.json"
    HEARTBEAT_JSON = ARTIFACT_DIR / "ATC26_temporal_similarity_heartbeat.json"


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


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
        raise RuntimeError(
            f"No PG19 books have enough tokens for max context/decode length: {min_tokens}"
        )
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


def language_layers(model) -> list[Any]:
    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return list(language_model.layers)


def mean_or_nan(values: list[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return float(np.mean(finite)) if finite else float("nan")


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    set_a = set(int(x) for x in a.tolist())
    set_b = set(int(x) for x in b.tolist())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    if b.size == 0:
        return float("nan")
    return len(set(int(x) for x in a.tolist()) & set(int(x) for x in b.tolist())) / float(b.size)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


class TemporalTraceBlockWisePress(BlockWisePress):
    def __post_init__(self):
        super().__post_init__()
        self.initial_key_lens: dict[int, int] = {}
        self.step_idx = -1
        self.window_query_size = 16
        self.records: dict[str, dict[int, dict[int, dict[str, Any]]]] = {
            "single": defaultdict(dict),
            "window": defaultdict(dict),
        }
        self.hidden_buffers: dict[int, list[torch.Tensor]] = defaultdict(list)

    def set_initial_key_lens(self, cache) -> None:
        self.initial_key_lens = {
            layer_idx: int(cache.get_seq_length(layer_idx)) for layer_idx in range(len(cache))
        }

    def set_step(self, step_idx: int) -> None:
        self.step_idx = step_idx

    def reset_trace(self) -> None:
        self.records = {"single": defaultdict(dict), "window": defaultdict(dict)}
        self.hidden_buffers = defaultdict(list)
        self.last_block_summary = {}

    def forward_hook(self, module, input, kwargs, output):
        del input
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        layer_idx = self._resolve_layer_idx(module)
        if layer_idx not in self.initial_key_lens or self.step_idx < 0:
            return output

        initial_len = self.initial_key_lens[layer_idx]
        keys, values = extract_keys_and_values(cache, layer_idx)
        keys = keys[:, :, :initial_len].contiguous()
        values = values[:, :, :initial_len].contiguous()
        self.hidden_buffers[layer_idx].append(hidden_states.detach())
        self.hidden_buffers[layer_idx] = self.hidden_buffers[layer_idx][-self.window_query_size :]

        single_hidden = hidden_states
        window_hidden = torch.cat(self.hidden_buffers[layer_idx], dim=1)
        self._record_query_mode("single", module, single_hidden, keys, values, output, kwargs)
        self._record_query_mode("window", module, window_hidden, keys, values, output, kwargs)
        return output

    def _record_query_mode(self, mode: str, module, hidden_states, keys, values, output, kwargs) -> None:
        layer_idx = self._resolve_layer_idx(module)
        plan = self.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            output[1] if len(output) > 1 else None,
            kwargs,
            force_refresh_summary=True,
        )
        score_np = plan["block_scores"][0].detach().float().cpu().numpy()
        full_np = plan["kept_block_indices"][0].detach().long().cpu().numpy()
        scored_np = self._select_scored_only(plan["block_scores"], keys.shape[2], plan["num_blocks"], plan["keep_budget"])[
            0
        ].detach().long().cpu().numpy()
        self.records[mode][layer_idx][self.step_idx] = {
            "scores": score_np.astype(np.float16),
            "selected_full": full_np.astype(np.int32),
            "selected_scored_only": scored_np.astype(np.int32),
            "num_blocks": int(plan["num_blocks"]),
            "keep_budget": int(plan["keep_budget"]),
        }

    def _select_scored_only(
        self,
        scores: torch.Tensor,
        key_len: int,
        num_blocks: int,
        keep_budget: int,
    ) -> torch.Tensor:
        has_partial_tail_block = key_len % self.block_size != 0
        protected = set(range(min(self.prefix_sink_blocks, num_blocks)))
        recent_count = min(self.protected_recent_blocks, num_blocks)
        protected |= set(range(max(0, num_blocks - recent_count), num_blocks))
        if has_partial_tail_block and num_blocks > 0:
            protected.add(num_blocks - 1)
        scored_budget = max(0, keep_budget - len(protected))
        candidates = [idx for idx in range(num_blocks) if idx not in protected]
        return self._select_top_block_indices(scores, candidates, scored_budget, scores.device).sort(dim=-1).values


@contextlib.contextmanager
def trace_context(press: TemporalTraceBlockWisePress, model):
    hooks = []
    try:
        layers = language_layers(model)
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for layer in layers:
            layer.self_attn.rotary_emb = language_model.rotary_emb
            hooks.append(layer.self_attn.register_forward_hook(press.forward_hook, with_kwargs=True))
        yield
    finally:
        for hook in hooks:
            hook.remove()


def summarize_trace(
    job: Job,
    press: TemporalTraceBlockWisePress,
    args: argparse.Namespace,
    score_path: Path,
) -> dict[str, Any]:
    job_summary: dict[str, Any] = {
        "job_id": job.job_id,
        "model_key": job.model_key,
        "dataset": args.dataset,
        "context_length": job.context_length,
        "sample_index": job.sample_index,
        "book_id": job.book_id,
        "decode_steps": args.decode_steps,
        "compression_ratio": args.compression_ratio,
        "block_size": args.block_size,
        "window_query_size": args.window_query_size,
        "score_arrays": str(score_path.relative_to(ROOT)),
        "modes": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for mode, per_layer in press.records.items():
        layer_summaries = []
        lag_summary = []
        reuse_summary = []
        score_layers = []
        full_layers = []
        scored_layers = []
        for layer_idx in sorted(per_layer):
            steps = per_layer[layer_idx]
            max_step = max(steps) if steps else -1
            if max_step + 1 < args.decode_steps:
                raise RuntimeError(f"Missing trace steps for layer={layer_idx} mode={mode}")
            scores = np.stack([steps[t]["scores"] for t in range(args.decode_steps)], axis=0)
            selected_full = np.stack([steps[t]["selected_full"] for t in range(args.decode_steps)], axis=0)
            selected_scored = np.stack([steps[t]["selected_scored_only"] for t in range(args.decode_steps)], axis=0)
            score_layers.append(scores)
            full_layers.append(selected_full)
            scored_layers.append(selected_scored)

            for lag in args.lags:
                if lag >= args.decode_steps:
                    continue
                j_vals = []
                o_vals = []
                c_vals = []
                for t in range(args.decode_steps - lag):
                    j_vals.append(jaccard(selected_scored[t], selected_scored[t + lag]))
                    o_vals.append(overlap(selected_scored[t], selected_scored[t + lag]))
                    c_vals.append(cosine(scores[t].astype(np.float32), scores[t + lag].astype(np.float32)))
                lag_summary.append(
                    {
                        "layer": layer_idx,
                        "lag": lag,
                        "jaccard": mean_or_nan(j_vals),
                        "overlap": mean_or_nan(o_vals),
                        "score_cosine": mean_or_nan(c_vals),
                    }
                )

            for interval in args.reuse_intervals:
                if interval > args.decode_steps:
                    continue
                j_vals = []
                o_vals = []
                for t in range(args.decode_steps):
                    anchor = (t // interval) * interval
                    j_vals.append(jaccard(selected_scored[anchor], selected_scored[t]))
                    o_vals.append(overlap(selected_scored[anchor], selected_scored[t]))
                reuse_summary.append(
                    {
                        "layer": layer_idx,
                        "reuse_interval": interval,
                        "reuse_jaccard": mean_or_nan(j_vals),
                        "reuse_recall": mean_or_nan(o_vals),
                        "refresh_reduction": 1.0 - 1.0 / interval,
                    }
                )

            layer_summaries.append(
                {
                    "layer": layer_idx,
                    "num_blocks": int(steps[0]["num_blocks"]),
                    "keep_budget": int(steps[0]["keep_budget"]),
                    "scored_budget": int(selected_scored.shape[1]),
                }
            )

        arrays[f"{mode}_selected_full"] = np.stack(full_layers, axis=0)
        arrays[f"{mode}_selected_scored_only"] = np.stack(scored_layers, axis=0)
        if args.save_full_scores:
            arrays[f"{mode}_scores"] = np.stack(score_layers, axis=0)
        job_summary["modes"][mode] = {
            "layers": layer_summaries,
            "lag_summary": lag_summary,
            "reuse_summary": reuse_summary,
        }
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_path, **arrays)
    return job_summary


def flatten_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "job_id": summary["job_id"],
        "model_key": summary["model_key"],
        "dataset": summary["dataset"],
        "context_length": summary["context_length"],
        "sample_index": summary["sample_index"],
        "book_id": summary["book_id"],
        "decode_steps": summary["decode_steps"],
        "compression_ratio": summary["compression_ratio"],
        "block_size": summary["block_size"],
        "window_query_size": summary["window_query_size"],
    }
    for mode, mode_payload in summary["modes"].items():
        for item in mode_payload["lag_summary"]:
            rows.append({**base, "mode": mode, "metric_group": "lag", **item})
        for item in mode_payload["reuse_summary"]:
            rows.append({**base, "mode": mode, "metric_group": "reuse", **item})
    return rows


def write_aggregate(all_summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in all_summaries:
        rows.extend(flatten_summary_rows(summary))
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with AGG_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in rows:
        key = (
            row.get("model_key"),
            row.get("context_length"),
            row.get("mode"),
            row.get("metric_group"),
            row.get("lag"),
            row.get("reuse_interval"),
        )
        bucket = grouped.setdefault(key, defaultdict(list))
        for metric in ("jaccard", "overlap", "score_cosine", "reuse_jaccard", "reuse_recall", "refresh_reduction"):
            value = row.get(metric)
            if value not in (None, "") and isinstance(value, (int, float)) and math.isfinite(float(value)):
                bucket[metric].append(float(value))

    aggregate_rows = []
    for key, bucket in sorted(grouped.items()):
        model_key, context_length, mode, metric_group, lag, reuse_interval = key
        out = {
            "model_key": model_key,
            "context_length": context_length,
            "mode": mode,
            "metric_group": metric_group,
            "lag": lag,
            "reuse_interval": reuse_interval,
        }
        for metric, values in bucket.items():
            out[f"{metric}_mean"] = mean_or_nan(values)
            out[f"{metric}_std"] = float(np.std(values)) if values else float("nan")
        aggregate_rows.append(out)

    write_json(
        AGG_JSON,
        {
            "experiment_name": EXPERIMENT_NAME,
            "updated_at": utc_now(),
            "raw_jsonl": str(RAW_JSONL.relative_to(ROOT)),
            "aggregate_csv": str(AGG_CSV.relative_to(ROOT)),
            "summaries": all_summaries,
            "aggregate": aggregate_rows,
        },
    )


def load_existing_summaries() -> list[dict[str, Any]]:
    if not RAW_JSONL.exists():
        return []
    summaries = []
    with RAW_JSONL.open() as f:
        for line in f:
            if line.strip():
                summaries.append(json.loads(line))
    return summaries


@torch.inference_mode()
def run_job(
    args: argparse.Namespace,
    job: Job,
    book: dict[str, Any],
    model,
    tokenizer,
    completed: int,
    total: int,
) -> dict[str, Any]:
    del tokenizer
    context_ids = torch.tensor([book["input_ids"][: job.context_length]], dtype=torch.long, device=args.device)
    target_ids = torch.tensor(
        [book["input_ids"][job.context_length : job.context_length + args.decode_steps]],
        dtype=torch.long,
        device=args.device,
    )
    cache = DynamicCache()
    model(input_ids=context_ids, past_key_values=cache, num_logits_to_keep=1)

    press = TemporalTraceBlockWisePress(compression_ratio=args.compression_ratio, **BLOCKWISE_CONFIG)
    press.block_size = args.block_size
    press.window_query_size = args.window_query_size
    press.set_initial_key_lens(cache)
    press.reset_trace()

    position_ids = torch.arange(
        job.context_length,
        job.context_length + args.decode_steps,
        device=args.device,
    ).unsqueeze(0)

    with trace_context(press, model):
        for step_idx in tqdm(range(args.decode_steps), desc=f"Trace {job.job_id}", leave=False):
            press.set_step(step_idx)
            write_heartbeat(
                {
                    "pid": os.getpid(),
                    "current_job": job.job_id,
                    "current_model": job.model_key,
                    "current_dataset": args.dataset,
                    "current_context_length": job.context_length,
                    "current_sample": job.sample_index,
                    "current_step": step_idx,
                    "completed_jobs": completed,
                    "total_jobs": total,
                }
            )
            model(
                input_ids=target_ids[:, step_idx : step_idx + 1],
                past_key_values=cache,
                position_ids=position_ids[:, step_idx : step_idx + 1],
                num_logits_to_keep=1,
            )

    score_path = SCORE_DIR / f"{job.job_id}.npz"
    summary = summarize_trace(job, press, args, score_path)
    summary["book"] = {
        "short_book_title": book["short_book_title"],
        "source_token_count": book["source_token_count"],
    }
    del context_ids, target_ids, cache, press
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def handle_signal(signum, frame):
    del frame
    write_heartbeat({"pid": os.getpid(), "status": f"received_signal_{signum}"})
    raise SystemExit(128 + signum)


def main() -> int:
    args = parse_args()
    configure_output_paths(args.run_tag)
    seed_everything(args.seed)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handle_signal)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG_JSON, vars(args))
    BLOCKWISE_CONFIG["block_size"] = args.block_size

    model_path = args.model or MODELS[args.model_key]
    print(f"[load] model_key={args.model_key} model={model_path} device={args.device}", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_path, args.device, args.dtype)
    books = tokenize_pg19_books(args, tokenizer)
    book_by_id = {str(book["book_id"]): book for book in books}
    jobs = build_jobs(args, books)
    write_json(
        MANIFEST_JSON,
        {
            "experiment_name": EXPERIMENT_NAME,
            "created_at": utc_now(),
            "jobs": [asdict(job) | {"job_id": job.job_id} for job in jobs],
            "books": [
                {
                    "book_id": book["book_id"],
                    "short_book_title": book["short_book_title"],
                    "source_token_count": book["source_token_count"],
                }
                for book in books
            ],
        },
    )

    done = completed_job_ids() if args.resume else set()
    if not args.resume and RAW_JSONL.exists():
        RAW_JSONL.unlink()
    summaries = load_existing_summaries() if args.resume else []

    for job_idx, job in enumerate(jobs):
        if job.job_id in done:
            print(f"[skip] {job.job_id}", flush=True)
            continue
        print(f"[run] {job.job_id}", flush=True)
        summary = run_job(
            args=args,
            job=job,
            book=book_by_id[job.book_id],
            model=model,
            tokenizer=tokenizer,
            completed=len(done),
            total=len(jobs),
        )
        append_jsonl(RAW_JSONL, summary)
        summaries.append(summary)
        done.add(job.job_id)
        write_aggregate(summaries)
        write_heartbeat(
            {
                "pid": os.getpid(),
                "status": "job_completed",
                "current_job": job.job_id,
                "completed_jobs": len(done),
                "total_jobs": len(jobs),
            }
        )
        print(f"[done] {job.job_id} ({job_idx + 1}/{len(jobs)})", flush=True)

    write_aggregate(summaries)
    write_heartbeat({"pid": os.getpid(), "status": "complete", "completed_jobs": len(done), "total_jobs": len(jobs)})
    print(AGG_JSON, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
