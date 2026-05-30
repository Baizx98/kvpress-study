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


EXPERIMENT_NAME = "ATC26_blockwise_ranked_topk_temporal_similarity"
RESULT_ROOT = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACT_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACT_DIR / "raw"
INDEX_DIR = ARTIFACT_DIR / "indices"
LOG_DIR = ARTIFACT_DIR / "logs"
AGG_JSON = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_aggregate.json"
AGG_CSV = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_aggregate.csv"
RAW_JSONL = RAW_DIR / "ATC26_ranked_topk_temporal_similarity_raw.jsonl"
MANIFEST_JSON = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_manifest.json"
CONFIG_JSON = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_config.json"
HEARTBEAT_JSON = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_heartbeat.json"

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
    parser = argparse.ArgumentParser(
        description="Collect ranked BlockWise top-k index traces from full KV without compression."
    )
    parser.add_argument("--model-key", choices=sorted(MODELS), default="llama31_8b_instruct")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", choices=["pg19"], default="pg19")
    parser.add_argument("--pg19-source-dataset", default=os.environ.get("PG19_SOURCE_DATASET", "/Tan/dataset/pg19-test"))
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--samples-per-length", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=1024)
    parser.add_argument("--compression-ratios", type=float, nargs="+", default=[0.7, 0.5, 0.3])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--window-query-size", type=int, default=16)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--reuse-intervals", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-tag", default="full")
    parser.add_argument("--compress-index-arrays", action="store_true")
    parser.add_argument("--max-books-to-tokenize", type=int, default=None)
    return parser.parse_args()


def configure_output_paths(run_tag: str) -> None:
    global ARTIFACT_DIR, RAW_DIR, INDEX_DIR, LOG_DIR
    global AGG_JSON, AGG_CSV, RAW_JSONL, MANIFEST_JSON, CONFIG_JSON, HEARTBEAT_JSON

    ARTIFACT_DIR = RESULT_ROOT / "artifacts" if run_tag == "full" else RESULT_ROOT / "artifacts" / run_tag
    RAW_DIR = ARTIFACT_DIR / "raw"
    INDEX_DIR = ARTIFACT_DIR / "indices"
    LOG_DIR = ARTIFACT_DIR / "logs"
    AGG_JSON = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_aggregate.json"
    AGG_CSV = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_aggregate.csv"
    RAW_JSONL = RAW_DIR / "ATC26_ranked_topk_temporal_similarity_raw.jsonl"
    MANIFEST_JSON = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_manifest.json"
    CONFIG_JSON = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_config.json"
    HEARTBEAT_JSON = ARTIFACT_DIR / "ATC26_ranked_topk_temporal_similarity_heartbeat.json"


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


def language_layers(model) -> list[Any]:
    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return list(language_model.layers)


def mean_or_nan(values: list[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return float(np.mean(finite)) if finite else float("nan")


def valid_list(row: np.ndarray, length: int) -> list[int]:
    return [int(x) for x in row[: int(length)].tolist() if int(x) >= 0]


def jaccard(a: list[int], b: list[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def overlap_recall(a: list[int], b: list[int]) -> float:
    if not b:
        return float("nan")
    return len(set(a) & set(b)) / float(len(b))


def rank_biased_overlap(a: list[int], b: list[int], p: float = 0.9) -> float:
    depth = min(len(a), len(b))
    if depth == 0:
        return float("nan")
    score = 0.0
    set_a: set[int] = set()
    set_b: set[int] = set()
    agreement = 0.0
    for d in range(1, depth + 1):
        set_a.add(a[d - 1])
        set_b.add(b[d - 1])
        agreement = len(set_a & set_b) / d
        score += (1.0 - p) * (p ** (d - 1)) * agreement
    score += (p ** depth) * agreement
    return float(score)


def common_rank_delta(a: list[int], b: list[int]) -> float:
    rank_a = {idx: rank for rank, idx in enumerate(a)}
    rank_b = {idx: rank for rank, idx in enumerate(b)}
    common = set(rank_a) & set(rank_b)
    if not common:
        return float("nan")
    return float(np.mean([abs(rank_a[idx] - rank_b[idx]) for idx in common]))


def build_topk_membership(
    ranked_indices: np.ndarray,
    ranked_origins: np.ndarray,
    lengths: np.ndarray,
    num_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    membership = np.zeros((ranked_indices.shape[0], num_blocks), dtype=bool)
    decode_membership = np.zeros_like(membership)
    for t, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        indices = ranked_indices[t, :length]
        valid = indices >= 0
        if not np.any(valid):
            continue
        indices = indices[valid]
        origins = ranked_origins[t, :length][valid]
        membership[t, indices] = True
        decode_membership[t, indices] = origins != 0
    return membership, decode_membership


def lag_metrics_from_membership(
    membership: np.ndarray,
    decode_membership: np.ndarray,
    lengths: np.ndarray,
    lag: int,
) -> dict[str, float]:
    if lag >= membership.shape[0]:
        return {
            "jaccard": float("nan"),
            "overlap": float("nan"),
            "decode_new_entry_ratio": float("nan"),
        }
    left = membership[:-lag]
    right = membership[lag:]
    inter = np.logical_and(left, right).sum(axis=1).astype(np.float32)
    len_left = lengths[:-lag].astype(np.float32)
    len_right = lengths[lag:].astype(np.float32)
    union = len_left + len_right - inter
    jaccard_vals = np.divide(inter, union, out=np.full_like(inter, np.nan), where=union > 0)
    overlap_vals = np.divide(inter, len_right, out=np.full_like(inter, np.nan), where=len_right > 0)

    new_entries = np.logical_and(right, ~left)
    new_count = new_entries.sum(axis=1).astype(np.float32)
    decode_new_count = np.logical_and(new_entries, decode_membership[lag:]).sum(axis=1).astype(np.float32)
    decode_new_ratio = np.divide(
        decode_new_count,
        new_count,
        out=np.full_like(decode_new_count, np.nan),
        where=new_count > 0,
    )
    return {
        "jaccard": float(np.nanmean(jaccard_vals)),
        "overlap": float(np.nanmean(overlap_vals)),
        "decode_new_entry_ratio": float(np.nanmean(decode_new_ratio)),
    }


def reuse_metrics_from_membership(
    membership: np.ndarray,
    lengths: np.ndarray,
    interval: int,
) -> dict[str, float]:
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


def block_origins(num_blocks: int, key_len: int, context_length: int, block_size: int) -> np.ndarray:
    origins = np.zeros(num_blocks, dtype=np.int8)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, key_len)
        if end <= context_length:
            origins[block_idx] = 0
        elif start >= context_length:
            origins[block_idx] = 1
        else:
            origins[block_idx] = 2
    return origins


class RankedTopKTraceBlockWisePress(BlockWisePress):
    def __post_init__(self):
        super().__post_init__()
        self.context_length = 0
        self.step_idx = -1
        self.window_query_size = 16
        self.compression_ratios: list[float] = [0.7, 0.5, 0.3]
        self.records: dict[str, dict[int, dict[int, dict[str, Any]]]] = {
            "single": defaultdict(dict),
            "window": defaultdict(dict),
        }
        self.hidden_buffers: dict[int, list[torch.Tensor]] = defaultdict(list)

    def set_context_length(self, context_length: int) -> None:
        self.context_length = context_length

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
        if self.step_idx < 0:
            return output

        keys, values = extract_keys_and_values(cache, layer_idx)
        keys = keys.contiguous()
        values = values.contiguous()
        self.hidden_buffers[layer_idx].append(hidden_states.detach())
        self.hidden_buffers[layer_idx] = self.hidden_buffers[layer_idx][-self.window_query_size :]

        self._record_query_mode("single", module, hidden_states, keys, values, output, kwargs)
        window_hidden = torch.cat(self.hidden_buffers[layer_idx], dim=1)
        self._record_query_mode("window", module, window_hidden, keys, values, output, kwargs)
        return output

    def _record_query_mode(self, mode: str, module, hidden_states, keys, values, output, kwargs) -> None:
        layer_idx = self._resolve_layer_idx(module)
        analysis = self.analyze_blocks(
            module,
            hidden_states,
            keys,
            values,
            output[1] if len(output) > 1 else None,
            kwargs,
            force_refresh_summary=True,
        )
        scores = analysis["block_scores"][0].detach().float()
        key_len = int(keys.shape[2])
        num_blocks = int(scores.shape[-1])
        origins = block_origins(num_blocks, key_len, self.context_length, self.block_size)
        full_order = torch.argsort(scores, descending=True).detach().long().cpu().numpy().astype(np.int32)
        full_ranks = np.empty(num_blocks, dtype=np.int32)
        full_ranks[full_order] = np.arange(num_blocks, dtype=np.int32)
        score_np = scores.detach().cpu().numpy().astype(np.float32)

        by_ratio: dict[str, dict[str, Any]] = {}
        decode_mask = origins != 0
        decode_order = [int(idx) for idx in full_order.tolist() if decode_mask[int(idx)]]
        best_decode_rank = int(full_ranks[decode_order[0]]) if decode_order else -1
        for ratio in self.compression_ratios:
            keep_budget = min(num_blocks, max(0, int(math.ceil(num_blocks * (1.0 - ratio)))))
            ranked = full_order[:keep_budget]
            ranked_scores = score_np[ranked] if keep_budget > 0 else np.empty(0, dtype=np.float32)
            ranked_origins = origins[ranked] if keep_budget > 0 else np.empty(0, dtype=np.int8)
            decode_in_topk_count = int(np.sum(ranked_origins != 0))
            by_ratio[f"{ratio:.2f}"] = {
                "keep_budget": keep_budget,
                "ranked_indices": ranked.astype(np.int32),
                "ranked_scores": ranked_scores.astype(np.float16),
                "ranked_origins": ranked_origins.astype(np.int8),
                "decode_in_topk_count": decode_in_topk_count,
                "decode_in_topk_ratio": float(decode_in_topk_count / keep_budget) if keep_budget else float("nan"),
            }

        self.records[mode][layer_idx][self.step_idx] = {
            "kv_len": key_len,
            "num_blocks": num_blocks,
            "origins": origins,
            "decode_block_indices_ranked": np.asarray(decode_order, dtype=np.int32),
            "decode_block_global_ranks": full_ranks[decode_order].astype(np.int32) if decode_order else np.empty(0, dtype=np.int32),
            "best_decode_global_rank": best_decode_rank,
            "ratios": by_ratio,
        }


@contextlib.contextmanager
def trace_context(press: RankedTopKTraceBlockWisePress, model):
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


def ratio_key(ratio: float) -> str:
    return f"{ratio:.2f}"


def summarize_trace(
    job: Job,
    press: RankedTopKTraceBlockWisePress,
    args: argparse.Namespace,
    index_path: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "job_id": job.job_id,
        "model_key": job.model_key,
        "dataset": args.dataset,
        "context_length": job.context_length,
        "sample_index": job.sample_index,
        "book_id": job.book_id,
        "decode_steps": args.decode_steps,
        "compression_ratios": args.compression_ratios,
        "block_size": args.block_size,
        "window_query_size": args.window_query_size,
        "index_arrays": str(index_path.relative_to(ROOT)),
        "modes": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for mode, per_layer in press.records.items():
        mode_summary: dict[str, Any] = {"layers": [], "lag_summary": [], "reuse_summary": [], "decode_summary": []}
        for layer_idx in sorted(per_layer):
            write_heartbeat(
                {
                    "pid": os.getpid(),
                    "status": "summarizing",
                    "current_job": job.job_id,
                    "current_mode": mode,
                    "current_layer": layer_idx,
                    "total_layers": len(per_layer),
                }
            )
            steps = per_layer[layer_idx]
            max_step = max(steps) if steps else -1
            if max_step + 1 < args.decode_steps:
                raise RuntimeError(f"Missing trace steps for layer={layer_idx} mode={mode}")

            kv_len = np.asarray([steps[t]["kv_len"] for t in range(args.decode_steps)], dtype=np.int32)
            num_blocks = np.asarray([steps[t]["num_blocks"] for t in range(args.decode_steps)], dtype=np.int32)
            best_decode_global_rank = np.asarray(
                [steps[t]["best_decode_global_rank"] for t in range(args.decode_steps)], dtype=np.int32
            )
            arrays[f"{mode}_layer{layer_idx:02d}_kv_len"] = kv_len
            arrays[f"{mode}_layer{layer_idx:02d}_num_blocks"] = num_blocks
            arrays[f"{mode}_layer{layer_idx:02d}_best_decode_global_rank"] = best_decode_global_rank

            for ratio in args.compression_ratios:
                key = ratio_key(ratio)
                lengths = np.asarray(
                    [steps[t]["ratios"][key]["keep_budget"] for t in range(args.decode_steps)], dtype=np.int32
                )
                max_k = int(lengths.max(initial=0))
                ranked_indices = np.full((args.decode_steps, max_k), -1, dtype=np.int32)
                ranked_origins = np.full((args.decode_steps, max_k), -1, dtype=np.int8)
                ranked_scores = np.full((args.decode_steps, max_k), np.nan, dtype=np.float16)
                decode_in_topk_count = np.zeros(args.decode_steps, dtype=np.int32)
                decode_in_topk_ratio = np.full(args.decode_steps, np.nan, dtype=np.float32)
                for t in range(args.decode_steps):
                    item = steps[t]["ratios"][key]
                    k = int(item["keep_budget"])
                    ranked_indices[t, :k] = item["ranked_indices"]
                    ranked_origins[t, :k] = item["ranked_origins"]
                    ranked_scores[t, :k] = item["ranked_scores"]
                    decode_in_topk_count[t] = int(item["decode_in_topk_count"])
                    decode_in_topk_ratio[t] = float(item["decode_in_topk_ratio"])

                prefix = f"{mode}_ratio{key}_layer{layer_idx:02d}"
                arrays[f"{prefix}_ranked_indices_all"] = ranked_indices
                arrays[f"{prefix}_ranked_scores_all"] = ranked_scores
                arrays[f"{prefix}_ranked_origins_all"] = ranked_origins
                arrays[f"{prefix}_topk_lengths"] = lengths
                arrays[f"{prefix}_decode_in_topk_count"] = decode_in_topk_count
                arrays[f"{prefix}_decode_in_topk_ratio"] = decode_in_topk_ratio
                membership, decode_membership = build_topk_membership(
                    ranked_indices,
                    ranked_origins,
                    lengths,
                    int(num_blocks.max(initial=0)),
                )

                mode_summary["decode_summary"].append(
                    {
                        "layer": layer_idx,
                        "compression_ratio": ratio,
                        "decode_in_topk_ratio_mean": float(np.nanmean(decode_in_topk_ratio)),
                        "decode_in_topk_ratio_last128_mean": float(np.nanmean(decode_in_topk_ratio[-128:])),
                        "best_decode_global_rank_mean": float(np.mean(best_decode_global_rank[best_decode_global_rank >= 0]))
                        if np.any(best_decode_global_rank >= 0)
                        else float("nan"),
                    }
                )

                for lag in args.lags:
                    if lag >= args.decode_steps:
                        continue
                    lag_metrics = lag_metrics_from_membership(membership, decode_membership, lengths, lag)
                    mode_summary["lag_summary"].append(
                        {
                            "layer": layer_idx,
                            "compression_ratio": ratio,
                            "lag": lag,
                            "jaccard": lag_metrics["jaccard"],
                            "overlap": lag_metrics["overlap"],
                            "rank_biased_overlap": float("nan"),
                            "common_rank_delta": float("nan"),
                            "decode_new_entry_ratio": lag_metrics["decode_new_entry_ratio"],
                        }
                    )

                for interval in args.reuse_intervals:
                    if interval > args.decode_steps:
                        continue
                    reuse_metrics = reuse_metrics_from_membership(membership, lengths, interval)
                    mode_summary["reuse_summary"].append(
                        {
                            "layer": layer_idx,
                            "compression_ratio": ratio,
                            "reuse_interval": interval,
                            "reuse_jaccard": reuse_metrics["reuse_jaccard"],
                            "reuse_recall": reuse_metrics["reuse_recall"],
                            "reuse_rank_biased_overlap": float("nan"),
                            "reuse_common_rank_delta": float("nan"),
                            "refresh_reduction": 1.0 - 1.0 / interval,
                        }
                    )

            mode_summary["layers"].append(
                {
                    "layer": layer_idx,
                    "num_blocks_initial": int(num_blocks[0]),
                    "num_blocks_final": int(num_blocks[-1]),
                    "kv_len_initial": int(kv_len[0]),
                    "kv_len_final": int(kv_len[-1]),
                }
            )
        summary["modes"][mode] = mode_summary

    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_heartbeat({"pid": os.getpid(), "status": "writing_index_arrays", "current_job": job.job_id})
    if args.compress_index_arrays:
        np.savez_compressed(index_path, **arrays)
    else:
        np.savez(index_path, **arrays)
    return summary


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
        "block_size": summary["block_size"],
        "window_query_size": summary["window_query_size"],
    }
    for mode, mode_payload in summary["modes"].items():
        for group_name in ("lag_summary", "reuse_summary", "decode_summary"):
            for item in mode_payload[group_name]:
                rows.append({**base, "mode": mode, "metric_group": group_name.replace("_summary", ""), **item})
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
            row.get("compression_ratio"),
            row.get("lag"),
            row.get("reuse_interval"),
        )
        bucket = grouped.setdefault(key, defaultdict(list))
        for metric, value in row.items():
            if metric in {
                "jaccard",
                "overlap",
                "rank_biased_overlap",
                "common_rank_delta",
                "decode_new_entry_ratio",
                "reuse_jaccard",
                "reuse_recall",
                "reuse_rank_biased_overlap",
                "reuse_common_rank_delta",
                "refresh_reduction",
                "decode_in_topk_ratio_mean",
                "decode_in_topk_ratio_last128_mean",
                "best_decode_global_rank_mean",
            } and isinstance(value, (int, float)) and math.isfinite(float(value)):
                bucket[metric].append(float(value))

    aggregate_rows = []
    for key, bucket in sorted(grouped.items()):
        model_key, context_length, mode, metric_group, compression_ratio, lag, reuse_interval = key
        out = {
            "model_key": model_key,
            "context_length": context_length,
            "mode": mode,
            "metric_group": metric_group,
            "compression_ratio": compression_ratio,
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

    press = RankedTopKTraceBlockWisePress(compression_ratio=0.0, **BLOCKWISE_CONFIG)
    press.block_size = args.block_size
    press.window_query_size = args.window_query_size
    press.compression_ratios = list(args.compression_ratios)
    press.set_context_length(job.context_length)
    press.reset_trace()

    position_ids = torch.arange(job.context_length, job.context_length + args.decode_steps, device=args.device).unsqueeze(0)

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

    index_path = INDEX_DIR / f"{job.job_id}.npz"
    summary = summarize_trace(job, press, args, index_path)
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
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
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
