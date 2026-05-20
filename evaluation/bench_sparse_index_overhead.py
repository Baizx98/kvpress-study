from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "sparse_index_overhead_snapkv_chunkkv_blockwise"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
SUMMARY_CSV = ARTIFACTS_DIR / "sparse_index_overhead_summary.csv"
LAYER_CSV = ARTIFACTS_DIR / "sparse_index_overhead_layers.csv"
METADATA_JSON = ARTIFACTS_DIR / "metadata.json"
RESULT_README = RESULT_ROOT / "README.md"


@dataclass
class FakeConfig:
    num_attention_heads: int
    num_key_value_heads: int


@dataclass
class FakeModule:
    config: FakeConfig
    head_dim: int
    layer_idx: int = 0


@dataclass
class RealProjectionWeights:
    vocab_size: int
    embed_weight: torch.Tensor
    q_proj_weights: dict[int, torch.Tensor]
    k_proj_weights: dict[int, torch.Tensor]
    layer_indices: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sparse index overhead for SnapKV, ChunkKV, and BlockWisePress.")
    parser.add_argument("--model", type=str, default="/Tan/model/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--reuse-steps", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--snap-kernel-size", type=int, default=5)
    parser.add_argument("--chunk-length", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--block-q-window", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-layer", type=int, default=0)
    parser.add_argument("--layers", nargs="+", default=["all"], help="Layer ids to test, or 'all'.")
    parser.add_argument("--no-real-model-weights", action="store_true")
    parser.add_argument("--force-exit", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def resolve_layer_indices(args: argparse.Namespace, model_metadata: dict[str, object]) -> list[int]:
    num_layers = int(model_metadata["num_hidden_layers"])
    if args.layers == ["all"]:
        return list(range(num_layers))
    layer_indices = sorted({int(layer) for layer in args.layers})
    invalid = [layer for layer in layer_indices if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Invalid layer ids {invalid}; model has {num_layers} layers.")
    return layer_indices


def load_llama_shape(model_path: str) -> tuple[int, int, int, dict[str, object]]:
    config_path = Path(model_path).expanduser() / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config.get("num_key_value_heads", num_heads))
    hidden_size = int(config["hidden_size"])
    head_dim = int(config.get("head_dim", hidden_size // num_heads))
    return num_heads, num_kv_heads, head_dim, {
        "model_path": model_path,
        "config_path": str(config_path),
        "model_type": config.get("model_type"),
        "hidden_size": hidden_size,
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def load_real_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> RealProjectionWeights | None:
    if args.no_real_model_weights:
        return None
    from safetensors import safe_open

    model_dir = Path(args.model).expanduser()
    with (model_dir / "config.json").open() as f:
        config = json.load(f)
    with (model_dir / "model.safetensors.index.json").open() as f:
        weight_map = json.load(f)["weight_map"]

    model_metadata = {
        "num_hidden_layers": int(config["num_hidden_layers"]),
    }
    layer_indices = resolve_layer_indices(args, model_metadata)
    names = {"embed_weight": "model.embed_tokens.weight"}
    for layer_idx in layer_indices:
        names[f"q_proj_weight.{layer_idx}"] = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        names[f"k_proj_weight.{layer_idx}"] = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    shard_names = {weight_map[name] for name in names.values()}
    loaded: dict[str, torch.Tensor] = {}
    for shard_name in sorted(shard_names):
        shard_path = model_dir / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for field_name, tensor_name in names.items():
                if weight_map[tensor_name] == shard_name:
                    loaded[field_name] = f.get_tensor(tensor_name).to(device=device, dtype=dtype, non_blocking=True)

    return RealProjectionWeights(
        vocab_size=int(config["vocab_size"]),
        embed_weight=loaded["embed_weight"],
        q_proj_weights={layer_idx: loaded[f"q_proj_weight.{layer_idx}"] for layer_idx in layer_indices},
        k_proj_weights={layer_idx: loaded[f"k_proj_weight.{layer_idx}"] for layer_idx in layer_indices},
        layer_indices=layer_indices,
    )


@torch.inference_mode()
def make_real_model_qk(
    weights: RealProjectionWeights,
    args: argparse.Namespace,
    batch_size: int,
    length: int,
    layer_idx: int,
    module: FakeModule,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed + batch_size * 100000 + length)
    input_ids = torch.randint(0, weights.vocab_size, (batch_size, length), device=device, generator=generator)
    hidden_states = F.embedding(input_ids, weights.embed_weight)

    q_window_hidden = hidden_states[:, -args.window_size :]
    q_proj_weight = weights.q_proj_weights[layer_idx]
    k_proj_weight = weights.k_proj_weights[layer_idx]
    q_window = F.linear(q_window_hidden, q_proj_weight)
    q_window = q_window.view(batch_size, args.window_size, module.config.num_attention_heads, module.head_dim)
    q_window = q_window.transpose(1, 2).contiguous()

    key_states = F.linear(hidden_states, k_proj_weight)
    key_states = key_states.view(batch_size, length, module.config.num_key_value_heads, module.head_dim)
    key_states = key_states.transpose(1, 2).contiguous()

    block_q_hidden = hidden_states[:, -args.block_q_window :]
    block_query_states = F.linear(block_q_hidden, q_proj_weight)
    groups = module.config.num_attention_heads // module.config.num_key_value_heads
    block_query_states = block_query_states.view(
        batch_size,
        args.block_q_window,
        module.config.num_key_value_heads,
        groups,
        module.head_dim,
    )
    block_query_states = block_query_states.mean(dim=3).transpose(1, 2).contiguous()

    del input_ids, hidden_states, q_window_hidden, block_q_hidden
    return q_window, key_states, block_query_states


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p10": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
    }


def time_cuda_and_wall(fn: Callable[[], torch.Tensor | dict[str, torch.Tensor]], warmup: int, repeat: int) -> dict[str, float]:
    keepalive = None
    for _ in range(warmup):
        keepalive = fn()
    torch.cuda.synchronize()

    cuda_values: list[float] = []
    wall_values: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        keepalive = fn()
        end.record()
        torch.cuda.synchronize()
        wall_values.append((time.perf_counter() - wall_start) * 1000.0)
        cuda_values.append(start.elapsed_time(end))

    cuda_stats = summarize(cuda_values)
    wall_stats = summarize(wall_values)
    del keepalive
    return {
        "cuda_ms_mean": cuda_stats["mean"],
        "cuda_ms_median": cuda_stats["median"],
        "cuda_ms_p10": cuda_stats["p10"],
        "cuda_ms_p90": cuda_stats["p90"],
        "wall_ms_mean": wall_stats["mean"],
        "wall_ms_median": wall_stats["median"],
        "wall_ms_p10": wall_stats["p10"],
        "wall_ms_p90": wall_stats["p90"],
    }


def snap_score_ops(q_window: torch.Tensor, keys: torch.Tensor, kernel_size: int, window_size: int) -> torch.Tensor:
    batch_size, kv_heads, kv_len, _ = keys.shape
    query_heads = q_window.shape[1]
    groups = query_heads // kv_heads
    key_states = keys.repeat_interleave(groups, dim=1)

    attn_weights = torch.matmul(q_window, key_states.transpose(2, 3)) / math.sqrt(q_window.shape[-1])
    attention_mask = torch.ones_like(attn_weights) * float("-inf")
    attention_mask = torch.triu(attention_mask, diagonal=kv_len - window_size + 1)
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_window.dtype)
    attn_weights = attn_weights[..., :-window_size]

    scores = attn_weights.mean(dim=-2)
    scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    scores = scores.view(batch_size, kv_heads, groups, kv_len - window_size).mean(2)
    return F.pad(scores, (0, window_size), value=scores.max().item())


def snap_topk_index_ops(scores: torch.Tensor, compression_ratio: float) -> torch.Tensor:
    keep = max(1, int(scores.shape[-1] * (1.0 - compression_ratio)))
    return scores.topk(keep, dim=-1).indices


def chunkkv_index_ops(global_scores: torch.Tensor, chunk_length: int, compression_ratio: float) -> torch.Tensor:
    batch_size = global_scores.shape[0]
    kv_len = global_scores.shape[-1]
    num_complete_chunks = kv_len // chunk_length
    remaining_tokens = kv_len % chunk_length

    if num_complete_chunks == 0:
        keep = max(1, int(kv_len * (1.0 - compression_ratio)))
        return global_scores.topk(keep, dim=-1).indices

    main_scores = global_scores[..., : num_complete_chunks * chunk_length]
    main_chunk_scores = main_scores.sum(dim=1).view(batch_size, num_complete_chunks, chunk_length).mean(dim=-1)
    if remaining_tokens > 0:
        remaining_scores = global_scores[..., -remaining_tokens:]
        remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
        chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
    else:
        chunk_scores = main_chunk_scores

    total_chunks = num_complete_chunks + int(remaining_tokens > 0)
    keep_chunks = max(1, int(total_chunks * (1.0 - compression_ratio)))
    top_chunks = chunk_scores.topk(keep_chunks, dim=-1).indices

    token_indices_per_batch = []
    for batch_idx in range(top_chunks.shape[0]):
        batch_token_indices = []
        for chunk_idx in top_chunks[batch_idx]:
            idx = int(chunk_idx.item())
            if idx < num_complete_chunks:
                start_idx = idx * chunk_length
                chunk_indices = torch.arange(start_idx, start_idx + chunk_length, device=global_scores.device)
            else:
                chunk_indices = torch.arange(num_complete_chunks * chunk_length, kv_len, device=global_scores.device)
            batch_token_indices.append(chunk_indices)
        token_indices_per_batch.append(torch.cat(batch_token_indices).sort()[0])
    return torch.stack(token_indices_per_batch, dim=0)


def blockwise_summary_build_ops(keys: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    batch_size, kv_heads, key_len, head_dim = keys.shape
    num_blocks = math.ceil(key_len / args.block_size)
    if num_blocks == 0:
        return {
            "num_blocks": torch.tensor(0, dtype=torch.long, device=keys.device),
            "mean_keys": keys.new_zeros((batch_size, kv_heads, 0, head_dim)),
            "topk_key_means": keys.new_zeros((batch_size, kv_heads, 0, head_dim)),
            "token_counts": torch.zeros((batch_size, 0), dtype=torch.long, device=keys.device),
        }

    topk = min(4, args.block_size)
    padded_len = num_blocks * args.block_size
    if padded_len == key_len:
        padded_keys = keys
    else:
        padded_keys = F.pad(keys, (0, 0, 0, padded_len - key_len))
    block_keys = padded_keys.view(batch_size, kv_heads, num_blocks, args.block_size, head_dim)

    token_counts_1d = torch.full((num_blocks,), args.block_size, dtype=torch.long, device=keys.device)
    tail_len = key_len - (num_blocks - 1) * args.block_size
    token_counts_1d[-1] = tail_len
    token_counts = token_counts_1d[None, :].expand(batch_size, -1)

    valid_mask = torch.arange(args.block_size, device=keys.device)[None, :] < token_counts_1d[:, None]
    valid_mask = valid_mask[None, None, :, :, None]
    valid_counts = token_counts_1d.to(keys.dtype).view(1, 1, num_blocks, 1).clamp_min(1)
    mean_keys = (block_keys * valid_mask).sum(dim=3) / valid_counts

    selector_scores = block_keys.norm(dim=-1).masked_fill(~valid_mask.squeeze(-1), float("-inf"))
    topk_indices = selector_scores.topk(topk, dim=-1).indices
    topk_gather = topk_indices[..., None].expand(-1, -1, -1, -1, head_dim)
    topk_keys = block_keys.gather(3, topk_gather)
    selected_valid = valid_mask.squeeze(-1).expand(batch_size, kv_heads, -1, -1).gather(3, topk_indices)
    topk_counts = token_counts_1d.clamp_max(topk).to(keys.dtype).view(1, 1, num_blocks, 1).clamp_min(1)
    topk_key_means = (topk_keys * selected_valid[..., None]).sum(dim=3) / topk_counts

    return {
        "num_blocks": torch.tensor(num_blocks, dtype=torch.long, device=keys.device),
        "mean_keys": mean_keys,
        "topk_key_means": topk_key_means,
        "token_counts": token_counts,
    }


def aggregate_head_scores_uniform(scores: torch.Tensor) -> torch.Tensor:
    if scores.shape[1] == 0:
        return scores.new_zeros((scores.shape[0], scores.shape[-1]))
    return scores.mean(dim=1)


def expand_blocks_to_token_indices(
    batch_size: int,
    key_len: int,
    block_size: int,
    block_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    token_indices_list = []
    expected_kept_len = None
    for batch_idx in range(batch_size):
        token_indices = []
        for block_idx in block_indices[batch_idx].tolist():
            start = block_idx * block_size
            end = min(start + block_size, key_len)
            token_indices.extend(range(start, end))
        token_tensor = torch.tensor(token_indices, dtype=torch.long, device=device)
        if expected_kept_len is None:
            expected_kept_len = token_tensor.numel()
        elif token_tensor.numel() != expected_kept_len:
            raise ValueError("BlockWise synthetic benchmark expects equal kept token counts across batch.")
        token_indices_list.append(token_tensor)
    return torch.stack(token_indices_list, dim=0) if token_indices_list else torch.empty(batch_size, 0, dtype=torch.long, device=device)


def select_top_block_indices(scores: torch.Tensor, candidates: list[int], count: int, device: torch.device) -> torch.Tensor:
    if count <= 0 or not candidates:
        return torch.empty(scores.shape[0], 0, dtype=torch.long, device=device)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    candidate_scores = scores.index_select(dim=-1, index=candidate_tensor)
    top_indices = candidate_scores.topk(min(count, candidate_tensor.numel()), dim=-1).indices
    return candidate_tensor[top_indices]


def blockwise_online_ops(
    args: argparse.Namespace,
    keys: torch.Tensor,
    kv_query_states: torch.Tensor,
    summary: dict[str, torch.Tensor],
    compression_ratio: float,
) -> dict[str, torch.Tensor]:
    weighted_anchors = 0.75 * summary["mean_keys"] + 0.25 * summary["topk_key_means"]
    summary_scores_per_head = torch.einsum("bhqd,bhkd->bhqk", kv_query_states, weighted_anchors) / math.sqrt(
        kv_query_states.shape[-1]
    )
    summary_scores_per_head = summary_scores_per_head.mean(dim=-2)
    scores = aggregate_head_scores_uniform(summary_scores_per_head)
    num_blocks = scores.shape[-1]
    keep_budget = min(num_blocks, max(0, int(math.ceil(num_blocks * (1.0 - compression_ratio)))))
    key_len = keys.shape[2]

    if keep_budget == 0:
        kept_block_indices = torch.empty(keys.shape[0], 0, dtype=torch.long, device=keys.device)
    elif keep_budget >= num_blocks:
        kept_block_indices = torch.arange(num_blocks, device=keys.device).expand(keys.shape[0], -1)
    else:
        sink_count = min(1, num_blocks)
        recent_count = min(2, num_blocks)
        protected_indices = set(range(sink_count))
        protected_indices |= set(range(max(0, num_blocks - recent_count), num_blocks))
        if key_len % args.block_size != 0 and num_blocks > 0:
            protected_indices.add(num_blocks - 1)

        if len(protected_indices) <= keep_budget:
            remaining_candidates = [idx for idx in range(num_blocks) if idx not in protected_indices]
            selected_remaining = select_top_block_indices(
                scores,
                remaining_candidates,
                keep_budget - len(protected_indices),
                keys.device,
            )
            protected_tensor = torch.tensor(sorted(protected_indices), dtype=torch.long, device=keys.device).expand(
                keys.shape[0], -1
            )
            kept_block_indices = torch.cat([protected_tensor, selected_remaining], dim=-1).sort(dim=-1).values
        else:
            kept_block_indices = select_top_block_indices(
                scores,
                list(range(num_blocks)),
                keep_budget,
                keys.device,
            ).sort(dim=-1).values

    token_indices = expand_blocks_to_token_indices(keys.shape[0], key_len, args.block_size, kept_block_indices, keys.device)
    return {
        "summary_scores_per_head": summary_scores_per_head,
        "block_scores": scores,
        "kept_block_indices": kept_block_indices,
        "token_indices": token_indices,
    }


def measure_case(
    *,
    sweep: str,
    batch_size: int,
    length: int,
    compression_ratio: float,
    reuse_steps: int,
    layer_idx: int,
    args: argparse.Namespace,
    module: FakeModule,
    weights: RealProjectionWeights | None,
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict[str, object]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if weights is None:
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_size * 100000 + length)
        q_window = torch.randn(
            batch_size,
            module.config.num_attention_heads,
            args.window_size,
            module.head_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        keys = torch.randn(
            batch_size,
            module.config.num_key_value_heads,
            length,
            module.head_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        kv_query_states = torch.randn(
            batch_size,
            module.config.num_key_value_heads,
            args.block_q_window,
            module.head_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        q_window, keys, kv_query_states = make_real_model_qk(
            weights,
            args,
            batch_size,
            length,
            layer_idx,
            module,
            dtype,
            device,
        )

    rows: list[dict[str, object]] = []
    snap_score_stats = time_cuda_and_wall(
        lambda: snap_score_ops(q_window, keys, args.snap_kernel_size, args.window_size),
        args.warmup,
        args.repeat,
    )
    scores = snap_score_ops(q_window, keys, args.snap_kernel_size, args.window_size)
    snap_topk_stats = time_cuda_and_wall(lambda: snap_topk_index_ops(scores, compression_ratio), args.warmup, args.repeat)
    chunk_index_stats = time_cuda_and_wall(
        lambda: chunkkv_index_ops(scores, args.chunk_length, compression_ratio),
        args.warmup,
        args.repeat,
    )

    summary_build_stats = time_cuda_and_wall(
        lambda: blockwise_summary_build_ops(keys, args),
        args.warmup,
        args.repeat,
    )
    summary = blockwise_summary_build_ops(keys, args)
    block_online_stats = time_cuda_and_wall(
        lambda: blockwise_online_ops(args, keys, kv_query_states, summary, compression_ratio),
        args.warmup,
        args.repeat,
    )

    snap_total_cuda_mean = snap_score_stats["cuda_ms_mean"] + snap_topk_stats["cuda_ms_mean"]
    snap_total_wall_mean = snap_score_stats["wall_ms_mean"] + snap_topk_stats["wall_ms_mean"]
    chunk_total_cuda_mean = snap_score_stats["cuda_ms_mean"] + chunk_index_stats["cuda_ms_mean"]
    chunk_total_wall_mean = snap_score_stats["wall_ms_mean"] + chunk_index_stats["wall_ms_mean"]
    block_amort_cuda_mean = block_online_stats["cuda_ms_mean"] + summary_build_stats["cuda_ms_mean"] / reuse_steps
    block_amort_wall_mean = block_online_stats["wall_ms_mean"] + summary_build_stats["wall_ms_mean"] / reuse_steps
    snap_total_cuda_median = snap_score_stats["cuda_ms_median"] + snap_topk_stats["cuda_ms_median"]
    snap_total_wall_median = snap_score_stats["wall_ms_median"] + snap_topk_stats["wall_ms_median"]
    chunk_total_cuda_median = snap_score_stats["cuda_ms_median"] + chunk_index_stats["cuda_ms_median"]
    chunk_total_wall_median = snap_score_stats["wall_ms_median"] + chunk_index_stats["wall_ms_median"]
    block_amort_cuda_median = block_online_stats["cuda_ms_median"] + summary_build_stats["cuda_ms_median"] / reuse_steps
    block_amort_wall_median = block_online_stats["wall_ms_median"] + summary_build_stats["wall_ms_median"] / reuse_steps
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    common = {
        "sweep": sweep,
        "batch_size": batch_size,
        "length": length,
        "compression_ratio": compression_ratio,
        "reuse_steps": reuse_steps,
        "layer_idx": layer_idx,
        "window_size": args.window_size,
        "chunk_length": args.chunk_length,
        "block_size": args.block_size,
        "block_q_window": args.block_q_window,
        "peak_memory_mb": peak_memory_mb,
    }
    rows.extend(
        [
            {
                **common,
                "method": "snapkv",
                "score_cuda_ms_mean": snap_score_stats["cuda_ms_mean"],
                "score_cuda_ms_median": snap_score_stats["cuda_ms_median"],
                "score_wall_ms_mean": snap_score_stats["wall_ms_mean"],
                "score_wall_ms_median": snap_score_stats["wall_ms_median"],
                "topk_index_cuda_ms_mean": snap_topk_stats["cuda_ms_mean"],
                "topk_index_cuda_ms_median": snap_topk_stats["cuda_ms_median"],
                "topk_index_wall_ms_mean": snap_topk_stats["wall_ms_mean"],
                "topk_index_wall_ms_median": snap_topk_stats["wall_ms_median"],
                "summary_build_cuda_ms_mean": 0.0,
                "summary_build_cuda_ms_median": 0.0,
                "summary_build_wall_ms_mean": 0.0,
                "summary_build_wall_ms_median": 0.0,
                "online_index_cuda_ms_mean": snap_total_cuda_mean,
                "online_index_cuda_ms_median": snap_total_cuda_median,
                "online_index_wall_ms_mean": snap_total_wall_mean,
                "online_index_wall_ms_median": snap_total_wall_median,
                "amortized_total_cuda_ms_mean": snap_total_cuda_mean,
                "amortized_total_cuda_ms_median": snap_total_cuda_median,
                "amortized_total_wall_ms_mean": snap_total_wall_mean,
                "amortized_total_wall_ms_median": snap_total_wall_median,
            },
            {
                **common,
                "method": "chunkkv",
                "score_cuda_ms_mean": snap_score_stats["cuda_ms_mean"],
                "score_cuda_ms_median": snap_score_stats["cuda_ms_median"],
                "score_wall_ms_mean": snap_score_stats["wall_ms_mean"],
                "score_wall_ms_median": snap_score_stats["wall_ms_median"],
                "topk_index_cuda_ms_mean": chunk_index_stats["cuda_ms_mean"],
                "topk_index_cuda_ms_median": chunk_index_stats["cuda_ms_median"],
                "topk_index_wall_ms_mean": chunk_index_stats["wall_ms_mean"],
                "topk_index_wall_ms_median": chunk_index_stats["wall_ms_median"],
                "summary_build_cuda_ms_mean": 0.0,
                "summary_build_cuda_ms_median": 0.0,
                "summary_build_wall_ms_mean": 0.0,
                "summary_build_wall_ms_median": 0.0,
                "online_index_cuda_ms_mean": chunk_total_cuda_mean,
                "online_index_cuda_ms_median": chunk_total_cuda_median,
                "online_index_wall_ms_mean": chunk_total_wall_mean,
                "online_index_wall_ms_median": chunk_total_wall_median,
                "amortized_total_cuda_ms_mean": chunk_total_cuda_mean,
                "amortized_total_cuda_ms_median": chunk_total_cuda_median,
                "amortized_total_wall_ms_mean": chunk_total_wall_mean,
                "amortized_total_wall_ms_median": chunk_total_wall_median,
            },
            {
                **common,
                "method": "blockwise",
                "score_cuda_ms_mean": block_online_stats["cuda_ms_mean"],
                "score_cuda_ms_median": block_online_stats["cuda_ms_median"],
                "score_wall_ms_mean": block_online_stats["wall_ms_mean"],
                "score_wall_ms_median": block_online_stats["wall_ms_median"],
                "topk_index_cuda_ms_mean": 0.0,
                "topk_index_cuda_ms_median": 0.0,
                "topk_index_wall_ms_mean": 0.0,
                "topk_index_wall_ms_median": 0.0,
                "summary_build_cuda_ms_mean": summary_build_stats["cuda_ms_mean"],
                "summary_build_cuda_ms_median": summary_build_stats["cuda_ms_median"],
                "summary_build_wall_ms_mean": summary_build_stats["wall_ms_mean"],
                "summary_build_wall_ms_median": summary_build_stats["wall_ms_median"],
                "online_index_cuda_ms_mean": block_online_stats["cuda_ms_mean"],
                "online_index_cuda_ms_median": block_online_stats["cuda_ms_median"],
                "online_index_wall_ms_mean": block_online_stats["wall_ms_mean"],
                "online_index_wall_ms_median": block_online_stats["wall_ms_median"],
                "amortized_total_cuda_ms_mean": block_amort_cuda_mean,
                "amortized_total_cuda_ms_median": block_amort_cuda_median,
                "amortized_total_wall_ms_mean": block_amort_wall_mean,
                "amortized_total_wall_ms_median": block_amort_wall_median,
            },
        ]
    )

    del q_window, keys, kv_query_states, scores, summary
    return rows


def write_readme() -> None:
    RESULT_README.write_text(
        f"""# {EXPERIMENT_NAME}

## Purpose

Compare sparse-index construction overhead for SnapKV, ChunkKV, and BlockWisePress.
The measurement includes score computation and top-k/index construction only; it excludes K/V gather.

## Model Shape

The benchmark loads `/Tan/model/Llama-3.1-8B-Instruct` weights by default.
It uses the real embedding table and every tested layer's Q/K projection weights to generate Q/K tensors before timing sparse-index logic.

## Artifacts

- `artifacts/sparse_index_overhead_summary.csv`
- `artifacts/sparse_index_overhead_layers.csv`
- `artifacts/metadata.json`
- Plan: `note/sparse_index_overhead_snapkv_chunkkv_blockwise_plan_zh.md`
"""
    )


def aggregate_layer_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    group_keys = [
        "sweep",
        "batch_size",
        "length",
        "compression_ratio",
        "reuse_steps",
        "method",
        "window_size",
        "chunk_length",
        "block_size",
        "block_q_window",
    ]
    numeric_average_fields = [
        "score_cuda_ms_mean",
        "score_cuda_ms_median",
        "score_wall_ms_mean",
        "score_wall_ms_median",
        "topk_index_cuda_ms_mean",
        "topk_index_cuda_ms_median",
        "topk_index_wall_ms_mean",
        "topk_index_wall_ms_median",
        "summary_build_cuda_ms_mean",
        "summary_build_cuda_ms_median",
        "summary_build_wall_ms_mean",
        "summary_build_wall_ms_median",
        "online_index_cuda_ms_mean",
        "online_index_cuda_ms_median",
        "online_index_wall_ms_mean",
        "online_index_wall_ms_median",
        "amortized_total_cuda_ms_mean",
        "amortized_total_cuda_ms_median",
        "amortized_total_wall_ms_mean",
        "amortized_total_wall_ms_median",
    ]

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        if "error" in row:
            key = tuple(row.get(field) for field in group_keys)
        else:
            key = tuple(row[field] for field in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for key, group_rows in grouped.items():
        aggregate = {field: value for field, value in zip(group_keys, key)}
        aggregate["layer_count"] = len({int(row["layer_idx"]) for row in group_rows if "layer_idx" in row})
        aggregate["layer_idx"] = "mean_all_layers"
        if any("error" in row for row in group_rows):
            aggregate["error"] = " | ".join(str(row.get("error")) for row in group_rows if "error" in row)
            aggregate_rows.append(aggregate)
            continue
        for field in numeric_average_fields:
            aggregate[field] = statistics.mean(float(row[field]) for row in group_rows)
        aggregate["peak_memory_mb"] = max(float(row["peak_memory_mb"]) for row in group_rows)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)

    num_heads, num_kv_heads, head_dim, model_metadata = load_llama_shape(args.model)
    module = FakeModule(FakeConfig(num_heads, num_kv_heads), head_dim=head_dim, layer_idx=0)
    print(
        f"[model] loading real projection weights from {args.model}"
        if not args.no_real_model_weights
        else "[model] synthetic tensors",
        flush=True,
    )
    weights = load_real_model(args, dtype, device)
    if weights is not None:
        print(f"[model] loaded real embedding/q_proj/k_proj weights on {device}: {torch.cuda.get_device_name(device)}", flush=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    write_readme()

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": EXPERIMENT_NAME,
        "model": model_metadata,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "args": vars(args),
        "notes": [
            "No K/V gather is included.",
            "BlockWisePress mean_values summary has been removed from the implementation.",
            "BlockWisePress multi_rep_keys summary has been removed from the implementation and benchmark.",
            "BlockWise amortized total = online_index + summary_build / reuse_steps.",
            "Q/K tensors are generated from real Llama-3.1-8B-Instruct embedding and layer projection weights unless --no-real-model-weights is set.",
        ],
        "real_model_weights_loaded": weights is not None,
        "loaded_weight_tensors": (
            ["model.embed_tokens.weight"]
            + [
                f"model.layers.{layer_idx}.self_attn.{proj}.weight"
                for layer_idx in (weights.layer_indices if weights is not None else [])
                for proj in ("q_proj", "k_proj")
            ]
        ),
        "layer_indices": weights.layer_indices if weights is not None else resolve_layer_indices(args, model_metadata),
        "layer_count": len(weights.layer_indices) if weights is not None else len(resolve_layer_indices(args, model_metadata)),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    all_rows: list[dict[str, object]] = []
    cases: list[tuple[str, int, int, float, int]] = []
    cases.extend(("length", 1, length, 0.5, 64) for length in args.lengths)
    cases.extend(("batch", batch_size, 8192, 0.5, 64) for batch_size in args.batch_sizes)
    cases.extend(("ratio", 1, 8192, ratio, 64) for ratio in args.ratios)
    cases.extend(("amortization", 1, 8192, 0.5, reuse_steps) for reuse_steps in args.reuse_steps)

    seen = set()
    unique_cases = []
    for case in cases:
        if case not in seen:
            unique_cases.append(case)
            seen.add(case)

    for sweep, batch_size, length, ratio, reuse_steps in unique_cases:
        if length <= args.window_size:
            continue
        layer_indices = weights.layer_indices if weights is not None else resolve_layer_indices(args, model_metadata)
        print(
            f"[start] sweep={sweep} B={batch_size} L={length} ratio={ratio} reuse={reuse_steps} layers={len(layer_indices)}",
            flush=True,
        )
        for layer_idx in layer_indices:
            try:
                rows = measure_case(
                    sweep=sweep,
                    batch_size=batch_size,
                    length=length,
                    compression_ratio=ratio,
                    reuse_steps=reuse_steps,
                    layer_idx=layer_idx,
                    args=args,
                    module=module,
                    weights=weights,
                    dtype=dtype,
                    device=device,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rows = [
                    {
                        "sweep": sweep,
                        "batch_size": batch_size,
                        "length": length,
                        "compression_ratio": ratio,
                        "reuse_steps": reuse_steps,
                        "layer_idx": layer_idx,
                        "method": method,
                        "error": f"OOM: {exc}",
                    }
                    for method in ("snapkv", "chunkkv", "blockwise")
                ]
            all_rows.extend(rows)
        print(f"[done] sweep={sweep} B={batch_size} L={length} ratio={ratio} reuse={reuse_steps}", flush=True)

    layer_fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with LAYER_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=layer_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    aggregate_rows = aggregate_layer_rows(all_rows)
    fieldnames = sorted({key for row in aggregate_rows for key in row.keys()})
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Wrote {LAYER_CSV}", flush=True)
    print(f"Wrote {SUMMARY_CSV}", flush=True)
    if args.force_exit:
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
