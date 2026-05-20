from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "scoring_overhead_snapkv_chunkkv"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
SUMMARY_CSV = ARTIFACTS_DIR / "scoring_overhead_summary.csv"
METADATA_JSON = ARTIFACTS_DIR / "metadata.json"

FLASH_ATTN_FUNC = None
FLASH_ATTN_IMPORT_ERROR: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SnapKV/ChunkKV scoring and index overhead against attention kernels.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--chunk-lengths", type=int, nargs="+", default=[20])
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-full-prefill", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def cuda_event_time_ms(fn: Callable[[], None], warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    values: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))

    values_sorted = sorted(values)
    return {
        "median": statistics.median(values_sorted),
        "p10": values_sorted[max(0, int(0.1 * (len(values_sorted) - 1)))],
        "p90": values_sorted[min(len(values_sorted) - 1, int(0.9 * (len(values_sorted) - 1)))],
    }


def setup_attention_backend() -> str:
    global FLASH_ATTN_FUNC
    global FLASH_ATTN_IMPORT_ERROR
    try:
        from flash_attn import flash_attn_func

        FLASH_ATTN_FUNC = flash_attn_func
        FLASH_ATTN_IMPORT_ERROR = None
        return "flash_attn.flash_attn_func"
    except Exception as exc:
        FLASH_ATTN_FUNC = None
        FLASH_ATTN_IMPORT_ERROR = repr(exc)
        return "torch.nn.functional.scaled_dot_product_attention forced FLASH_ATTENTION"


def attention_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    if FLASH_ATTN_FUNC is not None:
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        return FLASH_ATTN_FUNC(q_fa, k_fa, v_fa, causal=causal)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal,
                enable_gqa=q.shape[1] != k.shape[1],
            )
    except Exception:
        # Some PyTorch builds reject GQA under forced flash. Fall back to a pre-expanded
        # K/V view so the timed region still measures the attention call itself.
        groups = q.shape[1] // k.shape[1]
        k_rep = k.repeat_interleave(groups, dim=1)
        v_rep = v.repeat_interleave(groups, dim=1)
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=causal)


def snap_score_ops(
    q_window: torch.Tensor,
    keys: torch.Tensor,
    *,
    kernel_size: int,
    window_size: int,
) -> torch.Tensor:
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


def snap_topk_index_ops(
    scores: torch.Tensor,
    *,
    compression_ratio: float,
    head_dim: int,
) -> torch.Tensor:
    k_len = scores.shape[2]
    n_kept = int(k_len * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    return indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)


def chunkkv_index_ops(
    global_scores: torch.Tensor,
    *,
    chunk_length: int,
    compression_ratio: float,
) -> torch.Tensor:
    kv_len = global_scores.shape[2]
    num_complete_chunks = kv_len // chunk_length
    remaining_tokens = kv_len % chunk_length

    if num_complete_chunks == 0:
        n_kept = int(kv_len * (1 - compression_ratio))
        return global_scores.topk(n_kept, dim=-1).indices

    main_scores = global_scores[..., : num_complete_chunks * chunk_length]
    main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, chunk_length).mean(dim=-1)

    if remaining_tokens > 0:
        remaining_scores = global_scores[..., -remaining_tokens:]
        remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
        chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
    else:
        chunk_scores = main_chunk_scores

    total_chunks = num_complete_chunks + int(remaining_tokens > 0)
    n_chunks_kept = max(1, int(total_chunks * (1 - compression_ratio)))
    top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

    token_indices_per_batch = []
    for batch_idx in range(top_chunks.indices.shape[0]):
        batch_token_indices = []
        for chunk_idx in top_chunks.indices[batch_idx]:
            if chunk_idx < num_complete_chunks:
                start_idx = int(chunk_idx.item()) * chunk_length
                chunk_indices = torch.arange(start_idx, start_idx + chunk_length, device=global_scores.device)
            else:
                chunk_indices = torch.arange(num_complete_chunks * chunk_length, kv_len, device=global_scores.device)
            batch_token_indices.append(chunk_indices)
        token_indices_per_batch.append(torch.cat(batch_token_indices).sort()[0])

    return torch.stack(token_indices_per_batch, dim=0)


def fair_prefill_length(attention_pairs: int) -> int:
    return max(1, int((math.sqrt(1 + 8 * attention_pairs) - 1) // 2))


def summarize_ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator == 0:
        return None
    return numerator / denominator


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    attention_backend = setup_attention_backend()
    flash_attn_status = {
        "flash_attn_func": "available" if FLASH_ATTN_FUNC is not None else f"unavailable: {FLASH_ATTN_IMPORT_ERROR}",
    }

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "cuda_visible_devices": str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES")),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "attention_backend": attention_backend,
        "flash_attn_status": flash_attn_status,
        "args": vars(args),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    rows: list[dict[str, float | int | str | None]] = []
    for length in args.lengths:
        for window_size in args.window_sizes:
            if window_size >= length:
                continue
            for chunk_length in args.chunk_lengths:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

                score_attention_pairs = length * window_size
                prefill_len = fair_prefill_length(score_attention_pairs)
                decode_steps = window_size

                shape_q = (args.batch_size, args.query_heads, length, args.head_dim)
                shape_kv = (args.batch_size, args.kv_heads, length, args.head_dim)
                q = torch.randn(shape_q, device=device, dtype=dtype)
                k = torch.randn(shape_kv, device=device, dtype=dtype)
                v = torch.randn(shape_kv, device=device, dtype=dtype)
                q_window = q[:, :, -window_size:, :].contiguous()
                q_decode = q[:, :, -1:, :].contiguous()
                q_decode_window = q[:, :, -decode_steps:, :].contiguous()
                q_prefill = q[:, :, :prefill_len, :].contiguous()
                k_prefill = k[:, :, :prefill_len, :].contiguous()
                v_prefill = v[:, :, :prefill_len, :].contiguous()

                full_prefill_stats = None
                if not args.skip_full_prefill:
                    try:
                        full_prefill_stats = cuda_event_time_ms(
                        lambda: attention_kernel(q, k, v, causal=True),
                            args.warmup,
                            args.repeat,
                        )
                    except Exception as exc:
                        full_prefill_stats = {"median": None, "p10": None, "p90": None}
                        print(f"[warn] full prefill attention failed for L={length}: {exc}", flush=True)

                fair_prefill_stats = cuda_event_time_ms(
                    lambda: attention_kernel(q_prefill, k_prefill, v_prefill, causal=True),
                    args.warmup,
                    args.repeat,
                )
                decode_single_stats = cuda_event_time_ms(
                    lambda: attention_kernel(q_decode, k, v, causal=False),
                    args.warmup,
                    args.repeat,
                )
                decode_batched_stats = cuda_event_time_ms(
                    lambda: attention_kernel(q_decode_window, k, v, causal=False),
                    args.warmup,
                    args.repeat,
                )
                score_shape_fa_stats = cuda_event_time_ms(
                    lambda: attention_kernel(q_window, k, v, causal=False),
                    args.warmup,
                    args.repeat,
                )
                snap_stats = cuda_event_time_ms(
                    lambda: snap_score_ops(q_window, k, kernel_size=5, window_size=window_size),
                    args.warmup,
                    args.repeat,
                )

                scores = snap_score_ops(q_window, k, kernel_size=5, window_size=window_size)
                snap_topk_stats = cuda_event_time_ms(
                    lambda: snap_topk_index_ops(
                        scores,
                        compression_ratio=args.compression_ratio,
                        head_dim=args.head_dim,
                    ),
                    args.warmup,
                    args.repeat,
                )
                chunk_index_stats = cuda_event_time_ms(
                    lambda: chunkkv_index_ops(
                        scores,
                        chunk_length=chunk_length,
                        compression_ratio=args.compression_ratio,
                    ),
                    args.warmup,
                    args.repeat,
                )

                full_prefill_ms = None if full_prefill_stats is None else full_prefill_stats["median"]
                snap_ms = snap_stats["median"]
                snap_topk_ms = snap_topk_stats["median"]
                snap_total_ms = snap_ms + snap_topk_ms
                chunk_index_ms = chunk_index_stats["median"]
                chunk_total_ms = snap_ms + chunk_index_ms
                fair_prefill_pairs = prefill_len * (prefill_len + 1) // 2
                decode_pairs = decode_steps * length
                row = {
                    "length": length,
                    "window_size": window_size,
                    "chunk_length": chunk_length,
                    "compression_ratio": args.compression_ratio,
                    "dtype": args.dtype,
                    "score_attention_pairs": score_attention_pairs,
                    "fair_prefill_len": prefill_len,
                    "fair_prefill_attention_pairs": fair_prefill_pairs,
                    "decode_steps": decode_steps,
                    "decode_attention_pairs": decode_pairs,
                    "full_prefill_fa_ms_median": full_prefill_ms,
                    "full_prefill_fa_ms_p10": None if full_prefill_stats is None else full_prefill_stats["p10"],
                    "full_prefill_fa_ms_p90": None if full_prefill_stats is None else full_prefill_stats["p90"],
                    "fair_prefill_fa_ms_median": fair_prefill_stats["median"],
                    "fair_prefill_fa_ms_p10": fair_prefill_stats["p10"],
                    "fair_prefill_fa_ms_p90": fair_prefill_stats["p90"],
                    "decode_single_fa_ms_median": decode_single_stats["median"],
                    "decode_single_fa_ms_p10": decode_single_stats["p10"],
                    "decode_single_fa_ms_p90": decode_single_stats["p90"],
                    "decode_fair_batched_fa_ms_median": decode_batched_stats["median"],
                    "decode_fair_batched_fa_ms_p10": decode_batched_stats["p10"],
                    "decode_fair_batched_fa_ms_p90": decode_batched_stats["p90"],
                    "score_shape_fa_ms_median": score_shape_fa_stats["median"],
                    "score_shape_fa_ms_p10": score_shape_fa_stats["p10"],
                    "score_shape_fa_ms_p90": score_shape_fa_stats["p90"],
                    "snap_score_ms_median": snap_ms,
                    "snap_score_ms_p10": snap_stats["p10"],
                    "snap_score_ms_p90": snap_stats["p90"],
                    "snap_topk_index_ms_median": snap_topk_ms,
                    "snap_topk_index_ms_p10": snap_topk_stats["p10"],
                    "snap_topk_index_ms_p90": snap_topk_stats["p90"],
                    "snap_total_no_gather_ms_median": snap_total_ms,
                    "chunk_index_ms_median": chunk_index_ms,
                    "chunk_index_ms_p10": chunk_index_stats["p10"],
                    "chunk_index_ms_p90": chunk_index_stats["p90"],
                    "chunkkv_total_no_gather_ms_median": chunk_total_ms,
                    "snap_vs_fair_prefill_fa": summarize_ratio(snap_total_ms, fair_prefill_stats["median"]),
                    "snap_vs_decode_fair_batched_fa": summarize_ratio(snap_total_ms, decode_batched_stats["median"]),
                    "snap_vs_score_shape_fa": summarize_ratio(snap_total_ms, score_shape_fa_stats["median"]),
                    "chunk_vs_fair_prefill_fa": summarize_ratio(chunk_total_ms, fair_prefill_stats["median"]),
                    "chunk_vs_decode_fair_batched_fa": summarize_ratio(chunk_total_ms, decode_batched_stats["median"]),
                    "chunk_vs_score_shape_fa": summarize_ratio(chunk_total_ms, score_shape_fa_stats["median"]),
                    "decode_single_times_window_ms": decode_single_stats["median"] * window_size,
                    "snap_vs_decode_single_times_window": summarize_ratio(
                        snap_total_ms, decode_single_stats["median"] * window_size
                    ),
                    "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
                }
                rows.append(row)
                raw_path = RAW_DIR / f"L{length}_W{window_size}_C{chunk_length}.json"
                raw_path.write_text(json.dumps(row, indent=2, ensure_ascii=False))
                print(
                    "[bench] "
                    f"L={length} W={window_size} C={chunk_length} "
                    f"snap_no_gather={snap_total_ms:.3f}ms "
                    f"chunk_no_gather={chunk_total_ms:.3f}ms "
                    f"fair_prefill={fair_prefill_stats['median']:.3f}ms "
                    f"decode_batched={decode_batched_stats['median']:.3f}ms",
                    flush=True,
                )

                del q, k, v, q_window, q_decode, q_decode_window, q_prefill, k_prefill, v_prefill, scores

    if rows:
        with SUMMARY_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"[done] wrote {SUMMARY_CSV}", flush=True)
    print(f"[done] wrote {METADATA_JSON}", flush=True)


if __name__ == "__main__":
    main()
