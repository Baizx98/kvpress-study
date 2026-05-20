from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/Tan/dataset/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/Tan/dataset/hf_home/datasets")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Tan/dataset/hf_home/hub")

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import DynamicCache

from benchmarks.pg19.create_huggingface_dataset import load_pg19_source_dataframe
from evaluate import EvaluationConfig, EvaluationRunner
from kvpress import DecodingPress


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "pg19_dense_position_ppl_llama31_8b_snapkv_chunkkv_blockwise_ratio50"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACTS_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
LOG_DIR = ARTIFACTS_DIR / "logs"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
MODEL = os.environ.get("MODEL", "/Tan/model/Llama-3.1-8B-Instruct")
PG19_SOURCE_DATASET = os.environ.get("PG19_SOURCE_DATASET", "/Tan/dataset/pg19-test")

METHODS: dict[str, dict[str, object]] = {
    "snapkv": {
        "press_name": "snapkv",
    },
    "chunkkv": {
        "press_name": "chunkkv",
        "block_size": 16,
    },
    "blockwise": {
        "press_name": "block_wise",
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
    },
    "no_press": {
        "press_name": "no_press",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense PG19 next-token PPL by token position.")
    parser.add_argument("--methods", nargs="+", default=["snapkv", "chunkkv", "blockwise"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical_device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--target_tokens", type=int, default=256)
    parser.add_argument("--include_first_target", action="store_true")
    parser.add_argument("--start", type=int, default=1024)
    parser.add_argument("--end", type=int, default=32768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--max_books", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--pg19_source_dataset", default=PG19_SOURCE_DATASET)
    parser.add_argument("--include_no_press", action="store_true")
    parser.add_argument("--run_tag", default="full", help="Subdirectory under artifacts/ for this run.")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def token_lengths(start: int, end: int, stride: int) -> list[int]:
    if start <= 0 or end < start or stride <= 0:
        raise ValueError(f"Invalid token range: start={start}, end={end}, stride={stride}")
    values = list(range(start, end + 1, stride))
    if values[-1] != end:
        values.append(end)
    return values


def select_books(source_df: pd.DataFrame, fraction: float, max_books: int | None, seed: int) -> pd.DataFrame:
    df = source_df.reset_index(drop=True)
    if fraction < 1.0:
        sample_n = max(1, int(round(len(df) * fraction)))
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    if max_books is not None and len(df) > max_books:
        df = df.sample(n=max_books, random_state=seed)
    return df.reset_index(drop=True)


def tokenize_books(df: pd.DataFrame, tokenizer, min_length: int) -> list[dict[str, object]]:
    books: list[dict[str, object]] = []
    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing PG19 books"):
        input_ids = tokenizer(
            row["text"],
            add_special_tokens=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        if len(input_ids) <= min_length:
            continue
        books.append(
            {
                "book_id": str(row_idx),
                "short_book_title": str(row.get("short_book_title", row_idx)),
                "input_ids": input_ids,
                "source_token_count": len(input_ids),
            }
        )
    return books


def completed_lengths(path: Path) -> set[int]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "token_length" not in df:
        return set()
    return {int(value) for value in df["token_length"].tolist()}


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.inference_mode()
def eval_one_position(
    runner: EvaluationRunner,
    books: Iterable[dict[str, object]],
    k: int,
    target_tokens: int,
    include_first_target: bool,
) -> tuple[float, int, float, int]:
    assert runner.pipeline is not None
    model = runner.pipeline.model
    total_nll = 0.0
    total_tokens = 0
    total_nll_all = 0.0
    total_tokens_all = 0
    perform_prefill_compression = runner.press is not None and not isinstance(runner.press, DecodingPress)

    for book in books:
        input_ids = book["input_ids"]
        assert isinstance(input_ids, list)
        if len(input_ids) <= k + target_tokens:
            continue

        context_ids = torch.tensor([input_ids[:k]], dtype=torch.long, device=model.device)
        target_ids = torch.tensor(
            [input_ids[k : k + target_tokens]],
            dtype=torch.long,
            device=model.device,
        )
        cache = DynamicCache()

        with runner.press(model) if perform_prefill_compression else contextlib.nullcontext():
            context_outputs = model(
                input_ids=context_ids,
                past_key_values=cache,
                logits_to_keep=1,
            )

        first_nll = F.cross_entropy(
            context_outputs.logits[:, -1, :],
            target_ids[:, 0],
            reduction="sum",
        )
        total_nll_all += float(first_nll.item())
        total_tokens_all += 1
        if include_first_target:
            total_nll += float(first_nll.item())
            total_tokens += 1

        if target_ids.shape[1] > 1:
            position_ids = torch.arange(
                k,
                k + target_ids.shape[1] - 1,
                device=model.device,
            ).unsqueeze(0)
            continuation_outputs = model(
                input_ids=target_ids[:, :-1],
                past_key_values=cache,
                position_ids=position_ids,
            )
            continuation_nll = F.cross_entropy(
                continuation_outputs.logits.reshape(-1, continuation_outputs.logits.shape[-1]),
                target_ids[:, 1:].reshape(-1),
                reduction="sum",
            )
            continuation_count = target_ids.shape[1] - 1
            total_nll += float(continuation_nll.item())
            total_tokens += continuation_count
            total_nll_all += float(continuation_nll.item())
            total_tokens_all += continuation_count

            del continuation_outputs, continuation_nll, position_ids

        del context_ids, target_ids, cache, context_outputs, first_nll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if total_tokens == 0:
        return math.nan, 0, math.nan, 0
    return total_nll / total_tokens, total_tokens, total_nll_all / max(total_tokens_all, 1), total_tokens_all


def build_config(args: argparse.Namespace, method_key: str) -> EvaluationConfig:
    method_cfg = dict(METHODS[method_key])
    return EvaluationConfig(
        dataset="pg19",
        model=args.model,
        device=args.device,
        compression_ratio=args.compression_ratio,
        pg19_target_tokens=args.target_tokens,
        pg19_source_dataset=args.pg19_source_dataset,
        fraction=args.fraction,
        max_context_length=args.end,
        output_dir=str(RAW_DIR / method_key),
        log_level=args.log_level,
        model_kwargs={"torch_dtype": "auto"},
        seed=args.seed,
        **method_cfg,
    )


def run_method(args: argparse.Namespace, method_key: str, lengths: list[int]) -> Path:
    if method_key not in METHODS:
        raise KeyError(f"Unknown method: {method_key}. Available: {sorted(METHODS)}")

    method_dir = RAW_DIR / method_key
    method_dir.mkdir(parents=True, exist_ok=True)
    csv_path = method_dir / "per_position_metrics.csv"
    config_path = method_dir / "config.yaml"
    summary_path = method_dir / "summary.json"

    config = build_config(args, method_key)
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")

    runner = EvaluationRunner(config)
    runner._setup_press()
    runner._setup_model_pipeline()
    assert runner.pipeline is not None

    source_df = load_pg19_source_dataframe(dataset_id=args.pg19_source_dataset, split="test")
    selected_df = select_books(source_df, args.fraction, args.max_books, args.seed)
    books = tokenize_books(selected_df, runner.pipeline.tokenizer, min(lengths) + args.target_tokens)
    if not books:
        raise RuntimeError("No PG19 books are long enough for the requested token lengths.")

    (method_dir / "books.json").write_text(
        json.dumps(
            [
                {
                    "book_id": item["book_id"],
                    "short_book_title": item["short_book_title"],
                    "source_token_count": item["source_token_count"],
                }
                for item in books
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    fields = [
        "method",
        "token_length",
        "subword_ppl",
        "avg_nll",
        "subword_ppl_all_targets",
        "avg_nll_all_targets",
        "target_tokens_scored",
        "target_tokens_scored_all",
        "num_books",
        "target_tokens",
        "include_first_target",
        "compression_ratio",
        "stride",
        "model",
        "pg19_source_dataset",
        "physical_device",
    ]
    done = completed_lengths(csv_path)
    for k in tqdm(lengths, desc=f"Running {method_key} dense PG19 PPL"):
        if k in done:
            continue
        avg_nll, token_count, avg_nll_all, token_count_all = eval_one_position(
            runner,
            books,
            k,
            target_tokens=args.target_tokens,
            include_first_target=args.include_first_target,
        )
        num_books = token_count_all // args.target_tokens if args.target_tokens else 0
        ppl = math.exp(avg_nll) if num_books else math.nan
        ppl_all = math.exp(avg_nll_all) if num_books else math.nan
        append_csv_row(
            csv_path,
            fields,
            {
                "method": method_key,
                "token_length": k,
                "subword_ppl": round(ppl, 6) if num_books else "",
                "avg_nll": round(avg_nll, 8) if num_books else "",
                "subword_ppl_all_targets": round(ppl_all, 6) if num_books else "",
                "avg_nll_all_targets": round(avg_nll_all, 8) if num_books else "",
                "target_tokens_scored": token_count,
                "target_tokens_scored_all": token_count_all,
                "num_books": num_books,
                "target_tokens": args.target_tokens,
                "include_first_target": args.include_first_target,
                "compression_ratio": config.compression_ratio,
                "stride": args.stride,
                "model": args.model,
                "pg19_source_dataset": args.pg19_source_dataset,
                "physical_device": args.physical_device,
            },
        )

    summary_path.write_text(
        json.dumps(
            {
                "method": method_key,
                "csv_path": str(csv_path),
                "num_positions": len(lengths),
                "num_books": len(books),
                "min_book_tokens": min(int(item["source_token_count"]) for item in books),
                "max_book_tokens": max(int(item["source_token_count"]) for item in books),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path


def aggregate_outputs(methods: list[str]) -> Path:
    frames = []
    for method in methods:
        path = RAW_DIR / method / "per_position_metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("No per-position metrics found to aggregate.")
    df = pd.concat(frames, ignore_index=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "pg19_dense_position_ppl_metrics.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    global ARTIFACTS_DIR, RAW_DIR, LOG_DIR

    args = parse_args()
    if args.physical_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_device)

    if args.run_tag:
        ARTIFACTS_DIR = RESULT_ROOT / "artifacts" / args.run_tag
        RAW_DIR = ARTIFACTS_DIR / "raw"
        LOG_DIR = ARTIFACTS_DIR / "logs"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    methods = list(args.methods)
    if args.include_no_press and "no_press" not in methods:
        methods.append("no_press")
    lengths = token_lengths(args.start, args.end, args.stride)

    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "methods": methods,
        "token_lengths": lengths,
        "args": vars(args),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (ARTIFACTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for method in methods:
        run_method(args, method, lengths)
    aggregate_outputs(methods)
    return 0


if __name__ == "__main__":
    sys.exit(main())
