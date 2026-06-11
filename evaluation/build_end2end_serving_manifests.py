from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "end2end_serving_kvcore_vllm_infinigen_longreq"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "evaluation"
    / "results"
    / "experiments"
    / EXPERIMENT_NAME
    / "artifacts"
    / "manifests"
)
DEFAULT_RESULT_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
DEFAULT_PG19_PARQUET = Path(
    "/home/bzx/Tan/dataset/pg19-test/data/test-00000-of-00001-29a571947c0b5ccc.parquet"
)
DEFAULT_LONGBENCH_ROOT = Path("/home/bzx/Tan/dataset/LongBench")

MODEL_PATHS = {
    "llama31_8b_instruct": "/home/bzx/Tan/model/Llama-3.1-8B-Instruct",
    "qwen3_8b": "/home/bzx/Tan/model/Qwen3-8B",
    "mistral_7b_instruct_v03": "/home/bzx/Tan/model/Mistral-7B-Instruct-v0.3",
}

LONGBENCH_CONTEXT_PREFIX = {
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
    ),
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\n",
}

LONGBENCH_QUESTION_TEMPLATE = {
    "gov_report": "Now, write a one-page summary of the report.\n\n",
    "qmsum": "Query: {input}\n",
    "multi_news": "Now, write a one-page summary of all the news.\n\n",
}

LONGBENCH_ANSWER_PREFIX = {
    "gov_report": "Summary:",
    "qmsum": "Answer:",
    "multi_news": "Summary:",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    path: str


@dataclass(frozen=True)
class PromptSample:
    source_dataset: str
    source_id: str
    prompt: str
    prompt_token_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified end-to-end serving workload manifests.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS))
    parser.add_argument("--workloads", nargs="+", default=["pg19", "longbench"])
    parser.add_argument("--input-lens", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument("--output-lens", nargs="+", type=int, default=[1024, 2048, 4096, 6144, 8192])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 32])
    parser.add_argument("--num-requests-per-point", type=int, default=32)
    parser.add_argument(
        "--measured-batches-per-point",
        type=int,
        default=None,
        help="If set, requests per point becomes max(num_requests_per_point, batch_size * this value).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--pg19-parquet", type=Path, default=DEFAULT_PG19_PARQUET)
    parser.add_argument("--longbench-root", type=Path, default=DEFAULT_LONGBENCH_ROOT)
    parser.add_argument("--longbench-tasks", nargs="+", default=["gov_report", "qmsum", "multi_news"])
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def slug_len(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def stable_short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def get_git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return "unknown"


def load_tokenizer(model: ModelSpec):
    tokenizer_json = Path(model.path) / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"Missing tokenizer.json for {model.key}: {tokenizer_json}")
    return Tokenizer.from_file(str(tokenizer_json))


def encode_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def decode_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def trim_to_exact_token_len(tokenizer, text: str, token_len: int) -> str:
    ids = encode_ids(tokenizer, text)
    if len(ids) < token_len:
        raise ValueError(f"Prompt shorter than requested length: actual={len(ids)} requested={token_len}")
    return decode_ids(tokenizer, ids[:token_len])


def load_pg19_books(path: Path, tokenizer, min_tokens: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(path).reset_index(drop=True)
    books: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        token_ids = encode_ids(tokenizer, str(row["text"]))
        if len(token_ids) < min_tokens:
            continue
        books.append(
            {
                "row_idx": int(row_idx),
                "title": str(row.get("short_book_title", row_idx)),
                "url": str(row.get("url", "")),
                "token_ids": token_ids,
                "source_token_count": len(token_ids),
            }
        )
    return books


def build_pg19_samples(
    tokenizer,
    pg19_books: list[dict[str, Any]],
    input_len: int,
    num_requests: int,
    seed: int,
) -> list[PromptSample]:
    candidates = [book for book in pg19_books if len(book["token_ids"]) >= input_len]
    if not candidates:
        raise RuntimeError(f"No PG19 books have at least {input_len} tokens.")

    rng = random.Random(seed + input_len)
    samples: list[PromptSample] = []
    for sample_idx in range(num_requests):
        book = candidates[sample_idx % len(candidates)]
        available = len(book["token_ids"]) - input_len
        offset = 0 if available == 0 else rng.randrange(0, available + 1)
        prompt_ids = book["token_ids"][offset : offset + input_len]
        prompt = decode_ids(tokenizer, prompt_ids)
        source_id = f"pg19-test:row{book['row_idx']}:offset{offset}:title={book['title']}"
        samples.append(
            PromptSample(
                source_dataset="pg19",
                source_id=source_id,
                prompt=prompt,
                prompt_token_len=len(prompt_ids),
            )
        )
    return samples


def load_longbench_task(root: Path, task: str):
    return load_dataset(str(root), task, split="test", trust_remote_code=True)


def format_longbench_prompt(task: str, row: dict[str, Any]) -> str:
    context = LONGBENCH_CONTEXT_PREFIX[task].format(**row)
    question = LONGBENCH_QUESTION_TEMPLATE[task].format(**row)
    answer_prefix = LONGBENCH_ANSWER_PREFIX.get(task, "")
    return context + question + answer_prefix


def build_longbench_pool(root: Path, tasks: list[str], tokenizer, max_input_len: int) -> list[PromptSample]:
    pool: list[PromptSample] = []
    for task in tasks:
        if task not in LONGBENCH_CONTEXT_PREFIX:
            raise ValueError(f"LongBench task template is not configured: {task}")
        dataset = load_longbench_task(root, task)
        for row_idx, row in enumerate(dataset):
            row_dict = dict(row)
            full_prompt = format_longbench_prompt(task, row_dict)
            ids = encode_ids(tokenizer, full_prompt)
            if len(ids) < 1024:
                continue
            prompt = decode_ids(tokenizer, ids[: min(len(ids), max_input_len)])
            pool.append(
                PromptSample(
                    source_dataset=f"longbench:{task}",
                    source_id=f"{task}:{row_dict.get('_id', row_idx)}",
                    prompt=prompt,
                    prompt_token_len=min(len(ids), max_input_len),
                )
            )
    return pool


def build_longbench_samples(
    tokenizer,
    pool: list[PromptSample],
    input_len: int,
    num_requests: int,
    seed: int,
) -> list[PromptSample]:
    candidates: list[PromptSample] = []
    for sample in pool:
        ids = encode_ids(tokenizer, sample.prompt)
        if len(ids) >= input_len:
            prompt = decode_ids(tokenizer, ids[:input_len])
            candidates.append(
                PromptSample(
                    source_dataset=sample.source_dataset,
                    source_id=sample.source_id,
                    prompt=prompt,
                    prompt_token_len=input_len,
                )
            )
    if not candidates:
        raise RuntimeError(f"No LongBench prompts have at least {input_len} tokens.")

    rng = random.Random(seed + input_len + 17)
    selected = [candidates[i % len(candidates)] for i in range(num_requests)]
    rng.shuffle(selected)
    return selected


def manifest_filename(
    model_key: str,
    workload: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    seed: int,
) -> str:
    return (
        f"{model_key}__{workload}__in{slug_len(input_len)}_out{slug_len(output_len)}"
        f"__bs{batch_size}__seed{seed}.jsonl"
    )


def requests_for_batch_size(batch_size: int, min_requests: int, measured_batches: int | None) -> int:
    if measured_batches is None:
        return min_requests
    return max(min_requests, batch_size * measured_batches)


def make_request(
    sample: PromptSample,
    model: ModelSpec,
    tokenizer_path: str,
    workload: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    seed: int,
    repeat_id: int,
    repeat_count: int,
    request_idx: int,
) -> dict[str, Any]:
    request_id = (
        f"{workload}_{model.key}_in{input_len}_out{output_len}_bs{batch_size}"
        f"_seed{seed}_rep{repeat_id}_{request_idx:06d}_{stable_short_hash(sample.source_id)}"
    )
    return {
        "request_id": request_id,
        "model_key": model.key,
        "model_path": model.path,
        "tokenizer_path": tokenizer_path,
        "source_dataset": sample.source_dataset,
        "source_id": sample.source_id,
        "prompt": sample.prompt,
        "prompt_token_len": sample.prompt_token_len,
        "target_output_len": output_len,
        "max_new_tokens": output_len,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "ignore_eos": True,
            "seed": seed,
        },
        "workload": {
            "arrival_mode": "closed_loop_batch",
            "batch_size": batch_size,
            "input_len_bucket": input_len,
            "output_len_bucket": output_len,
            "repeat_id": repeat_id,
            "repeat_count": repeat_count,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_text(path: Path, text: str, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_experiment_readme(result_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# End-to-End Serving KVCore vs vLLM vs InfiniGen

## 实验目的

为 KVCore、vLLM、InfiniGen 生成统一端到端 serving workload manifests，保证三个系统后续读取同一批请求并将结果回填到本目录。

## 运行脚本

- `evaluation/build_end2end_serving_manifests.py`

## 数据集

- PG19 continuation: `{args.pg19_parquet}`
- LongBench selected tasks: `{args.longbench_root}`, tasks `{", ".join(args.longbench_tasks)}`

## 方法

每个 manifest 固定模型、tokenizer、prompt、输入长度 bucket、输出长度 bucket、batch size、seed、repeat count。系统 runner 只读取 manifest，不重新采样数据。

## Sweep

- models: `{", ".join(args.models)}`
- workloads: `{", ".join(args.workloads)}`
- input_lens: `{", ".join(map(str, args.input_lens))}`
- output_lens: `{", ".join(map(str, args.output_lens))}`
- batch_sizes: `{", ".join(map(str, args.batch_sizes))}`
- seeds: `{", ".join(map(str, args.seeds))}`
- repeat_count: `{args.repeat_count}`
- num_requests_per_point: `{args.num_requests_per_point}`
- measured_batches_per_point: `{args.measured_batches_per_point}`
- GPU target: single NVIDIA RTX A6000

## 产物位置

- manifests: `{output_dir}`
- configs: `{result_dir / "artifacts" / "configs"}`
- raw results target: `{result_dir / "artifacts" / "raw"}`
- logs target: `{result_dir / "artifacts" / "logs"}`
- summaries target: `{result_dir / "artifacts" / "summaries"}`

## 推荐优先查看

- `artifacts/configs/manifest_build_summary.json`
- `artifacts/manifests/`
- `note/end_to_end_system_performance_experiment_plan_zh.md`
"""
    write_text(result_dir / "README.md", text)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.result_dir
    (result_dir / "artifacts" / "configs").mkdir(parents=True, exist_ok=True)
    for subdir in ["raw/kvcore", "raw/vllm", "raw/infinigen", "logs/kvcore", "logs/vllm", "logs/infinigen", "summaries", "environment"]:
        (result_dir / "artifacts" / subdir).mkdir(parents=True, exist_ok=True)

    max_input_len = max(args.input_lens)
    summary: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kvpress_study_git_commit": get_git_commit(REPO_ROOT),
        "output_dir": str(args.output_dir),
        "models": args.models,
        "workloads": args.workloads,
        "input_lens": args.input_lens,
        "output_lens": args.output_lens,
        "batch_sizes": args.batch_sizes,
        "seeds": args.seeds,
        "repeat_count": args.repeat_count,
        "num_requests_per_point": args.num_requests_per_point,
        "measured_batches_per_point": args.measured_batches_per_point,
        "manifests": [],
        "skipped": [],
    }
    max_requests_per_point = max(
        requests_for_batch_size(batch_size, args.num_requests_per_point, args.measured_batches_per_point)
        for batch_size in args.batch_sizes
    )

    for model_key in args.models:
        if model_key not in MODEL_PATHS:
            raise ValueError(f"Unknown model key: {model_key}. Known: {sorted(MODEL_PATHS)}")
        model = ModelSpec(key=model_key, path=MODEL_PATHS[model_key])
        print(f"[manifest] loading tokenizer model={model.key} path={model.path}", flush=True)
        tokenizer = load_tokenizer(model)

        pg19_books = None
        longbench_pool = None
        if "pg19" in args.workloads:
            print(f"[manifest] tokenizing PG19 for model={model.key}", flush=True)
            pg19_books = load_pg19_books(args.pg19_parquet, tokenizer, min_tokens=max_input_len)
            summary.setdefault("pg19_books", {})[model.key] = len(pg19_books)
        if "longbench" in args.workloads:
            print(f"[manifest] loading LongBench pool for model={model.key}", flush=True)
            longbench_pool = build_longbench_pool(args.longbench_root, args.longbench_tasks, tokenizer, max_input_len)
            summary.setdefault("longbench_prompts", {})[model.key] = len(longbench_pool)

        for workload in args.workloads:
            for seed in args.seeds:
                for input_len in args.input_lens:
                    try:
                        if workload == "pg19":
                            assert pg19_books is not None
                            samples = build_pg19_samples(tokenizer, pg19_books, input_len, max_requests_per_point, seed)
                        elif workload == "longbench":
                            assert longbench_pool is not None
                            samples = build_longbench_samples(tokenizer, longbench_pool, input_len, max_requests_per_point, seed)
                        else:
                            raise ValueError(f"Unsupported workload: {workload}")
                    except RuntimeError as exc:
                        summary["skipped"].append(
                            {
                                "model_key": model.key,
                                "workload": workload,
                                "input_len": input_len,
                                "seed": seed,
                                "reason": str(exc),
                            }
                        )
                        print(
                            f"[manifest] skip model={model.key} workload={workload} input_len={input_len}: {exc}",
                            flush=True,
                        )
                        continue

                    for output_len in args.output_lens:
                        for batch_size in args.batch_sizes:
                            num_requests = requests_for_batch_size(
                                batch_size, args.num_requests_per_point, args.measured_batches_per_point
                            )
                            for repeat_id in range(args.repeat_count):
                                rows = [
                                    make_request(
                                        sample=sample,
                                        model=model,
                                        tokenizer_path=model.path,
                                        workload=workload,
                                        input_len=input_len,
                                        output_len=output_len,
                                        batch_size=batch_size,
                                        seed=seed,
                                        repeat_id=repeat_id,
                                        repeat_count=args.repeat_count,
                                        request_idx=request_idx,
                                    )
                                    for request_idx, sample in enumerate(samples[:num_requests])
                                ]
                                filename = manifest_filename(model.key, workload, input_len, output_len, batch_size, seed)
                                if args.repeat_count > 1:
                                    filename = filename.removesuffix(".jsonl") + f"__rep{repeat_id}.jsonl"
                                path = args.output_dir / filename
                                existed = path.exists()
                                write_jsonl(path, rows, overwrite=args.overwrite)
                                summary["manifests"].append(
                                    {
                                        "path": str(path),
                                        "model_key": model.key,
                                        "workload": workload,
                                        "input_len": input_len,
                                        "output_len": output_len,
                                        "batch_size": batch_size,
                                        "seed": seed,
                                        "repeat_id": repeat_id,
                                        "repeat_count": args.repeat_count,
                                        "measured_batches_per_point": args.measured_batches_per_point,
                                        "num_requests": len(rows),
                                        "status": "overwritten" if existed and args.overwrite else "exists" if existed else "created",
                                    }
                                )

    summary_name = "manifest_build_summary.json"
    if args.output_dir.name != "manifests":
        summary_name = f"manifest_build_summary_{args.output_dir.name}.json"
    summary_path = result_dir / "artifacts" / "configs" / summary_name
    write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False))
    write_experiment_readme(result_dir, args.output_dir, args)
    print(f"[manifest] wrote summary: {summary_path}", flush=True)
    print(f"[manifest] total manifest entries: {len(summary['manifests'])}", flush=True)


if __name__ == "__main__":
    main()
