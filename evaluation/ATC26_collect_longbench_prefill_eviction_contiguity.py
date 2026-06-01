from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/Tan/dataset/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/Tan/dataset/hf_home/datasets")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Tan/dataset/hf_home/hub")

import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvpress import SnapKVPress  # noqa: E402


EXPERIMENT_NAME = "ATC26_longbench_prefill_eviction_contiguity"
RESULT_ROOT = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACT_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACT_DIR / "raw"
SCORE_DIR = ARTIFACT_DIR / "scores"
RAW_JSONL = RAW_DIR / "ATC26_eviction_contiguity_raw.jsonl"
SUMMARY_CSV = ARTIFACT_DIR / "ATC26_eviction_contiguity_summary.csv"
SUMMARY_JSON = ARTIFACT_DIR / "ATC26_eviction_contiguity_summary.json"
MANIFEST_JSON = ARTIFACT_DIR / "ATC26_eviction_contiguity_manifest.json"

MODELS = {
    "llama31_8b_instruct": "/Tan/model/Llama-3.1-8B-Instruct",
    "mistral_7b_instruct_v03": "/Tan/model/Mistral-7B-Instruct-v0.3",
    "qwen3_8b": "/Tan/model/Qwen3-8B",
}


@dataclass
class SampleSpec:
    dataset_name: str
    dataset_row_index: int
    sample_index: int
    sample_id: str
    question: str
    context: str
    answers: Any
    context_sha1: str
    context_tokens: int


class TraceSnapKVScorePress(SnapKVPress):
    """Collect token-level SnapKV-style scores without modifying the KV cache."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.layer_scores: dict[int, torch.Tensor] = {}

    def compress(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[1] <= self.window_size:
            return keys, values
        layer_idx = int(getattr(module, "layer_idx", len(self.layer_scores)))
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        self.layer_scores[layer_idx] = scores.detach().float().cpu()
        return keys, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="llama31_8b_instruct")
    parser.add_argument("--datasets", default="hotpotqa,multifieldqa_en,qasper,gov_report")
    parser.add_argument("--ratios", default="0.5")
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-context-length", type=int, default=16384)
    parser.add_argument("--min-context-tokens", type=int, default=4096)
    parser.add_argument("--q-window-size", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--recent-tokens", type=int, default=64)
    parser.add_argument("--block-sizes", default="16,32,64")
    parser.add_argument("--random-repeats", type=int, default=100)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_layers(model: Any) -> int:
    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return len(language_model.layers)


def row_question(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("input") or "")


def row_answers(row: dict[str, Any]) -> Any:
    return row.get("answers", row.get("answer"))


def parse_explicit_indices(sample_indices: str) -> dict[str, list[int]]:
    parsed: dict[str, list[int]] = {}
    if not sample_indices:
        return parsed
    for part in sample_indices.split(";"):
        if not part.strip():
            continue
        name, values = part.split(":", 1)
        parsed[name.strip()] = [int(x) for x in values.split(",") if x.strip()]
    return parsed


def select_samples(
    tokenizer: Any,
    dataset_names: list[str],
    samples_per_dataset: int,
    seed: int,
    explicit_indices: str,
    min_context_tokens: int,
    max_context_length: int,
) -> list[SampleSpec]:
    rng = random.Random(seed)
    explicit = parse_explicit_indices(explicit_indices)
    samples: list[SampleSpec] = []
    for dataset_name in dataset_names:
        dataset = load_dataset("Xnhyacinth/LongBench", name=dataset_name, split="test")
        if dataset_name in explicit:
            candidates = explicit[dataset_name]
        else:
            candidates = list(range(len(dataset)))
            rng.shuffle(candidates)

        chosen = 0
        for row_idx in candidates:
            row = dict(dataset[int(row_idx)])
            context = str(row["context"])
            token_count = len(tokenizer.encode(context, add_special_tokens=False))
            effective_tokens = min(token_count, max_context_length)
            if effective_tokens < min_context_tokens:
                continue
            samples.append(
                SampleSpec(
                    dataset_name=dataset_name,
                    dataset_row_index=int(row_idx),
                    sample_index=chosen,
                    sample_id=str(row.get("_id", row_idx)),
                    question=row_question(row),
                    context=context,
                    answers=row_answers(row),
                    context_sha1=hashlib.sha1(context.encode("utf-8")).hexdigest(),
                    context_tokens=token_count,
                )
            )
            chosen += 1
            if chosen >= samples_per_dataset:
                break
        if chosen < samples_per_dataset:
            raise RuntimeError(
                f"Only selected {chosen}/{samples_per_dataset} samples for {dataset_name} "
                f"with min_context_tokens={min_context_tokens}."
            )
    return samples


def run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in mask.astype(bool).tolist():
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def run_stats(evicted: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    lengths = run_lengths(evicted[valid])
    evicted_count = int(evicted[valid].sum())
    if not lengths:
        return {
            "evicted_count": evicted_count,
            "num_evicted_runs": 0,
            "mean_evicted_run_length": 0.0,
            "median_evicted_run_length": 0.0,
            "p90_evicted_run_length": 0.0,
            "max_evicted_run_length": 0.0,
            "evicted_tokens_per_run": 0.0,
        }
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "evicted_count": evicted_count,
        "num_evicted_runs": int(len(lengths)),
        "mean_evicted_run_length": float(arr.mean()),
        "median_evicted_run_length": float(np.median(arr)),
        "p90_evicted_run_length": float(np.percentile(arr, 90)),
        "max_evicted_run_length": float(arr.max()),
        "evicted_tokens_per_run": float(evicted_count / len(lengths)),
    }


def build_masks(scores: np.ndarray, ratio: float, sink_tokens: int, recent_tokens: int) -> dict[str, np.ndarray | int]:
    k_len = int(scores.shape[0])
    protected = np.zeros(k_len, dtype=bool)
    protected[: min(sink_tokens, k_len)] = True
    if recent_tokens > 0:
        protected[max(0, k_len - recent_tokens) :] = True

    keep_budget = int(k_len * (1.0 - ratio))
    keep_budget = max(keep_budget, int(protected.sum()))
    keep = protected.copy()
    selectable = np.flatnonzero(~protected)
    additional = max(0, keep_budget - int(keep.sum()))
    if additional > 0 and selectable.size > 0:
        additional = min(additional, selectable.size)
        selected = selectable[np.argpartition(scores[selectable], -additional)[-additional:]]
        keep[selected] = True
    evicted = ~keep
    valid = ~protected
    return {
        "kept": keep,
        "evicted": evicted,
        "protected": protected,
        "valid": valid,
        "keep_budget": keep_budget,
    }


def random_run_stats(
    rng: np.random.Generator,
    evicted_count: int,
    protected: np.ndarray,
    repeats: int,
) -> dict[str, float]:
    valid_indices = np.flatnonzero(~protected)
    values = []
    for _ in range(repeats):
        evicted = np.zeros_like(protected, dtype=bool)
        chosen = rng.choice(valid_indices, size=evicted_count, replace=False)
        evicted[chosen] = True
        values.append(run_stats(evicted, ~protected))
    keys = [
        "num_evicted_runs",
        "mean_evicted_run_length",
        "median_evicted_run_length",
        "p90_evicted_run_length",
        "max_evicted_run_length",
        "evicted_tokens_per_run",
    ]
    out: dict[str, float] = {}
    for key in keys:
        arr = np.asarray([item[key] for item in values], dtype=np.float64)
        out[f"random_{key}_mean"] = float(arr.mean())
        out[f"random_{key}_p025"] = float(np.percentile(arr, 2.5))
        out[f"random_{key}_p975"] = float(np.percentile(arr, 97.5))
    return out


def block_projection_metrics(kept: np.ndarray, protected: np.ndarray, block_size: int) -> dict[str, float]:
    projected = kept.copy()
    valid = ~protected
    pure = 0
    majority_pure = 0
    block_count = 0
    for start in range(0, len(kept), block_size):
        end = min(len(kept), start + block_size)
        block_valid = valid[start:end]
        if not block_valid.any():
            continue
        block_kept = kept[start:end][block_valid]
        kept_fraction = float(block_kept.mean())
        block_decision = kept_fraction >= 0.5
        projected[start:end][block_valid] = block_decision
        block_count += 1
        majority = max(kept_fraction, 1.0 - kept_fraction)
        if majority >= 1.0:
            pure += 1
        if majority >= 0.8:
            majority_pure += 1

    mismatch = projected[valid] != kept[valid]
    false_eviction = (~projected[valid]) & kept[valid]
    false_keep = projected[valid] & (~kept[valid])
    valid_count = max(1, int(valid.sum()))
    kept_count = max(1, int(kept[valid].sum()))
    evicted_count = max(1, int((~kept[valid]).sum()))
    return {
        "block_size": int(block_size),
        "token_decision_mismatch_rate": float(mismatch.sum() / valid_count),
        "false_eviction_rate": float(false_eviction.sum() / kept_count),
        "false_keep_rate": float(false_keep.sum() / evicted_count),
        "pure_block_ratio": float(pure / max(1, block_count)),
        "majority_pure_block_ratio": float(majority_pure / max(1, block_count)),
    }


def collect_run(
    pipe: Any,
    model_key: str,
    ratio: float,
    sample: SampleSpec,
    args: argparse.Namespace,
    block_sizes: list[int],
) -> dict[str, Any]:
    press = TraceSnapKVScorePress(
        compression_ratio=ratio,
        window_size=args.q_window_size,
        kernel_size=args.kernel_size,
    )
    _ = pipe(
        sample.context,
        question=sample.question,
        press=press,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
    )
    if not press.layer_scores:
        raise RuntimeError(f"No layer scores collected for {sample.dataset_name} row={sample.dataset_row_index}")

    n_layers = get_num_layers(pipe.model)
    layer_scores = []
    for layer_idx in range(n_layers):
        if layer_idx in press.layer_scores:
            layer_scores.append(press.layer_scores[layer_idx][0].numpy())
    score_array = np.stack(layer_scores, axis=0)
    token_scores = score_array.mean(axis=(0, 1))

    masks = build_masks(token_scores, ratio, args.sink_tokens, args.recent_tokens)
    kept = masks["kept"]
    evicted = masks["evicted"]
    protected = masks["protected"]
    valid = masks["valid"]
    assert isinstance(kept, np.ndarray)
    assert isinstance(evicted, np.ndarray)
    assert isinstance(protected, np.ndarray)
    assert isinstance(valid, np.ndarray)

    stats = run_stats(evicted, valid)
    rng = np.random.default_rng(args.seed + sample.dataset_row_index + int(ratio * 1000))
    random_stats = random_run_stats(rng, int(stats["evicted_count"]), protected, args.random_repeats)
    block_stats = [block_projection_metrics(kept, protected, block_size) for block_size in block_sizes]

    run_id = (
        f"{model_key}__{sample.dataset_name}__row{sample.dataset_row_index}"
        f"__r{str(ratio).replace('.', 'p')}"
    )
    score_path = SCORE_DIR / f"{run_id}.npz"
    np.savez_compressed(
        score_path,
        layer_head_scores=score_array,
        token_scores=token_scores,
        kept_mask=kept.astype(np.uint8),
        evicted_mask=evicted.astype(np.uint8),
        protected_mask=protected.astype(np.uint8),
    )

    return {
        "run_id": run_id,
        "model_key": model_key,
        "model_path": MODELS[model_key],
        "compression_ratio": ratio,
        "sample": asdict(sample),
        "effective_context_tokens": int(token_scores.shape[0]),
        "q_window_size": args.q_window_size,
        "kernel_size": args.kernel_size,
        "sink_tokens": args.sink_tokens,
        "recent_tokens": args.recent_tokens,
        "keep_budget": int(masks["keep_budget"]),
        "actual_keep_count": int(kept.sum()),
        "actual_evicted_count": int(evicted.sum()),
        "actual_compression_ratio": float(evicted.sum() / len(evicted)),
        "run_stats": stats,
        "random_baseline": random_stats,
        "block_projection": block_stats,
        "score_arrays": str(score_path.relative_to(ROOT)),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary_csv(runs: list[dict[str, Any]], path: Path) -> None:
    import csv

    rows: list[dict[str, Any]] = []
    for run in runs:
        common = {
            "run_id": run["run_id"],
            "model_key": run["model_key"],
            "dataset_name": run["sample"]["dataset_name"],
            "dataset_row_index": run["sample"]["dataset_row_index"],
            "compression_ratio": run["compression_ratio"],
            "effective_context_tokens": run["effective_context_tokens"],
            "actual_compression_ratio": run["actual_compression_ratio"],
            **run["run_stats"],
            **run["random_baseline"],
        }
        for block_item in run["block_projection"]:
            row = dict(common)
            row.update(block_item)
            rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    mean_run = float(np.mean([run["run_stats"]["mean_evicted_run_length"] for run in runs]))
    mean_random = float(np.mean([run["random_baseline"]["random_mean_evicted_run_length_mean"] for run in runs]))
    by_block: dict[int, list[float]] = {}
    for run in runs:
        for item in run["block_projection"]:
            by_block.setdefault(int(item["block_size"]), []).append(float(item["token_decision_mismatch_rate"]))
    return {
        "run_count": len(runs),
        "mean_evicted_run_length": mean_run,
        "random_mean_evicted_run_length": mean_random,
        "run_length_gain_vs_random": float(mean_run / mean_random) if mean_random > 0 else None,
        "mean_mismatch_by_block_size": {
            str(block_size): float(np.mean(values)) for block_size, values in sorted(by_block.items())
        },
    }


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_JSONL.write_text("")

    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    block_sizes = [int(x.strip()) for x in args.block_sizes.split(",") if x.strip()]

    runs: list[dict[str, Any]] = []
    sample_manifest: list[dict[str, Any]] | None = None
    for model_key in model_keys:
        print(f"[collect] loading model={model_key} path={MODELS[model_key]}", flush=True)
        pipe = pipeline("kv-press-text-generation", model=MODELS[model_key], device=args.device, dtype="auto")
        samples = select_samples(
            pipe.tokenizer,
            dataset_names,
            args.samples_per_dataset,
            args.seed,
            args.sample_indices,
            args.min_context_tokens,
            args.max_context_length,
        )
        if sample_manifest is None:
            sample_manifest = [asdict(sample) for sample in samples]

        for ratio in ratios:
            for sample in samples:
                print(
                    f"[collect] model={model_key} dataset={sample.dataset_name} "
                    f"row={sample.dataset_row_index} ratio={ratio}",
                    flush=True,
                )
                run = collect_run(pipe, model_key, ratio, sample, args, block_sizes)
                runs.append(run)
                append_jsonl(RAW_JSONL, run)

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary_csv(runs, SUMMARY_CSV)
    summary = summarize(runs)
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset": "Xnhyacinth/LongBench",
        "models": {key: MODELS[key] for key in model_keys},
        "datasets": dataset_names,
        "ratios": ratios,
        "block_sizes": block_sizes,
        "seed": args.seed,
        "samples": sample_manifest or [],
        "args": vars(args),
        "summary": summary,
        "raw_jsonl": str(RAW_JSONL.relative_to(ROOT)),
        "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
    }
    write_json(MANIFEST_JSON, payload)
    write_json(SUMMARY_JSON, {"summary": summary, "runs": runs})
    print(SUMMARY_JSON, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
