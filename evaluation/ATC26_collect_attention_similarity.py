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

from kvpress import BlockWisePress  # noqa: E402


EXPERIMENT_NAME = "ATC26_blockwise_attention_similarity_hotpotqa_3samples"
RESULT_ROOT = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACT_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACT_DIR / "raw"
SCORE_DIR = ARTIFACT_DIR / "scores"
AGGREGATE_JSON = ARTIFACT_DIR / "ATC26_attention_similarity_aggregate.json"
RAW_JSONL = RAW_DIR / "ATC26_attention_similarity_raw.jsonl"
SAMPLE_MANIFEST = ARTIFACT_DIR / "ATC26_hotpotqa_sample_manifest.json"

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


@dataclass
class SampleSpec:
    sample_index: int
    dataset_row_index: int
    sample_id: str
    question: str
    context: str
    answers: Any
    context_sha1: str


class TraceBlockWisePress(BlockWisePress):
    """BlockWisePress variant that preserves per-layer and per-head block choices."""

    def __post_init__(self):
        super().__post_init__()
        self.last_kept_block_indices_per_head: dict[int, torch.Tensor] = {}
        self.last_block_scores_per_head_trace: dict[int, torch.Tensor] = {}
        self.last_block_scores_trace: dict[int, torch.Tensor] = {}

    def _select_kept_blocks_from_scores(
        self,
        scores: torch.Tensor,
        key_len: int,
        num_blocks: int,
        keep_budget: int,
        device: torch.device,
    ) -> torch.Tensor:
        if keep_budget <= 0:
            return torch.empty(scores.shape[0], 0, dtype=torch.long, device=device)
        if keep_budget >= num_blocks:
            return torch.arange(num_blocks, device=device).expand(scores.shape[0], -1)

        has_partial_tail_block = key_len % self.block_size != 0
        tail_block_idx = num_blocks - 1
        sink_count = min(self.prefix_sink_blocks, num_blocks)
        recent_count = min(self.protected_recent_blocks, num_blocks)
        protected_indices = set(range(sink_count))
        protected_indices |= set(range(max(0, num_blocks - recent_count), num_blocks))
        if has_partial_tail_block and num_blocks > 0:
            protected_indices.add(tail_block_idx)

        if len(protected_indices) <= keep_budget:
            remaining_candidates = [idx for idx in range(num_blocks) if idx not in protected_indices]
            additional_keeps = keep_budget - len(protected_indices)
            selected_remaining = self._select_top_block_indices(
                scores, remaining_candidates, additional_keeps, device
            )
            protected_tensor = torch.tensor(
                sorted(protected_indices), dtype=torch.long, device=device
            ).expand(scores.shape[0], -1)
            return torch.cat([protected_tensor, selected_remaining], dim=-1).sort(dim=-1).values

        return self._select_top_block_indices(
            scores, list(range(num_blocks)), keep_budget, device
        ).sort(dim=-1).values

    def compress(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        plan = self.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            force_refresh_summary=True,
        )
        layer_idx = self._resolve_layer_idx(module)
        self.last_kept_block_indices[layer_idx] = plan["kept_block_indices"].detach().clone()
        self.last_kept_token_indices[layer_idx] = plan["token_indices"].detach().clone()
        self.last_block_scores_trace[layer_idx] = plan["block_scores"].detach().float().cpu()

        per_head_scores = plan["summary_scores_per_head"]
        self.last_block_scores_per_head_trace[layer_idx] = per_head_scores.detach().float().cpu()
        per_head_kept = []
        for head_idx in range(per_head_scores.shape[1]):
            per_head_kept.append(
                self._select_kept_blocks_from_scores(
                    per_head_scores[:, head_idx, :],
                    key_len=keys.shape[2],
                    num_blocks=plan["num_blocks"],
                    keep_budget=plan["keep_budget"],
                    device=keys.device,
                )
            )
        self.last_kept_block_indices_per_head[layer_idx] = torch.stack(
            per_head_kept, dim=1
        ).detach().clone()

        compressed_keys, compressed_values = self.gather_by_token_indices(
            keys, values, plan["token_indices"]
        )
        self.build_or_refresh_block_summary(
            module, compressed_keys, compressed_values, force_refresh=True
        )
        return compressed_keys, compressed_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model keys.")
    parser.add_argument("--ratios", default="0.3,0.5,0.7", help="Comma-separated compression ratios.")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--sample-indices", default="", help="Optional comma-separated LongBench row indices.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-context-length", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hotpotqa_samples(sample_count: int, seed: int, sample_indices: str) -> list[SampleSpec]:
    dataset = load_dataset("Xnhyacinth/LongBench", name="hotpotqa", split="test")
    if sample_indices:
        row_indices = [int(x.strip()) for x in sample_indices.split(",") if x.strip()]
    else:
        rng = random.Random(seed)
        row_indices = sorted(rng.sample(range(len(dataset)), sample_count))

    samples = []
    for local_idx, row_idx in enumerate(row_indices):
        row = dict(dataset[int(row_idx)])
        question = str(row.get("question") or row.get("input") or "")
        context = str(row["context"])
        samples.append(
            SampleSpec(
                sample_index=local_idx,
                dataset_row_index=int(row_idx),
                sample_id=str(row.get("_id", row_idx)),
                question=question,
                context=context,
                answers=row.get("answers", row.get("answer")),
                context_sha1=hashlib.sha1(context.encode("utf-8")).hexdigest(),
            )
        )
    return samples


def make_press(ratio: float) -> TraceBlockWisePress:
    return TraceBlockWisePress(compression_ratio=ratio, **BLOCKWISE_CONFIG)


def get_num_layers(model: Any) -> int:
    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return len(language_model.layers)


def jaccard(a: list[int], b: list[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def upper_triangle_mean(matrix: list[list[float]]) -> float:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape[0] < 2:
        return float("nan")
    mask = np.triu(np.ones(arr.shape, dtype=bool), k=1)
    return float(arr[mask].mean())


def pairwise_jaccard(sets: list[list[int]]) -> list[list[float]]:
    return [[jaccard(sets[i], sets[j]) for j in range(len(sets))] for i in range(len(sets))]


def cosine_similarity_matrix(vectors: np.ndarray) -> list[list[float]]:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-12, None)
    return np.clip(normalized @ normalized.T, -1.0, 1.0).tolist()


def tensor_blocks_to_lists(tensor: torch.Tensor) -> list[list[int]]:
    arr = tensor.detach().cpu().long().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    return [[int(x) for x in row.tolist()] for row in arr]


def collect_run(
    pipe,
    model_key: str,
    ratio: float,
    sample: SampleSpec,
    n_layers: int,
    max_new_tokens: int,
    max_context_length: int | None,
) -> dict[str, Any]:
    press = make_press(ratio)
    kwargs: dict[str, Any] = {
        "question": sample.question,
        "press": press,
        "max_new_tokens": max_new_tokens,
    }
    if max_context_length is not None:
        kwargs["max_context_length"] = max_context_length
    _ = pipe(sample.context, **kwargs)

    layer_sets: list[list[int]] = []
    for layer_idx in range(n_layers):
        kept = press.last_kept_block_indices.get(layer_idx)
        layer_sets.append(tensor_blocks_to_lists(kept)[0] if kept is not None else [])
    layer_matrix = pairwise_jaccard(layer_sets)

    head_sets_by_layer: list[list[list[int]]] = []
    head_matrices_by_layer: list[list[list[float]]] = []
    head_cosine_by_layer: list[list[list[float]]] = []
    for layer_idx in range(n_layers):
        per_head = press.last_kept_block_indices_per_head.get(layer_idx)
        if per_head is None:
            head_sets = []
        else:
            head_sets = tensor_blocks_to_lists(per_head[0])
        head_sets_by_layer.append(head_sets)
        head_matrices_by_layer.append(pairwise_jaccard(head_sets) if head_sets else [])

        score_tensor = press.last_block_scores_per_head_trace.get(layer_idx)
        if score_tensor is not None:
            head_cosine_by_layer.append(cosine_similarity_matrix(score_tensor[0].numpy()))
        else:
            head_cosine_by_layer.append([])

    valid_head_matrices = [np.asarray(m, dtype=np.float64) for m in head_matrices_by_layer if m]
    valid_head_cosines = [np.asarray(m, dtype=np.float64) for m in head_cosine_by_layer if m]
    head_matrix = np.mean(valid_head_matrices, axis=0).tolist()
    head_score_cosine = np.mean(valid_head_cosines, axis=0).tolist()

    run_id = f"{model_key}__r{str(ratio).replace('.', 'p')}__sample{sample.sample_index:02d}"
    scores_path = SCORE_DIR / f"{run_id}.npz"
    save_score_arrays(press, n_layers, scores_path)

    return {
        "run_id": run_id,
        "model_key": model_key,
        "compression_ratio": ratio,
        "sample": {
            "sample_index": sample.sample_index,
            "dataset_row_index": sample.dataset_row_index,
            "sample_id": sample.sample_id,
            "question": sample.question,
            "answers": sample.answers,
            "context_sha1": sample.context_sha1,
            "context_chars": len(sample.context),
            "context_tokens_without_chat_template": len(
                pipe.tokenizer.encode(sample.context, add_special_tokens=False)
            ),
        },
        "blockwise_config": BLOCKWISE_CONFIG,
        "n_layers": n_layers,
        "n_kv_heads": len(head_sets_by_layer[0]) if head_sets_by_layer else 0,
        "layer_kept_blocks": layer_sets,
        "layer_similarity_matrix": layer_matrix,
        "layer_similarity_upper_mean": upper_triangle_mean(layer_matrix),
        "head_kept_blocks_by_layer": head_sets_by_layer,
        "head_similarity_matrix": head_matrix,
        "head_similarity_by_layer": [
            upper_triangle_mean(m) if m else float("nan") for m in head_matrices_by_layer
        ],
        "head_similarity_upper_mean": upper_triangle_mean(head_matrix),
        "head_score_cosine_matrix": head_score_cosine,
        "head_score_cosine_upper_mean": upper_triangle_mean(head_score_cosine),
        "score_arrays": str(scores_path.relative_to(ROOT)),
    }


def save_score_arrays(press: TraceBlockWisePress, n_layers: int, path: Path) -> None:
    layer_scores = []
    per_head_scores = []
    kept_layer_blocks = []
    kept_head_blocks = []
    for layer_idx in range(n_layers):
        layer_scores.append(press.last_block_scores_trace[layer_idx][0].numpy())
        per_head_scores.append(press.last_block_scores_per_head_trace[layer_idx][0].numpy())
        kept_layer_blocks.append(press.last_kept_block_indices[layer_idx][0].detach().cpu().numpy())
        kept_head_blocks.append(
            press.last_kept_block_indices_per_head[layer_idx][0].detach().cpu().numpy()
        )

    np.savez_compressed(
        path,
        layer_scores=np.stack(layer_scores, axis=0),
        per_head_scores=np.stack(per_head_scores, axis=0),
        kept_layer_blocks=np.stack(kept_layer_blocks, axis=0),
        kept_head_blocks=np.stack(kept_head_blocks, axis=0),
    )


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["model_key"], float(run["compression_ratio"])), []).append(run)

    configs = []
    for (model_key, ratio), items in sorted(grouped.items()):
        layer_mats = np.asarray([item["layer_similarity_matrix"] for item in items], dtype=np.float64)
        head_mats = np.asarray([item["head_similarity_matrix"] for item in items], dtype=np.float64)
        head_cosines = np.asarray([item["head_score_cosine_matrix"] for item in items], dtype=np.float64)
        configs.append(
            {
                "model_key": model_key,
                "compression_ratio": ratio,
                "sample_count": len(items),
                "layer_similarity_matrix_mean": layer_mats.mean(axis=0).tolist(),
                "layer_similarity_matrix_std": layer_mats.std(axis=0).tolist(),
                "layer_similarity_upper_mean": upper_triangle_mean(layer_mats.mean(axis=0).tolist()),
                "head_similarity_matrix_mean": head_mats.mean(axis=0).tolist(),
                "head_similarity_matrix_std": head_mats.std(axis=0).tolist(),
                "head_similarity_upper_mean": upper_triangle_mean(head_mats.mean(axis=0).tolist()),
                "head_score_cosine_matrix_mean": head_cosines.mean(axis=0).tolist(),
                "head_score_cosine_upper_mean": upper_triangle_mean(head_cosines.mean(axis=0).tolist()),
                "run_ids": [item["run_id"] for item in items],
            }
        )
    return {"configs": configs}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)

    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    samples = load_hotpotqa_samples(args.sample_count, args.seed, args.sample_indices)
    write_json(SAMPLE_MANIFEST, [asdict(sample) for sample in samples])
    RAW_JSONL.write_text("")

    runs: list[dict[str, Any]] = []
    for model_key in model_keys:
        model_path = MODELS[model_key]
        print(f"[collect] loading model={model_key} path={model_path}", flush=True)
        pipe = pipeline("kv-press-text-generation", model=model_path, device=args.device, dtype="auto")
        n_layers = get_num_layers(pipe.model)

        for ratio in ratios:
            for sample in samples:
                print(
                    f"[collect] model={model_key} ratio={ratio} sample={sample.sample_index} row={sample.dataset_row_index}",
                    flush=True,
                )
                run = collect_run(
                    pipe=pipe,
                    model_key=model_key,
                    ratio=ratio,
                    sample=sample,
                    n_layers=n_layers,
                    max_new_tokens=args.max_new_tokens,
                    max_context_length=args.max_context_length,
                )
                runs.append(run)
                append_jsonl(RAW_JSONL, run)

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset": "Xnhyacinth/LongBench",
        "dataset_config": "hotpotqa",
        "seed": args.seed,
        "models": {key: MODELS[key] for key in model_keys},
        "ratios": ratios,
        "sample_count": len(samples),
        "blockwise_config": BLOCKWISE_CONFIG,
        "raw_jsonl": str(RAW_JSONL.relative_to(ROOT)),
        "sample_manifest": str(SAMPLE_MANIFEST.relative_to(ROOT)),
        **aggregate_runs(runs),
    }
    write_json(AGGREGATE_JSON, aggregate)
    print(AGGREGATE_JSON, flush=True)


if __name__ == "__main__":
    main()
