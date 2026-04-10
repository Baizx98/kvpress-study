from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline

from evaluation.benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from kvpress import BlockWisePress, ChunkKVPress, SnapKVPress


ROOT = Path(__file__).resolve().parents[1]
MODEL = "/Tan/model/Llama-3.1-8B-Instruct"
DEVICE = "cuda:0"
BLOCK_SIZE = 16
SEED = 42
RATIOS = [float(x) for x in os.environ.get("RATIOS", "0.3,0.5,0.7").split(",")]
DEFAULT_SAMPLES = int(os.environ.get("SAMPLES_PER_DATASET", "4"))
NEEDLE_MAX_CONTEXT = int(os.environ.get("NEEDLE_MAX_CONTEXT", "16384"))
NEEDLE_DEPTHS = [int(x) for x in os.environ.get("NEEDLE_DEPTHS", "20,80").split(",")]

DATA_SPECS: list[dict[str, Any]] = [
    {
        "kind": "hf",
        "dataset_name": "Xnhyacinth/LongBench",
        "config": "triviaqa",
        "dataset_tag": "longbench_triviaqa",
        "sample_count": DEFAULT_SAMPLES,
    },
    {
        "kind": "hf",
        "dataset_name": "Xnhyacinth/LongBench",
        "config": "hotpotqa",
        "dataset_tag": "longbench_hotpotqa",
        "sample_count": DEFAULT_SAMPLES,
    },
    {
        "kind": "hf",
        "dataset_name": "Xnhyacinth/LongBench",
        "config": "multifieldqa_en",
        "dataset_tag": "longbench_multifieldqa_en",
        "sample_count": DEFAULT_SAMPLES,
    },
    {
        "kind": "hf",
        "dataset_name": "simonjegou/LongBench-v2",
        "config": "0shot",
        "dataset_tag": "longbenchv2_0shot",
        "sample_count": max(2, DEFAULT_SAMPLES - 1),
    },
    {
        "kind": "ruler",
        "dataset_name": "simonjegou/ruler",
        "config": "4096",
        "dataset_tag": "ruler_4096",
        "samples_per_task": 2,
        "tasks": ["niah_single_3", "niah_multikey_3", "qa_2"],
    },
    {
        "kind": "needle",
        "dataset_name": "alessiodevoto/paul_graham_essays",
        "config": None,
        "dataset_tag": "needle_in_haystack",
        "base_sample_count": 2,
    },
]

EXPERIMENT = "block_overlap_multidataset_full"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT
ARTIFACT_DIR = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT / "artifacts"


@dataclass
class SampleResult:
    dataset_tag: str
    sample_index: int
    sample_id: str
    question_preview: str
    ratios: dict[float, dict[str, Any]]


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _make_press(method: str, ratio: float):
    if method == "block_wise":
        return BlockWisePress(compression_ratio=ratio, block_size=BLOCK_SIZE)
    if method == "chunkkv":
        return ChunkKVPress(press=SnapKVPress(compression_ratio=ratio), chunk_length=20)
    raise ValueError(method)


def _sample_hf_dataset(dataset_name: str, config: str | None, sample_count: int) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, config, split="test")
    df = ds.to_pandas().sample(n=min(sample_count, len(ds)), random_state=SEED).reset_index(drop=False)
    return df.to_dict("records")


def _sample_ruler_dataset(
    dataset_name: str,
    config: str,
    tasks: list[str],
    samples_per_task: int,
) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, config, split="test")
    df = ds.to_pandas()
    selected = []
    for task_name in tasks:
        task_df = df[df["task"] == task_name]
        if task_df.empty:
            continue
        sample_n = min(samples_per_task, len(task_df))
        sampled = task_df.sample(n=sample_n, random_state=SEED).reset_index(drop=False)
        selected.append(sampled)
    if not selected:
        return []
    return pd.concat(selected, axis=0).reset_index(drop=True).to_dict("records")


def _sample_needle_dataset(
    dataset_name: str,
    sample_count: int,
    tokenizer,
) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split="test")
    df = ds.to_pandas().sample(n=min(sample_count, len(ds)), random_state=SEED).reset_index(drop=False)
    rows = []
    for local_idx in range(len(df)):
        base_df = df.iloc[[local_idx]].reset_index(drop=True)
        inserted = insert_needle_in_haystack(
            base_df,
            tokenizer,
            NEEDLE_MAX_CONTEXT,
            needle_depth=NEEDLE_DEPTHS,
        )
        for needle_row in inserted.to_dict("records"):
            needle_row["_id"] = f"essay{int(df.iloc[local_idx]['index'])}_depth{needle_row['needle_depth']}"
            rows.append(needle_row)
    return rows


def _sample_rows(spec: dict[str, Any], tokenizer) -> list[dict[str, Any]]:
    kind = spec["kind"]
    if kind == "hf":
        return _sample_hf_dataset(spec["dataset_name"], spec["config"], spec["sample_count"])
    if kind == "ruler":
        return _sample_ruler_dataset(
            spec["dataset_name"],
            spec["config"],
            spec["tasks"],
            spec["samples_per_task"],
        )
    if kind == "needle":
        return _sample_needle_dataset(spec["dataset_name"], spec["base_sample_count"], tokenizer)
    raise ValueError(f"Unsupported dataset kind: {kind}")


def _token_indices_to_block_ids(token_indices: torch.Tensor, block_size: int) -> list[int]:
    if token_indices.numel() == 0:
        return []
    return sorted({int(tok) // block_size for tok in token_indices.flatten().tolist()})


def _extract_layer_blocks(press, block_size: int) -> dict[int, list[int]]:
    layer_blocks: dict[int, list[int]] = {}
    if hasattr(press, "last_kept_block_indices") and getattr(press, "last_kept_block_indices"):
        for layer_idx, block_indices in press.last_kept_block_indices.items():
            layer_blocks[int(layer_idx)] = sorted({int(x) for x in block_indices.flatten().tolist() if int(x) >= 0})
        return layer_blocks

    for layer_idx, token_indices in getattr(press, "last_kept_token_indices", {}).items():
        layer_blocks[int(layer_idx)] = _token_indices_to_block_ids(token_indices, block_size)
    return layer_blocks


def _jaccard(a: list[int], b: list[int]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _pairwise_matrix(layer_blocks: dict[int, list[int]], n_layers: int) -> list[list[float]]:
    return [[_jaccard(layer_blocks.get(i, []), layer_blocks.get(j, [])) for j in range(n_layers)] for i in range(n_layers)]


def _same_layer_vector(a_blocks: dict[int, list[int]], b_blocks: dict[int, list[int]], n_layers: int) -> list[float]:
    return [_jaccard(a_blocks.get(i, []), b_blocks.get(i, [])) for i in range(n_layers)]


def _plot_sample(result: SampleResult, n_layers: int, dataset_dir: Path) -> Path:
    fig, axes = plt.subplots(len(RATIOS), 3, figsize=(16, 4.8 * len(RATIOS)), constrained_layout=True)
    if len(RATIOS) == 1:
        axes = [axes]

    for row_idx, ratio in enumerate(RATIOS):
        ratio_data = result.ratios[ratio]
        same = ratio_data["same_layer_similarity"]
        bw_mat = ratio_data["block_wise_matrix"]
        ck_mat = ratio_data["chunkkv_matrix"]

        ax = axes[row_idx][0]
        ax.plot(range(n_layers), same, marker="o", linewidth=1.4)
        ax.set_xlim(0, max(n_layers - 1, 0))
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_title(f"ratio={ratio} same-layer Jaccard")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Similarity")

        ax = axes[row_idx][1]
        im = ax.imshow(bw_mat, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-0.5, n_layers - 0.5)
        ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_yticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_title(f"ratio={ratio} BlockWise layer-pair")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row_idx][2]
        im = ax.imshow(ck_mat, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-0.5, n_layers - 0.5)
        ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_yticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_title(f"ratio={ratio} ChunkKV layer-pair")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{result.dataset_tag} | sample {result.sample_index} | {result.sample_id}\n{result.question_preview}",
        fontsize=13,
    )
    out = dataset_dir / f"sample_{result.sample_index:02d}_{result.sample_id[:24]}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_dataset_aggregate(dataset_tag: str, sample_results: list[SampleResult], n_layers: int, dataset_dir: Path) -> Path:
    fig, axes = plt.subplots(len(RATIOS), 3, figsize=(16, 4.8 * len(RATIOS)), constrained_layout=True)
    if len(RATIOS) == 1:
        axes = [axes]

    for row_idx, ratio in enumerate(RATIOS):
        same = np.array([r.ratios[ratio]["same_layer_similarity"] for r in sample_results]).mean(axis=0)
        bw_mat = np.array([r.ratios[ratio]["block_wise_matrix"] for r in sample_results]).mean(axis=0)
        ck_mat = np.array([r.ratios[ratio]["chunkkv_matrix"] for r in sample_results]).mean(axis=0)

        ax = axes[row_idx][0]
        ax.plot(range(n_layers), same, marker="o", linewidth=1.8)
        ax.set_xlim(0, max(n_layers - 1, 0))
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_title(f"ratio={ratio} mean same-layer Jaccard")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Similarity")

        ax = axes[row_idx][1]
        im = ax.imshow(bw_mat, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-0.5, n_layers - 0.5)
        ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_yticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_title(f"ratio={ratio} mean BlockWise layer-pair")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row_idx][2]
        im = ax.imshow(ck_mat, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-0.5, n_layers - 0.5)
        ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_yticks(np.arange(0, n_layers, max(1, n_layers // 8)))
        ax.set_title(f"ratio={ratio} mean ChunkKV layer-pair")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{dataset_tag} | averaged over {len(sample_results)} samples", fontsize=14)
    out = dataset_dir / f"{dataset_tag}_aggregate.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _run_single_method(pipe, press, row: dict[str, Any]) -> dict[int, list[int]]:
    combined_context = f"{row['context']}{row.get('question', '')}"
    _ = pipe(
        combined_context,
        question="",
        answer_prefix=row.get("answer_prefix", ""),
        press=press,
        max_new_tokens=min(int(row.get("max_new_tokens", 1)), 1),
    )
    return _extract_layer_blocks(press, BLOCK_SIZE)


def _save_partial_results(all_results: dict[str, list[dict[str, Any]]]) -> None:
    with open(ARTIFACT_DIR / "block_overlap_multidataset_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


def main() -> None:
    _seed_everything()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = pipeline("kv-press-text-generation", model=MODEL, device=DEVICE, dtype="auto")
    n_layers = len(pipe.model.model.layers)

    all_results: dict[str, list[dict[str, Any]]] = {}

    for spec in DATA_SPECS:
        dataset_tag = spec["dataset_tag"]
        dataset_dir = OUTDIR / dataset_tag
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = _sample_rows(spec, pipe.tokenizer)
        sample_results: list[SampleResult] = []

        for sample_idx, row in enumerate(rows):
            q_preview = str(row.get("question", "")).replace("\n", " ")[:120]
            sample_result = SampleResult(
                dataset_tag=dataset_tag,
                sample_index=sample_idx,
                sample_id=str(row.get("_id", sample_idx)),
                question_preview=q_preview,
                ratios={},
            )

            for ratio in RATIOS:
                method_blocks: dict[str, dict[int, list[int]]] = {}
                for method in ["block_wise", "chunkkv"]:
                    press = _make_press(method, ratio)
                    method_blocks[method] = _run_single_method(pipe, press, row)

                sample_result.ratios[ratio] = {
                    "same_layer_similarity": _same_layer_vector(method_blocks["block_wise"], method_blocks["chunkkv"], n_layers),
                    "block_wise_matrix": _pairwise_matrix(method_blocks["block_wise"], n_layers),
                    "chunkkv_matrix": _pairwise_matrix(method_blocks["chunkkv"], n_layers),
                    "block_wise_layer_blocks": method_blocks["block_wise"],
                    "chunkkv_layer_blocks": method_blocks["chunkkv"],
                }

            sample_results.append(sample_result)
            _plot_sample(sample_result, n_layers, dataset_dir)
            all_results.setdefault(dataset_tag, []).append(
                {
                    "sample_index": sample_result.sample_index,
                    "sample_id": sample_result.sample_id,
                    "question_preview": sample_result.question_preview,
                    "ratios": sample_result.ratios,
                }
            )
            _save_partial_results(all_results)

        _plot_dataset_aggregate(dataset_tag, sample_results, n_layers, dataset_dir)
        all_results[dataset_tag] = [
            {
                "sample_index": r.sample_index,
                "sample_id": r.sample_id,
                "question_preview": r.question_preview,
                "ratios": r.ratios,
            }
            for r in sample_results
        ]
        _save_partial_results(all_results)

    _save_partial_results(all_results)

    print(OUTDIR)


if __name__ == "__main__":
    main()
