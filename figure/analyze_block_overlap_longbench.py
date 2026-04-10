from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from kvpress import BlockWisePress, ChunkKVPress, SnapKVPress  # noqa: E402


MODEL = "/Tan/model/Llama-3.1-8B-Instruct"
DEVICE = "cuda:0"
DATASET = "Xnhyacinth/LongBench"
DATA_DIR = "triviaqa"
BLOCK_SIZE = 16
SAMPLE_COUNT = int(os.environ.get("SAMPLE_COUNT", "4"))
RATIOS = [float(x) for x in os.environ.get("RATIOS", "0.3,0.5,0.7").split(",")]
SEED = 42
OUTDIR = ROOT / "figure" / "experiments" / "block_overlap_longbench_triviaqa"
ARTIFACT_DIR = ROOT / "evaluation" / "results" / "experiments" / "block_overlap_longbench_triviaqa" / "artifacts"


@dataclass
class SampleResult:
    sample_index: int
    sample_id: str
    question_preview: str
    ratios: dict[float, dict[str, Any]]


def _sample_dataset(sample_count: int) -> list[dict[str, Any]]:
    ds = load_dataset(DATASET, DATA_DIR, split="test")
    df = ds.to_pandas().sample(n=sample_count, random_state=SEED).reset_index(drop=False)
    return df.to_dict("records")


def _make_press(method: str, ratio: float):
    if method == "block_wise":
        return BlockWisePress(compression_ratio=ratio, block_size=BLOCK_SIZE)
    if method == "chunkkv":
        return ChunkKVPress(press=SnapKVPress(compression_ratio=ratio), chunk_length=20)
    raise ValueError(method)


def _token_indices_to_block_ids(token_indices: torch.Tensor, block_size: int) -> list[int]:
    if token_indices.numel() == 0:
        return []
    ids = sorted({int(tok) // block_size for tok in token_indices.flatten().tolist()})
    return ids


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
    matrix = []
    for i in range(n_layers):
        row = []
        for j in range(n_layers):
            row.append(_jaccard(layer_blocks.get(i, []), layer_blocks.get(j, [])))
        matrix.append(row)
    return matrix


def _same_layer_vector(a_blocks: dict[int, list[int]], b_blocks: dict[int, list[int]], n_layers: int) -> list[float]:
    return [_jaccard(a_blocks.get(i, []), b_blocks.get(i, [])) for i in range(n_layers)]


def _plot_sample(result: SampleResult, n_layers: int) -> Path:
    fig, axes = plt.subplots(len(RATIOS), 3, figsize=(16, 4.8 * len(RATIOS)), constrained_layout=True)
    if len(RATIOS) == 1:
        axes = [axes]

    for row_idx, ratio in enumerate(RATIOS):
        ratio_data = result.ratios[ratio]
        same = ratio_data["same_layer_similarity"]
        bw_mat = ratio_data["block_wise_matrix"]
        ck_mat = ratio_data["chunkkv_matrix"]

        ax = axes[row_idx][0]
        ax.plot(range(n_layers), same, marker="o", linewidth=1.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"ratio={ratio} same-layer Jaccard")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Similarity")

        ax = axes[row_idx][1]
        im = ax.imshow(bw_mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"ratio={ratio} BlockWise layer-pair")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row_idx][2]
        im = ax.imshow(ck_mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"ratio={ratio} ChunkKV layer-pair")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Sample {result.sample_index} | {result.sample_id}\n{result.question_preview}",
        fontsize=13,
    )
    out = OUTDIR / f"sample_{result.sample_index:02d}_{result.sample_id[:8]}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = pipeline("kv-press-text-generation", model=MODEL, device=DEVICE, dtype="auto")
    n_layers = len(pipe.model.model.layers)

    samples = _sample_dataset(SAMPLE_COUNT)
    results: list[SampleResult] = []

    for sample_idx, row in enumerate(samples):
        q_preview = str(row["question"]).replace("\n", " ")[:120]
        sample_result = SampleResult(
            sample_index=sample_idx,
            sample_id=str(row.get("_id", sample_idx)),
            question_preview=q_preview,
            ratios={},
        )

        for ratio in RATIOS:
            method_blocks: dict[str, dict[int, list[int]]] = {}
            for method in ["block_wise", "chunkkv"]:
                press = _make_press(method, ratio)
                _ = pipe(
                    row["context"],
                    question=row["question"],
                    press=press,
                    max_new_tokens=1,
                )
                method_blocks[method] = _extract_layer_blocks(press, BLOCK_SIZE)

            sample_result.ratios[ratio] = {
                "same_layer_similarity": _same_layer_vector(
                    method_blocks["block_wise"], method_blocks["chunkkv"], n_layers
                ),
                "block_wise_matrix": _pairwise_matrix(method_blocks["block_wise"], n_layers),
                "chunkkv_matrix": _pairwise_matrix(method_blocks["chunkkv"], n_layers),
                "block_wise_layer_blocks": method_blocks["block_wise"],
                "chunkkv_layer_blocks": method_blocks["chunkkv"],
            }

        results.append(sample_result)
        _plot_sample(sample_result, n_layers)

    json_ready = []
    for r in results:
        json_ready.append(
            {
                "sample_index": r.sample_index,
                "sample_id": r.sample_id,
                "question_preview": r.question_preview,
                "ratios": r.ratios,
            }
        )

    with open(ARTIFACT_DIR / "block_overlap_results.json", "w") as f:
        json.dump(json_ready, f, indent=2)

    print(OUTDIR)


if __name__ == "__main__":
    main()
