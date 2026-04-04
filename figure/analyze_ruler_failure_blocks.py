from __future__ import annotations

import json
import math
import os
import re
import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from transformers import DynamicCache, pipeline
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress import BlockWisePress, ChunkKVPress, SnapKVPress
from kvpress.utils import get_prerope_query_states

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figure" / "experiments" / "ruler_failure_block_analysis"
RESULT_DIR = ROOT / "evaluation" / "results" / "experiments" / "ruler_failure_block_analysis" / "artifacts"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "/Tan/model/Llama-3.1-8B-Instruct"
BLOCK_SIZE = 16
RATIO = 0.7
FRACTION = 0.5
TARGET_TASKS = ("niah_multikey_3", "niah_single_3")
CASES_PER_TASK = 3
SEED = 42


def parse_refs(answer_text: str) -> list[str]:
    refs = re.findall(r"'([^']*)'", str(answer_text))
    if refs:
        return refs
    refs = re.findall(r"\[([^\]]+)\]", str(answer_text))
    if refs:
        return [x.strip() for x in refs[0].split(",")]
    return [str(answer_text)]


def row_correct(task: str, pred: str, answer: str) -> bool:
    pred = str(pred).lower()
    refs = [r.lower() for r in parse_refs(answer)]
    if task.startswith("qa_"):
        return any(r in pred for r in refs)
    return all(r in pred for r in refs)


def load_sampled_ruler() -> pd.DataFrame:
    df = load_dataset("simonjegou/ruler", name="4096", split="test").to_pandas()
    sample_n = int(round(len(df) * FRACTION))
    df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    return df


def load_predictions(path: str) -> pd.DataFrame:
    return pd.read_csv(path).reset_index(drop=True)


def select_failure_cases(sampled_df: pd.DataFrame) -> list[dict]:
    bw_path = ROOT / "evaluation" / "results" / "experiments" / "prefill_compare_50pct_blockwise_chunkkv" / "artifacts" / (
        "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__block_wise__0.70__fraction0.500__query_aware"
    ) / "predictions.csv"
    ck_path = ROOT / "evaluation" / "results" / "experiments" / "prefill_compare_50pct_blockwise_chunkkv" / "artifacts" / (
        "ruler__4096__--Tan--model--Llama-3.1-8B-Instruct__chunkkv__0.70__fraction0.500__query_aware"
    ) / "predictions.csv"
    bw = load_predictions(str(bw_path))
    ck = load_predictions(str(ck_path))

    assert len(bw) == len(sampled_df) == len(ck), "Prediction rows do not align with sampled dataset"

    cases = []
    for task in TARGET_TASKS:
        task_cases = []
        for i in range(len(sampled_df)):
            if sampled_df.loc[i, "task"] != task:
                continue
            bw_ok = row_correct(task, bw.loc[i, "predicted_answer"], bw.loc[i, "answer"])
            ck_ok = row_correct(task, ck.loc[i, "predicted_answer"], ck.loc[i, "answer"])
            if (not bw_ok) and ck_ok:
                task_cases.append(
                    {
                        "sample_index": int(i),
                        "task": task,
                        "answer": str(sampled_df.loc[i, "answer"]),
                        "question": str(sampled_df.loc[i, "question"]),
                        "blockwise_prediction": str(bw.loc[i, "predicted_answer"]),
                        "chunkkv_prediction": str(ck.loc[i, "predicted_answer"]),
                    }
                )
        cases.extend(task_cases[:CASES_PER_TASK])
    return cases


def build_context_and_question(row: pd.Series) -> tuple[str, str]:
    context = str(row["context"]) + str(row["question"])
    question = ""
    return context, question


def preprocess_for_prefill(pipe, context: str):
    processed = pipe.preprocess(
        context=context,
        questions=[""],
        answer_prefix="",
        max_context_length=min(pipe.tokenizer.model_max_length, int(1e10)),
        enable_thinking=False,
    )
    return processed["context_ids"]


def locate_answer_token_spans(pipe, full_context: str, refs: list[str]) -> tuple[list[tuple[int, int]], int]:
    context_ids = preprocess_for_prefill(pipe, full_context)
    full_len = context_ids.shape[1]
    spans = []
    for ref in refs:
        char_pos = full_context.find(ref)
        if char_pos < 0:
            continue
        prefix_ids = preprocess_for_prefill(pipe, full_context[:char_pos])
        ref_ids = preprocess_for_prefill(pipe, full_context[: char_pos + len(ref)])
        start = prefix_ids.shape[1]
        end = ref_ids.shape[1]
        spans.append((start, max(start, end - 1)))
    return spans, full_len


class DebugBlockWisePress(BlockWisePress):
    def __post_init__(self):
        super().__post_init__()
        self.layer_debug: dict[int, dict] = {}

    def _token_scores(self, module, hidden_states, keys):
        q_window = min(hidden_states.shape[1], self.q_window_size)
        query_states = get_prerope_query_states(module, hidden_states[:, -q_window:])
        num_key_value_heads = keys.shape[1]
        num_groups = module.config.num_attention_heads // num_key_value_heads
        key_states = repeat_kv(keys, num_groups)
        token_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(module.head_dim)
        token_scores = token_scores.view(
            keys.shape[0], num_key_value_heads, num_groups, q_window, keys.shape[2]
        ).mean(dim=2)
        token_scores = token_scores.mean(dim=-2).mean(dim=1)
        return token_scores

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        plan = self.build_block_plan(
            module, hidden_states, keys, values, attentions, kwargs, force_refresh_summary=True
        )
        layer_idx = int(module.layer_idx)
        self.layer_debug[layer_idx] = {
            "key_len": int(keys.shape[2]),
            "num_blocks": int(plan["num_blocks"]),
            "keep_budget": int(plan["keep_budget"]),
            "kept_block_indices": plan["kept_block_indices"][0].detach().cpu().tolist(),
            "block_scores": plan["block_scores"][0].detach().cpu().tolist(),
            "token_scores": self._token_scores(module, hidden_states, keys)[0].detach().cpu().tolist(),
        }
        compressed_keys, compressed_values = self.gather_by_token_indices(keys, values, plan["token_indices"])
        self.build_or_refresh_block_summary(module, compressed_keys, compressed_values, force_refresh=True)
        return compressed_keys, compressed_values


class DebugChunkKVPress(ChunkKVPress):
    analysis_block_size: int = BLOCK_SIZE

    def __post_init__(self):
        super().__post_init__()
        self.layer_debug: dict[int, dict] = {}

    def _token_indices_to_block_coverage(self, key_len: int, token_indices: torch.Tensor) -> list[float]:
        num_blocks = math.ceil(key_len / self.analysis_block_size)
        coverage = []
        kept_set = set(int(x) for x in token_indices.tolist())
        for block_idx in range(num_blocks):
            start = block_idx * self.analysis_block_size
            end = min(start + self.analysis_block_size, key_len)
            block_len = end - start
            kept = sum(1 for idx in range(start, end) if idx in kept_set)
            coverage.append(kept / block_len)
        return coverage

    def _token_scores_to_block_scores(self, token_scores: torch.Tensor, key_len: int) -> list[float]:
        num_blocks = math.ceil(key_len / self.analysis_block_size)
        block_scores = []
        for block_idx in range(num_blocks):
            start = block_idx * self.analysis_block_size
            end = min(start + self.analysis_block_size, key_len)
            block_scores.append(float(token_scores[start:end].mean().item()))
        return block_scores

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.press.compression_ratio == 0:
            return keys, values

        global_scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        kv_len = keys.shape[2]
        num_complete_chunks = kv_len // self.chunk_length
        remaining_tokens = kv_len % self.chunk_length

        if num_complete_chunks == 0:
            return self.press.compress(module, hidden_states, keys, values, attentions, kwargs)

        main_scores = global_scores[..., : num_complete_chunks * self.chunk_length]
        main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, self.chunk_length).mean(dim=-1)
        if remaining_tokens > 0:
            remaining_scores = global_scores[..., -remaining_tokens:]
            remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
            chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
        else:
            chunk_scores = main_chunk_scores

        n_chunks_kept = max(
            1,
            int((num_complete_chunks + (remaining_tokens > 0)) * (1 - self.press.compression_ratio)),
        )
        top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

        indices = []
        for chunk_idx in top_chunks.indices[0]:
            if chunk_idx < num_complete_chunks:
                start_idx = chunk_idx * self.chunk_length
                chunk_indices = torch.arange(start_idx, start_idx + self.chunk_length, device=keys.device)
            else:
                chunk_indices = torch.arange(num_complete_chunks * self.chunk_length, kv_len, device=keys.device)
            indices.append(chunk_indices)
        token_indices = torch.cat(indices).sort()[0]

        layer_idx = int(module.layer_idx)
        mean_token_scores = global_scores.mean(dim=1)[0].detach().cpu()
        self.layer_debug[layer_idx] = {
            "key_len": int(kv_len),
            "chunk_scores": chunk_scores[0].detach().cpu().tolist(),
            "kept_chunk_indices": top_chunks.indices[0].detach().cpu().tolist(),
            "kept_token_indices": token_indices.detach().cpu().tolist(),
            "token_scores": mean_token_scores.tolist(),
            "block_scores": self._token_scores_to_block_scores(mean_token_scores, kv_len),
            "block_coverage": self._token_indices_to_block_coverage(kv_len, token_indices.detach().cpu()),
        }

        gather_indices = token_indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)
        keys = keys.gather(2, gather_indices).contiguous()
        values = values.gather(2, gather_indices).contiguous()
        return keys, values


def run_prefill_debug(pipe, context: str, press):
    input_tensors = pipe.preprocess(
        context=context,
        questions=[""],
        answer_prefix="",
        max_context_length=min(pipe.tokenizer.model_max_length, int(1e10)),
        enable_thinking=False,
    )
    cache = DynamicCache()
    context_ids = input_tensors["context_ids"].to(pipe.model.device)
    with torch.inference_mode():
        with press(pipe.model):
            outputs = pipe.model.model(input_ids=context_ids, past_key_values=cache)

    del outputs
    del cache
    del context_ids
    del input_tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return press


def plot_case(case: dict, bw_press: DebugBlockWisePress, ck_press: DebugChunkKVPress, answer_spans, question_start_token):
    layers = sorted(bw_press.layer_debug.keys())
    num_blocks = bw_press.layer_debug[layers[0]]["num_blocks"]
    answer_blocks = sorted({span[0] // BLOCK_SIZE for span in answer_spans})
    question_start_block = question_start_token // BLOCK_SIZE

    bw_keep = np.zeros((len(layers), num_blocks), dtype=float)
    ck_cov = np.zeros((len(layers), num_blocks), dtype=float)
    for row_idx, layer in enumerate(layers):
        bw_debug = bw_press.layer_debug[layer]
        ck_debug = ck_press.layer_debug[layer]
        for idx in bw_debug["kept_block_indices"]:
            if 0 <= idx < num_blocks:
                bw_keep[row_idx, idx] = 1.0
        cov = ck_debug["block_coverage"]
        ck_cov[row_idx, : min(num_blocks, len(cov))] = cov[:num_blocks]

    last_layer = layers[-1]
    bw_last = bw_press.layer_debug[last_layer]
    ck_last = ck_press.layer_debug[last_layer]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    ax = axes[0, 0]
    im = ax.imshow(bw_keep, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("BlockWise kept blocks by layer")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    ax = axes[0, 1]
    im2 = ax.imshow(ck_cov, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    ax.set_title("ChunkKV kept-token coverage by block")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    for subplot in (axes[0, 0], axes[0, 1]):
        for b in answer_blocks:
            subplot.axvline(b, color="red", linestyle="--", linewidth=1)
        subplot.axvline(question_start_block, color="green", linestyle=":", linewidth=1)

    ax = axes[1, 0]
    x = np.arange(len(bw_last["block_scores"]))
    ax.plot(x, bw_last["block_scores"], label="BlockWise block score", linewidth=2)
    ax.plot(x, ck_last["block_scores"], label="ChunkKV token->block mean score", linewidth=2)
    for b in answer_blocks:
        ax.axvspan(b - 0.5, b + 0.5, color="red", alpha=0.15)
    ax.axvline(question_start_block, color="green", linestyle=":", linewidth=1, label="question start")
    ax.set_title(f"Last-layer block scores (layer={last_layer})")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if answer_spans:
        center = answer_spans[0][0]
        lo = max(0, center - 64)
        hi = min(len(bw_last["token_scores"]), answer_spans[0][1] + 64)
    else:
        lo, hi = 0, min(256, len(bw_last["token_scores"]))
    tx = np.arange(lo, hi)
    ax.plot(tx, bw_last["token_scores"][lo:hi], label="BlockWise token score", linewidth=2)
    ax.plot(tx, ck_last["token_scores"][lo:hi], label="ChunkKV token score", linewidth=2)
    for start, end in answer_spans:
        if end < lo or start >= hi:
            continue
        ax.axvspan(max(start, lo), min(end, hi - 1), color="red", alpha=0.2)
    ax.axvline(question_start_token, color="green", linestyle=":", linewidth=1)
    ax.set_title("Token scores near answer span (last layer)")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sample {case['sample_index']} | {case['task']} | BW wrong / ChunkKV correct",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = FIGURE_DIR / f"case_{case['sample_index']:04d}_{case['task']}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    sampled_df = load_sampled_ruler()
    selected_cases = select_failure_cases(sampled_df)
    with open(RESULT_DIR / "case_selection.json", "w") as f:
        json.dump(selected_cases, f, ensure_ascii=False, indent=2)

    pipe = pipeline("kv-press-text-generation", model=MODEL, device="cuda:0", trust_remote_code=True, model_kwargs={"dtype": "auto"})
    pipe.model.eval()

    analysis_summary = []
    for case in selected_cases:
        row = sampled_df.iloc[case["sample_index"]]
        full_context, _ = build_context_and_question(row)
        raw_context = str(row["context"])
        answer_spans, _ = locate_answer_token_spans(pipe, full_context, parse_refs(row["answer"]))
        question_start_token = preprocess_for_prefill(pipe, raw_context).shape[1]

        bw_press = DebugBlockWisePress(compression_ratio=RATIO)
        ck_press = DebugChunkKVPress(press=SnapKVPress(compression_ratio=RATIO), chunk_length=20)

        bw_press = run_prefill_debug(pipe, full_context, bw_press)
        ck_press = run_prefill_debug(pipe, full_context, ck_press)
        fig_path = plot_case(case, bw_press, ck_press, answer_spans, question_start_token)
        debug_path = RESULT_DIR / f"case_{case['sample_index']:04d}_{case['task']}.json"
        with open(debug_path, "w") as f:
            json.dump(
                {
                    "case": case,
                    "answer_spans": answer_spans,
                    "question_start_token": question_start_token,
                    "blockwise": bw_press.layer_debug,
                    "chunkkv": ck_press.layer_debug,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        analysis_summary.append(
            {
                **case,
                "answer_spans": answer_spans,
                "question_start_token": question_start_token,
                "figure_path": str(fig_path),
                "debug_json_path": str(debug_path),
            }
        )

    with open(RESULT_DIR / "analysis_summary.json", "w") as f:
        json.dump(analysis_summary, f, ensure_ascii=False, indent=2)

    print(str(RESULT_DIR / "analysis_summary.json"))
    for case in analysis_summary:
        print(case["figure_path"])


if __name__ == "__main__":
    main()
