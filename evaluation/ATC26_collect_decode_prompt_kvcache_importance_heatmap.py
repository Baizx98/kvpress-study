from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/Tan/dataset/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/Tan/dataset/hf_home/datasets")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Tan/dataset/hf_home/hub")

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_decode_prompt_kvcache_importance_heatmap_longbench"
RESULT_ROOT = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
ARTIFACT_DIR = RESULT_ROOT / "artifacts"
RAW_DIR = ARTIFACT_DIR / "raw"
SAMPLE_MANIFEST = ARTIFACT_DIR / "sample_manifest.json"
RUN_CONFIG = ARTIFACT_DIR / "run_config.json"
SUMMARY_CSV = ARTIFACT_DIR / "summary_metrics.csv"
TRACE_JSONL = ARTIFACT_DIR / "trace_summary.jsonl"


@dataclass
class SampleSpec:
    dataset: str
    sample_index: int
    dataset_row_index: int
    prompt: str
    context: str
    question: str
    answer_prefix: str
    answers: Any
    prompt_sha1: str
    prompt_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/Tan/model/Llama-3.1-8B-Instruct")
    parser.add_argument("--datasets", default="gov_report,qmsum,multi_news")
    parser.add_argument("--sample-count-per-dataset", type=int, default=1)
    parser.add_argument("--sample-indices", default="", help="Optional mapping like gov_report:0,3;qmsum:4")
    parser.add_argument("--decode-steps", default="256,512")
    parser.add_argument("--compression-ratios", default="0.3,0.5,0.7")
    parser.add_argument("--min-prompt-tokens", type=int, default=4000)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer-window", type=int, default=8, help="Average the last N layers.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default=None, help="Optional Transformers attention implementation override.")
    parser.add_argument("--stop-on-eos", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-sample traces.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_sample_indices(text: str) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    if not text.strip():
        return mapping
    for group in text.split(";"):
        if not group.strip():
            continue
        dataset, values = group.split(":", 1)
        mapping[dataset.strip()] = parse_csv_ints(values)
    return mapping


def dtype_from_arg(dtype: str):
    if dtype == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def load_longbench_config(dataset_name: str):
    return load_dataset("Xnhyacinth/LongBench", name=dataset_name, split="test")


def build_prompt(row: dict[str, Any]) -> str:
    return f"{row['context']}{row.get('question', '')}{row.get('answer_prefix', '')}"


def select_samples(
    tokenizer,
    dataset_names: list[str],
    sample_count_per_dataset: int,
    min_prompt_tokens: int,
    max_prompt_tokens: int | None,
    seed: int,
    sample_indices: dict[str, list[int]],
) -> list[SampleSpec]:
    samples: list[SampleSpec] = []
    rng = random.Random(seed)

    for dataset_name in dataset_names:
        dataset = load_longbench_config(dataset_name)
        if dataset_name in sample_indices:
            candidate_indices = sample_indices[dataset_name]
        else:
            candidate_indices = list(range(len(dataset)))
            rng.shuffle(candidate_indices)

        selected_for_dataset = 0
        for row_idx in candidate_indices:
            row = dict(dataset[int(row_idx)])
            prompt = build_prompt(row)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_tokens < min_prompt_tokens:
                continue
            if max_prompt_tokens is not None and prompt_tokens > max_prompt_tokens:
                continue
            samples.append(
                SampleSpec(
                    dataset=dataset_name,
                    sample_index=len(samples),
                    dataset_row_index=int(row_idx),
                    prompt=prompt,
                    context=str(row["context"]),
                    question=str(row.get("question", "")),
                    answer_prefix=str(row.get("answer_prefix", "")),
                    answers=row.get("answers", row.get("answer")),
                    prompt_sha1=hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
                    prompt_tokens=prompt_tokens,
                )
            )
            selected_for_dataset += 1
            if selected_for_dataset >= sample_count_per_dataset:
                break

        if selected_for_dataset < sample_count_per_dataset:
            raise RuntimeError(
                f"Only selected {selected_for_dataset}/{sample_count_per_dataset} samples "
                f"for {dataset_name} with min_prompt_tokens={min_prompt_tokens}."
            )

    return samples


def topk_mask(scores: np.ndarray, compression_ratio: float) -> np.ndarray:
    prompt_len = int(scores.shape[-1])
    keep_count = int(np.ceil(prompt_len * (1.0 - compression_ratio)))
    keep_count = min(max(keep_count, 1), prompt_len)
    mask = np.zeros(scores.shape, dtype=np.bool_)
    if keep_count == prompt_len:
        mask[...] = True
        return mask
    indices = np.argpartition(scores, prompt_len - keep_count, axis=1)[:, prompt_len - keep_count :]
    rows = np.arange(scores.shape[0])[:, None]
    mask[rows, indices] = True
    return mask


def jaccard_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    return float(inter / union) if union else 1.0


def lag_jaccard(mask: np.ndarray, lag: int) -> float:
    if mask.shape[0] <= lag:
        return float("nan")
    values = [jaccard_bool(mask[i], mask[i + lag]) for i in range(mask.shape[0] - lag)]
    return float(np.mean(values))


def summarize_mask(mask: np.ndarray, scores: np.ndarray, compression_ratio: float) -> dict[str, float]:
    keep_frequency = mask.mean(axis=0)
    return {
        "compression_ratio": compression_ratio,
        "steps": int(mask.shape[0]),
        "prompt_tokens": int(mask.shape[1]),
        "mean_keep_frequency": float(keep_frequency.mean()),
        "std_keep_frequency": float(keep_frequency.std()),
        "min_keep_frequency": float(keep_frequency.min()),
        "max_keep_frequency": float(keep_frequency.max()),
        "adjacent_jaccard": lag_jaccard(mask, 1),
        "lag16_jaccard": lag_jaccard(mask, 16),
        "lag32_jaccard": lag_jaccard(mask, 32),
        "lag64_jaccard": lag_jaccard(mask, 64),
        "lag128_jaccard": lag_jaccard(mask, 128),
        "mean_prompt_attention_mass": float(scores.sum(axis=1).mean()),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_attention_implementation(model, implementation: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(implementation)
        return
    if hasattr(model, "config"):
        model.config._attn_implementation = implementation


def trace_sample(
    model,
    tokenizer,
    sample: SampleSpec,
    max_decode_steps: int,
    layer_window: int,
    device: str,
    stop_on_eos: bool,
) -> dict[str, Any]:
    encoded = tokenizer(sample.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_len = int(input_ids.shape[1])

    set_attention_implementation(model, "sdpa")
    prefill = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = prefill.past_key_values
    next_token = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
    set_attention_implementation(model, "eager")

    scores: list[np.ndarray] = []
    generated: list[int] = []
    first_eos_step: int | None = None
    eos_token_id = tokenizer.eos_token_id

    for step in range(max_decode_steps):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
        )
        past_key_values = outputs.past_key_values
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attentions. Use eager attention implementation.")

        selected_layers = attentions[-layer_window:] if layer_window > 0 else attentions
        per_layer_scores = []
        for attn in selected_layers:
            prompt_attn = attn[0, :, -1, :prompt_len].detach().float()
            per_layer_scores.append(prompt_attn.mean(dim=0))
        step_scores = torch.stack(per_layer_scores, dim=0).mean(dim=0)
        scores.append(step_scores.cpu().numpy().astype(np.float16))

        token_id = int(next_token.item())
        generated.append(token_id)
        if eos_token_id is not None and token_id == int(eos_token_id) and first_eos_step is None:
            first_eos_step = step + 1
            if stop_on_eos:
                break

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        del outputs, attentions, selected_layers, per_layer_scores, step_scores
        if torch.cuda.is_available() and (step + 1) % 32 == 0:
            torch.cuda.empty_cache()

    del prefill, past_key_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    score_matrix = np.stack(scores, axis=0)
    return {
        "scores": score_matrix,
        "generated_token_ids": np.asarray(generated, dtype=np.int64),
        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
        "first_eos_step": first_eos_step,
        "prompt_len": prompt_len,
    }


def save_trace(
    sample: SampleSpec,
    trace: dict[str, Any],
    decode_steps: list[int],
    compression_ratios: list[float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    run_id = f"{sample.dataset}__row{sample.dataset_row_index}__sample{sample.sample_index:02d}"
    output_path = RAW_DIR / f"{run_id}.npz"
    arrays: dict[str, Any] = {
        "scores": trace["scores"],
        "generated_token_ids": trace["generated_token_ids"],
    }

    rows: list[dict[str, Any]] = []
    for steps in decode_steps:
        if trace["scores"].shape[0] < steps:
            continue
        step_scores = trace["scores"][:steps].astype(np.float32)
        for ratio in compression_ratios:
            mask = topk_mask(step_scores, ratio)
            key = f"keep_mask_s{steps}_r{str(ratio).replace('.', 'p')}"
            arrays[key] = mask
            summary = summarize_mask(mask, step_scores, ratio)
            summary.update(
                {
                    "run_id": run_id,
                    "dataset": sample.dataset,
                    "dataset_row_index": sample.dataset_row_index,
                    "sample_index": sample.sample_index,
                    "decode_steps": steps,
                    "model": args.model,
                    "layer_window": args.layer_window,
                    "first_eos_step": trace["first_eos_step"],
                    "npz": str(output_path.relative_to(ROOT)),
                }
            )
            rows.append(summary)

    np.savez_compressed(output_path, **arrays)

    metadata_path = RAW_DIR / f"{run_id}.json"
    metadata = {
        "run_id": run_id,
        "sample": asdict(sample) | {"prompt": None, "context": None},
        "npz": str(output_path.relative_to(ROOT)),
        "generated_text": trace["generated_text"],
        "generated_tokens": int(trace["generated_token_ids"].shape[0]),
        "first_eos_step": trace["first_eos_step"],
        "decode_steps_requested": decode_steps,
        "compression_ratios": compression_ratios,
        "model": args.model,
        "layer_window": args.layer_window,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    append_jsonl(TRACE_JSONL, metadata)
    return rows


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_JSONL.write_text("")

    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    decode_steps = sorted(parse_csv_ints(args.decode_steps))
    compression_ratios = parse_csv_floats(args.compression_ratios)
    max_decode_steps = max(decode_steps)

    run_config = {
        "experiment_name": EXPERIMENT_NAME,
        "model": args.model,
        "datasets": dataset_names,
        "decode_steps": decode_steps,
        "compression_ratios": compression_ratios,
        "compression_ratio_semantics": "discard fraction; keep top 1-ratio prompt tokens per decode step",
        "granularity": "token",
        "seed": args.seed,
        "min_prompt_tokens": args.min_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "layer_window": args.layer_window,
        "device": args.device,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    RUN_CONFIG.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")

    print(f"[collect] loading tokenizer {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = select_samples(
        tokenizer=tokenizer,
        dataset_names=dataset_names,
        sample_count_per_dataset=args.sample_count_per_dataset,
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        seed=args.seed,
        sample_indices=parse_sample_indices(args.sample_indices),
    )
    manifest = [asdict(sample) | {"prompt": None, "context": None} for sample in samples]
    SAMPLE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"[collect] loading model {args.model}", flush=True)
    model_kwargs = {
        "torch_dtype": dtype_from_arg(args.dtype),
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(args.device)
    model.eval()

    all_rows: list[dict[str, Any]] = []
    for sample in samples:
        run_id = f"{sample.dataset}__row{sample.dataset_row_index}__sample{sample.sample_index:02d}"
        output_path = RAW_DIR / f"{run_id}.npz"
        if output_path.exists() and not args.force:
            print(f"[collect] skip existing {output_path}", flush=True)
            continue
        print(
            f"[collect] dataset={sample.dataset} row={sample.dataset_row_index} "
            f"prompt_tokens={sample.prompt_tokens} max_steps={max_decode_steps}",
            flush=True,
        )
        with torch.inference_mode():
            trace = trace_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                max_decode_steps=max_decode_steps,
                layer_window=args.layer_window,
                device=args.device,
                stop_on_eos=args.stop_on_eos,
            )
        all_rows.extend(save_trace(sample, trace, decode_steps, compression_ratios, args))
        write_summary_csv(all_rows)

    write_summary_csv(all_rows)
    print(f"[collect] wrote {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
