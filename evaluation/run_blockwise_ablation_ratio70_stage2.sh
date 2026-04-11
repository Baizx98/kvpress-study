#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/experiments/blockwise_ablation_ratio70_stage2/artifacts}"
FRACTION="${FRACTION:-0.2}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${OUTPUT_DIR}/failed_tasks.txt}"

mkdir -p "${OUTPUT_DIR}"
: > "${FAILED_TASKS_FILE}"

COMMON_ARGS=(
  --model "${MODEL}"
  --device "${DEVICE}"
  --compression_ratio 0.7
  --block_size 16
  --fraction "${FRACTION}"
  --query_aware true
  --prefill_skip_first_layers 1
  --q_window_size 64
  --summary_topk_keys 4
  --mean_key_weight 0.75
  --representative_k 4
  --multi_rep_k 4
  --query_topr 16
  --head_topk 1
  --output_dir "${OUTPUT_DIR}"
)

run_one() {
  local tag="$1"
  shift
  echo "Running stage2 tag=${tag}"
  local cmd=(./.venv/bin/python evaluation/evaluate.py "${COMMON_ARGS[@]}" "$@")
  if ! PYTORCH_ALLOC_CONF=expandable_segments:True "${cmd[@]}"; then
    echo "FAILED tag=${tag}" | tee -a "${FAILED_TASKS_FILE}"
    sleep 2
  fi
}

for dataset_name in qasper hotpotqa 2wikimqa musique; do
  run_one "longbench_${dataset_name}_current_baseline" \
    --dataset longbench \
    --data_dir "${dataset_name}" \
    --max_new_tokens 128 \
    --samples_per_task 8 \
    --press_name block_wise_prefill_per_layer \
    --summary_mode mean_plus_norm_topk_mean \
    --representative_mode key_norm \
    --query_agg_mode mean \
    --head_agg_mode uniform_mean

  run_one "longbench_${dataset_name}_best_blockwise" \
    --dataset longbench \
    --data_dir "${dataset_name}" \
    --max_new_tokens 128 \
    --samples_per_task 8 \
    --press_name block_wise_prefill_per_layer \
    --summary_mode multi_rep_max \
    --representative_mode tail_query_relevance \
    --query_agg_mode topr_mean \
    --head_agg_mode strength_weighted

  run_one "longbench_${dataset_name}_quest_prefill" \
    --dataset longbench \
    --data_dir "${dataset_name}" \
    --max_new_tokens 128 \
    --samples_per_task 8 \
    --press_name quest_blockwise_prefill_per_layer \
    --summary_mode mean_plus_norm_topk_mean \
    --representative_mode key_norm \
    --query_agg_mode topr_mean \
    --head_agg_mode strength_weighted
done

run_one "needle_best_blockwise" \
  --dataset needle_in_haystack \
  --needle_depth 50 \
  --max_context_length 16384 \
  --max_new_tokens 50 \
  --press_name block_wise_prefill_per_layer \
  --summary_mode multi_rep_max \
  --representative_mode tail_query_relevance \
  --query_agg_mode topr_mean \
  --head_agg_mode strength_weighted

run_one "needle_quest_prefill" \
  --dataset needle_in_haystack \
  --needle_depth 50 \
  --max_context_length 16384 \
  --max_new_tokens 50 \
  --press_name quest_blockwise_prefill_per_layer \
  --summary_mode mean_plus_norm_topk_mean \
  --representative_mode key_norm \
  --query_agg_mode topr_mean \
  --head_agg_mode strength_weighted
