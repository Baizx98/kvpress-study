#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/experiments/quest_prefill_ratio70/artifacts}"
FRACTION="${FRACTION:-0.2}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${OUTPUT_DIR}/failed_tasks.txt}"

mkdir -p "${OUTPUT_DIR}"
: > "${FAILED_TASKS_FILE}"

run_one() {
  local dataset="$1"
  local extra_args="$2"
  echo "Running quest prefill dataset=${dataset}"
  local cmd=(
    ./.venv/bin/python evaluation/evaluate.py
    --dataset "${dataset}"
    --model "${MODEL}"
    --device "${DEVICE}"
    --press_name quest_blockwise_prefill_per_layer
    --compression_ratio 0.7
    --block_size 16
    --fraction "${FRACTION}"
    --query_aware true
    --prefill_skip_first_layers 1
    --q_window_size 64
    --query_agg_mode topr_mean
    --head_agg_mode strength_weighted
    --query_topr 16
    --output_dir "${OUTPUT_DIR}"
  )
  if [[ -n "${extra_args}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${extra_args} )
    cmd+=("${extra[@]}")
  fi

  if ! PYTORCH_ALLOC_CONF=expandable_segments:True "${cmd[@]}"; then
    echo "FAILED dataset=${dataset}" | tee -a "${FAILED_TASKS_FILE}"
    sleep 2
  fi
}

run_one "ruler" "--data_dir 4096 --max_new_tokens 128 --task_filter niah_single_3,niah_multikey_3,qa_2 --samples_per_task 8"
for dataset_name in qasper hotpotqa 2wikimqa musique; do
  run_one "longbench" "--data_dir ${dataset_name} --max_new_tokens 128 --samples_per_task 8"
done
run_one "needle_in_haystack" "--needle_depth 50 --max_context_length 16384 --max_new_tokens 50"
