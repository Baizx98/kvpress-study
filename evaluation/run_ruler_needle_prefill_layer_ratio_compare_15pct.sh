#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/experiments/ruler_needle_prefill_layer_ratio_compare_15pct/artifacts}"
FRACTION="${FRACTION:-0.15}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${OUTPUT_DIR}/failed_tasks.txt}"

mkdir -p "${OUTPUT_DIR}"
: > "${FAILED_TASKS_FILE}"

PRESSES=(
  "block_wise_prefill_per_layer"
  "chunkkv_prefill_per_layer"
)

RATIOS=(
  "0.3"
  "0.5"
  "0.7"
)

SKIP_FIRSTS=(
  "0"
  "1"
  "2"
)

run_one() {
  local dataset="$1"
  local data_dir="$2"
  local press="$3"
  local ratio="$4"
  local skip_first="$5"
  local max_new_tokens="$6"
  local extra_args="$7"

  echo "Running dataset=${dataset} data_dir=${data_dir} press=${press} ratio=${ratio} skip_first=${skip_first}"
  local cmd=(
    ./.venv/bin/python evaluation/evaluate.py
    --dataset "${dataset}"
    --model "${MODEL}"
    --device "${DEVICE}"
    --press_name "${press}"
    --compression_ratio "${ratio}"
    --block_size 16
    --fraction "${FRACTION}"
    --max_new_tokens "${max_new_tokens}"
    --query_aware true
    --prefill_skip_first_layers "${skip_first}"
    --output_dir "${OUTPUT_DIR}"
  )
  if [[ -n "${data_dir}" ]]; then
    cmd+=(--data_dir "${data_dir}")
  fi
  if [[ -n "${extra_args}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${extra_args} )
    cmd+=("${extra[@]}")
  fi

  if ! PYTORCH_ALLOC_CONF=expandable_segments:True "${cmd[@]}"; then
    echo "FAILED dataset=${dataset} data_dir=${data_dir} press=${press} ratio=${ratio} skip_first=${skip_first}" | tee -a "${FAILED_TASKS_FILE}"
    sleep 2
  fi
}

# RULER: representative tasks only, keep runtime manageable
RULER_EXTRA_ARGS="--task_filter niah_single_3,niah_multikey_3,qa_2 --samples_per_task 6"
for press in "${PRESSES[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    for skip_first in "${SKIP_FIRSTS[@]}"; do
      run_one "ruler" "4096" "${press}" "${ratio}" "${skip_first}" "128" "${RULER_EXTRA_ARGS}"
    done
  done
done

# Needle in a Haystack: use representative middle depth
NEEDLE_EXTRA_ARGS="--needle_depth 50 --max_context_length 16384"
for press in "${PRESSES[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    for skip_first in "${SKIP_FIRSTS[@]}"; do
      run_one "needle_in_haystack" "" "${press}" "${ratio}" "${skip_first}" "50" "${NEEDLE_EXTRA_ARGS}"
    done
  done
done
