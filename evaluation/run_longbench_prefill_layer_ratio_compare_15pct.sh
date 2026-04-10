#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/experiments/longbench_prefill_layer_ratio_compare_15pct/artifacts}"
FRACTION="${FRACTION:-0.15}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${OUTPUT_DIR}/failed_tasks.txt}"

mkdir -p "${OUTPUT_DIR}"
: > "${FAILED_TASKS_FILE}"

DATASETS=(
  "hotpotqa"
  "multifieldqa_en"
  "triviaqa"
)

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

for dataset in "${DATASETS[@]}"; do
  case "${dataset}" in
    hotpotqa)
      DATASET_MAX_NEW_TOKENS=52
      ;;
    multifieldqa_en)
      DATASET_MAX_NEW_TOKENS=84
      ;;
    triviaqa)
      DATASET_MAX_NEW_TOKENS=52
      ;;
    *)
      DATASET_MAX_NEW_TOKENS=64
      ;;
  esac
  for press in "${PRESSES[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for skip_first in "${SKIP_FIRSTS[@]}"; do
        echo "Running dataset=${dataset} press=${press} ratio=${ratio} skip_first=${skip_first} max_new_tokens=${DATASET_MAX_NEW_TOKENS}"
        if ! PYTORCH_ALLOC_CONF=expandable_segments:True ./.venv/bin/python evaluation/evaluate.py \
          --dataset longbench \
          --data_dir "${dataset}" \
          --model "${MODEL}" \
          --device "${DEVICE}" \
          --press_name "${press}" \
          --compression_ratio "${ratio}" \
          --block_size 16 \
          --fraction "${FRACTION}" \
          --max_new_tokens "${DATASET_MAX_NEW_TOKENS}" \
          --query_aware true \
          --prefill_skip_first_layers "${skip_first}" \
          --output_dir "${OUTPUT_DIR}"; then
          echo "FAILED dataset=${dataset} press=${press} ratio=${ratio} skip_first=${skip_first}" | tee -a "${FAILED_TASKS_FILE}"
          sleep 2
        fi
      done
    done
  done
done
