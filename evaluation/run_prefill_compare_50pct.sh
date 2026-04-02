#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_prefill_compare_50pct}"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
FRACTION="${FRACTION:-0.5}"

RATIOS=(0.3 0.5 0.7)
PRESSES=("block_wise" "block_wise_legacy" "snapkv" "chunkkv")
GPUS=(0 1)

run_eval() {
  local dataset="$1"
  local data_dir="$2"
  local press="$3"
  local ratio="$4"
  local device="$5"
  shift 5

  echo "Running dataset=${dataset} data_dir=${data_dir} press=${press} ratio=${ratio} device=${device}"
  if [[ -n "${data_dir}" ]]; then
    "${PYTHON_BIN}" evaluate.py \
      --dataset "${dataset}" \
      --data_dir "${data_dir}" \
      --model "${MODEL}" \
      --press_name "${press}" \
      --compression_ratio "${ratio}" \
      --block_size 16 \
      --fraction "${FRACTION}" \
      --device "cuda:${device}" \
      --output_dir "${OUTPUT_DIR}" \
      "$@"
  else
    "${PYTHON_BIN}" evaluate.py \
      --dataset "${dataset}" \
      --model "${MODEL}" \
      --press_name "${press}" \
      --compression_ratio "${ratio}" \
      --block_size 16 \
      --fraction "${FRACTION}" \
      --device "cuda:${device}" \
      --output_dir "${OUTPUT_DIR}" \
      "$@"
  fi
}

run_dataset_grid() {
  local dataset="$1"
  local data_dir="$2"
  shift 2
  local extra_args=("$@")
  local job_idx=0

  for press in "${PRESSES[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      local gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
      run_eval "${dataset}" "${data_dir}" "${press}" "${ratio}" "${gpu}" "${extra_args[@]}" &
      job_idx=$((job_idx + 1))

      if (( job_idx % ${#GPUS[@]} == 0 )); then
        wait
      fi
    done
  done
  wait
}

mkdir -p "${OUTPUT_DIR}"

run_dataset_grid "ruler" "4096"
run_dataset_grid "longbench" "triviaqa"
run_dataset_grid "needle_in_haystack" "" --max_context_length 8192 --needle_depth "[0,25,50,75,100]"
