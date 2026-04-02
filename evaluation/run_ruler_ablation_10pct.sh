#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_ruler_ablation_10pct}"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
FRACTION="${FRACTION:-0.1}"
DATA_DIR="${DATA_DIR:-4096}"
GPU_LIST="${GPU_LIST:-0,1}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
RATIOS=(0.5 0.7)

DEFAULT_Q_WINDOW=64
DEFAULT_TOPK=2
DEFAULT_RECENT=4
DEFAULT_MEAN_WEIGHT=0.5

run_eval() {
  local ratio="$1"
  local device="$2"
  local tag="$3"
  local q_window="$4"
  local topk="$5"
  local recent="$6"
  local mean_weight="$7"

  echo "Running tag=${tag} ratio=${ratio} device=${device} q_window=${q_window} topk=${topk} recent=${recent} mean_weight=${mean_weight}"
  "${PYTHON_BIN}" evaluate.py \
    --dataset ruler \
    --data_dir "${DATA_DIR}" \
    --model "${MODEL}" \
    --press_name block_wise \
    --compression_ratio "${ratio}" \
    --block_size 16 \
    --fraction "${FRACTION}" \
    --device "cuda:${device}" \
    --output_dir "${OUTPUT_DIR}" \
    --q_window_size "${q_window}" \
    --summary_topk_keys "${topk}" \
    --protected_recent_blocks "${recent}" \
    --mean_key_weight "${mean_weight}"
}

schedule_jobs() {
  local -a specs=("$@")
  local job_idx=0
  local gpu_count="${#GPUS[@]}"

  for spec in "${specs[@]}"; do
    IFS='|' read -r tag ratio q_window topk recent mean_weight <<< "${spec}"
    local gpu="${GPUS[$((job_idx % gpu_count))]}"
    run_eval "${ratio}" "${gpu}" "${tag}" "${q_window}" "${topk}" "${recent}" "${mean_weight}" &
    job_idx=$((job_idx + 1))
    if (( job_idx % gpu_count == 0 )); then
      wait
    fi
  done
  wait
}

mkdir -p "${OUTPUT_DIR}"

declare -a JOB_SPECS=()

for ratio in "${RATIOS[@]}"; do
  for q_window in 32 64 96; do
    JOB_SPECS+=("q_window|${ratio}|${q_window}|${DEFAULT_TOPK}|${DEFAULT_RECENT}|${DEFAULT_MEAN_WEIGHT}")
  done
  for topk in 1 2 4; do
    JOB_SPECS+=("summary_topk|${ratio}|${DEFAULT_Q_WINDOW}|${topk}|${DEFAULT_RECENT}|${DEFAULT_MEAN_WEIGHT}")
  done
  for recent in 2 4 8; do
    JOB_SPECS+=("protected_recent|${ratio}|${DEFAULT_Q_WINDOW}|${DEFAULT_TOPK}|${recent}|${DEFAULT_MEAN_WEIGHT}")
  done
  for mean_weight in 0.25 0.5 0.75; do
    JOB_SPECS+=("mean_key_weight|${ratio}|${DEFAULT_Q_WINDOW}|${DEFAULT_TOPK}|${DEFAULT_RECENT}|${mean_weight}")
  done
done

schedule_jobs "${JOB_SPECS[@]}"
