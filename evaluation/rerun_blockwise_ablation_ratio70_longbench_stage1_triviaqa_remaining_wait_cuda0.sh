#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
GPU_INDEX="${GPU_INDEX:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results/experiments/blockwise_ablation_ratio70_longbench_stage1/artifacts}"
FRACTION="${FRACTION:-0.2}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${OUTPUT_DIR}/failed_tasks_triviaqa_remaining.txt}"
RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/run.log}"

mkdir -p "${OUTPUT_DIR}"
: > "${FAILED_TASKS_FILE}"
exec > >(tee -a "${RUN_LOG}") 2>&1

DATASET="triviaqa"
RATIO="0.7"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
Q_WINDOW="${Q_WINDOW:-64}"
SUMMARY_TOPK="${SUMMARY_TOPK:-4}"
MEAN_WEIGHT="${MEAN_WEIGHT:-0.75}"
REPRESENTATIVE_K="${REPRESENTATIVE_K:-4}"
MULTI_REP_K="${MULTI_REP_K:-4}"
QUERY_TOPR="${QUERY_TOPR:-16}"
HEAD_TOPK="${HEAD_TOPK:-1}"
MAX_NEW_TOKENS=52

COMMON_ARGS=(
  --dataset longbench
  --data_dir "${DATASET}"
  --model "${MODEL}"
  --device "${DEVICE}"
  --compression_ratio "${RATIO}"
  --block_size "${BLOCK_SIZE}"
  --fraction "${FRACTION}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --query_aware true
  --q_window_size "${Q_WINDOW}"
  --summary_topk_keys "${SUMMARY_TOPK}"
  --mean_key_weight "${MEAN_WEIGHT}"
  --representative_k "${REPRESENTATIVE_K}"
  --multi_rep_k "${MULTI_REP_K}"
  --query_topr "${QUERY_TOPR}"
  --head_topk "${HEAD_TOPK}"
  --output_dir "${OUTPUT_DIR}"
)

wait_for_gpu_free() {
  while true; do
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((GPU_INDEX + 1))p" | tr -d ' ')"
    if [[ -n "${free_mb}" ]] && (( free_mb >= MIN_FREE_MB )); then
      echo "CUDA:${GPU_INDEX} free memory ${free_mb}MB >= ${MIN_FREE_MB}MB, starting next missing run."
      return 0
    fi
    echo "CUDA:${GPU_INDEX} free memory ${free_mb:-unknown}MB < ${MIN_FREE_MB}MB, waiting ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
}

run_one() {
  local tag="$1"
  shift
  wait_for_gpu_free
  echo "Rerunning remaining triviaqa config tag=${tag} device=${DEVICE}"
  local cmd=(./.venv/bin/python evaluation/evaluate.py "${COMMON_ARGS[@]}" "$@")
  if ! PYTORCH_ALLOC_CONF=expandable_segments:True "${cmd[@]}"; then
    echo "FAILED rerun tag=${tag} dataset=${DATASET}" | tee -a "${FAILED_TASKS_FILE}"
    sleep 2
  fi
}

run_postprocess() {
  echo "Postprocessing longbench stage1 experiment results"
  ./.venv/bin/python evaluation/postprocess_blockwise_ablation_ratio70_longbench_stage1.py
}

trap run_postprocess EXIT

run_one "A_multi_rep_max" \
  --press_name block_wise_prefill_per_layer \
  --summary_mode multi_rep_max \
  --representative_mode key_norm \
  --query_agg_mode mean \
  --head_agg_mode uniform_mean

run_one "C_max" \
  --press_name block_wise_prefill_per_layer \
  --summary_mode mean_plus_norm_topk_mean \
  --representative_mode key_norm \
  --query_agg_mode max \
  --head_agg_mode uniform_mean

run_one "C_topr_mean" \
  --press_name block_wise_prefill_per_layer \
  --summary_mode mean_plus_norm_topk_mean \
  --representative_mode key_norm \
  --query_agg_mode topr_mean \
  --head_agg_mode uniform_mean

run_one "D_strength_weighted" \
  --press_name block_wise_prefill_per_layer \
  --summary_mode mean_plus_norm_topk_mean \
  --representative_mode key_norm \
  --query_agg_mode mean \
  --head_agg_mode strength_weighted

run_one "D_top_head_only" \
  --press_name block_wise_prefill_per_layer \
  --summary_mode mean_plus_norm_topk_mean \
  --representative_mode key_norm \
  --query_agg_mode mean \
  --head_agg_mode top_head_only

run_one "baseline" \
  --press_name block_wise_prefill_per_layer \
  --summary_mode mean_plus_norm_topk_mean \
  --representative_mode key_norm \
  --query_agg_mode mean \
  --head_agg_mode uniform_mean
