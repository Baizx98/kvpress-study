#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/experiments/blockwise_ablation_ratio70_stage1/artifacts}"
FRACTION="${FRACTION:-0.2}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${OUTPUT_DIR}/failed_tasks.txt}"
RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/run.log}"

mkdir -p "${OUTPUT_DIR}"
: > "${FAILED_TASKS_FILE}"
: > "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

PRESS="block_wise_prefill_per_layer"
RATIO="0.7"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SKIP_FIRST="${SKIP_FIRST:-1}"
Q_WINDOW="${Q_WINDOW:-64}"
SUMMARY_TOPK="${SUMMARY_TOPK:-4}"
MEAN_WEIGHT="${MEAN_WEIGHT:-0.75}"
REPRESENTATIVE_K="${REPRESENTATIVE_K:-4}"
MULTI_REP_K="${MULTI_REP_K:-4}"
QUERY_TOPR="${QUERY_TOPR:-16}"
HEAD_TOPK="${HEAD_TOPK:-1}"

COMMON_ARGS=(
  --dataset ruler
  --data_dir 4096
  --model "${MODEL}"
  --device "${DEVICE}"
  --press_name "${PRESS}"
  --compression_ratio "${RATIO}"
  --block_size "${BLOCK_SIZE}"
  --fraction "${FRACTION}"
  --max_new_tokens 128
  --query_aware true
  --prefill_skip_first_layers "${SKIP_FIRST}"
  --q_window_size "${Q_WINDOW}"
  --summary_topk_keys "${SUMMARY_TOPK}"
  --mean_key_weight "${MEAN_WEIGHT}"
  --representative_k "${REPRESENTATIVE_K}"
  --multi_rep_k "${MULTI_REP_K}"
  --query_topr "${QUERY_TOPR}"
  --head_topk "${HEAD_TOPK}"
  --output_dir "${OUTPUT_DIR}"
  --task_filter niah_single_3,niah_multikey_3,qa_2
)

run_one() {
  local tag="$1"
  shift
  echo "Running stage1 tag=${tag}"
  local cmd=(./.venv/bin/python evaluation/evaluate.py "${COMMON_ARGS[@]}" "$@")
  if ! PYTORCH_ALLOC_CONF=expandable_segments:True "${cmd[@]}"; then
    echo "FAILED tag=${tag}" | tee -a "${FAILED_TASKS_FILE}"
    sleep 2
  fi
}

run_one "baseline" \
  --summary_mode mean_plus_norm_topk_mean \
  --representative_mode key_norm \
  --query_agg_mode mean \
  --head_agg_mode uniform_mean

for summary_mode in mean_only norm_topk_mean_only mean_plus_norm_topk_mean multi_rep_max; do
  run_one "A_${summary_mode}" \
    --summary_mode "${summary_mode}" \
    --representative_mode key_norm \
    --query_agg_mode mean \
    --head_agg_mode uniform_mean
done

for representative_mode in key_norm tail_query_relevance random_topk; do
  if [[ "${representative_mode}" == "random_topk" ]]; then
    for seed in 42 43 44; do
      run_one "B_${representative_mode}_seed${seed}" \
        --summary_mode mean_plus_norm_topk_mean \
        --representative_mode "${representative_mode}" \
        --query_agg_mode mean \
        --head_agg_mode uniform_mean \
        --random_seed "${seed}"
    done
  else
    run_one "B_${representative_mode}" \
      --summary_mode mean_plus_norm_topk_mean \
      --representative_mode "${representative_mode}" \
      --query_agg_mode mean \
      --head_agg_mode uniform_mean
  fi
done

for query_agg_mode in mean max topr_mean; do
  run_one "C_${query_agg_mode}" \
    --summary_mode mean_plus_norm_topk_mean \
    --representative_mode key_norm \
    --query_agg_mode "${query_agg_mode}" \
    --head_agg_mode uniform_mean
done

for head_agg_mode in uniform_mean strength_weighted top_head_only; do
  run_one "D_${head_agg_mode}" \
    --summary_mode mean_plus_norm_topk_mean \
    --representative_mode key_norm \
    --query_agg_mode mean \
    --head_agg_mode "${head_agg_mode}"
done

run_one "quest_prefill" \
  --press_name quest_blockwise_prefill_per_layer \
  --summary_mode mean_plus_norm_topk_mean \
  --representative_mode key_norm \
  --query_agg_mode mean \
  --head_agg_mode uniform_mean \
  --quest_score_mode minmax
