#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/ruler_residual_ablation_fast/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
DATA_DIR="${DATA_DIR:-4096}"
DEVICE="${DEVICE:-cuda:0}"
RATIO="${RATIO:-0.7}"
TASK_FILTER="${TASK_FILTER:-niah_multikey_2,niah_multikey_3,niah_single_3,qa_1,qa_2}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-6}"
WEIGHTS=(${WEIGHTS:-0.0 0.1 0.2 0.3 0.5})

mkdir -p "${OUTPUT_DIR}"

for weight in "${WEIGHTS[@]}"; do
  echo "Running residual ablation weight=${weight} device=${DEVICE}"
  "${PYTHON_BIN}" evaluate.py \
    --dataset ruler \
    --data_dir "${DATA_DIR}" \
    --model "${MODEL}" \
    --press_name block_wise \
    --compression_ratio "${RATIO}" \
    --block_size 16 \
    --fraction 1.0 \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}" \
    --task_filter "${TASK_FILTER}" \
    --samples_per_task "${SAMPLES_PER_TASK}" \
    --cross_layer_score_residual_weight "${weight}"
done
