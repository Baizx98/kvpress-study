#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/ruler_cross_layer_residual_50pct/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
FRACTION="${FRACTION:-0.5}"
DATA_DIR="${DATA_DIR:-4096}"
DEVICE="${DEVICE:-cuda:0}"
RATIO="${RATIO:-0.7}"
LAYER_RES_W="${LAYER_RES_W:-0.2}"

mkdir -p "${OUTPUT_DIR}"

echo "Running RULER cross-layer residual validation:"
echo "  model=${MODEL}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  fraction=${FRACTION}"
echo "  ratio=${RATIO}"
echo "  device=${DEVICE}"
echo "  layer_res_w=${LAYER_RES_W}"

"${PYTHON_BIN}" evaluate.py \
  --dataset ruler \
  --data_dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --press_name block_wise \
  --compression_ratio "${RATIO}" \
  --block_size 16 \
  --fraction "${FRACTION}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}" \
  --cross_layer_score_residual_weight "${LAYER_RES_W}"
