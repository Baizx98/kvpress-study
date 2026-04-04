#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/ruler_token_correction_50pct/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
FRACTION="${FRACTION:-0.5}"
DATA_DIR="${DATA_DIR:-4096}"
DEVICE="${DEVICE:-cuda:0}"
RATIO="${RATIO:-0.7}"

mkdir -p "${OUTPUT_DIR}"

echo "Running RULER token-correction validation:"
echo "  model=${MODEL}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  fraction=${FRACTION}"
echo "  ratio=${RATIO}"
echo "  device=${DEVICE}"

"${PYTHON_BIN}" evaluate.py \
  --dataset ruler \
  --data_dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --press_name block_wise \
  --compression_ratio "${RATIO}" \
  --block_size 16 \
  --fraction "${FRACTION}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}"
