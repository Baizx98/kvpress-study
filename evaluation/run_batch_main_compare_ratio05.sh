#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/batch_main_compare_ratio05/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
RATIO="${RATIO:-0.5}"
FRACTION="${FRACTION:-1}"

mkdir -p "${OUTPUT_DIR}"

PRESSES=("block_wise" "chunkkv")

SPECS=(
  "longbench|hotpotqa"
  "longbench|multifieldqa_en"
  "longbench|triviaqa"
  "longbench-v2|0shot"
  "infinitebench|passkey"
  "infinitebench|kv_retrieval"
  "infinitebench|longbook_qa_eng"
  "loogle|shortdep_qa"
  "loogle|longdep_qa"
  "loogle|longdep_summarization"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r dataset data_dir <<< "${spec}"
  for press in "${PRESSES[@]}"; do
    echo "Running dataset=${dataset} data_dir=${data_dir} press=${press} ratio=${RATIO} fraction=${FRACTION} device=${DEVICE}"
    "${PYTHON_BIN}" evaluate.py \
      --dataset "${dataset}" \
      --data_dir "${data_dir}" \
      --model "${MODEL}" \
      --press_name "${press}" \
      --compression_ratio "${RATIO}" \
      --fraction "${FRACTION}" \
      --device "${DEVICE}" \
      --output_dir "${OUTPUT_DIR}"
  done
done
