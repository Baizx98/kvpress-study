#!/usr/bin/env bash
set -uo pipefail

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/batch_main_compare_ratio05/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
RATIO="${RATIO:-0.5}"
FRACTION="${FRACTION:-1}"
SAFE_MAX_CONTEXT="${SAFE_MAX_CONTEXT:-32768}"

mkdir -p "${OUTPUT_DIR}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

PRESSES=("block_wise" "chunkkv")

MISSING_SPECS=(
  "longbench-v2|0shot|${SAFE_MAX_CONTEXT}"
  "infinitebench|passkey|${SAFE_MAX_CONTEXT}"
  "infinitebench|kv_retrieval|${SAFE_MAX_CONTEXT}"
  "infinitebench|longbook_qa_eng|${SAFE_MAX_CONTEXT}"
  "loogle|shortdep_qa|${SAFE_MAX_CONTEXT}"
  "loogle|longdep_qa|${SAFE_MAX_CONTEXT}"
  "loogle|longdep_summarization|${SAFE_MAX_CONTEXT}"
)

for spec in "${MISSING_SPECS[@]}"; do
  IFS='|' read -r dataset data_dir max_context_length <<< "${spec}"
  for press in "${PRESSES[@]}"; do
    metrics_glob="${OUTPUT_DIR}/${dataset}__${data_dir}__*__${press}__${RATIO}0__max_context${max_context_length}__query_aware/metrics.json"
    if compgen -G "${metrics_glob}" > /dev/null; then
      echo "Skipping completed dataset=${dataset} data_dir=${data_dir} press=${press} ratio=${RATIO} max_context_length=${max_context_length}"
      continue
    fi

    echo "Resuming dataset=${dataset} data_dir=${data_dir} press=${press} ratio=${RATIO} fraction=${FRACTION} device=${DEVICE} max_context_length=${max_context_length}"
    if ! "${PYTHON_BIN}" evaluate.py \
      --dataset "${dataset}" \
      --data_dir "${data_dir}" \
      --model "${MODEL}" \
      --press_name "${press}" \
      --compression_ratio "${RATIO}" \
      --fraction "${FRACTION}" \
      --max_context_length "${max_context_length}" \
      --device "${DEVICE}" \
      --output_dir "${OUTPUT_DIR}"; then
      echo "FAILED dataset=${dataset} data_dir=${data_dir} press=${press} ratio=${RATIO} max_context_length=${max_context_length}" >&2
    fi
  done
done
