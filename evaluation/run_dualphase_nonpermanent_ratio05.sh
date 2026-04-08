#!/usr/bin/env bash
set -uo pipefail

MODEL="${MODEL:-/Tan/model/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/dualphase_nonpermanent_ratio05/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
RATIO="${RATIO:-0.5}"
FRACTION="${FRACTION:-1}"

mkdir -p "${OUTPUT_DIR}"

SPECS=(
  "longbench|hotpotqa|"
  "longbench|multifieldqa_en|"
  "longbench|triviaqa|"
  "longbench-v2|0shot|32768"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r dataset data_dir max_context_length <<< "${spec}"

  suffix=""
  if [[ -n "${max_context_length}" ]]; then
    suffix="__max_context${max_context_length}"
  fi
  metrics_path="${OUTPUT_DIR}/${dataset}__${data_dir}__--Tan--model--Llama-3.1-8B-Instruct__dual_phase_per_layer__0.50${suffix}__query_aware/metrics.json"
  if [[ -f "${metrics_path}" ]]; then
    echo "Skipping completed dataset=${dataset} data_dir=${data_dir}"
    continue
  fi

  echo "Running dataset=${dataset} data_dir=${data_dir} press=dual_phase_per_layer ratio=${RATIO} fraction=${FRACTION} device=${DEVICE} max_context_length=${max_context_length:-none}"
  cmd=(
    "${PYTHON_BIN}" evaluate.py
    --dataset "${dataset}"
    --data_dir "${data_dir}"
    --model "${MODEL}"
    --press_name dual_phase_per_layer
    --compression_ratio "${RATIO}"
    --fraction "${FRACTION}"
    --device "${DEVICE}"
    --output_dir "${OUTPUT_DIR}"
  )
  if [[ -n "${max_context_length}" ]]; then
    cmd+=(--max_context_length "${max_context_length}")
  fi

  if ! "${cmd[@]}"; then
    echo "FAILED dataset=${dataset} data_dir=${data_dir} ratio=${RATIO} device=${DEVICE}" >&2
  fi
done
