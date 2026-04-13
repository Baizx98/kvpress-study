#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/experiments/blockwise_ablation_ratio70_longbench_stage1/artifacts}"
RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/run.log}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-30000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${OUTPUT_DIR}"
: >> "${RUN_LOG}"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}" | tee -a "${RUN_LOG}"
}

wait_for_gpu0_idle() {
  while true; do
    local line used total util free
    line="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F',' '$1 ~ /0/ {gsub(/ /, "", $0); print; exit}')"
    used="$(echo "${line}" | cut -d',' -f2)"
    total="$(echo "${line}" | cut -d',' -f3)"
    util="$(echo "${line}" | cut -d',' -f4)"
    free=$(( total - used ))

    if [[ -n "${free}" && "${free}" -ge "${MIN_FREE_MEMORY_MB}" ]]; then
      log "CUDA:0 has enough free memory (used=${used}MB free=${free}MB total=${total}MB util=${util}%). Starting LongBench stage1."
      return 0
    fi

    log "CUDA:0 does not have enough free memory yet (used=${used}MB free=${free}MB total=${total}MB util=${util}%). Waiting ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
}

wait_for_gpu0_idle
bash evaluation/run_blockwise_ablation_ratio70_longbench_stage1.sh
