#!/usr/bin/env bash
set -euo pipefail

ROOT="/home10T/bzx/workspace/kvpress-study/evaluation"
MAIN_SESSION="dualphase_nonpermanent_ratio05_main"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/experiments/dualphase_nonpermanent_ratio05/artifacts}"
LOG_FILE="${OUTPUT_DIR}/watchdog.log"
RUN_SCRIPT="${ROOT}/run_dualphase_nonpermanent_ratio05.sh"
EXPECTED_RESULTS=4

mkdir -p "${OUTPUT_DIR}"

count_results() {
  find "${OUTPUT_DIR}" -name metrics.json | wc -l | tr -d ' '
}

while true; do
  completed="$(count_results)"
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  if [[ "${completed}" -ge "${EXPECTED_RESULTS}" ]]; then
    echo "${timestamp} watchdog: all ${completed}/${EXPECTED_RESULTS} results completed, stopping watchdog." | tee -a "${LOG_FILE}"
    exit 0
  fi

  if ! tmux has-session -t "${MAIN_SESSION}" 2>/dev/null; then
    echo "${timestamp} watchdog: main session missing with ${completed}/${EXPECTED_RESULTS} results, restarting." | tee -a "${LOG_FILE}"
    tmux new-session -d -s "${MAIN_SESSION}" "cd ${ROOT} && ./run_dualphase_nonpermanent_ratio05.sh 2>&1 | tee -a results/experiments/dualphase_nonpermanent_ratio05/artifacts/run.log"
  else
    echo "${timestamp} watchdog: main session alive, ${completed}/${EXPECTED_RESULTS} results present." | tee -a "${LOG_FILE}"
  fi

  sleep "${CHECK_INTERVAL}"
done
