#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="$REPO_ROOT/evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv"
ARTIFACTS_DIR="$EXP_DIR/artifacts"
WATCH_LOG="$ARTIFACTS_DIR/ATC26_watch_skip_gov_report.log"
RUNNER_LOG="$ARTIFACTS_DIR/ATC26_watch_skip_gov_report.runner.log"
LOCK_FILE="$ARTIFACTS_DIR/ATC26_watch_skip_gov_report.lock"
PROGRESS_MD="$ARTIFACTS_DIR/ATC26_progress.md"

mkdir -p "$ARTIFACTS_DIR"
cd "$REPO_ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '[%s] Another watchdog is already running; exiting.\n' "$(date '+%F %T')" | tee -a "$WATCH_LOG"
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$WATCH_LOG"
}

is_complete() {
  [[ -f "$PROGRESS_MD" ]] || return 1
  grep -q -- '- Pending: `0`' "$PROGRESS_MD" && grep -q -- '- Running: `0`' "$PROGRESS_MD"
}

export ATC26_SKIP_LONGBENCH_TASKS="${ATC26_SKIP_LONGBENCH_TASKS:-gov_report}"
export ATC26_GPUS="${ATC26_GPUS:-0,2}"
export ATC26_MIN_FREE_MB="${ATC26_MIN_FREE_MB:-0:36000,2:24000}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export POLL_SECONDS="${POLL_SECONDS:-60}"

log "watchdog started: skip=$ATC26_SKIP_LONGBENCH_TASKS gpus=$ATC26_GPUS min_free=$ATC26_MIN_FREE_MB"

while true; do
  if is_complete; then
    log "progress reports completion; watchdog exiting."
    exit 0
  fi

  log "starting runner with --resume"
  .venv/bin/python -u evaluation/ATC26_run_longbench16_prefill_sweep.py --mode full --resume >>"$RUNNER_LOG" 2>&1
  rc=$?
  log "runner exited with rc=$rc"

  if is_complete; then
    log "progress reports completion after runner exit; watchdog exiting."
    exit 0
  fi

  log "progress incomplete; restarting after 60s"
  sleep 60
done
