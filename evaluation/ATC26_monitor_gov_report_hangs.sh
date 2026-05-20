#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="$REPO_ROOT/evaluation/results/experiments/ATC26_longbench16_prefill_sweep_blockwise_snapkv_chunkkv"
ARTIFACTS_DIR="$EXP_DIR/artifacts"
MONITOR_LOG="$ARTIFACTS_DIR/ATC26_monitor_gov_report_hangs.log"
LOCK_FILE="$ARTIFACTS_DIR/ATC26_monitor_gov_report_hangs.lock"

mkdir -p "$ARTIFACTS_DIR"
cd "$REPO_ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '[%s] Another gov_report hang monitor is already running; exiting.\n' "$(date '+%F %T')" | tee -a "$MONITOR_LOG"
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MONITOR_LOG"
}

threshold_seconds="${ATC26_HANG_D_SECONDS:-300}"
poll_seconds="${ATC26_HANG_POLL_SECONDS:-60}"

log "hang monitor started: threshold=${threshold_seconds}s poll=${poll_seconds}s"

while true; do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u || true)"
  ps -eo pid=,ppid=,stat=,etimes=,cmd= | while read -r pid ppid stat etimes cmd; do
    case "$cmd" in
      *"evaluation/evaluate.py"*gov_report*)
        if [[ "$stat" == D* ]] && (( etimes >= threshold_seconds )); then
          if ! printf '%s\n' "$gpu_pids" | grep -qx "$pid"; then
            log "killing hung gov_report pid=$pid ppid=$ppid stat=$stat etimes=${etimes}s cmd=$cmd"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 5
            kill -KILL "$pid" 2>/dev/null || true
          fi
        fi
        ;;
    esac
  done
  sleep "$poll_seconds"
done
