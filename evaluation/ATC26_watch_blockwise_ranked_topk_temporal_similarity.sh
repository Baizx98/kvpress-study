#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXP="ATC26_blockwise_ranked_topk_temporal_similarity"
RUN_TAG="${RUN_TAG:-decode1024}"
DECODE_STEPS="${DECODE_STEPS:-1024}"
if [[ "$RUN_TAG" == "full" ]]; then
  ART="evaluation/results/experiments/${EXP}/artifacts"
else
  ART="evaluation/results/experiments/${EXP}/artifacts/${RUN_TAG}"
fi
LOG_DIR="$ART/logs"
WATCHDOG_LOG="$LOG_DIR/ATC26_ranked_topk_temporal_similarity_watchdog.log"
RUN_LOG="$LOG_DIR/ATC26_ranked_topk_temporal_similarity_run.log"
COMMAND_FILE="$ART/ATC26_ranked_topk_temporal_similarity_command.txt"
HEARTBEAT="$ART/ATC26_ranked_topk_temporal_similarity_heartbeat.json"
AGG="$ART/ATC26_ranked_topk_temporal_similarity_aggregate.json"
PID_FILE="$ART/ATC26_ranked_topk_temporal_similarity_watchdog.pid"

mkdir -p "$LOG_DIR" "$ART/raw" "$ART/indices"
echo "$$" > "$PID_FILE"

COMMAND=(
  .venv/bin/python
  evaluation/ATC26_collect_blockwise_ranked_topk_temporal_similarity.py
  --device cuda:0
  --model-key llama31_8b_instruct
  --dataset pg19
  --context-lengths 8192 16384
  --samples-per-length 4
  --decode-steps "$DECODE_STEPS"
  --compression-ratios 0.7 0.5 0.3
  --block-size 16
  --window-query-size 16
  --lags 1 2 4 8 16 32 64 128 256 512
  --reuse-intervals 2 4 8 16 32 64 128 256 512
  --seed 42
  --run-tag "$RUN_TAG"
  --resume
)

printf "%q " "CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=2" "PYTHONUNBUFFERED=1" "${COMMAND[@]}" > "$COMMAND_FILE"
printf "\n" >> "$COMMAND_FILE"

log() {
  echo "[$(date +"%F %T")] $*" | tee -a "$WATCHDOG_LOG"
}

heartbeat_age_seconds() {
  if [[ ! -s "$HEARTBEAT" ]]; then
    if [[ -e "$HEARTBEAT" ]]; then
      local now hb
      now=$(date +%s)
      hb=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)
      echo $((now - hb))
    else
      echo 999999
    fi
    return
  fi
  local now hb
  now=$(date +%s)
  hb=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)
  echo $((now - hb))
}

kill_tree() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return
  fi
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  sleep 15
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

attempt=0
failures=0
while true; do
  attempt=$((attempt + 1))
  log "attempt=${attempt} start"
  cat > "$HEARTBEAT" <<EOF
{"updated_at":"$(date -Iseconds)","status":"watchdog_starting_child","attempt":${attempt},"pid":$$}
EOF

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "${COMMAND[@]}" >> "$RUN_LOG" 2>&1 &
  child=$!
  log "child_pid=${child}"

  stale=0
  while kill -0 "$child" 2>/dev/null; do
    age=$(heartbeat_age_seconds)
    if (( age > 1200 )); then
      log "heartbeat stale age=${age}s; collecting diagnostics and restarting child=${child}"
      {
        echo "===== $(date +"%F %T") ps ====="
        ps -o pid,ppid,stat,etime,cmd -p "$child" --forest
        echo "===== $(date +"%F %T") nvidia-smi ====="
        nvidia-smi
        echo "===== recent run log ====="
        tail -n 80 "$RUN_LOG"
      } >> "$WATCHDOG_LOG" 2>&1
      stale=1
      kill_tree "$child"
      break
    fi
    sleep 60
  done

  wait "$child"
  rc=$?
  log "child exit rc=${rc} stale=${stale}"

  if [[ -s "$AGG" ]]; then
    status=$(python - "$HEARTBEAT" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if p.exists():
    print(json.loads(p.read_text()).get("status", ""))
PY
)
    if [[ "$status" == "complete" ]]; then
      log "success aggregate=${AGG}"
      exit 0
    fi
  fi

  failures=$((failures + 1))
  if (( failures >= 5 )); then
    log "too many failures=${failures}; exiting"
    exit 1
  fi

  log "restart after rc=${rc}; failures=${failures}"
  sleep 60
done
