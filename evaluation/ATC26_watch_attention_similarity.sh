#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ART="evaluation/results/experiments/ATC26_blockwise_attention_similarity_hotpotqa_3samples/artifacts"
WATCHDOG_LOG="$ART/ATC26_collect_watchdog.log"
RUN_LOG="$ART/ATC26_collect_run.log"
COMMAND_FILE="$ART/ATC26_collect_command.txt"
AGG="$ART/ATC26_attention_similarity_aggregate.json"

mkdir -p "$ART/logs" "$ART/raw" "$ART/scores"

COMMAND=(
  .venv/bin/python
  evaluation/ATC26_collect_attention_similarity.py
  --device
  cuda:2
)

printf "%q " "${COMMAND[@]}" > "$COMMAND_FILE"
printf "\n" >> "$COMMAND_FILE"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date +"%F %T")] watchdog attempt=$attempt start" >> "$WATCHDOG_LOG"

  PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID "${COMMAND[@]}" >> "$RUN_LOG" 2>&1
  rc=$?

  if [[ -s "$AGG" ]]; then
    echo "[$(date +"%F %T")] watchdog success aggregate=$AGG rc=$rc" >> "$WATCHDOG_LOG"
    exit 0
  fi

  echo "[$(date +"%F %T")] watchdog restart rc=$rc aggregate_missing_or_empty=$AGG" >> "$WATCHDOG_LOG"
  sleep 60
done
