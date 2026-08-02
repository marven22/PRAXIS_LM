#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <log_prefix> <command...>"
  echo "Example: $0 logs/praxis_eval8 python few-shot/eval-self-edits.py ..."
  exit 1
fi

log_prefix="$1"
shift

mkdir -p "$(dirname "$log_prefix")"

stdout_log="${log_prefix}.stdout.log"
stderr_log="${log_prefix}.stderr.log"
time_log="${log_prefix}.time.log"
gpu_log="${log_prefix}.gpu.csv"
meta_log="${log_prefix}.meta.log"

echo "start_time=$(date --iso-8601=seconds)" > "$meta_log"
echo "cwd=$(pwd)" >> "$meta_log"
printf 'command=' >> "$meta_log"
printf '%q ' "$@" >> "$meta_log"
printf '\n' >> "$meta_log"

gpu_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  {
    echo "timestamp,memory.used,memory.total,utilization.gpu,utilization.memory"
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits 2>/dev/null || true
      sleep 2
    done
  } > "$gpu_log" &
  gpu_pid=$!
  echo "gpu_logger_pid=${gpu_pid}" >> "$meta_log"
fi

cleanup() {
  if [[ -n "${gpu_pid}" ]]; then
    kill "${gpu_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

/usr/bin/time -v -o "$time_log" "$@" >"$stdout_log" 2>"$stderr_log"

echo "end_time=$(date --iso-8601=seconds)" >> "$meta_log"
