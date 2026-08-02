#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for pid_file in logs/local_ttt_*.pid logs/local_vllm_*.pid; do
  [[ -f "${pid_file}" ]] || continue
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${pid} from ${pid_file}"
    kill "${pid}" || true
  fi
  rm -f "${pid_file}"
done
