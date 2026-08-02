#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SEAL-main

for pidfile in logs/runpod_ttt_server.pid logs/runpod_vllm_server.pid; do
  if [[ -f "${pidfile}" ]]; then
    pid="$(cat "${pidfile}")"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Stopping ${pid} from ${pidfile}"
      kill "${pid}" || true
    fi
    rm -f "${pidfile}"
  fi
done
