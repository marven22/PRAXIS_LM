#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B}"
PORT="${PORT:-18001}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
HOST="${HOST:-127.0.0.1}"
VLLM_API_URL="http://${HOST}:${PORT}"
LOG_PATH="${LOG_PATH:-logs/local_vllm_${PORT}.log}"
PID_PATH="${PID_PATH:-logs/local_vllm_${PORT}.pid}"

source .venv/bin/activate
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

if curl --silent --fail "${VLLM_API_URL}/v1/models" >/dev/null 2>&1; then
  echo "vLLM endpoint already ready at ${VLLM_API_URL}"
  exit 0
fi

if [[ -f "${PID_PATH}" ]] && kill -0 "$(cat "${PID_PATH}")" 2>/dev/null; then
  echo "vLLM already running with PID $(cat "${PID_PATH}")"
  exit 0
fi

echo "Starting vLLM for ${MODEL_NAME} on ${VLLM_API_URL}"
nohup vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_SEQ_LENGTH}" \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  > "${LOG_PATH}" 2>&1 &
echo $! > "${PID_PATH}"

echo "Waiting for vLLM..."
until curl --silent --fail "${VLLM_API_URL}/v1/models" >/dev/null; do
  sleep 5
done
echo "vLLM ready. Log: ${LOG_PATH}"
