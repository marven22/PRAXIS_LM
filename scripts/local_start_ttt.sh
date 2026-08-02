#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B}"
PORT="${PORT:-18001}"
ZMQ_PORT="${ZMQ_PORT:-5555}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-64}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.42}"
HOST="${HOST:-127.0.0.1}"
VLLM_API_URL="http://${HOST}:${PORT}"

source .venv/bin/activate
set -a
[[ -f .env ]] && source <(tr -d '\r' < .env)
set +a

MODEL_NAME="${MODEL_NAME}" PORT="${PORT}" MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
MAX_LORA_RANK="${MAX_LORA_RANK:-64}" VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
LOG_PATH="logs/local_vllm_ttt_${PORT}.log" PID_PATH="logs/local_vllm_ttt_${PORT}.pid" \
  bash scripts/local_start_vllm.sh

if [[ -f logs/local_ttt_${ZMQ_PORT}.pid ]] && kill -0 "$(cat logs/local_ttt_${ZMQ_PORT}.pid)" 2>/dev/null; then
  echo "TTT server already running with PID $(cat logs/local_ttt_${ZMQ_PORT}.pid)"
  exit 0
fi

echo "Starting TTT server on tcp://127.0.0.1:${ZMQ_PORT}"
nohup python -m general-knowledge.src.inner.TTT_server \
  --vllm_api_url "${VLLM_API_URL}" \
  --model "${MODEL_NAME}" \
  --zmq_port "${ZMQ_PORT}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --eval_max_tokens "${EVAL_MAX_TOKENS}" \
  --eval_temperature "${EVAL_TEMPERATURE}" \
  --eval_top_p "${EVAL_TOP_P}" \
  > "logs/local_ttt_${ZMQ_PORT}.log" 2>&1 &
echo $! > "logs/local_ttt_${ZMQ_PORT}.pid"
echo "TTT server starting. Log: logs/local_ttt_${ZMQ_PORT}.log"
