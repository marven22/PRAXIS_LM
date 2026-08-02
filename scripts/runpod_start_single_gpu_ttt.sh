#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SEAL-main
mkdir -p logs models

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B}"
PORT="${PORT:-18001}"
ZMQ_PORT="${ZMQ_PORT:-5555}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-64}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"

HOST="127.0.0.1"
VLLM_API_URL="http://${HOST}:${PORT}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export CUDA_VISIBLE_DEVICES=0

echo "Starting vLLM for ${MODEL_NAME} on ${VLLM_API_URL}"
nohup vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_SEQ_LENGTH}" \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  > logs/runpod_vllm_server.log 2>&1 &
echo $! > logs/runpod_vllm_server.pid

echo "Waiting for vLLM model endpoint..."
until curl --silent --fail "${VLLM_API_URL}/v1/models" | grep -q "${MODEL_NAME}"; do
  sleep 5
done
echo "vLLM ready."

echo "Starting TTT server on tcp://127.0.0.1:${ZMQ_PORT}"
nohup python -m general-knowledge.src.inner.TTT_server \
  --vllm_api_url "${VLLM_API_URL}" \
  --model "${MODEL_NAME}" \
  --zmq_port "${ZMQ_PORT}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --eval_max_tokens "${EVAL_MAX_TOKENS}" \
  --eval_temperature "${EVAL_TEMPERATURE}" \
  --eval_top_p "${EVAL_TOP_P}" \
  > logs/runpod_ttt_server.log 2>&1 &
echo $! > logs/runpod_ttt_server.pid

echo "Started. Logs:"
echo "  logs/runpod_vllm_server.log"
echo "  logs/runpod_ttt_server.log"
