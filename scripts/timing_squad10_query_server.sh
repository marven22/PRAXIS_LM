#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs analysis_results/timing_calibration

ZMQ_PORT="${ZMQ_PORT:-5555}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B}"
PORT="${PORT:-18001}"

source .venv/bin/activate
set -a
[[ -f .env ]] && source <(tr -d '\r' < .env)
set +a

MODEL_NAME="${MODEL_NAME}" PORT="${PORT}" ZMQ_PORT="${ZMQ_PORT}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}" \
  bash scripts/local_start_ttt.sh

run_timed() {
  local name="$1"
  local module="$2"
  local dataset="$3"
  shift 3
  local start end elapsed
  start="$(python - <<'PY'
import time
print(time.time())
PY
)"
  echo "=== START ${name} $(date -Is) ==="
  python -u -m "${module}" \
    --exp_name "${name}" \
    --dataset "${dataset}" \
    --output_dir general-knowledge/results/query_server \
    --server_host 127.0.0.1 \
    --zmq_port "${ZMQ_PORT}" \
    --start_article 0 \
    --n_articles 10 \
    --k_completions 4 \
    --eval_times 3 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0 \
    --finetune_epochs 10 \
    --finetune_lr 1e-3 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --split_newlines \
    --reward_mode ttt \
    "$@" 2>&1 | tee "logs/${name}.log"
  end="$(python - <<'PY'
import time
print(time.time())
PY
)"
  elapsed="$(python - <<PY
start = float("${start}")
end = float("${end}")
print(round(end - start, 3))
PY
)"
  python - <<PY
import json
from pathlib import Path
elapsed = float("${elapsed}")
Path("analysis_results/timing_calibration/${name}.runtime.json").write_text(json.dumps({
    "name": "${name}",
    "n": 10,
    "hardware": "local",
    "total_seconds": round(elapsed, 3),
    "seconds_per_example": round(elapsed / 10.0, 3),
}, indent=2), encoding="utf-8")
PY
  echo "=== END ${name} elapsed=${elapsed} ==="
}

run_timed \
  "squad10_context_only" \
  "general-knowledge.src.query.query_server" \
  "general-knowledge/data/synthetic_data/train/iter0_train_50_passage_only.json"

run_timed \
  "squad10_praxis" \
  "praxis.general-knowledge.src.query.query_server" \
  "general-knowledge/data/synthetic_data/train/iter0_train_praxis_50_grounded.json" \
  --archive_path praxis/general-knowledge/archive/strategies.json

run_timed \
  "squad10_seal" \
  "general-knowledge.src.query.query_server" \
  "general-knowledge/data/synthetic_data/train/iter0_train_50.json"
