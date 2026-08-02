#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs analysis_results/ablations/component_suite/squad_query_server

source .venv/bin/activate
set -a
[[ -f .env ]] && source <(tr -d '\r' < .env)
set +a

SQUAD_ARTICLES="${SQUAD_ARTICLES:-10}"
ZMQ_PORT="${ZMQ_PORT:-5555}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="logs/squad_component_ablation_resume_${STAMP}.log"

log() {
  echo "$(date -Is) $*" | tee -a "${MASTER_LOG}"
}

run_step() {
  local name="$1"
  shift
  local log_path="logs/${name}.log"
  log "=== START ${name} ==="
  "$@" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  log "=== END ${name} exit=${status} log=${log_path} ==="
  return 0
}

COMMON_ARGS=(
  --dataset general-knowledge/data/synthetic_data/train/iter0_train_praxis.json
  --output_dir analysis_results/ablations/component_suite/squad_query_server
  --server_host 127.0.0.1
  --zmq_port "${ZMQ_PORT}"
  --start_article 0
  --n_articles "${SQUAD_ARTICLES}"
  --eval_times 3
  --lora_rank 32
  --lora_alpha 64
  --lora_dropout 0
  --finetune_epochs 10
  --finetune_lr 1e-3
  --batch_size 1
  --gradient_accumulation_steps 1
  --split_newlines
  --reward_mode ttt
)

log "=== SQUAD COMPONENT ABLATION RESUME START articles=${SQUAD_ARTICLES} zmq=${ZMQ_PORT} ==="

run_step "component_squad_full_praxis_archive" \
  python -u -m praxis.general-knowledge.src.query.query_server \
    --exp_name component_squad_full_praxis_archive \
    "${COMMON_ARGS[@]}" \
    --k_completions 4 \
    --archive_path praxis/general-knowledge/archive/strategies.json

run_step "component_squad_no_archive" \
  python -u -m praxis.general-knowledge.src.query.query_server \
    --exp_name component_squad_no_archive \
    "${COMMON_ARGS[@]}" \
    --k_completions 4

run_step "component_squad_fixed_generator_one_completion" \
  python -u -m praxis.general-knowledge.src.query.query_server \
    --exp_name component_squad_fixed_generator_one_completion \
    "${COMMON_ARGS[@]}" \
    --k_completions 1 \
    --archive_path praxis/general-knowledge/archive/strategies.json

run_step "component_squad_passage_only_no_generated_edit" \
  bash scripts/timing_squad10_passage_only_sft.sh

python scripts/summarize_praxis_component_ablations.py \
  --out analysis_results/ablations/component_suite/component_ablation_table.json | tee -a "${MASTER_LOG}"

log "=== SQUAD COMPONENT ABLATION RESUME COMPLETE ==="
