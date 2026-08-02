#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs analysis_results/rq3_crossmodel

source .venv/bin/activate
set -a
[[ -f .env ]] && source <(tr -d '\r' < .env)
set +a

MASTER_LOG="logs/rq3_crossmodel_all_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "$(date -Is) $*" | tee -a "${MASTER_LOG}"
}

run_cmd() {
  local name="$1"
  shift
  local log_path="logs/${name}.log"
  log "=== START ${name} ==="
  "$@" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  log "=== END ${name} exit=${status} log=${log_path} ==="
  return 0
}

run_clutrr() {
  run_cmd "rq3_clutrr_rule_proposer_qwen25_3b" \
    python scripts/run_clutrr_llm_rule_proposer.py \
      --model "Qwen/Qwen2.5-3B" \
      --n 200 \
      --output "analysis_results/rq3_crossmodel/clutrr_rule_proposer_qwen25_3b_200.json"

  run_cmd "rq3_clutrr_rule_proposer_phi35_mini" \
    python scripts/run_clutrr_llm_rule_proposer.py \
      --model "microsoft/Phi-3.5-mini-instruct" \
      --n 200 \
      --output "analysis_results/rq3_crossmodel/clutrr_rule_proposer_phi35_mini_200.json"

  run_cmd "rq3_clutrr_rule_proposer_llama32_3b" \
    python scripts/run_clutrr_llm_rule_proposer.py \
      --model "meta-llama/Llama-3.2-3B-Instruct" \
      --n 200 \
      --output "analysis_results/rq3_crossmodel/clutrr_rule_proposer_llama32_3b_200.json"
}

run_graphlog() {
  run_cmd "rq3_graphlog_praxis_qwen25_3b" \
    python scripts/run_graphlog_binary_adapters.py \
      --mode praxis \
      --phase train_eval \
      --experiment_name "rq3_graphlog_rule56_r7plus100_praxis_qwen25_3b" \
      --dataset "data/graphlog/graphlog_binary_rule56_r7plus_100.json" \
      --model_name "Qwen/Qwen2.5-3B" \
      --n_tasks 100 \
      --epochs 5 \
      --lr 5e-4 \
      --batch_size 1 \
      --gradient_accumulation_steps 1 \
      --score_method logprob \
      --output_root "loras/rq3-graphlog-binary" \
      --results_root "analysis_results/rq3_crossmodel/graphlog_binary"

  run_cmd "rq3_graphlog_praxis_phi35_mini" \
    python scripts/run_graphlog_binary_adapters.py \
      --mode praxis \
      --phase train_eval \
      --experiment_name "rq3_graphlog_rule56_r7plus100_praxis_phi35_mini" \
      --dataset "data/graphlog/graphlog_binary_rule56_r7plus_100.json" \
      --model_name "microsoft/Phi-3.5-mini-instruct" \
      --n_tasks 100 \
      --epochs 5 \
      --lr 5e-4 \
      --batch_size 1 \
      --gradient_accumulation_steps 1 \
      --score_method logprob \
      --output_root "loras/rq3-graphlog-binary" \
      --results_root "analysis_results/rq3_crossmodel/graphlog_binary"

  run_cmd "rq3_graphlog_praxis_llama32_3b" \
    python scripts/run_graphlog_binary_adapters.py \
      --mode praxis \
      --phase train_eval \
      --experiment_name "rq3_graphlog_rule56_r7plus100_praxis_llama32_3b" \
      --dataset "data/graphlog/graphlog_binary_rule56_r7plus_100.json" \
      --model_name "meta-llama/Llama-3.2-3B-Instruct" \
      --n_tasks 100 \
      --epochs 5 \
      --lr 5e-4 \
      --batch_size 1 \
      --gradient_accumulation_steps 1 \
      --score_method logprob \
      --output_root "loras/rq3-graphlog-binary" \
      --results_root "analysis_results/rq3_crossmodel/graphlog_binary"
}

run_squad_one() {
  local tag="$1"
  local model="$2"
  local port="$3"
  local zmq_port="$4"
  local exp_name="rq3_squad_praxis_${tag}"

  run_cmd "rq3_squad_stop_before_${tag}" bash scripts/local_stop_servers.sh

  run_cmd "rq3_squad_start_${tag}" env \
    MODEL_NAME="${model}" \
    PORT="${port}" \
    ZMQ_PORT="${zmq_port}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}" \
    bash scripts/local_start_ttt.sh

  run_cmd "${exp_name}" \
    python -u -m praxis.general-knowledge.src.query.query_server \
      --exp_name "${exp_name}" \
      --dataset "general-knowledge/data/synthetic_data/train/iter0_train_praxis.json" \
      --output_dir "analysis_results/rq3_crossmodel/squad_query_server" \
      --server_host 127.0.0.1 \
      --zmq_port "${zmq_port}" \
      --start_article 0 \
      --n_articles "${RQ3_SQUAD_ARTICLES:-10}" \
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
      --archive_path praxis/general-knowledge/archive/strategies.json
}

run_squad() {
  run_squad_one "qwen25_3b" "Qwen/Qwen2.5-3B" "18001" "5555"
  run_squad_one "phi35_mini" "microsoft/Phi-3.5-mini-instruct" "18001" "5555"
  run_squad_one "llama32_3b" "meta-llama/Llama-3.2-3B-Instruct" "18001" "5555"
  run_cmd "rq3_squad_stop_after_all" bash scripts/local_stop_servers.sh
}

log "=== RQ3 CROSS-MODEL SUITE START ==="
run_cmd "rq3_initial_stop_servers" bash scripts/local_stop_servers.sh
if [[ "${RQ3_SKIP_CLUTRR:-0}" == "1" ]]; then
  log "=== SKIP run_clutrr because RQ3_SKIP_CLUTRR=1 ==="
else
  run_clutrr
fi
run_graphlog
run_squad
log "=== RQ3 CROSS-MODEL SUITE COMPLETE ==="
