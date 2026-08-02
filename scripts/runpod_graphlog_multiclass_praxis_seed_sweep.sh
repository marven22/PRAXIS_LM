#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-/workspace/SEAL-main}"

mkdir -p logs
mkdir -p analysis_results/full_scale/graphlog_multiclass_praxis_seed_sweep
mkdir -p loras/graphlog-multiclass-praxis-seed-sweep

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
DATASET="${DATASET:-data/graphlog/graphlog_rule56_full1000.json}"
N_TASKS="${N_TASKS:-1000}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-4}"
SEEDS="${SEEDS:-42 43 44}"

echo "GraphLog multiclass PRAXIS seed sweep"
echo "model=${MODEL_NAME}"
echo "dataset=${DATASET}"
echo "n_tasks=${N_TASKS}"
echo "epochs=${EPOCHS}"
echo "lr=${LR}"
echo "seeds=${SEEDS}"
echo "started=$(date -Is)"

for seed in ${SEEDS}; do
  exp="graphlog_multiclass_praxis_full1000_seed${seed}"
  log="logs/${exp}.log"
  echo "=== START ${exp} $(date -Is) ===" | tee "${log}"
  python scripts/run_graphlog_multiclass_adapters.py \
    --mode praxis \
    --phase train_eval \
    --experiment_name "${exp}" \
    --dataset "${DATASET}" \
    --n_tasks "${N_TASKS}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --results_root analysis_results/full_scale/graphlog_multiclass_praxis_seed_sweep \
    --output_root loras/graphlog-multiclass-praxis-seed-sweep \
    --seed "${seed}" \
    2>&1 | tee -a "${log}"
  echo "=== END ${exp} $(date -Is) exit=${PIPESTATUS[0]} ===" | tee -a "${log}"
done

echo "completed=$(date -Is)"
