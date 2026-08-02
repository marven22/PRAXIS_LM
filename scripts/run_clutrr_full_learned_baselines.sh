#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B}"

mkdir -p analysis_results/full_scale logs

python scripts/run_clutrr_roberta_baseline.py \
  --eval_dataset data/clutrr/clutrr_test_full1048.json \
  --reference_results analysis_results/full_scale/clutrr_symbolic_baselines_full1048.json \
  --output analysis_results/full_scale/clutrr_distilroberta_full1048_seed42.json \
  --seed 42 \
  2>&1 | tee logs/clutrr_distilroberta_full1048_seed42.log

python scripts/run_clutrr_seal_edits.py \
  --phase eval \
  --eval_dataset data/clutrr/clutrr_test_full1048.json \
  --experiment_name clutrr_seal_edits_full1062_e5 \
  --model_name "${MODEL_NAME}" \
  --n_eval 1048 \
  --results_root analysis_results/full_scale/clutrr_edit_sft \
  2>&1 | tee logs/clutrr_edit_sft_full1048.log

python scripts/run_clutrr_official_seal.py \
  --experiment_name clutrr_official_seal_full1048_supportonly_e1 \
  --model_name "${MODEL_NAME}" \
  --n_tasks 0 \
  --support_n 8 \
  --n_self_edits 1 \
  --temperature 0 \
  --epochs 1 \
  --max_generated_epochs 1 \
  --results_root analysis_results/full_scale/clutrr_official_seal \
  --output_root loras/full_scale/clutrr-official-seal \
  2>&1 | tee logs/clutrr_official_seal_full1048.log

python scripts/summarize_clutrr_full_table.py \
  --distilroberta analysis_results/full_scale/clutrr_distilroberta_full1048_seed42.json \
  --edit_sft analysis_results/full_scale/clutrr_edit_sft/clutrr_seal_edits_full1062_e5/final_results.json \
  --seal_style analysis_results/full_scale/clutrr_official_seal/clutrr_official_seal_full1048_supportonly_e1/final_results.json \
  --out analysis_results/full_scale/clutrr_full_table_metrics.json
