#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs analysis_results

source .venv/bin/activate

run_one() {
  local name="$1"
  local model="$2"
  local out="$3"
  local log="$4"
  echo "=== START ${name} $(date -Is) ==="
  python scripts/run_clutrr_archive_conditioned_llm.py \
    --model "${model}" \
    --n 200 \
    --output "${out}" 2>&1 | tee "${log}"
  echo "=== END ${name} $(date -Is) ==="
}

run_one \
  "qwen25_3b" \
  "Qwen/Qwen2.5-3B" \
  "analysis_results/clutrr_archive_conditioned_qwen25_3b_200.json" \
  "logs/clutrr_archive_conditioned_qwen25_3b_200.log"

run_one \
  "phi35_mini" \
  "microsoft/Phi-3.5-mini-instruct" \
  "analysis_results/clutrr_archive_conditioned_phi35_mini_200.json" \
  "logs/clutrr_archive_conditioned_phi35_mini_200.log"

run_one \
  "llama32_3b" \
  "meta-llama/Llama-3.2-3B-Instruct" \
  "analysis_results/clutrr_archive_conditioned_llama32_3b_200.json" \
  "logs/clutrr_archive_conditioned_llama32_3b_200.log"
