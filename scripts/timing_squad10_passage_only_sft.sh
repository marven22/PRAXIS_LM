#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs analysis_results/timing_calibration general-knowledge/results/cpt

ZMQ_PORT="${ZMQ_PORT:-5555}"
DATASET="general-knowledge/data/synthetic_data/train/iter0_train_50_passage_only.json"
NAME="squad10_passage_only_sft"

source .venv/bin/activate
set -a
[[ -f .env ]] && source <(tr -d '\r' < .env)
set +a

start="$(python - <<'PY'
import time
print(time.time())
PY
)"

echo "=== START ${NAME} $(date -Is) ==="
python -u -m general-knowledge.src.query.CPT \
  --dataset "${DATASET}" \
  --output_dir general-knowledge/results/cpt \
  --server_host 127.0.0.1 \
  --zmq_port "${ZMQ_PORT}" \
  --k_completions 0 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0 \
  --finetune_epochs 3 \
  --finetune_lr 2e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --n_articles 10 \
  --split_newlines \
  --baseline_eval 2>&1 | tee "logs/${NAME}.log"

end="$(python - <<'PY'
import time
print(time.time())
PY
)"

elapsed="$(python - <<PY
start=float("${start}")
end=float("${end}")
print(round(end-start, 3))
PY
)"

python - <<PY
import json
from pathlib import Path

elapsed = float("${elapsed}")
payload = {
    "name": "${NAME}",
    "dataset": "${DATASET}",
    "n": 10,
    "hardware": "local",
    "timing_scope": "passage_only_cpt_train_plus_eval",
    "total_seconds": round(elapsed, 3),
    "seconds_per_example": round(elapsed / 10.0, 3),
}
Path("analysis_results/timing_calibration/${NAME}.runtime.json").write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
)
PY

echo "=== END ${NAME} elapsed=${elapsed} ==="
