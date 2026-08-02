from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ROWS = [
    ("Qwen2.5-3B", "clutrr_rule_proposer_qwen25_3b_full1000.json"),
    ("Phi-3.5-mini", "clutrr_rule_proposer_phi35_mini_full1000.json"),
    ("Llama-3.2-3B", "clutrr_rule_proposer_llama32_3b_full1000.json"),
]


def pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 2)


def row_from_result(model_label: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"dataset": "CLUTRR", "model": model_label, "status": "missing", "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": "CLUTRR",
        "model": model_label,
        "status": "ok",
        "n": data.get("n"),
        "initial": pct(data.get("initial")),
        "final": pct(data.get("final")),
        "gain": pct(data.get("gain")),
        "pgr": pct(data.get("pgr")),
        "dr": pct(data.get("dr")),
        "time": round(float(data.get("time_per_example_seconds", 0.0)), 4),
        "candidate_rule_count": data.get("candidate_rule_count"),
        "accepted_rule_count": data.get("accepted_rule_count"),
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="analysis_results/rq3_crossmodel_full1000")
    parser.add_argument("--out", default="analysis_results/rq3_crossmodel_full1000/clutrr_rq3_model_family_table.json")
    args = parser.parse_args()

    results_root = REPO_ROOT / args.results_root
    rows = [row_from_result(label, results_root / filename) for label, filename in MODEL_ROWS]
    summary = {"dataset": "CLUTRR/v1", "protocol": "rq3_model_family_full1000", "rows": rows}

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
