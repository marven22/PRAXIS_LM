from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def correctness_by_id(rows: list[dict[str, Any]], key: str = "correct") -> dict[str, bool]:
    return {row["id"]: bool(row[key]) for row in rows}


def paired_metrics(reference: dict[str, bool], candidate: dict[str, bool]) -> dict[str, float]:
    ids = [item for item in reference if item in candidate]
    ref_vals = [reference[item] for item in ids]
    cand_vals = [candidate[item] for item in ids]
    return {
        "n": len(ids),
        "ref": sum(ref_vals) / len(ids),
        "final": sum(cand_vals) / len(ids),
        "gain": (sum(cand_vals) - sum(ref_vals)) / len(ids),
        "pgr": sum((not r) and c for r, c in zip(ref_vals, cand_vals)) / len(ids),
        "dr": sum(r and (not c) for r, c in zip(ref_vals, cand_vals)) / len(ids),
    }


def method_from_symbolic(symbolic: dict[str, Any], name: str) -> dict[str, bool]:
    for method in symbolic["methods"]:
        if method["method"] == name:
            return correctness_by_id(method["results"])
    raise KeyError(name)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def add_replicated(rows: list[dict[str, Any]], method: str, metrics: dict[str, float]) -> None:
    for seed in SEEDS:
        rows.append({"method": method, "seed": seed, **metrics})


def add_from_paths(rows: list[dict[str, Any]], method: str, reference: dict[str, bool], pattern: str) -> None:
    for seed in SEEDS:
        path = REPO_ROOT / pattern.format(seed=seed)
        if not path.exists():
            rows.append({"method": method, "seed": seed, "status": "missing", "path": str(path)})
            continue
        data = load_json(path)
        rows.append({"method": method, "seed": seed, "status": "ok", **paired_metrics(reference, correctness_by_id(data["results"]))})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbolic", default="analysis_results/full_scale/clutrr_symbolic_baselines_full1048.json")
    parser.add_argument("--praxis", default="analysis_results/full_scale/clutrr_praxis_symbolic_full1048.json")
    parser.add_argument("--out", default="analysis_results/full_scale/clutrr_seed_sweep_metrics.json")
    args = parser.parse_args()

    symbolic = load_json(REPO_ROOT / args.symbolic)
    reference = method_from_symbolic(symbolic, "left_to_right_archive")
    rows: list[dict[str, Any]] = []

    add_replicated(rows, "Majority label", paired_metrics(reference, method_from_symbolic(symbolic, "majority_train_label")))
    add_replicated(rows, "Last-edge heuristic", paired_metrics(reference, method_from_symbolic(symbolic, "last_edge")))
    add_from_paths(rows, "Edit-SFT", reference, "analysis_results/full_scale/clutrr_edit_sft_seeds/clutrr_edit_sft_full1048_seed{seed}/final_results.json")
    add_from_paths(rows, "DistilRoBERTa", reference, "analysis_results/full_scale/clutrr_distilroberta_full1048_seed{seed}.json")
    praxis = correctness_by_id(load_json(REPO_ROOT / args.praxis)["results"])
    add_replicated(rows, "PRAXIS", paired_metrics(reference, praxis))

    aggregate = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method and row.get("status", "ok") == "ok"]
        if not method_rows:
            aggregate[method] = {"status": "missing"}
            continue
        aggregate[method] = {
            metric: summarize([row[metric] for row in method_rows])
            for metric in ["final", "gain", "pgr", "dr"]
        }

    summary = {"dataset": "CLUTRR/v1", "seeds": SEEDS, "reference": "left_to_right_archive", "rows": rows, "aggregate": aggregate}
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
