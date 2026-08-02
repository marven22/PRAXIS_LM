from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def correctness_by_id(rows: list[dict[str, Any]], key: str = "correct") -> dict[str, bool]:
    return {row["id"]: bool(row[key]) for row in rows}


def paired_metrics(reference: dict[str, bool], candidate: dict[str, bool]) -> dict[str, float]:
    ids = [item for item in reference if item in candidate]
    if not ids:
        return {"n": 0, "ref": 0.0, "final": 0.0, "gain": 0.0, "pgr": 0.0, "dr": 0.0}
    ref_vals = [reference[item] for item in ids]
    cand_vals = [candidate[item] for item in ids]
    ref = sum(ref_vals) / len(ids)
    final = sum(cand_vals) / len(ids)
    pgr = sum((not r) and c for r, c in zip(ref_vals, cand_vals)) / len(ids)
    dr = sum(r and (not c) for r, c in zip(ref_vals, cand_vals)) / len(ids)
    return {"n": len(ids), "ref": ref, "final": final, "gain": final - ref, "pgr": pgr, "dr": dr}


def method_from_symbolic(symbolic: dict[str, Any], name: str) -> dict[str, bool]:
    for method in symbolic["methods"]:
        if method["method"] == name:
            return correctness_by_id(method["results"])
    raise KeyError(name)


def method_from_flat_result(path: str, result_key: str = "results", correct_key: str = "correct") -> dict[str, bool]:
    data = load_json(path)
    return correctness_by_id(data[result_key], key=correct_key)


def maybe_add(rows: list[dict[str, Any]], name: str, reference: dict[str, bool], candidate: dict[str, bool] | None) -> None:
    if candidate is None:
        rows.append({"method": name, "status": "missing"})
        return
    metrics = paired_metrics(reference, candidate)
    rows.append({"method": name, "status": "ok", **metrics})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbolic", default="analysis_results/full_scale/clutrr_symbolic_baselines_full1048.json")
    parser.add_argument("--praxis", default="analysis_results/full_scale/clutrr_praxis_symbolic_full1048.json")
    parser.add_argument("--distilroberta", default="")
    parser.add_argument("--edit_sft", default="")
    parser.add_argument("--seal_style", default="")
    parser.add_argument("--out", default="analysis_results/full_scale/clutrr_full_table_metrics.json")
    args = parser.parse_args()

    symbolic = load_json(args.symbolic)
    reference = method_from_symbolic(symbolic, "left_to_right_archive")
    rows: list[dict[str, Any]] = []
    maybe_add(rows, "Majority label", reference, method_from_symbolic(symbolic, "majority_train_label"))
    maybe_add(rows, "Last-edge heuristic", reference, method_from_symbolic(symbolic, "last_edge"))

    if args.edit_sft:
        maybe_add(rows, "Edit-SFT", reference, method_from_flat_result(args.edit_sft))
    else:
        maybe_add(rows, "Edit-SFT", reference, None)

    if args.seal_style:
        seal_data = load_json(args.seal_style)
        if "results" in seal_data:
            if seal_data["results"] and "vote_correct" in seal_data["results"][0]:
                seal = correctness_by_id(seal_data["results"], key="vote_correct")
            elif seal_data["results"] and "best_score_correct" in seal_data["results"][0]:
                seal = correctness_by_id(seal_data["results"], key="best_score_correct")
            else:
                seal = correctness_by_id(seal_data["results"])
            maybe_add(rows, "SEAL-style", reference, seal)
        else:
            maybe_add(rows, "SEAL-style", reference, None)
    else:
        maybe_add(rows, "SEAL-style", reference, None)

    if args.distilroberta:
        maybe_add(rows, "DistilRoBERTa", reference, method_from_flat_result(args.distilroberta))
    else:
        maybe_add(rows, "DistilRoBERTa", reference, None)

    praxis_data = load_json(args.praxis)
    maybe_add(rows, "PRAXIS", reference, correctness_by_id(praxis_data["results"]))

    summary = {
        "dataset": "CLUTRR/v1",
        "reference": "left_to_right_archive",
        "n_reference": len(reference),
        "rows": rows,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
