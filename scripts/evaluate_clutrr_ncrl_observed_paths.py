from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ncrl_dataset(ncrl_root: Path, data_name: str):
    code_dir = ncrl_root / "code"
    sys.path.insert(0, str(code_dir))
    from data import Dataset as NcrlDataset  # type: ignore

    return NcrlDataset(data_root=str(ncrl_root / "datasets" / data_name) + "/", inv=True)


def load_reference(reference_path: Path, method_name: str) -> dict[str, bool]:
    obj = load_json(reference_path)
    for method in obj.get("methods", []):
        if method.get("method") == method_name:
            return {str(row["id"]): bool(row["correct"]) for row in method.get("results", [])}
    raise KeyError(f"Could not find reference method {method_name!r} in {reference_path}")


def load_test_rows(ncrl_data_dir: Path) -> list[dict[str, Any]]:
    row_map = load_json(ncrl_data_dir / "row_map.json")
    rows = [row for row in row_map if row.get("split") == "test"]
    if not rows:
        raise ValueError(f"No test rows found in {ncrl_data_dir / 'row_map.json'}")
    return rows


def load_model(model_path: Path, device: torch.device):
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    model.to(device)
    model.eval()
    return model


def predict_batch(model, inputs: list[list[int]], allowed_indices: list[int], device: torch.device) -> list[int]:
    tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    with torch.no_grad():
        scores, _ = model(tensor)
        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed_indices] = scores[:, allowed_indices]
        return mask.argmax(dim=-1).detach().cpu().tolist()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    ncrl_root = (REPO_ROOT / args.ncrl_root).resolve()
    data_dir = ncrl_root / "datasets" / args.data
    rows = load_test_rows(data_dir)
    reference = load_reference(REPO_ROOT / args.reference_results, args.reference_method)
    dataset = load_ncrl_dataset(ncrl_root, args.data)
    head_rdict = dataset.get_head_relation_dict()

    allowed_relations = [
        relation.strip()
        for relation in (data_dir / "relations.txt").read_text(encoding="utf-8").splitlines()
        if relation.strip()
    ]
    allowed_indices = [head_rdict.rel2idx[relation] for relation in allowed_relations]

    skipped = []
    encoded: list[dict[str, Any]] = []
    for row in rows:
        try:
            body = [head_rdict.rel2idx[str(relation)] for relation in row["edge_types"]]
        except KeyError as exc:
            skipped.append({"id": row.get("id"), "reason": f"missing relation {exc}"})
            continue
        if len(body) == 0:
            skipped.append({"id": row.get("id"), "reason": "empty path"})
            continue
        encoded.append({**row, "body_idx": body})

    dry_summary = {
        "dataset": "CLUTRR/v1",
        "method": "NCRL observed-path evaluator",
        "n_rows": len(rows),
        "n_encoded": len(encoded),
        "n_skipped": len(skipped),
        "n_allowed_relations": len(allowed_relations),
        "path_lengths": {
            str(length): count
            for length, count in sorted(
                {
                    length: sum(1 for row in encoded if len(row["body_idx"]) == length)
                    for length in {len(row["body_idx"]) for row in encoded}
                }.items()
            )
        },
        "skipped": skipped[:20],
    }
    if args.dry_run:
        return dry_summary

    model_path = ncrl_root / "results" / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"NCRL model not found: {model_path}")

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(model_path, device)

    predictions: dict[int, str] = {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in encoded:
        grouped[len(row["body_idx"])].append(row)

    for length, group in sorted(grouped.items()):
        for start in range(0, len(group), args.batch_size):
            batch = group[start : start + args.batch_size]
            pred_indices = predict_batch(
                model,
                [row["body_idx"] for row in batch],
                allowed_indices,
                device,
            )
            for row, pred_idx in zip(batch, pred_indices):
                predictions[id(row)] = head_rdict.idx2rel[pred_idx]

    results = []
    for row in encoded:
        prediction = predictions[id(row)]
        target = str(row["target"])
        row_id = str(row["id"])
        correct = prediction == target
        ref_correct = bool(reference.get(row_id, False))
        results.append(
            {
                "id": row_id,
                "target": target,
                "prediction": prediction,
                "edge_types": row["edge_types"],
                "chain_len": len(row["edge_types"]),
                "reference_correct": ref_correct,
                "correct": correct,
                "improved": (not ref_correct) and correct,
                "degraded": ref_correct and (not correct),
            }
        )

    n = len(results)
    ref = sum(row["reference_correct"] for row in results) / max(n, 1)
    final = sum(row["correct"] for row in results) / max(n, 1)
    pgr = sum(row["improved"] for row in results) / max(n, 1)
    dr = sum(row["degraded"] for row in results) / max(n, 1)
    runtime = time.time() - started
    by_len: dict[int, list[bool]] = defaultdict(list)
    for row in results:
        by_len[int(row["chain_len"])].append(bool(row["correct"]))

    return {
        **dry_summary,
        "method": "NCRL observed-path",
        "model": str(model_path),
        "device": str(device),
        "reference": args.reference_method,
        "n": n,
        "initial": ref,
        "final": final,
        "gain": final - ref,
        "pgr": pgr,
        "dr": dr,
        "time_per_example_seconds": runtime / max(n, 1),
        "runtime_seconds": runtime,
        "accuracy_by_chain_len": {
            str(length): sum(values) / len(values) for length, values in sorted(by_len.items())
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncrl_root", default="external/NCRL")
    parser.add_argument("--data", default="clutrr_praxis")
    parser.add_argument("--model", default="clutrr_praxis")
    parser.add_argument("--reference_results", default="analysis_results/full_scale/clutrr_symbolic_baselines_full1048.json")
    parser.add_argument("--reference_method", default="left_to_right_archive")
    parser.add_argument("--output", default="analysis_results/full_scale/clutrr_ncrl_observed_paths_full1048.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    summary = evaluate(args)
    if not args.dry_run:
        out = REPO_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key not in {"results", "skipped"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
