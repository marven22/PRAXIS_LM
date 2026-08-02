from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset

from run_clutrr_praxis_symbolic import learn_binary_archive, parse_edge_types, predict_relation


def chain_len(row: dict) -> int:
    return len(parse_edge_types(row["edge_types"]))


def make_subset(rows, n: int, min_len: int, max_len: int) -> list:
    if n <= 0:
        selected = []
        for row in rows:
            length = chain_len(row)
            if min_len <= length <= max_len:
                item = dict(row)
                item["chain_len"] = length
                selected.append(item)
        return selected
    selected = []
    by_len = Counter()
    max_per_len = max(4, n // max(1, max_len - min_len + 1) + 2)
    for row in rows:
        length = chain_len(row)
        if length < min_len or length > max_len:
            continue
        if by_len[length] >= max_per_len:
            continue
        item = dict(row)
        item["chain_len"] = length
        selected.append(item)
        by_len[length] += 1
        if len(selected) >= n:
            break
    return selected


def majority_label(train_rows: Iterable[dict]) -> str:
    return Counter(row["target_text"] for row in train_rows).most_common(1)[0][0]


def direct_pair_lookup(edges: List[str], archive: Dict[Tuple[str, str], str]) -> str | None:
    if len(edges) != 2:
        return None
    return archive.get((edges[0], edges[1]))


def left_to_right(edges: List[str], archive: Dict[Tuple[str, str], str]) -> str | None:
    if not edges:
        return None
    current = edges[0]
    for next_relation in edges[1:]:
        current = archive.get((current, next_relation))
        if current is None:
            return None
    return current


def greedy_recomposition(edges: List[str], archive: Dict[Tuple[str, str], str]) -> str | None:
    if not edges:
        return None
    current = list(edges)
    while len(current) > 1:
        reduced = False
        for index in range(len(current) - 1):
            composed = archive.get((current[index], current[index + 1]))
            if composed is None:
                continue
            current = current[:index] + [composed] + current[index + 2 :]
            reduced = True
            break
        if not reduced:
            return None
    return current[0]


def beam_recomposition(edges: List[str], archive: Dict[Tuple[str, str], str], beam_size: int) -> str | None:
    if not edges:
        return None
    n = len(edges)
    chart: list[list[Counter[str]]] = [[Counter() for _ in range(n)] for _ in range(n)]
    for index, relation in enumerate(edges):
        chart[index][index][relation] = 1

    for width in range(2, n + 1):
        for start in range(0, n - width + 1):
            end = start + width - 1
            candidates: Counter[str] = Counter()
            for split in range(start, end):
                for left, left_count in chart[start][split].items():
                    for right, right_count in chart[split + 1][end].items():
                        predicted = archive.get((left, right))
                        if predicted is not None:
                            candidates[predicted] += left_count * right_count
            if candidates:
                chart[start][end].update(dict(candidates.most_common(beam_size)))

    if not chart[0][n - 1]:
        return None
    return chart[0][n - 1].most_common(1)[0][0]


def last_edge(edges: List[str]) -> str | None:
    return edges[-1] if edges else None


def evaluate(name: str, subset: list, predictor) -> dict:
    results = []
    by_len = defaultdict(list)
    for row in subset:
        edges = parse_edge_types(row["edge_types"])
        pred = predictor(edges)
        correct = pred == row["target_text"]
        results.append(
            {
                "id": row["id"],
                "target": row["target_text"],
                "edge_types": edges,
                "chain_len": len(edges),
                "prediction": pred,
                "correct": correct,
            }
        )
        by_len[len(edges)].append(correct)
    return {
        "method": name,
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "coverage": sum(r["prediction"] is not None for r in results) / len(results),
        "accuracy_by_chain_len": {str(k): sum(v) / len(v) for k, v in sorted(by_len.items())},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gen_train234_test2to10")
    parser.add_argument("--output", default="analysis_results/clutrr_symbolic_baselines_50.json")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=10)
    args = parser.parse_args()

    train = load_dataset("CLUTRR/v1", args.config, split="train")
    test = load_dataset("CLUTRR/v1", args.config, split="test")
    subset = make_subset(test, args.n, args.min_len, args.max_len)
    archive = learn_binary_archive(train)
    majority = majority_label(train)

    methods = [
        evaluate("majority_train_label", subset, lambda edges: majority),
        evaluate("last_edge", subset, last_edge),
        evaluate("direct_pair_lookup_len2_only", subset, lambda edges: direct_pair_lookup(edges, archive)),
        evaluate("left_to_right_archive", subset, lambda edges: left_to_right(edges, archive)),
        evaluate("greedy_recomposition", subset, lambda edges: greedy_recomposition(edges, archive)),
        evaluate("beam1_recomposition", subset, lambda edges: beam_recomposition(edges, archive, 1)),
        evaluate("beam2_recomposition", subset, lambda edges: beam_recomposition(edges, archive, 2)),
        evaluate("praxis_dp_archive", subset, lambda edges: predict_relation(edges, archive)[0]),
    ]
    summary = {
        "dataset": "CLUTRR/v1",
        "config": args.config,
        "n": len(subset),
        "archive_size": len(archive),
        "methods": methods,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({m["method"]: {"accuracy": m["accuracy"], "coverage": m["coverage"]} for m in methods}, indent=2))


if __name__ == "__main__":
    main()
