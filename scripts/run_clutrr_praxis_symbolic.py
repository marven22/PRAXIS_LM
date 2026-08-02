from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


def parse_edge_types(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in ast.literal_eval(value)]


def chain_len(row: dict) -> int:
    return len(parse_edge_types(row["edge_types"]))


def learn_binary_archive(train_rows: Iterable[dict]) -> Dict[Tuple[str, str], str]:
    votes: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        edges = parse_edge_types(row["edge_types"])
        if len(edges) == 2:
            votes[(edges[0], edges[1])][row["target_text"]] += 1
        proof_state = row.get("proof_state")
        if proof_state:
            try:
                proof_steps = ast.literal_eval(proof_state) if isinstance(proof_state, str) else proof_state
            except (SyntaxError, ValueError):
                proof_steps = []
            for proof in proof_steps:
                if not isinstance(proof, dict):
                    continue
                for conclusion, premises in proof.items():
                    if not isinstance(conclusion, tuple) or len(conclusion) < 3:
                        continue
                    if not isinstance(premises, list) or len(premises) != 2:
                        continue
                    left, right = premises
                    if len(left) >= 3 and len(right) >= 3:
                        votes[(str(left[1]), str(right[1]))][str(conclusion[1])] += 1
    return {pair: counts.most_common(1)[0][0] for pair, counts in votes.items()}


def predict_relation(edge_types: List[str], archive: Dict[Tuple[str, str], str]) -> tuple[str | None, list]:
    """Compose a relation chain over all split points, not only left-to-right."""
    trace = []
    n = len(edge_types)
    if not edge_types:
        return None, trace
    chart: list[list[Counter[str]]] = [[Counter() for _ in range(n)] for _ in range(n)]
    for index, relation in enumerate(edge_types):
        chart[index][index][relation] = 1

    for width in range(2, n + 1):
        for start in range(0, n - width + 1):
            end = start + width - 1
            for split in range(start, end):
                for left, left_count in chart[start][split].items():
                    for right, right_count in chart[split + 1][end].items():
                        predicted = archive.get((left, right))
                        trace.append(
                            {
                                "span": [start, end],
                                "split": split,
                                "left": left,
                                "right": right,
                                "prediction": predicted,
                            }
                        )
                        if predicted is not None:
                            chart[start][end][predicted] += left_count * right_count
    if not chart[0][n - 1]:
        return None, trace
    return chart[0][n - 1].most_common(1)[0][0], trace


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gen_train234_test2to10")
    parser.add_argument("--output", default="analysis_results/clutrr_praxis_symbolic_50.json")
    parser.add_argument("--subset_output", default="data/clutrr/clutrr_test_50.json")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=10)
    args = parser.parse_args()

    train = load_dataset("CLUTRR/v1", args.config, split="train")
    test = load_dataset("CLUTRR/v1", args.config, split="test")
    archive = learn_binary_archive(train)
    subset = make_subset(test, args.n, args.min_len, args.max_len)

    results = []
    for row in subset:
        edges = parse_edge_types(row["edge_types"])
        pred, trace = predict_relation(edges, archive)
        correct = pred == row["target_text"]
        results.append(
            {
                "id": row["id"],
                "story": row["story"],
                "query": row["query"],
                "target": row["target_text"],
                "edge_types": edges,
                "chain_len": len(edges),
                "prediction": pred,
                "correct": correct,
                "trace": trace,
            }
        )

    by_len: Dict[int, list] = defaultdict(list)
    for row in results:
        by_len[row["chain_len"]].append(row["correct"])
    summary = {
        "dataset": "CLUTRR/v1",
        "config": args.config,
        "n_train": len(train),
        "n_test_available": len(test),
        "n": len(results),
        "archive_size": len(archive),
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "coverage": sum(r["prediction"] is not None for r in results) / len(results),
        "accuracy_by_chain_len": {
            str(length): sum(values) / len(values) for length, values in sorted(by_len.items())
        },
        "count_by_chain_len": {str(length): len(values) for length, values in sorted(by_len.items())},
        "archive": {f"{a}+{b}": c for (a, b), c in sorted(archive.items())},
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(args.subset_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.subset_output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"archive", "results"}}, indent=2))


if __name__ == "__main__":
    main()
