from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from build_graphlog_subset import choose_worlds, format_graph, read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def graph_text(row: dict[str, Any], target: str) -> str:
    return (
        f"Target relation: {target}\n"
        f"{format_graph(row)}\n"
        "Does the query edge have the target relation?"
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    root = (REPO_ROOT / args.graphlog_root).resolve()
    worlds = choose_worlds(root, args.world_split, args.difficulty, args.n_worlds)
    if args.world_names:
        requested = set(args.world_names.split(","))
        worlds = [world for world in worlds if world.name in requested]
    if not worlds:
        raise RuntimeError(f"No GraphLog worlds found in {root / args.world_split}")
    target_filter = set(args.target_relations.split(",")) if args.target_relations else None
    path_lengths = {int(item) for item in args.path_lengths.split(",")} if args.path_lengths else None

    tasks = []
    per_world = max(1, args.n_tasks // len(worlds))
    remainder = args.n_tasks % len(worlds)
    for world_idx, world in enumerate(worlds):
        n_for_world = per_world + int(world_idx < remainder)
        support_pool = read_jsonl(world / "train.jsonl")
        query_pool = read_jsonl(world / args.query_split)
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in support_pool:
            by_label.setdefault(row["query"][2], []).append(row)
        query_by_label: dict[str, list[dict[str, Any]]] = {}
        for row in query_pool:
            query_by_label.setdefault(row["query"][2], []).append(row)
        labels = sorted(by_label)
        if len(labels) < 2:
            continue
        for rows in by_label.values():
            rng.shuffle(rows)

        local_idx = 0
        attempts = 0
        made_for_world = 0
        max_attempts = max(n_for_world * 50, 500)
        while made_for_world < n_for_world and attempts < max_attempts:
            attempts += 1
            make_positive = local_idx % 2 == 0
            if target_filter is not None:
                fixed_targets = sorted(
                    label
                    for label in target_filter
                    if label in by_label and label in query_by_label and len(by_label[label]) >= args.support_examples // 2
                )
                if not fixed_targets:
                    break
                fixed_target = fixed_targets[(made_for_world + args.seed) % len(fixed_targets)]
                if make_positive:
                    candidates = query_by_label.get(fixed_target, [])
                    if not candidates:
                        continue
                    query = candidates[(local_idx * 997 + args.seed) % len(candidates)]
                    target = fixed_target
                else:
                    candidates = [row for label, rows in query_by_label.items() if label != fixed_target for row in rows]
                    if not candidates:
                        continue
                    query = candidates[(local_idx * 997 + args.seed) % len(candidates)]
                    target = fixed_target
                local_idx += 1
            elif make_positive:
                query = query_pool[(local_idx * 997 + args.seed) % len(query_pool)]
                local_idx += 1
                true_label = query["query"][2]
                target = true_label
            else:
                query = query_pool[(local_idx * 997 + args.seed) % len(query_pool)]
                local_idx += 1
                true_label = query["query"][2]
                alternatives = [label for label in labels if label != true_label]
                target = alternatives[(local_idx + args.seed) % len(alternatives)]
            descriptor = query.get("descriptor", "")
            query_path_len = len([item for item in descriptor.split(",") if item])
            if path_lengths is not None and query_path_len not in path_lengths:
                continue
            positives = by_label.get(target, [])
            negatives = [row for label, rows in by_label.items() if label != target for row in rows[: max(args.support_examples, 12)]]
            if len(positives) < args.support_examples // 2 or len(negatives) < args.support_examples // 2:
                continue
            pos_start = (local_idx * 3) % max(1, len(positives) - args.support_examples // 2)
            neg_start = (local_idx * 5) % max(1, len(negatives) - args.support_examples // 2)
            support_rows = positives[pos_start : pos_start + args.support_examples // 2] + negatives[
                neg_start : neg_start + args.support_examples // 2
            ]
            rng.shuffle(support_rows)
            task_id = f"graphlog_bin_{world.name}_{local_idx:04d}"
            tasks.append(
                {
                    "id": task_id,
                    "world": world.name,
                    "target_relation": target,
                    "train": [
                        {
                            "text": graph_text(row, target),
                            "answer": "YES" if row["query"][2] == target else "NO",
                            "relation": row["query"][2],
                            "descriptor": row.get("descriptor", ""),
                        }
                        for row in support_rows
                    ],
                    "query": {
                        "text": graph_text(query, target),
                        "answer": "YES" if query["query"][2] == target else "NO",
                        "relation": query["query"][2],
                        "descriptor": query.get("descriptor", ""),
                    },
                    "metadata": {
                        "dataset": "GraphLog",
                        "task": "binary_target_relation",
                        "world": world.name,
                        "target_relation": target,
                        "world_split": args.world_split,
                        "query_split": args.query_split,
                        "difficulty": args.difficulty,
                    },
                }
            )
            made_for_world += 1
            if len(tasks) >= args.n_tasks:
                break
        if len(tasks) >= args.n_tasks:
            break
    return {"source": "GraphLog v1.1 official release, binary target-relation formulation", "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphlog_root", default="data/graphlog/graphlog_v1.1")
    parser.add_argument("--out", default="data/graphlog/graphlog_binary_test_50.json")
    parser.add_argument("--n_tasks", type=int, default=50)
    parser.add_argument("--support_examples", type=int, default=8)
    parser.add_argument("--world_split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--query_split", choices=["train.jsonl", "valid.jsonl", "test.jsonl"], default="test.jsonl")
    parser.add_argument("--difficulty", choices=["easy", "moderate", "hard", "all"], default="all")
    parser.add_argument("--n_worlds", type=int, default=5)
    parser.add_argument("--world_names", default="")
    parser.add_argument("--target_relations", default="")
    parser.add_argument("--path_lengths", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = build(args)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(data['tasks'])} binary GraphLog tasks to {out}")


if __name__ == "__main__":
    main()
