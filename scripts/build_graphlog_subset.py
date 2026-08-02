from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def edge_lookup(row: dict[str, Any]) -> dict[tuple[int, int], str]:
    lookup = {}
    for src, dst, rel in row["edges"]:
        lookup[(src, dst)] = rel
    return lookup


def format_graph(row: dict[str, Any], *, include_answer: bool = False, representation: str = "path") -> str:
    if representation == "path" and row.get("resolution_path"):
        lookup = edge_lookup(row)
        path = row["resolution_path"]
        edge_parts = []
        for src, dst in zip(path, path[1:]):
            rel = lookup.get((src, dst), "UNKNOWN")
            edge_parts.append(f"{src}->{dst}:{rel}")
        edges = ", ".join(edge_parts)
        descriptor = row.get("descriptor", "")
        qsrc, qdst, rel = row["query"]
        text = (
            f"Reasoning path: {edges}\n"
            f"Path relation sequence: {descriptor}\n"
            f"Query: {qsrc}->{qdst}\n"
            "Predict the relation label for the query edge."
        )
        if include_answer:
            text += f"\nRELATION: {rel}"
        return text

    edges = ", ".join(f"{src}->{dst}:{rel}" for src, dst, rel in row["edges"])
    qsrc, qdst, rel = row["query"]
    text = (
        f"Edges: {edges}\n"
        f"Query: {qsrc}->{qdst}\n"
        "Predict the relation label for the query edge."
    )
    if include_answer:
        text += f"\nRELATION: {rel}"
    return text


def collect_labels(world_dir: Path) -> list[str]:
    labels = set()
    for split in ["train", "valid", "test"]:
        for row in read_jsonl(world_dir / f"{split}.jsonl", limit=2000):
            labels.add(row["query"][2])
            labels.update(edge[2] for edge in row["edges"])
    return sorted(labels)


def choose_worlds(root: Path, split: str, difficulty: str, n_worlds: int | None) -> list[Path]:
    split_dir = root / split
    worlds = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name.split("_")[1]))
    meta_path = root.parent / "meta.json"
    if meta_path.exists() and difficulty != "all":
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        diff = meta.get("graphlog_v1.0", {}).get("difficulty", {})
        ids = set(diff.get(difficulty, []))
        worlds = [p for p in worlds if int(p.name.split("_")[1]) in ids]
    if n_worlds is not None:
        worlds = worlds[:n_worlds]
    return worlds


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    root = (REPO_ROOT / args.graphlog_root).resolve()
    worlds = choose_worlds(root, args.world_split, args.difficulty, args.n_worlds)
    if args.world_names:
        requested = set(args.world_names.split(","))
        worlds = [world for world in worlds if world.name in requested]
    if not worlds:
        raise RuntimeError(f"No GraphLog worlds found in {root / args.world_split}")

    tasks = []
    per_world = max(1, args.n_tasks // len(worlds))
    remainder = args.n_tasks % len(worlds)
    for world_idx, world in enumerate(worlds):
        n_for_world = per_world + int(world_idx < remainder)
        support_pool = read_jsonl(world / "train.jsonl")
        query_pool = read_jsonl(world / args.query_split)
        labels = collect_labels(world)
        if len(support_pool) < args.support_examples or not query_pool:
            continue
        support_indices = list(range(len(support_pool)))
        rng.shuffle(support_indices)
        for local_idx in range(n_for_world):
            query = query_pool[(local_idx * 997 + args.seed) % len(query_pool)]
            start = (local_idx * args.support_examples) % max(1, len(support_indices) - args.support_examples)
            support = [support_pool[i] for i in support_indices[start : start + args.support_examples]]
            task_id = f"graphlog_{world.name}_{local_idx:04d}"
            tasks.append(
                {
                    "id": task_id,
                    "world": world.name,
                    "labels": labels,
                    "train": [
                        {
                            "text": format_graph(row),
                            "answer": row["query"][2],
                            "query": row["query"][:2],
                            "descriptor": row.get("descriptor", ""),
                        }
                        for row in support
                    ],
                    "query": {
                        "text": format_graph(query),
                        "answer": query["query"][2],
                        "query": query["query"][:2],
                        "descriptor": query.get("descriptor", ""),
                    },
                    "metadata": {
                        "dataset": "GraphLog",
                        "world": world.name,
                        "world_split": args.world_split,
                        "query_split": args.query_split,
                        "difficulty": args.difficulty,
                    },
                }
            )
            if len(tasks) >= args.n_tasks:
                break
        if len(tasks) >= args.n_tasks:
            break
    return {"source": "GraphLog v1.1 official release", "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphlog_root", default="data/graphlog/graphlog_v1.1")
    parser.add_argument("--out", default="data/graphlog/graphlog_test_50.json")
    parser.add_argument("--n_tasks", type=int, default=50)
    parser.add_argument("--support_examples", type=int, default=8)
    parser.add_argument("--world_split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--query_split", choices=["train.jsonl", "valid.jsonl", "test.jsonl"], default="test.jsonl")
    parser.add_argument("--difficulty", choices=["easy", "moderate", "hard", "all"], default="all")
    parser.add_argument("--n_worlds", type=int, default=None)
    parser.add_argument("--world_names", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = build(args)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(data['tasks'])} GraphLog tasks to {out}")


if __name__ == "__main__":
    main()
