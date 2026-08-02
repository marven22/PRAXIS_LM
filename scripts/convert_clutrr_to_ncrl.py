from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def parse_edge_types(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in ast.literal_eval(value)]


def target_text(row: dict) -> str:
    return str(row.get("target_text", row.get("target")))


def ncrl_relation(value: str) -> str:
    # NCRL uses '-' and '|' as internal rule delimiters, so relation labels
    # cannot contain either character.
    return str(value).replace("-", "_").replace("|", "_")


def row_id(row: dict, split: str, index: int) -> str:
    raw = str(row.get("id") or f"{split}_{index}")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)


def ncrl_fact(head: str, relation: str, tail: str) -> str:
    # NCRL's parser interprets rows as tail, relation, head.
    return f"{tail}\t{relation}\t{head}"


def make_path_facts(row: dict, split: str, index: int) -> tuple[list[str], str, list[str]]:
    edges = parse_edge_types(row["edge_types"])
    prefix = row_id(row, split, index)
    entities = [f"{split}_{prefix}_n{i}" for i in range(len(edges) + 1)]
    facts = [ncrl_fact(entities[i], ncrl_relation(relation), entities[i + 1]) for i, relation in enumerate(edges)]
    query = ncrl_fact(entities[0], ncrl_relation(target_text(row)), entities[-1])
    return facts, query, entities


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_inverse_facts(facts: Iterable[str]) -> list[str]:
    result: list[str] = []
    for line in facts:
        tail, relation, head = line.split("\t")
        result.append(line)
        result.append(f"{head}\tinv_{relation}\t{tail}")
    return result


def convert(args: argparse.Namespace) -> dict:
    train_rows = list(load_dataset("CLUTRR/v1", args.config, split="train"))
    valid_rows = list(load_dataset("CLUTRR/v1", args.config, split="validation"))
    test_rows = list(load_dataset("CLUTRR/v1", args.config, split="test"))

    if args.max_test > 0:
        test_rows = test_rows[: args.max_test]
    if args.max_train > 0:
        train_rows = train_rows[: args.max_train]
    if args.max_valid > 0:
        valid_rows = valid_rows[: args.max_valid]

    facts: list[str] = []
    train_queries: list[str] = []
    valid_queries: list[str] = []
    test_queries: list[str] = []
    entities: set[str] = set()
    relations: set[str] = set()
    row_map: list[dict] = []

    def consume(rows: list[dict], split: str, sink: list[str]) -> None:
        for index, row in enumerate(rows):
            path_facts, query, row_entities = make_path_facts(row, split, index)
            facts.extend(path_facts)
            sink.append(query)
            entities.update(row_entities)
            relations.update(ncrl_relation(relation) for relation in parse_edge_types(row["edge_types"]))
            relations.add(ncrl_relation(target_text(row)))
            row_map.append(
                {
                    "split": split,
                    "index": index,
                    "id": row.get("id"),
                    "edge_types": [ncrl_relation(relation) for relation in parse_edge_types(row["edge_types"])],
                    "target": ncrl_relation(target_text(row)),
                    "edge_types_raw": parse_edge_types(row["edge_types"]),
                    "target_raw": target_text(row),
                    "ncrl_query": query,
                }
            )

    consume(train_rows, "train", train_queries)
    consume(valid_rows, "valid", valid_queries)
    consume(test_rows, "test", test_queries)

    # Training and validation query edges are known facts for rule discovery.
    fact_lines = sorted(set(facts + train_queries + valid_queries))
    out_dir = Path(args.output_dir)
    write_lines(out_dir / "entities.txt", sorted(entities))
    write_lines(out_dir / "relations.txt", sorted(relations))
    write_lines(out_dir / "facts.txt", fact_lines)
    write_lines(out_dir / "facts.txt.inv", add_inverse_facts(fact_lines))
    write_lines(out_dir / "train.txt", train_queries)
    write_lines(out_dir / "valid.txt", valid_queries)
    write_lines(out_dir / "test.txt", test_queries)
    write_lines(out_dir / "test_path_facts.txt", sorted(set(facts) - set(train_queries) - set(valid_queries)))
    (out_dir / "row_map.json").write_text(json.dumps(row_map, indent=2), encoding="utf-8")

    manifest = {
        "dataset": "CLUTRR/v1",
        "config": args.config,
        "format": "NCRL KG triples",
        "output_dir": str(out_dir),
        "n_train_queries": len(train_queries),
        "n_valid_queries": len(valid_queries),
        "n_test_queries": len(test_queries),
        "n_entities": len(entities),
        "n_relations": len(relations),
        "n_background_facts": len(fact_lines),
        "n_background_facts_with_inverse": len(fact_lines) * 2,
        "conversion_note": (
            "Each CLUTRR story is represented as a synthetic entity path. "
            "Path edges are background facts and the target relation between "
            "the path endpoints is the train, validation, or test query. "
            "Relation labels are sanitized because NCRL uses '-' and '|' as "
            "internal delimiters."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gen_train234_test2to10")
    parser.add_argument("--output_dir", default="external/NCRL/datasets/clutrr_praxis")
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_valid", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(convert(args), indent=2))


if __name__ == "__main__":
    main()
