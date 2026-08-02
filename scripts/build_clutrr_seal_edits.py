from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from run_clutrr_llm_baseline import LABELS


def parse_edge_types(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in ast.literal_eval(value)]


def relation_chain(row: dict) -> str:
    return " -> ".join(parse_edge_types(row["edge_types"]))


def parse_proof_state(value) -> list:
    if not value:
        return []
    try:
        return ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError):
        return []


def extract_pair_rules(rows) -> dict[tuple[str, str], str]:
    votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        edges = parse_edge_types(row["edge_types"])
        if len(edges) == 2:
            votes[(edges[0], edges[1])][row["target_text"]] += 1
        for proof in parse_proof_state(row.get("proof_state")):
            if not isinstance(proof, dict):
                continue
            for conclusion, premises in proof.items():
                if not isinstance(conclusion, tuple) or not isinstance(premises, list) or len(premises) != 2:
                    continue
                left, right = premises
                if len(left) >= 3 and len(right) >= 3:
                    votes[(str(left[1]), str(right[1]))][str(conclusion[1])] += 1
    return {pair: counts.most_common(1)[0][0] for pair, counts in sorted(votes.items())}


def make_messages(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def atomic_edit(pair: tuple[str, str], output: str) -> dict:
    left, right = pair
    user = (
        "Learn this family-relation composition edit.\n"
        f"Edit: when a relation chain contains '{left} -> {right}', that span composes to '{output}'.\n\n"
        f"Relation chain: {left} -> {right}\n"
        f"Candidate labels: {', '.join(LABELS)}\n"
        "Final relation:"
    )
    return make_messages(user, f"RELATION: {output}")


def full_chain_edit(row: dict) -> dict:
    chain = relation_chain(row)
    proof_lines = []
    for proof in parse_proof_state(row.get("proof_state")):
        if not isinstance(proof, dict):
            continue
        for conclusion, premises in proof.items():
            if isinstance(conclusion, tuple) and isinstance(premises, list) and len(premises) == 2:
                left, right = premises
                if len(left) >= 3 and len(right) >= 3:
                    proof_lines.append(f"{left[1]} + {right[1]} -> {conclusion[1]}")
    proof_text = "\n".join(proof_lines[:8]) if proof_lines else "Compose adjacent relation spans until one relation remains."
    user = (
        "Apply family-relation composition edits to solve the chain.\n"
        f"Relation chain: {chain}\n"
        f"Useful intermediate edits:\n{proof_text}\n"
        f"Candidate labels: {', '.join(LABELS)}\n"
        "Final relation:"
    )
    return make_messages(user, f"RELATION: {row['target_text']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gen_train234_test2to10")
    parser.add_argument("--output", default="data/clutrr/seal_edits_train.jsonl")
    parser.add_argument("--max_full_chain", type=int, default=1000)
    args = parser.parse_args()

    rows = list(load_dataset("CLUTRR/v1", args.config, split="train"))
    pair_rules = extract_pair_rules(rows)
    edits = [atomic_edit(pair, output) for pair, output in pair_rules.items()]
    for row in rows[: args.max_full_chain]:
        edits.append(full_chain_edit(row))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in edits:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    meta = {
        "dataset": "CLUTRR/v1",
        "config": args.config,
        "n_atomic_rules": len(pair_rules),
        "n_full_chain": min(args.max_full_chain, len(rows)),
        "n_edits": len(edits),
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
