"""Convert a stratified DROP subset into the article/question experiment format."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


CATEGORIES = ("count", "arithmetic", "comparison", "temporal", "reference")


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def classify_question(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(how many|number of)\b", q):
        if re.search(r"\b(total|altogether|combined|in all|more|fewer|difference)\b", q):
            return "arithmetic"
        return "count"
    if re.search(r"\b(sum|total|altogether|combined|difference|average|percent|yards? (?:more|less))\b", q):
        return "arithmetic"
    if re.search(r"\b(more|fewer|higher|lower|larger|smaller|longer|shorter|most|least|largest|smallest)\b", q):
        return "comparison"
    if re.search(r"\b(first|last|before|after|earlier|later|how long|what year|when)\b", q):
        return "temporal"
    return "reference"


def _load(dataset_name: str, split: str) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Hugging Face datasets is required to load DROP.") from exc
    return load_dataset(dataset_name, split=split)


def convert_item(item: Dict[str, Any], category: str, max_context_chars: int) -> Dict[str, Any]:
    question = _normalize(str(item.get("question", "")))
    passage = _normalize(str(item.get("passage", "")))
    if max_context_chars > 0 and len(passage) > max_context_chars:
        passage = passage[:max_context_chars].rsplit(" ", 1)[0].strip()

    answer_data = item.get("answers_spans", {}) or {}
    aliases = [_normalize(str(v)) for v in answer_data.get("spans", []) if _normalize(str(v))]
    answer = aliases[0] if aliases else ""
    source_id = str(item.get("query_id", "unknown"))

    return {
        "title": f"Question: {question}",
        "context": passage,
        "questions": [{"question": question, "answer": answer}],
        "source_dataset": "drop",
        "source_id": source_id,
        "section_id": str(item.get("section_id", "")),
        "answer_aliases": aliases,
        "answer_types": list(answer_data.get("types", []) or []),
        "operation_category": category,
        "strategy_hint": {
            "count": "count_and_enumerate",
            "arithmetic": "arithmetic_derivation",
            "comparison": "comparison_resolution",
            "temporal": "temporal_event_order",
            "reference": "entity_reference_resolution",
        }[category],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", default="ucinlp/drop")
    parser.add_argument("--hf_split", default="validation")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--n_per_category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_chars", type=int, default=6000)
    args = parser.parse_args()

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _load(args.hf_dataset, args.hf_split):
        category = classify_question(str(row.get("question", "")))
        converted = convert_item(dict(row), category, args.max_context_chars)
        if converted["context"] and converted["questions"][0]["answer"]:
            buckets[category].append(converted)

    rng = random.Random(args.seed)
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for category in CATEGORIES:
        rng.shuffle(buckets[category])
        chosen = buckets[category][: args.n_per_category]
        if len(chosen) < args.n_per_category:
            raise ValueError(f"Only {len(chosen)} usable examples for category {category}")
        selected.extend(chosen)
        counts[category] = len(chosen)
    rng.shuffle(selected)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(output.suffix + ".meta").write_text(
        json.dumps(
            {
                "source_dataset": args.hf_dataset,
                "split": args.hf_split,
                "seed": args.seed,
                "n": len(selected),
                "category_counts": counts,
                "max_context_chars": args.max_context_chars,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(selected)} DROP examples to {output}: {counts}")


if __name__ == "__main__":
    main()
