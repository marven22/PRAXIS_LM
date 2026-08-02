"""Convert BoolQ examples into the existing article/question JSON format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _load_boolq(split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Hugging Face datasets is required to load BoolQ.") from exc
    return [dict(row) for row in load_dataset("google/boolq", split=split)]


def convert_item(item: Dict[str, Any], idx: int, max_context_chars: int) -> Dict[str, Any]:
    title = f"BoolQ::{idx:05d}"
    context = _normalize_ws(str(item.get("passage", "")))
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars].rsplit(" ", 1)[0].strip()

    question = _normalize_ws(str(item.get("question", "")))
    if question and not question.endswith("?"):
        question = f"{question}?"
    answer = "yes" if bool(item.get("answer")) else "no"

    return {
        "title": title,
        "context": context,
        "questions": [{"question": question, "answer": answer}],
        "source_dataset": "boolq",
        "source_id": str(idx),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", help="BoolQ split, usually train or validation")
    p.add_argument("--output_json", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_context_chars", type=int, default=3500)
    args = p.parse_args()

    raw = _load_boolq(args.split)
    indexed = list(enumerate(raw))
    rng = random.Random(args.seed)
    rng.shuffle(indexed)
    subset = indexed[args.start : args.start + args.n] if args.n > 0 else indexed[args.start :]

    converted = [
        convert_item(item, idx, args.max_context_chars)
        for idx, item in subset
        if _normalize_ws(str(item.get("passage", ""))) and _normalize_ws(str(item.get("question", "")))
    ]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(converted, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    meta = {
        "source_dataset": "google/boolq",
        "split": args.split,
        "output_json": args.output_json,
        "n_requested": args.n,
        "n_emitted": len(converted),
        "start": args.start,
        "seed": args.seed,
        "max_context_chars": args.max_context_chars,
    }
    json.dump(meta, open(out_path.with_suffix(out_path.suffix + ".meta"), "w", encoding="utf-8"), indent=2)
    print(f"Saved {len(converted)} BoolQ examples to {out_path}")


if __name__ == "__main__":
    main()
