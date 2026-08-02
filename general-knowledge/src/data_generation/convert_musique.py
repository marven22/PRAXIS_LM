"""Convert MuSiQue examples into the existing article/question JSON format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _load_musique(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Hugging Face datasets is required to load MuSiQue.") from exc
    return [dict(row) for row in load_dataset(dataset_name, split=split)]


def _paragraph_block(paragraph: Dict[str, Any]) -> str:
    title = _normalize_ws(str(paragraph.get("title", "")))
    text = _normalize_ws(str(paragraph.get("paragraph_text", "")))
    if title and text:
        return f"{title}\n{text}"
    return title or text


def _context_from_musique_item(
    item: Dict[str, Any],
    *,
    context_mode: str,
    max_distractors: int,
    max_context_chars: int,
) -> tuple[str, List[str]]:
    paragraphs = list(item.get("paragraphs", []) or [])
    supporting = [p for p in paragraphs if bool(p.get("is_supporting"))]
    distractors = [p for p in paragraphs if not bool(p.get("is_supporting"))]

    if context_mode == "supporting":
        chosen = supporting
    elif context_mode == "supporting_plus_distractors":
        chosen = supporting + distractors[: max(0, max_distractors)]
        chosen = sorted(chosen, key=lambda p: int(p.get("idx", 0)))
    else:
        chosen = paragraphs

    blocks = [_paragraph_block(p) for p in chosen]
    context = "\n\n".join(block for block in blocks if block.strip()).strip()
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars].rsplit(" ", 1)[0].strip()

    support_titles = [
        _normalize_ws(str(p.get("title", "")))
        for p in supporting
        if _normalize_ws(str(p.get("title", "")))
    ]
    return context, list(dict.fromkeys(support_titles))


def _decomposition_steps(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = []
    for step in item.get("question_decomposition", []) or []:
        steps.append(
            {
                "id": step.get("id", ""),
                "question": _normalize_ws(str(step.get("question", ""))),
                "answer": _normalize_ws(str(step.get("answer", ""))),
                "paragraph_support_idx": step.get("paragraph_support_idx", None),
            }
        )
    return steps


def convert_item(
    item: Dict[str, Any],
    *,
    context_mode: str,
    max_distractors: int,
    max_context_chars: int,
) -> Dict[str, Any]:
    context, support_titles = _context_from_musique_item(
        item,
        context_mode=context_mode,
        max_distractors=max_distractors,
        max_context_chars=max_context_chars,
    )
    source_id = str(item.get("id", "unknown"))
    question = _normalize_ws(str(item.get("question", "")))
    answer = _normalize_ws(str(item.get("answer", "")))
    title = " | ".join(support_titles[:2]) if support_titles else f"MuSiQue::{source_id}"

    return {
        "title": title,
        "context": context,
        "questions": [{"question": question, "answer": answer}],
        "source_dataset": "musique",
        "source_id": source_id,
        "answer_aliases": list(item.get("answer_aliases", []) or []),
        "supporting_titles": support_titles,
        "question_decomposition": _decomposition_steps(item),
    }


def _filter_items(items: Iterable[Dict[str, Any]], include_unanswerable: bool) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        if not include_unanswerable and not bool(item.get("answerable", True)):
            continue
        if not _normalize_ws(str(item.get("question", ""))) or not _normalize_ws(str(item.get("answer", ""))):
            continue
        if not item.get("paragraphs"):
            continue
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dataset", default="dgslibisey/MuSiQue")
    p.add_argument("--hf_split", default="validation")
    p.add_argument("--output_json", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--context_mode",
        choices=["supporting", "supporting_plus_distractors", "full"],
        default="supporting_plus_distractors",
    )
    p.add_argument("--max_distractors", type=int, default=4)
    p.add_argument("--max_context_chars", type=int, default=6000)
    p.add_argument("--include_unanswerable", action="store_true")
    args = p.parse_args()

    raw = _filter_items(
        _load_musique(args.hf_dataset, args.hf_split),
        include_unanswerable=args.include_unanswerable,
    )
    rng = random.Random(args.seed)
    rng.shuffle(raw)
    subset = raw[args.start : args.start + args.n] if args.n > 0 else raw[args.start :]

    converted = [
        convert_item(
            item,
            context_mode=args.context_mode,
            max_distractors=args.max_distractors,
            max_context_chars=args.max_context_chars,
        )
        for item in subset
    ]
    converted = [item for item in converted if item["context"]]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(converted, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    meta = {
        "source_dataset": args.hf_dataset,
        "split": args.hf_split,
        "output_json": args.output_json,
        "n_requested": args.n,
        "n_emitted": len(converted),
        "start": args.start,
        "seed": args.seed,
        "context_mode": args.context_mode,
        "max_distractors": args.max_distractors,
        "max_context_chars": args.max_context_chars,
        "include_unanswerable": args.include_unanswerable,
    }
    json.dump(meta, open(out_path.with_suffix(out_path.suffix + ".meta"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved {len(converted)} MuSiQue examples to {out_path}")


if __name__ == "__main__":
    main()
