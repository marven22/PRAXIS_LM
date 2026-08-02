"""
Convert HotpotQA examples into the article/question JSON format already used by
the existing general-knowledge SQuAD-style pipelines.

This script is intentionally additive: it does not modify any SQuAD codepaths or
existing dataset files. The output format matches:

[
  {
    "title": "...",
    "context": "...",
    "questions": [{"question": "...", "answer": "..."}],
    ...
  }
]

Supported input modes:
1. Hugging Face datasets name/config loading, e.g.:
      --hf_dataset hotpotqa/hotpot_qa --hf_config distractor --hf_split validation
2. Local JSON file loading, where the file contains a list of HotpotQA-style examples:
      --input_json path/to/hotpot_validation.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _sentence_block(title: str, sentences: List[str]) -> str:
    clean = [_normalize_ws(s) for s in sentences if _normalize_ws(s)]
    body = " ".join(clean).strip()
    return f"{title}\n{body}".strip()


def _context_from_hotpot_item(
    item: Dict[str, Any],
    *,
    context_mode: str,
    max_context_chars: int,
) -> str:
    ctx = item.get("context", {}) or {}
    titles = list(ctx.get("title", []) or [])
    sentence_groups = list(ctx.get("sentences", []) or [])

    support = item.get("supporting_facts", {}) or {}
    support_titles = list(support.get("title", []) or [])
    support_sent_ids = list(support.get("sent_id", []) or [])

    blocks: List[str] = []

    if context_mode == "supporting":
        support_map: Dict[str, set[int]] = {}
        for t, sent_id in zip(support_titles, support_sent_ids):
            support_map.setdefault(t, set()).add(int(sent_id))

        for title, sents in zip(titles, sentence_groups):
            if title not in support_map:
                continue
            chosen = []
            for idx in sorted(support_map[title]):
                if 0 <= idx < len(sents):
                    chosen.append(sents[idx])
            if chosen:
                blocks.append(_sentence_block(title, chosen))
    elif context_mode == "supporting_plus_context":
        support_map: Dict[str, set[int]] = {}
        for t, sent_id in zip(support_titles, support_sent_ids):
            support_map.setdefault(t, set()).add(int(sent_id))

        for title, sents in zip(titles, sentence_groups):
            if title not in support_map:
                continue
            chosen = set()
            for idx in support_map[title]:
                for neighbor in (idx - 1, idx, idx + 1):
                    if 0 <= neighbor < len(sents):
                        chosen.add(neighbor)
            if chosen:
                ordered = [sents[idx] for idx in sorted(chosen)]
                blocks.append(_sentence_block(title, ordered))
    else:
        for title, sents in zip(titles, sentence_groups):
            blocks.append(_sentence_block(title, sents))

    context = "\n\n".join(block for block in blocks if block.strip()).strip()
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[: max_context_chars].rsplit(" ", 1)[0].strip()
    return context


def _title_from_hotpot_item(item: Dict[str, Any]) -> str:
    support = item.get("supporting_facts", {}) or {}
    support_titles = list(dict.fromkeys(support.get("title", []) or []))
    if support_titles:
        return " | ".join(support_titles[:2])
    ctx = item.get("context", {}) or {}
    titles = list(ctx.get("title", []) or [])
    if titles:
        return " | ".join(titles[:2])
    return f"HotpotQA::{item.get('id', 'unknown')}"


def convert_item(
    item: Dict[str, Any],
    *,
    context_mode: str,
    max_context_chars: int,
) -> Dict[str, Any]:
    title = _title_from_hotpot_item(item)
    context = _context_from_hotpot_item(
        item,
        context_mode=context_mode,
        max_context_chars=max_context_chars,
    )
    answer = _normalize_ws(str(item.get("answer", "")))
    question = _normalize_ws(str(item.get("question", "")))

    return {
        "title": title,
        "context": context,
        "questions": [{"question": question, "answer": answer}],
        "source_dataset": "hotpotqa",
        "source_id": item.get("id", ""),
        "hotpot_type": item.get("type", ""),
        "hotpot_level": item.get("level", ""),
        "supporting_titles": list(dict.fromkeys((item.get("supporting_facts", {}) or {}).get("title", []) or [])),
    }


def _load_from_hf(dataset_name: str, config: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Hugging Face datasets is not installed. Install `datasets` or use --input_json."
        ) from exc

    ds = load_dataset(dataset_name, config, split=split)
    return [dict(row) for row in ds]


def _load_from_json(path: str) -> List[Dict[str, Any]]:
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError("Expected input_json to contain a list of HotpotQA items.")


def _filter_items(items: Iterable[Dict[str, Any]], include_yesno: bool) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        answer = _normalize_ws(str(item.get("answer", ""))).lower()
        if not include_yesno and answer in {"yes", "no"}:
            continue
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_dataset", help="HF dataset name, e.g. hotpotqa/hotpot_qa")
    src.add_argument("--input_json", help="Local JSON file containing HotpotQA items")

    p.add_argument("--hf_config", default="distractor", help="HF config, e.g. distractor or fullwiki")
    p.add_argument("--hf_split", default="validation", help="HF split, e.g. train / validation")
    p.add_argument("--output_json", required=True, help="Converted output path")
    p.add_argument("--n", type=int, default=50, help="Number of examples to emit")
    p.add_argument("--start", type=int, default=0, help="Start offset after shuffling/filtering")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p.add_argument(
        "--context_mode",
        choices=["supporting", "supporting_plus_context", "full"],
        default="supporting_plus_context",
        help="How much Hotpot context to preserve in the flattened context field",
    )
    p.add_argument("--max_context_chars", type=int, default=5000, help="Truncate serialized context to this many chars (0 disables)")
    p.add_argument("--include_yesno", action="store_true", help="Keep yes/no questions (off by default)")
    args = p.parse_args()

    if args.hf_dataset:
        raw = _load_from_hf(args.hf_dataset, args.hf_config, args.hf_split)
    else:
        raw = _load_from_json(args.input_json)

    raw = _filter_items(raw, include_yesno=args.include_yesno)

    rng = random.Random(args.seed)
    rng.shuffle(raw)
    subset = raw[args.start : args.start + args.n] if args.n > 0 else raw[args.start :]

    converted = [
        convert_item(
            item,
            context_mode=args.context_mode,
            max_context_chars=args.max_context_chars,
        )
        for item in subset
    ]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(converted, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    meta = {
        "source": {
            "hf_dataset": args.hf_dataset,
            "hf_config": args.hf_config if args.hf_dataset else None,
            "hf_split": args.hf_split if args.hf_dataset else None,
            "input_json": args.input_json,
        },
        "output_json": args.output_json,
        "n": len(converted),
        "start": args.start,
        "seed": args.seed,
        "context_mode": args.context_mode,
        "max_context_chars": args.max_context_chars,
        "include_yesno": args.include_yesno,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"Wrote {len(converted)} converted HotpotQA items -> {out_path}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
