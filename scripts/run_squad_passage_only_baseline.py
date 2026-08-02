from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_bool_list(values: Any) -> list[bool] | None:
    if values is None:
        return None
    return [bool(v) for v in values]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    passage = load_json(Path(args.passage_only_result))
    reference = load_json(Path(args.reference_result))

    passage_correct = to_bool_list(passage.get("per_question", {}).get("adapter_correct"))
    ref_base = to_bool_list(reference.get("per_question", {}).get("baseline_correct"))
    if passage_correct is None:
        raise ValueError(f"{args.passage_only_result} does not contain per_question.adapter_correct")
    if ref_base is None:
        raise ValueError(f"{args.reference_result} does not contain per_question.baseline_correct")

    n = min(len(passage_correct), len(ref_base))
    passage_correct = passage_correct[:n]
    ref_base = ref_base[:n]
    final = sum(passage_correct) / max(n, 1)
    initial = sum(ref_base) / max(n, 1)
    pgr = sum((not b) and a for b, a in zip(ref_base, passage_correct)) / max(n, 1)
    dr = sum(b and (not a) for b, a in zip(ref_base, passage_correct)) / max(n, 1)

    summary = {
        "dataset": "SQuAD-style QA",
        "method": "Passage-only adaptation",
        "role": "CPT on raw passage text without generated edits or archive material",
        "source": str(Path(args.passage_only_result)),
        "reference_source": str(Path(args.reference_result)),
        "n_articles": passage.get("n_articles"),
        "n_questions": n,
        "initial": initial,
        "final": final,
        "gain": final - initial,
        "pgr": pgr,
        "dr": dr,
        "time_per_example_seconds": args.time_per_example_seconds,
        "runtime_seconds": time.time() - started,
        "results": [
            {
                "index": i,
                "baseline_correct": b,
                "adapter_correct": a,
                "improved": (not b) and a,
                "degraded": b and (not a),
            }
            for i, (b, a) in enumerate(zip(ref_base, passage_correct))
        ],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passage_only_result", default="general-knowledge/results/cpt/200/naive_200.json")
    parser.add_argument("--reference_result", default="general-knowledge/results/cpt/200/base_200.json")
    parser.add_argument("--output", default="analysis_results/native_baselines/squad_passage_only_200.json")
    parser.add_argument(
        "--time_per_example_seconds",
        type=float,
        default=None,
        help="Optional externally measured wall-clock seconds per example.",
    )
    args = parser.parse_args()

    summary = summarize(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "method": summary["method"],
                "n_questions": summary["n_questions"],
                "initial": summary["initial"],
                "final": summary["final"],
                "gain": summary["gain"],
                "pgr": summary["pgr"],
                "dr": summary["dr"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
