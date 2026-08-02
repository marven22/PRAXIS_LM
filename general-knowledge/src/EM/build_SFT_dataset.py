# general-knowledge/src/EM/build_SFT_dataset.py
"""
Convert general-knowledge/results/query_server/run_*.json into an SFT JSONL

Each row keeps exactly the prompt that was fed to vLLM (with
<|im_start|> tags if instruct model) plus the top-k completions ranked by adapter_mean

Output:  general-knowledge/data/synthetic_data/EM_SFT/sft_best<k>of<k2>_<timestamp>.jsonl

Example usage:
    python3 general-knowledge/src/EM/build_SFT_dataset.py general-knowledge/results/query_server/train/rank_iter0.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ------------------------- helper funcs ------------------------------
def _select_completions(
    comps: List[Dict[str, Any]],
    k: int,
    metric: str,
    min_threshold: float,
    allow_fallback: bool,
) -> List[str]:
    """Return up to k completion texts, filtered by min_threshold with optional fallback."""
    ranked = sorted(
        comps,
        key=lambda c: c["stats"].get(metric, 0),
        reverse=True,
    )
    filtered = [
        c["text"].strip()
        for c in ranked
        if c["text"].strip() and c["stats"].get(metric, 0) >= min_threshold
    ]
    if not filtered and allow_fallback and ranked:
        top = ranked[0]["text"].strip()
        return [top] if top else []
    return filtered[:k]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("query_json", nargs="?", help="run_*.json from query_server")
    p.add_argument("--query_jsons", nargs="*", default=[],
                   help="multiple run_*.json files to aggregate")
    p.add_argument("--output_dir", default="general-knowledge/data/synthetic_data/EM_SFT",
                   help="destination folder for the JSONL")
    p.add_argument("--k_best", type=int, default=1,
                   help="top-k completions per article to keep")
    p.add_argument(
        "--metric", choices=["adapter_mean", "proxy_mean"], default="adapter_mean",
        help="stats metric to rank completions by"
    )
    p.add_argument("--min_threshold", type=float, default=0.0,
                   help="minimum metric value required to keep a completion")
    p.add_argument("--no_fallback", action="store_true",
                   help="disable fallback to top-1 when nothing meets threshold")
    return p.parse_args()

# ----------------------------- main ----------------------------------
def main() -> None:
    args = _parse_args()

    if not args.query_json and not args.query_jsons:
        raise SystemExit("Provide a query_json or --query_jsons")

    run_files = []
    if args.query_json:
        run_files.append(args.query_json)
    run_files.extend(args.query_jsons)

    runs: List[Dict[str, Any]] = [
        json.load(open(p, encoding="utf-8")) for p in run_files
    ]

    timestamp = runs[-1].get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sft_best{args.k_best}of{len(runs[-1]['articles'][0]['completions'])}_{timestamp}.jsonl"

    n_rows = 0
    fallback_count = 0
    # aggregate completions across runs by prompt
    by_prompt: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        for art in run["articles"]:
            prompt = art["prompt"]
            entry = by_prompt.setdefault(prompt, {"completions": [], "seen": set()})
            for comp in art.get("completions", []):
                text = comp.get("text", "").strip()
                if not text:
                    continue
                if text in entry["seen"]:
                    continue
                entry["seen"].add(text)
                entry["completions"].append(comp)

    with out_path.open("w", encoding="utf-8") as fout:
        for prompt, entry in by_prompt.items():
            meets_threshold = any(
                c.get("text", "").strip() and c["stats"].get(args.metric, 0) >= args.min_threshold
                for c in entry["completions"]
            )
            selected = _select_completions(
                entry["completions"],
                args.k_best,
                args.metric,
                args.min_threshold,
                allow_fallback=not args.no_fallback,
            )
            if not meets_threshold and not args.no_fallback:
                fallback_count += 1
            if not selected:
                continue
            for comp in selected:
                row = {
                    "prompt":     prompt,
                    "completion": comp,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"wrote {n_rows} examples → {out_path}")
    if args.min_threshold > 0.0 and not args.no_fallback:
        print(f"fallback used for {fallback_count} articles")


if __name__ == "__main__":
    main()
