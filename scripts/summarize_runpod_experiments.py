import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median, pstdev


ARTICLE_RE = re.compile(
    r"^Article\s+(?P<idx>\d+)\s+base\s+(?P<base>[\d.]+)%\s+\|\s+adapter\s+(?P<adapter>[\d.]+)%.*gain\s+(?P<gain>[+-]?[\d.]+)%"
)
COMP_RE = re.compile(
    r"^\[(?P<idx>\d+)\.(?P<comp>\d+)\]\s+base\s+(?P<base>[\d.]+)%\s+\|\s+adapter\s+(?P<adapter>[\d.]+)%.*gain\s+(?P<gain>[+-]?[\d.]+)%"
)


def pct_to_float(value: str) -> float:
    return float(value) / 100.0


def load_json_run(path: Path, index_offset: int = 0) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for idx, article in enumerate(data.get("articles", [])):
        stats = article["stats"]
        completions = []
        for comp in article.get("completions", []):
            cstats = comp.get("stats", {})
            if "adapter_mean" in cstats:
                completions.append(float(cstats["adapter_mean"]))
        out.append(
            {
                "idx": idx + index_offset,
                "baseline": float(stats["baseline_accuracy"]),
                "adapter": float(stats["adapter_mean_accuracy"]),
                "gain": float(stats["mean_gain"]),
                "completions": completions,
                "source": "json",
            }
        )
    return out


def load_log_run(path: Path, index_offset: int = 0) -> list[dict]:
    articles: dict[int, dict] = {}
    comps: dict[int, list[float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cm = COMP_RE.match(line)
        if cm:
            idx = int(cm.group("idx")) + index_offset
            comps.setdefault(idx, []).append(pct_to_float(cm.group("adapter")))
            continue
        am = ARTICLE_RE.match(line)
        if am:
            idx = int(am.group("idx")) + index_offset
            articles[idx] = {
                "idx": idx,
                "baseline": pct_to_float(am.group("base")),
                "adapter": pct_to_float(am.group("adapter")),
                "gain": pct_to_float(am.group("gain")),
                "completions": comps.get(idx, []),
                "source": "log",
            }
    for idx, values in comps.items():
        articles.setdefault(
            idx,
            {
                "idx": idx,
                "baseline": math.nan,
                "adapter": mean(values) if values else math.nan,
                "gain": math.nan,
                "completions": values,
                "source": "log_incomplete",
            },
        )
    return [articles[idx] for idx in sorted(articles)]


def summarize(name: str, articles: list[dict]) -> dict:
    complete = [a for a in articles if not math.isnan(a["baseline"])]
    comp_values = [a["completions"] for a in complete if a.get("completions")]
    all_comps = [x for xs in comp_values for x in xs]
    best = [max(xs) for xs in comp_values if xs]
    worst = [min(xs) for xs in comp_values if xs]
    return {
        "name": name,
        "n_articles": len(complete),
        "baseline": mean(a["baseline"] for a in complete),
        "adapter": mean(a["adapter"] for a in complete),
        "gain": mean(a["gain"] for a in complete),
        "adapter_std_article": pstdev([a["adapter"] for a in complete]) if len(complete) > 1 else 0.0,
        "median_gain": median(a["gain"] for a in complete),
        "positive_gain_rate": mean(1.0 if a["gain"] > 0 else 0.0 for a in complete),
        "negative_gain_rate": mean(1.0 if a["gain"] < 0 else 0.0 for a in complete),
        "same_gain_rate": mean(1.0 if a["gain"] == 0 else 0.0 for a in complete),
        "catastrophic_drop_rate": mean(1.0 if a["gain"] <= -0.25 else 0.0 for a in complete),
        "best_of_k": mean(best) if best else math.nan,
        "worst_of_k": mean(worst) if worst else math.nan,
        "mean_completion_adapter": mean(all_comps) if all_comps else math.nan,
    }


def compare_by_index(left_name: str, left: list[dict], right_name: str, right: list[dict]) -> dict:
    lmap = {a["idx"]: a for a in left if not math.isnan(a["baseline"]) and not math.isnan(a["gain"])}
    rmap = {a["idx"]: a for a in right if not math.isnan(a["baseline"]) and not math.isnan(a["gain"])}
    common = sorted(set(lmap) & set(rmap))
    adapter_diffs = [lmap[i]["adapter"] - rmap[i]["adapter"] for i in common]
    gain_diffs = [lmap[i]["gain"] - rmap[i]["gain"] for i in common]
    return {
        "left": left_name,
        "right": right_name,
        "n_common": len(common),
        "left_adapter_win_rate": mean(1.0 if d > 0 else 0.0 for d in adapter_diffs),
        "left_gain_win_rate": mean(1.0 if d > 0 else 0.0 for d in gain_diffs),
        "adapter_diff_mean": mean(adapter_diffs),
        "gain_diff_mean": mean(gain_diffs),
        "adapter_diff_median": median(adapter_diffs),
        "gain_diff_median": median(gain_diffs),
    }


def fmt_pct(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def load_overall(path: Path) -> dict:
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("overall", {})


def attach_runtime(summary: dict, overall: dict) -> None:
    for key in [
        "wall_clock_total_seconds",
        "wall_clock_per_article_seconds",
        "wall_clock_per_completion_seconds",
        "gain_per_minute",
        "mean_completion_chars",
        "median_completion_chars",
        "mean_completion_words",
        "median_completion_words",
        "mean_prompt_chars",
        "mean_context_chars",
        "mean_context_compression_ratio",
    ]:
        summary[key] = overall.get(key, math.nan)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--praxis-json", type=Path, required=True)
    parser.add_argument("--seal-log", type=Path, required=True)
    parser.add_argument("--seal-resume-json", type=Path, required=True)
    parser.add_argument("--praxis-sft-log", type=Path)
    parser.add_argument("--seal-sft-log", type=Path)
    parser.add_argument("--context-log", type=Path)
    parser.add_argument("--out", type=Path, default=Path("analysis_results/runpod_summary.json"))
    args = parser.parse_args()

    praxis = load_json_run(args.praxis_json)
    seal_first = load_log_run(args.seal_log)
    seal_resume = load_json_run(args.seal_resume_json, index_offset=107)
    seal = sorted(seal_first + seal_resume, key=lambda item: item["idx"])

    runs = {
        "praxis_200": praxis,
        "seal_200_combined": seal,
        "seal_first_107": seal_first,
        "seal_resume_93": seal_resume,
    }
    if args.praxis_sft_log and args.praxis_sft_log.exists():
        runs["praxis_sft_partial"] = load_log_run(args.praxis_sft_log)
    if args.seal_sft_log and args.seal_sft_log.exists():
        runs["seal_sft_partial"] = load_log_run(args.seal_sft_log)
    if args.context_log and args.context_log.exists():
        runs["context_only_partial"] = load_log_run(args.context_log)

    summaries = {name: summarize(name, articles) for name, articles in runs.items()}
    attach_runtime(summaries["praxis_200"], load_overall(args.praxis_json))
    attach_runtime(summaries["seal_resume_93"], load_overall(args.seal_resume_json))
    # The first SEAL segment is log-only because the original client crashed before
    # writing its JSON summary. Combined runtime is therefore intentionally left blank.
    for key in [
        "wall_clock_total_seconds",
        "wall_clock_per_article_seconds",
        "wall_clock_per_completion_seconds",
        "gain_per_minute",
        "mean_completion_chars",
        "median_completion_chars",
        "mean_completion_words",
        "median_completion_words",
        "mean_prompt_chars",
        "mean_context_chars",
        "mean_context_compression_ratio",
    ]:
        summaries["seal_200_combined"][key] = math.nan
        summaries["seal_first_107"][key] = math.nan
    comparisons = {
        "praxis_minus_seal_all": compare_by_index("praxis", praxis, "seal", seal),
        "praxis_minus_seal_first_107": compare_by_index("praxis", praxis[:107], "seal", seal_first),
        "praxis_minus_seal_resume_93": compare_by_index("praxis", praxis[107:], "seal", seal_resume),
    }
    if "praxis_sft_partial" in runs and "seal_sft_partial" in runs:
        comparisons["praxis_sft_minus_seal_sft_common"] = compare_by_index(
            "praxis_sft", runs["praxis_sft_partial"], "seal_sft", runs["seal_sft_partial"]
        )

    result = {"summaries": summaries, "comparisons": comparisons}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    metric_fields = [
        "name",
        "n_articles",
        "baseline",
        "adapter",
        "gain",
        "adapter_std_article",
        "median_gain",
        "positive_gain_rate",
        "same_gain_rate",
        "negative_gain_rate",
        "catastrophic_drop_rate",
        "best_of_k",
        "worst_of_k",
        "mean_completion_adapter",
        "wall_clock_total_seconds",
        "wall_clock_per_article_seconds",
        "wall_clock_per_completion_seconds",
        "gain_per_minute",
        "mean_completion_chars",
        "median_completion_chars",
        "mean_completion_words",
        "median_completion_words",
        "mean_prompt_chars",
        "mean_context_chars",
        "mean_context_compression_ratio",
    ]
    comparison_fields = [
        "comparison",
        "left",
        "right",
        "n_common",
        "left_adapter_win_rate",
        "left_gain_win_rate",
        "adapter_diff_mean",
        "gain_diff_mean",
        "adapter_diff_median",
        "gain_diff_median",
    ]
    write_csv(args.out.with_name("runpod_metric_table.csv"), list(summaries.values()), metric_fields)
    comp_rows = [dict({"comparison": key}, **value) for key, value in comparisons.items()]
    write_csv(args.out.with_name("runpod_comparison_table.csv"), comp_rows, comparison_fields)

    print("Run summaries")
    for name, summary in summaries.items():
        print(
            f"{name:24s} n={summary['n_articles']:3d} "
            f"base={fmt_pct(summary['baseline'])} adapter={fmt_pct(summary['adapter'])} "
            f"gain={fmt_pct(summary['gain'])} best-of-k={fmt_pct(summary['best_of_k'])} "
            f"pos-rate={fmt_pct(summary['positive_gain_rate'])}"
        )
    print("\nComparisons")
    for name, comp in comparisons.items():
        print(
            f"{name:34s} n={comp['n_common']:3d} "
            f"adapter_diff={fmt_pct(comp['adapter_diff_mean'])} "
            f"gain_diff={fmt_pct(comp['gain_diff_mean'])} "
            f"adapter_win={fmt_pct(comp['left_adapter_win_rate'])} "
            f"gain_win={fmt_pct(comp['left_gain_win_rate'])}"
        )
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.out.with_name('runpod_metric_table.csv')}")
    print(f"Wrote {args.out.with_name('runpod_comparison_table.csv')}")


if __name__ == "__main__":
    main()
