# src/query/query_server.py
"""
Query TTT server on SQuAD synthetic data. This drives the inner-loop TTT server:
sample k synthetic completions per article, run `eval_times` fine-tune+eval 
cycles, and write an aggregated JSON report. This is used in both ReST-EM 
training and evaluation with n=1.

The results are written to a JSON file. The overall summary contains:
    baseline_mean_accuracy                - mean of article-level baseline means
    baseline_std_of_article_means         - std-dev across those means
    adapter_mean_accuracy                 - mean of article-level adapter means
    adapter_std_of_article_means          - std-dev across those means
    mean_adapter_std_over_completions     - average per-article adapter std-dev 
                                            (across the k completions);
                                            higher ⇒ more variance the RL selector
                                            can exploit
    mean_adapter_std_within_completions   - avg std-dev within each completion,
                                            across TTT eval runs; 
                                            lower ⇒ more stable signal
    mean_gain                             - adapter_mean - baseline_mean

This requires first running TTT_server.py to set up the inner-loop server.
"""
import argparse
import datetime as _dt
import json
import pathlib
import random
import statistics as _stats
import sys
import time
import zmq
from typing import Any, Dict, List, Optional, Tuple

import math
try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_EMBEDDER = None

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    if SentenceTransformer is None:
        return None
    try:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _EMBEDDER = None
    return _EMBEDDER

from ..archive_utils import canonical_prompt_key, canonical_strategy_name, normalize_archive
from ..utils import (
    build_train_sequences,
)


def _safe_mean(values: List[float]) -> float:
    return _stats.mean(values) if values else 0.0


def _safe_median(values: List[float]) -> float:
    return _stats.median(values) if values else 0.0


def _completion_length_metrics(articles: List[Dict[str, Any]]) -> Dict[str, float]:
    completion_chars: List[int] = []
    completion_words: List[int] = []
    prompt_chars: List[int] = []
    context_chars: List[int] = []
    compression_ratios: List[float] = []

    for article in articles:
        prompt_chars.append(len(article.get("prompt", "") or ""))
        context_len = len(article.get("context", "") or "")
        context_chars.append(context_len)
        for comp in article.get("completions", []):
            text = comp.get("text", "") or ""
            char_len = len(text)
            completion_chars.append(char_len)
            completion_words.append(len(text.split()))
            if context_len > 0:
                compression_ratios.append(char_len / context_len)

    return {
        "mean_completion_chars": round(_safe_mean(completion_chars), 4),
        "median_completion_chars": round(_safe_median(completion_chars), 4),
        "mean_completion_words": round(_safe_mean(completion_words), 4),
        "median_completion_words": round(_safe_median(completion_words), 4),
        "mean_prompt_chars": round(_safe_mean(prompt_chars), 4),
        "mean_context_chars": round(_safe_mean(context_chars), 4),
        "mean_context_compression_ratio": round(_safe_mean(compression_ratios), 4),
    }


def _article_outcome_metrics(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_vals = [float(a["stats"]["baseline_accuracy"]) for a in articles]
    adapter_vals = [float(a["stats"]["adapter_mean_accuracy"]) for a in articles]
    gains = [float(a["stats"]["mean_gain"]) for a in articles]

    best_completion_accs: List[float] = []
    worst_completion_accs: List[float] = []
    for article in articles:
        comp_adapters = [float(c["stats"].get("adapter_mean", 0.0)) for c in article.get("completions", [])]
        if comp_adapters:
            best_completion_accs.append(max(comp_adapters))
            worst_completion_accs.append(min(comp_adapters))

    improved = sum(1 for g in gains if g > 0)
    same = sum(1 for g in gains if g == 0)
    worsened = sum(1 for g in gains if g < 0)

    return {
        "median_article_baseline": round(_safe_median(baseline_vals), 4),
        "median_article_adapter": round(_safe_median(adapter_vals), 4),
        "median_article_gain": round(_safe_median(gains), 4),
        "relative_gain": round((_safe_mean(adapter_vals) - _safe_mean(baseline_vals)) / max(_safe_mean(baseline_vals), 1e-8), 4),
        "articles_improved_count": improved,
        "articles_same_count": same,
        "articles_worsened_count": worsened,
        "positive_gain_rate": round(improved / max(1, len(articles)), 4),
        "catastrophic_drop_rate": round(sum(1 for g in gains if g <= -0.25) / max(1, len(articles)), 4),
        "best_completion_accuracy_mean": round(_safe_mean(best_completion_accs), 4),
        "worst_completion_accuracy_mean": round(_safe_mean(worst_completion_accs), 4),
    }


def _strategy_metrics(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    usage_counts: Dict[str, int] = {}
    gain_sums: Dict[str, float] = {}
    positive_counts: Dict[str, int] = {}

    for article in articles:
        strategy = article.get("strategy") or article.get("prompt_key") or "unknown"
        usage_counts[strategy] = usage_counts.get(strategy, 0) + 1
        gain = float(article["stats"].get("mean_gain", 0.0))
        gain_sums[strategy] = gain_sums.get(strategy, 0.0) + gain
        if gain > 0:
            positive_counts[strategy] = positive_counts.get(strategy, 0) + 1

    strategy_summary: Dict[str, Dict[str, float]] = {}
    total = max(1, len(articles))
    for strategy, count in usage_counts.items():
        strategy_summary[strategy] = {
            "usage_count": count,
            "usage_fraction": round(count / total, 4),
            "mean_gain": round(gain_sums.get(strategy, 0.0) / count, 4),
            "positive_gain_rate": round(positive_counts.get(strategy, 0) / count, 4),
        }

    return {
        "strategy_usage_counts": usage_counts,
        "strategy_summary": strategy_summary,
    }

# -------------------------- ARGPARSE / CONFIG ------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="rank_iter0")
    p.add_argument("--dataset", default="data/synthetic_data/train/iter0_train.json")
    p.add_argument("--output_dir", default="results/query_server")
    p.add_argument("--server_host", default="127.0.0.1")
    p.add_argument("--zmq_port", type=int, default=5555)

    p.add_argument("--n_articles", type=int, default=3)
    p.add_argument("--start_article", type=int, default=0)
    p.add_argument("--k_completions", type=int, default=5)
    p.add_argument("--eval_times", type=int, default=3)

    # LoRA / optimization hyperparams
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--end_mask_substring", default="")
    p.add_argument("--split_newlines", action="store_true")
    p.add_argument("--chain_of_thought", action="store_true")
    p.add_argument("--reward_mode", choices=["ttt", "proxy", "both"], default="ttt")
    p.add_argument("--proxy_mode", choices=["rubric", "grounded"], default="rubric",
                   help="Proxy scoring mode when reward_mode includes proxy")
    # Two-stage eval (proxy -> TTT)
    p.add_argument("--two_stage", action="store_true",
                   help="First score with proxy, then run TTT only on top-m per article")
    p.add_argument("--proxy_top_m", type=int, default=2,
                   help="Number of top proxy-scored completions to run TTT on (two-stage only)")
    # Adaptive eval_times
    p.add_argument("--adaptive_eval", action="store_true",
                   help="Adapt eval_times based on first-run adapter_mean")
    p.add_argument("--eval_times_min", type=int, default=1)
    p.add_argument("--eval_times_max", type=int, default=2)
    p.add_argument("--adaptive_low", type=float, default=0.4)
    p.add_argument("--adaptive_high", type=float, default=0.6)
    # PRAXIS-style archive updates
    p.add_argument("--archive_path", default="", help="Path to PRAXIS strategy archive JSON")
    p.add_argument("--archive_beta", type=float, default=2.0, help="Gibbs temperature for archive updates")
    p.add_argument("--archive_min_weight", type=float, default=0.02, help="Minimum weight floor after update")
    p.add_argument("--archive_update", action="store_true", help="Update archive weights after run")
    p.add_argument("--archive_update_mode", choices=["none", "end", "batch"], default="end")
    p.add_argument("--archive_batch_size", type=int, default=5, help="Batch size (in passages) for within-run updates")
    p.add_argument("--archive_log_path", default="", help="Optional JSONL log of archive updates")
    p.add_argument("--growth_gain_threshold", type=float, default=0.0, help="Mean gain threshold for stability streak")
    p.add_argument("--growth_streak_needed", type=int, default=3, help="Consecutive batches above gain threshold")
    p.add_argument("--prune_weight_threshold", type=float, default=0.01, help="Weight threshold for pruning")
    p.add_argument("--prune_streak_needed", type=int, default=5, help="Consecutive low-weight batches before pruning")
    p.add_argument("--min_uses_to_prune", type=int, default=10, help="Min uses before a strategy can be pruned")
    p.add_argument("--seed_grace_batches", type=int, default=3, help="Batches to protect seed strategies from pruning")
    p.add_argument("--enable_growth", action="store_true", help="Enable archive growth")
    p.add_argument("--enable_prune", action="store_true", help="Enable archive pruning")
    p.add_argument("--enable_embedding_novelty", action="store_true", help="Use embedding novelty")
    p.add_argument("--enable_structure_novelty", action="store_true", help="Use structure novelty")
    p.add_argument("--embedding_similarity_threshold", type=float, default=0.82, help="Cosine similarity threshold")
    # Validity gate (ablation-friendly; off by default)
    p.add_argument("--enable_validity_gate", action="store_true",
                   help="Filter obviously invalid/self-contaminated completions before selection")
    p.add_argument("--validity_min_chars", type=int, default=40,
                   help="Minimum non-whitespace characters required for a completion to be valid")
    p.add_argument("--validity_fallback_on_empty", action="store_true",
                   help="If all completions are invalid, evaluate one fallback completion so article metrics remain defined")
    # Blame-aware archive scoring (ablation-friendly; off by default)
    p.add_argument("--archive_use_blame", action="store_true",
                   help="Use reward-minus-blame scoring instead of raw mean gain for archive updates")
    p.add_argument("--archive_blame_lambda", type=float, default=1.0,
                   help="Global multiplier on blame penalties during archive updates")
    p.add_argument("--archive_invalid_penalty", type=float, default=0.10,
                   help="Additional blame penalty applied to invalid completions")
    p.add_argument("--seed", type=int, default=None, help="Shuffle dataset with this seed")
    return p.parse_args()


def send_round_trip(
    ctx: zmq.Context,
    endpoint: str,
    train_sequences: List[str],
    comp_raw: str,
    questions: List[Dict[str, str]],
    args: argparse.Namespace,
    reward_mode: str,
    max_retries: int = 2,
    timeout_ms: int = 600_000,  # 10 minutes
) -> Dict[str, Any]:
    """
    Send one request to the TTT server and wait (≤10 min) for its reply.
    On timeout we recreate the REQ socket and retry, up to `max_retries`.
    """
    # we recreate the socket on each retry, so wrap the whole thing in a loop
    for attempt in range(1, max_retries + 1):
        # ------ (re)create a REQ socket ------
        sock = ctx.socket(zmq.REQ)
        sock.connect(endpoint)
        sock.setsockopt(zmq.LINGER, 0)  # don't block on close

        # register this socket with a poller
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        # build & send the request -------------------------
        payload = {
            "train_sequences": train_sequences,
            "eval_questions": questions,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "end_mask_substring": args.end_mask_substring,
            "chain_of_thought": args.chain_of_thought,
            "reward_mode": reward_mode,
            "proxy_mode": args.proxy_mode,
            "comp_raw": comp_raw,
        }

        sock.send_json(payload)
        print(f"Sent request to TTT server (attempt {attempt}/{max_retries}, waiting ≤10 min)")

        # ------------------ wait for the reply --------------------
        events = poller.poll(timeout_ms)
        if events:  # reply arrived in time
            reply = sock.recv_json()
            sock.close()
            if "error" not in reply:
                return reply
            print(f"TTT server error: {reply['error']}")  # fall through → retry
        else:  # timed out
            print("No reply after 10 minutes - retrying…")

        # clean up before the next attempt
        poller.unregister(sock)
        sock.close()

    # if we reach here every attempt failed
    raise RuntimeError(f"TTT server unreachable after {max_retries} attempts")

def evaluate_completion(ctx, endpoint, item: Dict[str, Any], comp_raw: str, args,
                        reward_mode_override: str | None = None,
                        eval_times_override: int | None = None):
    """Run fine-tune / eval cycles for one completion."""
    title, context = item["title"], item["context"]
    questions = [
        {
            "title": title,
            "context": context,
            "question": f"Topic: {title}\n{q['question']}",
            "answer": q["answer"],
        }
        for q in item["questions"]
    ]
    train_sequences = build_train_sequences(comp_raw, context, title, split_newlines=args.split_newlines)

    base_accs, adpt_accs, gains = [], [], []
    q_details: List[Dict[str, Any]] = []
    proxy_payload = []
    proxy_finals = []

    reward_mode = reward_mode_override or args.reward_mode

    eval_times = eval_times_override or args.eval_times
    if args.adaptive_eval and reward_mode in ("ttt", "both"):
        eval_times = max(args.eval_times_min, 1)

    i = 0
    while i < eval_times:
        rep = send_round_trip(ctx, endpoint, train_sequences, comp_raw, questions, args, reward_mode)

        if reward_mode in ("proxy", "both"):
            # Capture per-sequence proxy results and stop; no TTT stats to aggregate here
            proxy_scores = rep.get("proxy_scores", [])
            proxy_payload.append(proxy_scores)
            proxy_finals.append(proxy_scores.get("final", 0))
            print(f"[proxy] {item['title']!r} {proxy_scores}")
        if reward_mode == "proxy":
            i += 1
            continue
        base_accs.append(rep["baseline_accuracy"])
        adpt_accs.append(rep["adapter_accuracy"])
        gains.append(rep["adapter_gain"])

        q_details_rep = []
        baseline_texts = rep.get("baseline_texts", []) or []
        adapter_texts = rep.get("adapter_texts", []) or []
        baseline_correct = rep.get("baseline_correct", []) or []
        adapter_correct = rep.get("adapter_correct", []) or []
        for qi, q in enumerate(item["questions"]):
            q_details_rep.append(
                {
                    "rep": i,
                    "question": q["question"],
                    "answer": q["answer"],
                    "baseline_answer": baseline_texts[qi] if qi < len(baseline_texts) else "",
                    "adapter_answer": adapter_texts[qi] if qi < len(adapter_texts) else "",
                    "baseline_correct": baseline_correct[qi] if qi < len(baseline_correct) else False,
                    "adapter_correct": adapter_correct[qi] if qi < len(adapter_correct) else False,
                }
            )
        q_details.extend(q_details_rep)

        if args.adaptive_eval and reward_mode in ("ttt", "both") and i == 0:
            first = adpt_accs[0]
            if args.adaptive_low < first < args.adaptive_high:
                eval_times = max(eval_times, args.eval_times_max)
            else:
                eval_times = max(eval_times, args.eval_times_min)

        i += 1

    if reward_mode == "proxy":
        stats_only = {
            "baseline_mean": 0.0,
            "baseline_std": 0.0,
            "adapter_mean": 0.0,
            "adapter_std": 0.0,
            "gain_mean": 0.0,
            "proxy_mean": (sum(proxy_finals)/len(proxy_finals)) if proxy_finals else 0.0,
        }
        return stats_only, [], proxy_payload  # q_details meaningless in proxy mode

    stats_only = {
        "baseline_mean": _stats.mean(base_accs),
        "baseline_std": _stats.stdev(base_accs) if len(base_accs) > 1 else 0.0,
        "adapter_mean": _stats.mean(adpt_accs),
        "adapter_std": _stats.stdev(adpt_accs) if len(adpt_accs) > 1 else 0.0,
        "gain_mean": _stats.mean(gains),
        "proxy_mean": (sum(proxy_finals)/len(proxy_finals)) if proxy_finals else 0.0
    }

    return stats_only, q_details, proxy_payload

def send_shutdown(sock):
    sock.send_json({"cmd": "shutdown"})
    sock.recv_json()  # consume the "bye"

def _load_archive(path: str) -> Dict[str, Any]:
    # Accept both plain UTF-8 and UTF-8-with-BOM archives. PowerShell's
    # Set-Content can emit a BOM depending on version/configuration.
    archive = normalize_archive(json.load(open(path, encoding="utf-8-sig")))
    if "strategies" not in archive or not isinstance(archive["strategies"], list):
        raise ValueError("Archive is missing a 'strategies' list")
    return archive

def _write_archive(path: str, archive: Dict[str, Any]) -> None:
    normalized = normalize_archive(archive)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

def _archive_snapshot_path(snapshot_dir: pathlib.Path, ts: str, stage: str) -> pathlib.Path:
    return snapshot_dir / f"archive_{stage}_{ts}.json"

def _snapshot_archive(archive_path: str, snapshot_dir: pathlib.Path, ts: str, stage: str) -> Optional[str]:
    if not archive_path:
        return None
    src = pathlib.Path(archive_path)
    if not src.exists():
        return None
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    dst = _archive_snapshot_path(snapshot_dir, ts, stage)
    archive = _load_archive(str(src))
    _write_archive(str(dst), archive)
    return str(dst)

_INVALID_PATTERNS = [
    "you are an ai assistant",
    "user will you give you a",
    "while answering think step-by-step",
    "while answering think step by step",
    "think step-by-step and justify",
    "think step by step and justify",
    "helps people find information",
    "<|im_start|>",
]


def _validate_completion_text(comp_raw: str, *, min_chars: int) -> Dict[str, Any]:
    text = (comp_raw or "").strip()
    reasons: List[str] = []
    if len("".join(text.split())) < min_chars:
        reasons.append("too_short")
    lowered = text.lower()
    hit_patterns = [pat for pat in _INVALID_PATTERNS if pat in lowered]
    if hit_patterns:
        reasons.append("meta_leakage")
    valid = len(reasons) == 0
    return {
        "valid": valid,
        "reasons": reasons,
        "matched_patterns": hit_patterns,
        "char_len": len(text),
    }


def _placeholder_completion_entry(comp_raw: str, validity: Dict[str, Any], *, fallback_evaluated: bool = False) -> Dict[str, Any]:
    validity_payload = dict(validity)
    if fallback_evaluated:
        validity_payload["fallback_evaluated"] = True
    return {
        "text": comp_raw,
        "stats": {
            "baseline_mean": 0.0,
            "baseline_std": 0.0,
            "adapter_mean": 0.0,
            "adapter_std": 0.0,
            "gain_mean": 0.0,
            "proxy_mean": 0.0,
        },
        "questions": [],
        "proxy_scores": [],
        "validity": validity_payload,
    }


def _article_archive_score(
    art: Dict[str, Any],
    *,
    use_blame: bool,
    blame_lambda: float,
    invalid_penalty: float,
) -> float:
    if not use_blame:
        stats = art.get("stats", {}) or {}
        gain = stats.get("mean_gain")
        if gain is None:
            gain = stats.get("gain_mean", 0.0)
        return float(gain)

    comps = art.get("completions", []) or []
    if not comps:
        return 0.0

    scores = []
    for comp in comps:
        cstats = comp.get("stats", {}) or {}
        gain = float(cstats.get("gain_mean", 0.0))
        positive = max(gain, 0.0)
        blame = max(-gain, 0.0)
        validity = comp.get("validity", {}) or {}
        if validity and not validity.get("valid", True):
            blame += invalid_penalty
        scores.append(positive - blame_lambda * blame)
    return sum(scores) / len(scores)

def _update_archive(
    archive_path: str,
    articles: List[Dict[str, Any]],
    beta: float,
    min_weight: float,
    log_path: str,
    *,
    update_mode: str,
    batch_index: int,
    batch_mean_gain: float,
    growth_gain_threshold: float,
    growth_streak_needed: int,
    prune_weight_threshold: float,
    prune_streak_needed: int,
    min_uses_to_prune: int,
    seed_grace_batches: int,
    enable_growth: bool,
    enable_prune: bool,
    enable_embedding_novelty: bool,
    enable_structure_novelty: bool,
    embedding_similarity_threshold: float,
    use_blame: bool,
    blame_lambda: float,
    invalid_penalty: float,
) -> None:
    """
    Update archive weights using a Gibbs rule on per-strategy mean gain.
    Uses article-level stats['mean_gain'] as reward signal.
    """
    archive = _load_archive(archive_path)
    strategies = archive.get("strategies", [])
    config = archive.get("config", {})
    state = archive.get("state", {})
    seed_set = set(archive.get("seed_strategies", []))
    candidate_pool = list(archive.get("candidate_strategies", []))

    # Allow archive config to act as defaults unless CLI explicitly enables features.
    if not enable_growth:
        enable_growth = bool(config.get("enable_growth", False))
    if not enable_prune:
        enable_prune = bool(config.get("enable_prune", False))
    if not enable_embedding_novelty:
        enable_embedding_novelty = bool(config.get("enable_embedding_novelty", False))
    if not enable_structure_novelty:
        enable_structure_novelty = bool(config.get("enable_structure_novelty", False))

    strat_rewards: Dict[str, List[float]] = {}
    for art in articles:
        strat = canonical_strategy_name(art.get("strategy") or art.get("prompt_key") or "unknown")
        reward_value = _article_archive_score(
            art,
            use_blame=use_blame,
            blame_lambda=blame_lambda,
            invalid_penalty=invalid_penalty,
        )
        strat_rewards.setdefault(strat, []).append(float(reward_value))

    # update running stats
    for s in strategies:
        name = canonical_strategy_name(s.get("name", s.get("prompt_key")))
        rewards = strat_rewards.get(name, [])
        if not rewards:
            continue
        mean_gain = sum(rewards) / len(rewards)
        stats = s.setdefault("stats", {})
        uses = int(stats.get("uses", 0)) + len(rewards)
        prev_mean = float(stats.get("mean_reward", 0.0))
        new_mean = prev_mean + (mean_gain - prev_mean) * (len(rewards) / max(1, uses))
        stats["uses"] = uses
        stats["mean_reward"] = new_mean

    # Gibbs weight update
    weights = []
    for s in strategies:
        mean_r = float((s.get("stats", {}) or {}).get("mean_reward", 0.0))
        weights.append(max(0.0, pow(2.718281828, beta * mean_r)))
    total = sum(weights)
    if total <= 0:
        total = 1.0
        weights = [1.0 for _ in weights]

    for s, w in zip(strategies, weights):
        new_w = w / total
        if min_weight > 0.0:
            new_w = max(min_weight, new_w)
        s["weight"] = new_w

    if min_weight > 0.0:
        total = sum(float(s.get("weight", 0.0)) for s in strategies)
        if total > 0:
            for s in strategies:
                s["weight"] = float(s.get("weight", 0.0)) / total

    # stability tracking for growth
    stable_streak = int(state.get("stable_gain_streak", 0))
    if batch_mean_gain > growth_gain_threshold:
        stable_streak += 1
    else:
        stable_streak = 0
    state["stable_gain_streak"] = stable_streak
    growth_enabled = bool(state.get("growth_enabled", False))
    if enable_growth and (stable_streak >= growth_streak_needed):
        growth_enabled = True
    state["growth_enabled"] = growth_enabled

    # novelty helpers
    def _structure_signature(text: str) -> Dict[str, Any]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        num_lines = len(lines)
        has_q = any("?" in ln for ln in lines)
        has_list = any(ln[:2].isdigit() or ln.startswith("-") for ln in lines)
        avg_len = sum(len(ln) for ln in lines) / max(1, num_lines)
        return {"num_lines": num_lines, "has_q": has_q, "has_list": has_list, "avg_len": avg_len}

    embedder = _get_embedder() if enable_embedding_novelty else None

    def _cos_sim(a, b) -> float:
        if np is None:
            return 0.0
        a = np.asarray(a); b = np.asarray(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _is_novel(text: str) -> bool:
        novel_structure = True
        if enable_structure_novelty:
            sig = _structure_signature(text)
            for s in strategies:
                examples = s.get("examples", []) or []
                if not examples:
                    continue
                ex_sig = _structure_signature(examples[-1])
                if (sig["has_q"] == ex_sig["has_q"]) and (sig["has_list"] == ex_sig["has_list"]) and (abs(sig["avg_len"] - ex_sig["avg_len"]) < 20):
                    novel_structure = False
                    break
        novel_embed = True
        if enable_embedding_novelty and embedder is not None and np is not None:
            emb = embedder.encode([text])[0]
            for s in strategies:
                exs = s.get("examples", []) or []
                if not exs:
                    continue
                ex_emb = embedder.encode([exs[-1]])[0]
                if _cos_sim(emb, ex_emb) >= embedding_similarity_threshold:
                    novel_embed = False
                    break
        if enable_embedding_novelty and (embedder is None or np is None):
            novel_embed = False
        return (not enable_structure_novelty or novel_structure) and (not enable_embedding_novelty or novel_embed)

    def _structure_to_candidate(text: str) -> Optional[str]:
        sig = _structure_signature(text)
        if sig["has_q"]:
            return None
        if sig["has_list"] and sig["num_lines"] >= 8:
            return "implications-very-long"
        if sig["has_list"]:
            return "extractive-facts"
        return "rewrite"

    # growth: add new strategy from candidate pool
    if growth_enabled and candidate_pool:
        # use best completion per article as candidate
        for art in articles:
            comps = art.get("completions", [])
            if not comps:
                continue
            # choose completion with best gain
            best = max(comps, key=lambda c: (c.get("stats", {}) or {}).get("gain_mean", 0.0))
            best_text = best.get("text", "")
            best_gain = (best.get("stats", {}) or {}).get("gain_mean", 0.0)
            if best_gain <= growth_gain_threshold:
                continue
            if not _is_novel(best_text):
                continue
            candidate = canonical_prompt_key(_structure_to_candidate(best_text))
            if candidate and candidate in candidate_pool:
                strategies.append(
                    {
                        "name": canonical_strategy_name(candidate),
                        "prompt_key": candidate,
                        "weight": min_weight,
                        "stats": {"uses": 0, "mean_reward": 0.0},
                        "examples": [best_text],
                    }
                )
                candidate_pool.remove(candidate)
                break

    # pruning: track low-weight streaks and remove
    low_streaks = state.get("low_weight_streaks", {})
    new_strategies = []
    for s in strategies:
        name = canonical_strategy_name(s.get("name", s.get("prompt_key")))
        if not enable_prune:
            new_strategies.append(s); continue
        if name in seed_set and batch_index < seed_grace_batches:
            new_strategies.append(s); continue
        uses = int((s.get("stats", {}) or {}).get("uses", 0))
        if uses < min_uses_to_prune:
            new_strategies.append(s); continue
        weight = float(s.get("weight", 0.0))
        if weight < prune_weight_threshold:
            low_streaks[name] = low_streaks.get(name, 0) + 1
        else:
            low_streaks[name] = 0
        if low_streaks.get(name, 0) >= prune_streak_needed:
            continue
        new_strategies.append(s)

    strategies = new_strategies
    state["low_weight_streaks"] = low_streaks
    state["batch_index"] = batch_index

    archive["strategies"] = strategies
    archive["candidate_strategies"] = candidate_pool
    archive["state"] = state
    archive["last_update"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    _write_archive(archive_path, archive)

    if log_path:
        log_entry = {
            "timestamp": archive["last_update"],
            "archive_path": archive_path,
            "beta": beta,
            "min_weight": min_weight,
            "strategy_rewards": {k: sum(v) / len(v) for k, v in strat_rewards.items()},
            "use_blame": use_blame,
            "blame_lambda": blame_lambda,
            "invalid_penalty": invalid_penalty,
            "weights": {canonical_strategy_name(s.get("name", s.get("prompt_key"))): s.get("weight", 0.0) for s in strategies},
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ------------------------------- MAIN ------------------------------- #
def main() -> None:
    args = parse_args()
    run_start = time.time()

    data_path = pathlib.Path(args.dataset)
    try:
        dataset: List[Dict[str, Any]] = json.load(data_path.open(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"[!] Dataset not found: {data_path}")

    if args.seed is not None:
        random.Random(args.seed).shuffle(dataset)
    dataset = dataset[args.start_article :]
    dataset = dataset[: args.n_articles] if args.n_articles else dataset

    out_dir = pathlib.Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%m%d_%H%M%S")
    archive_snapshot_dir = out_dir / "archive_snapshots"
    archive_snapshot_before = _snapshot_archive(args.archive_path, archive_snapshot_dir, ts, "before")

    ctx = zmq.Context()
    endpoint = f"tcp://{args.server_host}:{args.zmq_port}"

    articles_out: List[Dict[str, Any]] = []
    overall_base_means, overall_adpt_means, adapter_std_list, completion_run_std_list = [], [], [], []
    overall_proxy_means = []
    overall_valid_completion_count = 0
    overall_invalid_completion_count = 0

    # ---------------- iterate over articles -------------------------- #
    batch_articles: List[Dict[str, Any]] = []
    batch_index = 0
    for art_idx, item in enumerate(dataset):
        completions = (
            item["completions"] if isinstance(item.get("completions"), list)
            else [item.get("completion", "")]
        )
        completions = [c for c in completions if c.strip()][: args.k_completions]
        if not completions:
            completions = [""]
        comp_entries: List[Dict[str, Any]] = []
        completion_candidates: List[Tuple[int, str, Dict[str, Any]]] = []
        for comp_idx, comp_raw in enumerate(completions):
            validity = (
                _validate_completion_text(comp_raw, min_chars=args.validity_min_chars)
                if args.enable_validity_gate else
                {"valid": True, "reasons": [], "matched_patterns": [], "char_len": len(comp_raw.strip())}
            )
            completion_candidates.append((comp_idx, comp_raw, validity))

        valid_candidates = [entry for entry in completion_candidates if entry[2].get("valid", True)]
        valid_completion_count = len(valid_candidates)
        invalid_completion_count = len(completion_candidates) - valid_completion_count

        if args.two_stage:
            proxy_scored = []
            for comp_idx, comp_raw, validity in valid_candidates:
                p_stats, _, p_details = evaluate_completion(
                    ctx, endpoint, item, comp_raw, args,
                    reward_mode_override="proxy",
                    eval_times_override=1,
                )
                proxy_scored.append((comp_idx, comp_raw, validity, p_stats, p_details))

            fallback_entry: Optional[Tuple[int, str, Dict[str, Any]]] = None
            if not proxy_scored and args.validity_fallback_on_empty and completion_candidates:
                fallback_entry = completion_candidates[0]
                comp_idx, comp_raw, validity = fallback_entry
                p_stats, _, p_details = evaluate_completion(
                    ctx, endpoint, item, comp_raw, args,
                    reward_mode_override="proxy",
                    eval_times_override=1,
                )
                proxy_scored.append((comp_idx, comp_raw, validity, p_stats, p_details))

            proxy_scored.sort(key=lambda x: x[3].get("proxy_mean", 0), reverse=True)
            top_m = proxy_scored[: max(1, min(args.proxy_top_m, len(proxy_scored)))] if proxy_scored else []
            evaluated_completion_keys = set()
            for comp_idx, comp_raw, validity, _, p_details in top_m:
                stats, q_details, _ = evaluate_completion(ctx, endpoint, item, comp_raw, args)
                completion_run_std_list.append(stats["adapter_std"])
                validity_payload = dict(validity)
                if fallback_entry is not None and comp_idx == fallback_entry[0] and comp_raw == fallback_entry[1]:
                    validity_payload["fallback_evaluated"] = True
                comp_entries.append(
                    {
                        "text": comp_raw,
                        "stats": stats,
                        "questions": q_details,
                        "proxy_scores": p_details,
                        "validity": validity_payload,
                    }
                )
                evaluated_completion_keys.add((comp_idx, comp_raw))

            for comp_idx, comp_raw, validity in completion_candidates:
                if (comp_idx, comp_raw) not in evaluated_completion_keys and not validity.get("valid", True):
                    comp_entries.append(
                        _placeholder_completion_entry(
                            comp_raw,
                            validity,
                            fallback_evaluated=(
                                fallback_entry is not None and
                                comp_idx == fallback_entry[0] and
                                comp_raw == fallback_entry[1]
                            ),
                        )
                    )
        else:
            evaluated_any = False
            fallback_entry = None
            for comp_idx, comp_raw, validity in valid_candidates:
                stats, q_details, proxy_details = evaluate_completion(ctx, endpoint, item, comp_raw, args)
                completion_run_std_list.append(stats["adapter_std"])
                comp_entries.append(
                    {
                        "text": comp_raw,
                        "stats": stats,
                        "questions": q_details,
                        "proxy_scores": proxy_details,
                        "validity": dict(validity),
                    }
                )
                evaluated_any = True

                print(f"[{art_idx:02d}.{comp_idx:02d}] "
                      f"base {stats['baseline_mean']*100:.2f}% | "
                      f"adapter {stats['adapter_mean']*100:.2f}% +/- {stats['adapter_std']*100:.2f}% "
                      f"gain {stats['gain_mean']*100:+.2f}%")

            if not evaluated_any and args.validity_fallback_on_empty and completion_candidates:
                fallback_entry = completion_candidates[0]
                comp_idx, comp_raw, validity = fallback_entry
                stats, q_details, proxy_details = evaluate_completion(ctx, endpoint, item, comp_raw, args)
                completion_run_std_list.append(stats["adapter_std"])
                comp_entries.append(
                    {
                        "text": comp_raw,
                        "stats": stats,
                        "questions": q_details,
                        "proxy_scores": proxy_details,
                        "validity": {**dict(validity), "fallback_evaluated": True},
                    }
                )

            for comp_idx, comp_raw, validity in completion_candidates:
                already_added = any(c["text"] == comp_raw for c in comp_entries)
                if not already_added and not validity.get("valid", True):
                    comp_entries.append(
                        _placeholder_completion_entry(
                            comp_raw,
                            validity,
                            fallback_evaluated=(
                                fallback_entry is not None and
                                comp_idx == fallback_entry[0] and
                                comp_raw == fallback_entry[1]
                            ),
                        )
                    )

        if not comp_entries and completion_candidates:
            comp_idx, comp_raw, validity = completion_candidates[0]
            comp_entries.append(_placeholder_completion_entry(comp_raw, validity))

        # article-level aggregates
        evaluated_entries = [c for c in comp_entries if c["stats"].get("evaluated", False)]
        aggregate_entries = evaluated_entries if evaluated_entries else comp_entries
        base_mean_article = _stats.mean(c["stats"]["baseline_mean"] for c in aggregate_entries)
        adpt_mean_article = _stats.mean(c["stats"]["adapter_mean"] for c in aggregate_entries)
        adpt_std_article = _stats.stdev([c["stats"]["adapter_mean"] for c in aggregate_entries]) \
                           if len(aggregate_entries) > 1 else 0.0
        mean_run_std_article = _stats.mean(c["stats"]["adapter_std"] for c in aggregate_entries)
        gain_mean_article = _stats.mean(c["stats"]["gain_mean"] for c in aggregate_entries)
        proxy_mean_article = _stats.mean(c["stats"]["proxy_mean"] for c in aggregate_entries)

        overall_valid_completion_count += valid_completion_count
        overall_invalid_completion_count += invalid_completion_count

        overall_base_means.append(base_mean_article)
        overall_adpt_means.append(adpt_mean_article)
        overall_proxy_means.append(proxy_mean_article)
        adapter_std_list.append(adpt_std_article)

        if args.reward_mode in ("proxy", "both"):
            print(f"Article {art_idx:02d}  mean proxy {proxy_mean_article:.2f}/20")
        if args.reward_mode in ("ttt", "both"):
            print(f"Article {art_idx:02d}  base {base_mean_article*100:.2f}% | "
                  f"adapter {adpt_mean_article*100:.2f}% ± {adpt_std_article*100:.2f}% "
                  f"gain {gain_mean_article*100:+.2f}%")

        cur_base = _stats.mean(overall_base_means)
        cur_adapt = _stats.mean(overall_adpt_means)
        cur_gain = cur_adapt - cur_base

        print(f"[progress] overall ({len(overall_base_means)} articles)  "
            f"baseline {cur_base*100:.2f}% | "
            f"adapter {cur_adapt*100:.2f}% | "
            f"gain {cur_gain*100:+.2f}%")

        articles_out.append(
            {
                "stats": {
                    "baseline_accuracy": round(base_mean_article, 4),
                    "adapter_mean_accuracy": round(adpt_mean_article, 4),
                    "adapter_std_over_completions": round(adpt_std_article, 4),
                    "mean_adapter_std_within_completions": round(mean_run_std_article, 4),
                    "mean_gain": round(gain_mean_article, 4),
                    "proxy_mean": round(proxy_mean_article, 4),
                    "valid_completion_count": valid_completion_count,
                    "invalid_completion_count": invalid_completion_count,
                },
                "title": item["title"],
                "context": item["context"],
                "completions": comp_entries,
                "prompt": item.get("prompt", ""),
                "strategy": canonical_strategy_name(item.get("strategy", item.get("prompt_key", ""))),
                "prompt_key": canonical_prompt_key(item.get("prompt_key", "")),
            }
        )

        # batch-level archive update
        if args.archive_update and args.archive_update_mode == "batch" and args.archive_path:
            batch_articles.append(articles_out[-1])
            if len(batch_articles) >= args.archive_batch_size:
                batch_mean_gain = _stats.mean([a["stats"]["mean_gain"] for a in batch_articles])
                _update_archive(
                    args.archive_path,
                    batch_articles,
                    beta=args.archive_beta,
                    min_weight=args.archive_min_weight,
                    log_path=args.archive_log_path,
                    update_mode="batch",
                    batch_index=batch_index,
                    batch_mean_gain=batch_mean_gain,
                    growth_gain_threshold=args.growth_gain_threshold,
                    growth_streak_needed=args.growth_streak_needed,
                    prune_weight_threshold=args.prune_weight_threshold,
                    prune_streak_needed=args.prune_streak_needed,
                    min_uses_to_prune=args.min_uses_to_prune,
                    seed_grace_batches=args.seed_grace_batches,
                    enable_growth=args.enable_growth,
                    enable_prune=args.enable_prune,
                    enable_embedding_novelty=args.enable_embedding_novelty,
                    enable_structure_novelty=args.enable_structure_novelty,
                    embedding_similarity_threshold=args.embedding_similarity_threshold,
                    use_blame=args.archive_use_blame,
                    blame_lambda=args.archive_blame_lambda,
                    invalid_penalty=args.archive_invalid_penalty,
                )
                batch_articles = []
                batch_index += 1

    # flush any remaining batch updates
    if args.archive_update and args.archive_update_mode == "batch" and args.archive_path and batch_articles:
        batch_mean_gain = _stats.mean([a["stats"]["mean_gain"] for a in batch_articles])
        _update_archive(
            args.archive_path,
            batch_articles,
            beta=args.archive_beta,
            min_weight=args.archive_min_weight,
            log_path=args.archive_log_path,
            update_mode="batch",
            batch_index=batch_index,
            batch_mean_gain=batch_mean_gain,
            growth_gain_threshold=args.growth_gain_threshold,
            growth_streak_needed=args.growth_streak_needed,
            prune_weight_threshold=args.prune_weight_threshold,
            prune_streak_needed=args.prune_streak_needed,
            min_uses_to_prune=args.min_uses_to_prune,
            seed_grace_batches=args.seed_grace_batches,
            enable_growth=args.enable_growth,
            enable_prune=args.enable_prune,
            enable_embedding_novelty=args.enable_embedding_novelty,
            enable_structure_novelty=args.enable_structure_novelty,
            embedding_similarity_threshold=args.embedding_similarity_threshold,
            use_blame=args.archive_use_blame,
            blame_lambda=args.archive_blame_lambda,
            invalid_penalty=args.archive_invalid_penalty,
        )

    # ------------- overall summary & JSON write ---------------------- #
    if args.archive_update and args.archive_update_mode == "end" and args.archive_path:
        batch_mean_gain = _stats.mean([a["stats"]["mean_gain"] for a in articles_out]) if articles_out else 0.0
        _update_archive(
            args.archive_path,
            articles_out,
            beta=args.archive_beta,
            min_weight=args.archive_min_weight,
            log_path=args.archive_log_path,
            update_mode="end",
            batch_index=batch_index,
            batch_mean_gain=batch_mean_gain,
            growth_gain_threshold=args.growth_gain_threshold,
            growth_streak_needed=args.growth_streak_needed,
            prune_weight_threshold=args.prune_weight_threshold,
            prune_streak_needed=args.prune_streak_needed,
            min_uses_to_prune=args.min_uses_to_prune,
            seed_grace_batches=args.seed_grace_batches,
            enable_growth=args.enable_growth,
            enable_prune=args.enable_prune,
            enable_embedding_novelty=args.enable_embedding_novelty,
            enable_structure_novelty=args.enable_structure_novelty,
            embedding_similarity_threshold=args.embedding_similarity_threshold,
            use_blame=args.archive_use_blame,
            blame_lambda=args.archive_blame_lambda,
            invalid_penalty=args.archive_invalid_penalty,
        )

    archive_snapshot_after = _snapshot_archive(args.archive_path, archive_snapshot_dir, ts, "after")

    overall_base_mean = _stats.mean(overall_base_means) if overall_base_means else 0.0
    overall_base_std = _stats.stdev(overall_base_means) if len(overall_base_means) > 1 else 0.0
    overall_adpt_mean = _stats.mean(overall_adpt_means) if overall_adpt_means else 0.0
    overall_adpt_std = _stats.stdev(overall_adpt_means) if len(overall_adpt_means) > 1 else 0.0
    mean_adapter_std_accuracy = _stats.mean(adapter_std_list) if adapter_std_list else 0.0
    mean_adapter_std_within_completion = _stats.mean(completion_run_std_list) if completion_run_std_list else 0.0
    runtime_seconds = time.time() - run_start
    length_metrics = _completion_length_metrics(articles_out)
    article_metrics = _article_outcome_metrics(articles_out)
    strategy_metrics = _strategy_metrics(articles_out)

    out_path = out_dir / f"run_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "overall": {
                "baseline_mean_accuracy": round(overall_base_mean, 4),
                "baseline_std_of_article_means": round(overall_base_std, 4),
                "adapter_mean_accuracy": round(overall_adpt_mean, 4),
                "adapter_std_of_article_means": round(overall_adpt_std, 4),
                "mean_adapter_std_over_completions": round(mean_adapter_std_accuracy, 4),
                "mean_adapter_std_within_completions": round(mean_adapter_std_within_completion, 4),
                "mean_gain": round(overall_adpt_mean - overall_base_mean, 4),
                "mean_proxy_score": round(_stats.mean(overall_proxy_means), 4) if overall_proxy_means else 0.0,
                "valid_completion_count": overall_valid_completion_count,
                "invalid_completion_count": overall_invalid_completion_count,
                "valid_completion_rate": round(
                    overall_valid_completion_count / max(1, overall_valid_completion_count + overall_invalid_completion_count),
                    4,
                ),
                "wall_clock_total_seconds": round(runtime_seconds, 4),
                "wall_clock_per_article_seconds": round(runtime_seconds / max(1, len(articles_out)), 4),
                "wall_clock_per_completion_seconds": round(runtime_seconds / max(1, len(articles_out) * max(1, args.k_completions)), 4),
                "gain_per_minute": round((overall_adpt_mean - overall_base_mean) / max(runtime_seconds / 60.0, 1e-8), 6),
                **article_metrics,
                **length_metrics,
            },
            "timestamp": ts,
            "exp_name": args.exp_name,
            "dataset": str(data_path),
            "reward_mode": args.reward_mode,
            "proxy_mode": args.proxy_mode,
            "split_newlines": args.split_newlines,
            "n_articles": len(articles_out),
            "start_article": args.start_article,
            "k_completions": args.k_completions,
            "eval_times": args.eval_times,
            "two_stage": args.two_stage,
            "proxy_top_m": args.proxy_top_m,
            "adaptive_eval": args.adaptive_eval,
            "eval_times_min": args.eval_times_min,
            "eval_times_max": args.eval_times_max,
            "adaptive_low": args.adaptive_low,
            "adaptive_high": args.adaptive_high,
            "lora_params": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "epochs": args.finetune_epochs,
                "lr": args.finetune_lr,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.gradient_accumulation_steps,
            },
            "archive": {
                "path": args.archive_path or None,
                "snapshot_before": archive_snapshot_before,
                "snapshot_after": archive_snapshot_after,
                "update_enabled": args.archive_update,
                "update_mode": args.archive_update_mode,
                "use_blame": args.archive_use_blame,
                "blame_lambda": args.archive_blame_lambda,
                "invalid_penalty": args.archive_invalid_penalty,
            },
            "validity_gate": {
                "enabled": args.enable_validity_gate,
                "min_chars": args.validity_min_chars,
                "fallback_on_empty": args.validity_fallback_on_empty,
            },
            "strategy_metrics": strategy_metrics,
            "articles": articles_out,
        },
        out_path.open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"\nWrote summary → {out_path}")

    # Optional: tell the server to shut down when finished
    # send_shutdown(ctx, endpoint)
    ctx.term()


if __name__ == "__main__":
    main()
