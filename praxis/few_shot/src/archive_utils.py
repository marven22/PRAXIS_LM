from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _canonical(name: str | None) -> str:
    if not name:
        return ""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_archive(archive: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(archive)
    strategies = []
    for strategy in out.get("strategies", []):
        name = _canonical(strategy.get("name"))
        if not name:
            continue
        stats = strategy.get("stats", {}) or {}
        strategies.append(
            {
                "name": name,
                "weight": float(strategy.get("weight", 0.0)),
                "stats": {
                    "uses": int(stats.get("uses", 0)),
                    "mean_reward": float(stats.get("mean_reward", 0.0)),
                },
                "examples": list(strategy.get("examples", [])),
            }
        )
    out["strategies"] = strategies
    return out


def load_archive(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return normalize_archive(json.load(handle))


def save_archive(path: str | Path, archive: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(archive, handle, indent=2, ensure_ascii=False)


def _normalized_weights(strategies: List[Dict[str, Any]], min_weight: float = 1e-6) -> List[float]:
    raw = [max(float(s.get("weight", 0.0)), min_weight) for s in strategies]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(strategies)] * len(strategies)
    return [w / total for w in raw]


def sample_strategy(archive: Dict[str, Any], rng: random.Random) -> Tuple[Dict[str, Any], List[float]]:
    strategies = archive.get("strategies", [])
    if not strategies:
        raise ValueError("Archive has no strategies to sample from.")
    weights = _normalized_weights(strategies)
    idx = rng.choices(range(len(strategies)), weights=weights, k=1)[0]
    return deepcopy(strategies[idx]), weights


def update_archive(archive: Dict[str, Any], strategy_name: str, reward: float, beta: float = 0.3) -> Dict[str, Any]:
    strategy_name = _canonical(strategy_name)
    updated = deepcopy(archive)
    strategies = updated.get("strategies", [])
    target = None
    for strategy in strategies:
        if _canonical(strategy.get("name")) == strategy_name:
            target = strategy
            break
    if target is None:
        raise KeyError(f"Unknown strategy: {strategy_name}")

    uses = int(target["stats"].get("uses", 0)) + 1
    prev_mean = float(target["stats"].get("mean_reward", 0.0))
    new_mean = prev_mean + (float(reward) - prev_mean) / uses
    target["stats"]["uses"] = uses
    target["stats"]["mean_reward"] = new_mean

    for strategy in strategies:
        base = max(float(strategy["weight"]), 1e-6)
        bonus = beta * float(strategy["stats"].get("mean_reward", 0.0))
        strategy["weight"] = max(base + bonus, 1e-6)

    total = sum(float(s["weight"]) for s in strategies)
    for strategy in strategies:
        strategy["weight"] = float(strategy["weight"]) / total

    return updated
