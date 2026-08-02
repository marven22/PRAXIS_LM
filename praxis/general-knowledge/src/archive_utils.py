from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

_PROMPT_KEY_ALIASES = {
    "implications": "implications",
    "rewrite": "rewrite",
    "none": "none",
    "extractive-facts": "extractive-facts",
    "extractive_facts": "extractive-facts",
    "grounded-exposition": "grounded-exposition",
    "grounded_exposition": "grounded-exposition",
    "implications-long": "implications-long",
    "implications_long": "implications-long",
    "implications-very-long": "implications-very-long",
    "implications_very_long": "implications-very-long",
    "implications-chain-of-thought": "implications-chain-of-thought",
    "implications_chain_of_thought": "implications-chain-of-thought",
    "supporting-fact-extraction": "supporting-fact-extraction",
    "supporting_fact_extraction": "supporting-fact-extraction",
    "bridge-entity-chain": "bridge-entity-chain",
    "bridge_entity_chain": "bridge-entity-chain",
    "comparison-table": "comparison-table",
    "comparison_table": "comparison-table",
    "question-aware-evidence-selection": "question-aware-evidence-selection",
    "question_aware_evidence_selection": "question-aware-evidence-selection",
    "multi-hop-explanation": "multi-hop-explanation",
    "multi_hop_explanation": "multi-hop-explanation",
    "answer-oriented-summary": "answer-oriented-summary",
    "answer_oriented_summary": "answer-oriented-summary",
    "bridge-step-reasoning": "bridge-step-reasoning",
    "bridge_step_reasoning": "bridge-step-reasoning",
    "answer-bearing-fact": "answer-bearing-fact",
    "answer_bearing_fact": "answer-bearing-fact",
    "multi-hop-qa-rehearsal": "multi-hop-qa-rehearsal",
    "multi_hop_qa_rehearsal": "multi-hop-qa-rehearsal",
    "question-decomposition": "question-decomposition",
    "question_decomposition": "question-decomposition",
}


def canonical_prompt_key(value: str | None) -> str:
    if not value:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    lowered = raw.lower().replace(" ", "_")
    return _PROMPT_KEY_ALIASES.get(lowered, lowered.replace("_", "-"))


def canonical_strategy_name(value: str | None) -> str:
    prompt_key = canonical_prompt_key(value)
    if not prompt_key:
        return ""
    return prompt_key.replace("-", "_")


def normalize_strategy_record(strategy: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(strategy)
    prompt_key = canonical_prompt_key(normalized.get("prompt_key") or normalized.get("name"))
    if prompt_key:
        normalized["prompt_key"] = prompt_key
        normalized["name"] = canonical_strategy_name(normalized.get("name") or prompt_key)
    return normalized


def normalize_archive(archive: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(archive)
    normalized["strategies"] = [
        normalize_strategy_record(s) for s in normalized.get("strategies", [])
    ]
    normalized["seed_strategies"] = [
        canonical_strategy_name(s) for s in normalized.get("seed_strategies", []) if canonical_strategy_name(s)
    ]
    normalized["candidate_strategies"] = [
        canonical_prompt_key(s) for s in normalized.get("candidate_strategies", []) if canonical_prompt_key(s)
    ]

    state = normalized.get("state")
    if isinstance(state, dict) and isinstance(state.get("low_weight_streaks"), dict):
        state["low_weight_streaks"] = {
            canonical_strategy_name(k): v
            for k, v in state["low_weight_streaks"].items()
            if canonical_strategy_name(k)
        }

    return normalized
