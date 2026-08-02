BASE_JSON_SCHEMA = """
Respond with a valid JSON object only. Do not include markdown or explanation.
Use this schema:
{
  "data_generation": {
    "use_basic_augmentations": true_or_false,
    "use_size_augmentations": true_or_false,
    "use_chain_augmentations": true_or_false,
    "use_repeat_augmentations": true_or_false
  },
  "training": {
    "strategy": "train_using_all_tokens" or "train_using_output_tokens",
    "learning_rate": float,
    "num_train_epochs": integer
  }
}

Conservative configuration rules:
- Prefer enabling only the augmentation families that are clearly justified by the demonstrations.
- Do not turn on every augmentation family unless the task strongly suggests that many invariances are present.
- Use small, stable learning rates. Prefer values between 0.00005 and 0.0005.
- Prefer short adaptation. Usually choose 1 to 3 epochs.
- If you are uncertain, choose a simpler configuration rather than a more aggressive one.
""".strip()

SYSTEM_MESSAGE = "You are a careful research assistant configuring a temporary adaptation pipeline for an ARC task."

STRATEGY_PROMPTS = {
    "rule_abstraction": (
        "Study the examples and infer the most likely abstract rule. "
        "Choose a conservative training configuration that focuses on learning the rule without overfitting. "
        "Prefer simpler augmentation choices unless the examples clearly suggest symmetry or geometric variation. "
        "Bias toward low learning rates and short training."
    ),
    "augmentation_plan": (
        "Study the examples and decide which augmentation families are most likely to preserve the task rule. "
        "Be proactive about enabling useful augmentations when pattern-preserving transformations seem plausible, "
        "but avoid turning on all augmentation families by default. "
        "Only enable a family if you can justify it from the demonstrations."
    ),
    "decomposition": (
        "Study the examples as if the solution may require multiple sub-steps. "
        "Choose a training configuration that gives the model enough room to learn intermediate structure, "
        "but avoid extreme settings. "
        "Prefer compact training schedules over aggressive ones."
    ),
    "invariance_hypothesis": (
        "Study the examples and focus on invariances: reflections, rotations, transpositions, repetitions, or scale-like effects. "
        "Only enable augmentation families that are plausibly supported by the demonstrations. "
        "If the evidence for an invariance is weak, leave that augmentation family disabled."
    ),
}


def build_strategy_prompt(strategy_name: str) -> str:
    if strategy_name not in STRATEGY_PROMPTS:
        raise KeyError(f"Unknown strategy prompt: {strategy_name}")
    return STRATEGY_PROMPTS[strategy_name] + "\n\n" + BASE_JSON_SCHEMA
