from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from .answers import extract_answers

FEATURES = (
    "majority_size_std",
    "per_prompt_answer_entropy_median",
    "completion_length_variance",
    "first_correct_sample_position_median",
)

LABEL_FREE_FEATURES = FEATURES[:3]

Prompt = Mapping[str, Sequence]


def _valid_answers(prompt: Prompt, domain: str) -> list[str]:
    completions = list(prompt.get("completions") or [])
    if len(completions) < 2:
        return []
    return [answer for answer in extract_answers(completions, domain) if answer is not None]


def modal_fraction(answers: Sequence[str]) -> float:
    if not answers:
        return float("nan")
    return Counter(answers).most_common(1)[0][1] / len(answers)


def answer_entropy(answers: Sequence[str]) -> float:
    if not answers:
        return float("nan")
    total = len(answers)
    return float(-sum((n / total) * math.log(n / total) for n in Counter(answers).values()))


def majority_size_std(prompts: Sequence[Prompt], domain: str) -> float:
    fractions = [modal_fraction(a) for a in (_valid_answers(p, domain) for p in prompts) if a]
    if len(fractions) < 2:
        return float("nan")
    return float(np.std(fractions))


def per_prompt_answer_entropy_median(prompts: Sequence[Prompt], domain: str) -> float:
    entropies = [answer_entropy(a) for a in (_valid_answers(p, domain) for p in prompts) if a]
    return float(np.median(entropies)) if entropies else float("nan")


def completion_length_variance(prompts: Sequence[Prompt]) -> float:
    spreads = []
    for prompt in prompts:
        completions = list(prompt.get("completions") or [])
        if len(completions) >= 2:
            spreads.append(float(np.std([len(c) for c in completions])))
    return float(np.mean(spreads)) if spreads else float("nan")


def first_correct_position(correct: Sequence) -> int | None:
    if not correct or any(c is None for c in correct):
        return None
    for index, value in enumerate(correct):
        if int(value) == 1:
            return index
    return None


def first_correct_sample_position_median(prompts: Sequence[Prompt]) -> float:
    positions = []
    for prompt in prompts:
        position = first_correct_position(list(prompt.get("correct") or []))
        if position is not None:
            positions.append(position)
    return float(np.median(positions)) if positions else float("nan")


def compact_features(prompts: Sequence[Prompt], domain: str) -> dict[str, float]:
    return {
        "majority_size_std": majority_size_std(prompts, domain),
        "per_prompt_answer_entropy_median": per_prompt_answer_entropy_median(prompts, domain),
        "completion_length_variance": completion_length_variance(prompts),
        "first_correct_sample_position_median": first_correct_sample_position_median(prompts),
    }
