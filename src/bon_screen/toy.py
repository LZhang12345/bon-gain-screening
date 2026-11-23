from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .features import compact_features
from .gain import bon_gain, pass_at_1

TEMPERATURES = (0.3, 0.7, 1.0)
GROUPS = ("A-math", "A-code", "B-math", "B-code", "C-math", "C-code", "D-math", "D-code")


@dataclass
class ToyRow:
    configuration: str
    group: str
    temperature: float
    features: dict
    gain: float
    pass1: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _completion(answer: str, length: int, domain: str) -> str:
    filler = "step " * max(1, length // 5)
    if domain == "code":
        return f"{filler}\ndef {answer}(x):\n    return x"
    return f"{filler}\\boxed{{{answer}}}"


def simulate_configuration(
    rng: np.random.Generator,
    skill: float,
    sharpness: float,
    verifier_noise: float,
    temperature: float,
    difficulty: np.ndarray,
    domain: str,
    k: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    slope = (1.0 + 4.0 * sharpness) / (0.55 + 0.75 * temperature)
    p_correct = _sigmoid(slope * (skill - difficulty))
    n_wrong = 1 + round((1.0 - sharpness) * 5 * temperature + 2 * temperature)
    prompts, correct_rows, score_rows = [], [], []
    for j, p in enumerate(p_correct):
        correct = (rng.random(k) < p).astype(int)
        weights = 1.0 / np.arange(1, n_wrong + 1) ** (1.5 - temperature)
        wrong = rng.choice(n_wrong, size=k, p=weights / weights.sum())
        answers = [
            (f"f{j}" if c else f"g{j}_{w}") if domain == "code" else (str(7 * j + 3) if c else str(1000 + 10 * j + w))
            for c, w in zip(correct, wrong)
        ]
        spread = 30.0 + 260.0 * float(p * (1.0 - p)) * (0.5 + temperature)
        lengths = np.clip(rng.normal(320.0, spread, size=k), 40, None).astype(int)
        prompts.append(
            {
                "completions": [_completion(a, n, domain) for a, n in zip(answers, lengths)],
                "correct": correct.tolist(),
            }
        )
        correct_rows.append(correct)
        score_rows.append(correct * 1.0 + rng.normal(0.0, verifier_noise, size=k))
    return prompts, np.asarray(correct_rows), np.asarray(score_rows)


def toy_rows(
    seed: int = 0,
    per_group: int = 5,
    n_prompts: int = 60,
    k: int = 16,
    temperatures: Sequence[float] = TEMPERATURES,
) -> list[ToyRow]:
    rng = np.random.default_rng(seed)
    difficulty = {"math": rng.normal(0.0, 1.3, n_prompts), "code": rng.normal(0.3, 1.1, n_prompts)}
    rows: list[ToyRow] = []
    for g, group in enumerate(GROUPS):
        domain = group.split("-")[1]
        base_skill = rng.normal(0.0, 0.5)
        for i in range(per_group):
            skill = base_skill + rng.normal(0.0, 0.6)
            sharpness = float(rng.uniform(0.0, 1.0))
            verifier_noise = float(rng.uniform(0.5, 1.0))
            name = f"{group}-{i + 1}"
            for t in temperatures:
                prompts, correct, scores = simulate_configuration(
                    rng, skill, sharpness, verifier_noise, t, difficulty[domain], domain, k
                )
                rows.append(
                    ToyRow(
                        configuration=name,
                        group=group,
                        temperature=float(t),
                        features=compact_features(prompts, domain),
                        gain=bon_gain(scores, correct),
                        pass1=pass_at_1(correct),
                    )
                )
    return rows
