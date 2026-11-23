from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have shape (prompts, candidates)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def pass_at_1(correct: Sequence[Sequence[int]]) -> float:
    return float(_matrix(correct, "correct").mean())


def best_of_n_accuracy(scores: Sequence[Sequence[float]], correct: Sequence[Sequence[int]]) -> float:
    s = _matrix(scores, "scores")
    c = _matrix(correct, "correct")
    if s.shape != c.shape:
        raise ValueError("scores and correct must have the same shape")
    chosen = np.argmax(s, axis=1)
    return float(c[np.arange(len(c)), chosen].mean())


def bon_gain(scores: Sequence[Sequence[float]], correct: Sequence[Sequence[int]]) -> float:
    return best_of_n_accuracy(scores, correct) - pass_at_1(correct)
