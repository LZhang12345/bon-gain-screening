from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .features import FEATURES
from .predictor import GainPredictor, design_matrix, top_k


@dataclass
class ScreeningResult:
    configurations: list[str]
    predicted_gain: np.ndarray
    selected: list[str]
    alpha: float

    def ranking(self) -> list[tuple[str, float]]:
        order = np.argsort(-self.predicted_gain, kind="mergesort")
        return [(self.configurations[i], float(self.predicted_gain[i])) for i in order]


def average_by_configuration(names: Sequence[str], values: Sequence[float]) -> tuple[list[str], np.ndarray]:
    order: list[str] = []
    totals: dict[str, list[float]] = {}
    for name, value in zip(names, values):
        if name not in totals:
            order.append(name)
            totals[name] = []
        totals[name].append(float(value))
    return order, np.asarray([np.mean(totals[name]) for name in order])


def screen(
    support: Sequence[Mapping],
    candidates: Sequence[Mapping],
    k: int,
    names: Sequence[str] = FEATURES,
    alpha: float | None = None,
) -> ScreeningResult:
    candidate_names = [r["configuration"] for r in candidates]
    overlap = {r["configuration"] for r in support}.intersection(candidate_names)
    if overlap:
        raise ValueError(f"Configurations must be entirely scored or unscored: {', '.join(sorted(overlap))}")
    x_candidates = design_matrix([r["features"] for r in candidates], [r["temperature"] for r in candidates], names)
    invalid = np.flatnonzero(~np.isfinite(x_candidates).all(axis=1))
    if len(invalid):
        raise ValueError(f"Candidate {candidate_names[invalid[0]]!r} has a non-finite feature or temperature")

    x_support = design_matrix([r["features"] for r in support], [r["temperature"] for r in support], names)
    y_support = [r["gain"] for r in support]
    groups = [r["group"] for r in support]
    model = GainPredictor(alpha=alpha).fit(x_support, y_support, None if alpha is not None else groups)
    configurations, predicted = average_by_configuration(candidate_names, model.predict(x_candidates))
    chosen = top_k(predicted, min(k, len(configurations)))
    return ScreeningResult(
        configurations=configurations,
        predicted_gain=predicted,
        selected=[configurations[i] for i in chosen],
        alpha=model.fit_.alpha,
    )
