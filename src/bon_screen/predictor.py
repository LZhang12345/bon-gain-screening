from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .features import FEATURES

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
REFERENCE_TEMPERATURE = 0.7


def design_matrix(
    features: Sequence[Mapping[str, float]],
    temperatures: Sequence[float],
    names: Sequence[str] = FEATURES,
    reference_temperature: float = REFERENCE_TEMPERATURE,
) -> np.ndarray:
    if len(features) != len(temperatures):
        raise ValueError("features and temperatures must have the same length")
    rows = [
        [float(row[name]) for name in names] + [float(t) - reference_temperature]
        for row, t in zip(features, temperatures)
    ]
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names) + 1)


def group_kfold(groups: Sequence, n_splits: int) -> list[np.ndarray]:
    labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
    if n_splits < 2 or n_splits > len(labels):
        raise ValueError("n_splits must be between 2 and the number of groups")
    sizes = np.bincount(inverse)
    load = np.zeros(n_splits)
    fold_of_group = np.zeros(len(labels), dtype=int)
    for group in np.argsort(sizes)[::-1]:
        fold = int(np.argmin(load))
        fold_of_group[group] = fold
        load[fold] += sizes[group]
    fold_of_row = fold_of_group[inverse]
    return [np.flatnonzero(fold_of_row == k) for k in range(n_splits)]


@dataclass
class RidgeFit:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return z @ self.coef + self.intercept


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeFit:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    z = (x - mean) / scale
    y_mean = float(y.mean())
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y_mean))
    return RidgeFit(mean=mean, scale=scale, coef=coef, intercept=y_mean, alpha=float(alpha))


def select_alpha(
    x: np.ndarray, y: np.ndarray, groups: Sequence, alphas: Sequence[float] = ALPHAS
) -> float:
    n_groups = len(set(np.asarray(groups).tolist()))
    if n_groups < 2:
        raise ValueError("choosing the penalty by grouped cross-validation needs at least two groups")
    folds = group_kfold(groups, min(5, n_groups))
    rows = np.arange(len(y))
    errors = []
    for alpha in alphas:
        fold_errors = []
        for test in folds:
            train = np.setdiff1d(rows, test)
            fit = fit_ridge(x[train], y[train], alpha)
            fold_errors.append(float(np.mean((fit.predict(x[test]) - y[test]) ** 2)))
        errors.append(float(np.mean(fold_errors)))
    return float(alphas[int(np.argmin(errors))])


@dataclass
class GainPredictor:
    alphas: Sequence[float] = ALPHAS
    alpha: float | None = None
    fit_: RidgeFit | None = field(default=None, init=False, repr=False)

    def fit(self, x: np.ndarray, y: Sequence[float], groups: Sequence | None = None) -> GainPredictor:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("training rows must be finite")
        if self.alpha is not None:
            alpha = float(self.alpha)
        elif groups is not None:
            alpha = select_alpha(x, y, groups, self.alphas)
        else:
            raise ValueError("pass groups to choose the penalty, or set alpha")
        self.fit_ = fit_ridge(x, y, alpha)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("call fit first")
        return self.fit_.predict(x)


def leave_one_group_out(
    x: np.ndarray, y: Sequence[float], groups: Sequence, alphas: Sequence[float] = ALPHAS
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    group_array = np.asarray(groups)
    predictions = np.full(len(y), np.nan)
    for held_out in np.unique(group_array):
        test = group_array == held_out
        model = GainPredictor(alphas=alphas).fit(x[~test], y[~test], group_array[~test])
        predictions[test] = model.predict(x[test])
    return predictions


def rank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr))
    sorted_values = arr[order]
    start = 0
    while start < len(arr):
        stop = start
        while stop + 1 < len(arr) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(rank(a[keep]), rank(b[keep]))[0, 1])


def grouped_bootstrap_interval(
    predictions: Sequence[float],
    targets: Sequence[float],
    groups: Sequence,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    group_array = np.asarray(groups)
    labels = np.unique(group_array)
    members = {label: np.flatnonzero(group_array == label) for label in labels}
    values = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(labels, size=len(labels), replace=True)
        idx = np.concatenate([members[label] for label in drawn])
        value = spearman(p[idx], t[idx])
        if np.isfinite(value):
            values.append(value)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def top_k(predicted_gain: Sequence[float], k: int) -> np.ndarray:
    scores = np.asarray(predicted_gain, dtype=np.float64)
    if not 1 <= k <= len(scores):
        raise ValueError("k must be between 1 and the number of candidates")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Predicted gains must be finite")
    return np.argsort(-scores, kind="mergesort")[:k]
