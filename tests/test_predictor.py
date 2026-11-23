import numpy as np
import pytest

from bon_screen import (
    GainPredictor,
    best_of_n_accuracy,
    bon_gain,
    design_matrix,
    leave_one_group_out,
    pass_at_1,
    spearman,
    top_k,
)
from bon_screen.predictor import fit_ridge, group_kfold


def test_gain_is_best_of_n_minus_pass_at_1():
    correct = [[0, 1, 0, 0], [0, 0, 0, 0]]
    scores = [[0.1, 0.9, 0.2, 0.3], [0.5, 0.1, 0.2, 0.3]]
    assert pass_at_1(correct) == pytest.approx(0.125)
    assert best_of_n_accuracy(scores, correct) == pytest.approx(0.5)
    assert bon_gain(scores, correct) == pytest.approx(0.375)


def test_design_matrix_appends_centered_temperature():
    x = design_matrix([{"a": 1.0, "b": 2.0}], [1.0], names=("a", "b"))
    assert x.tolist() == [[1.0, 2.0, pytest.approx(0.3)]]


def test_ridge_matches_the_closed_form_on_standardized_inputs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=30)
    fit = fit_ridge(x, y, alpha=2.0)
    z = (x - x.mean(0)) / x.std(0)
    expected = np.linalg.solve(z.T @ z + 2.0 * np.eye(3), z.T @ (y - y.mean()))
    assert np.allclose(fit.coef, expected)
    assert np.allclose(fit.predict(x), z @ expected + y.mean())


def test_group_kfold_never_splits_a_group():
    groups = ["a"] * 3 + ["b"] * 6 + ["c"] * 3 + ["d"] * 3
    folds = group_kfold(groups, 3)
    seen = {}
    for k, idx in enumerate(folds):
        for i in idx:
            assert seen.setdefault(groups[i], k) == k
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(groups)))


def test_leave_one_group_out_does_not_see_the_held_out_group():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(24, 2))
    y = x[:, 0] + 0.05 * rng.normal(size=24)
    groups = np.repeat(["a", "b", "c", "d"], 6)
    base = leave_one_group_out(x, y, groups)
    shifted = y.copy()
    shifted[groups == "a"] += 100.0
    moved = leave_one_group_out(x, shifted, groups)
    assert np.allclose(base[groups == "a"], moved[groups == "a"])
    assert spearman(base, y) > 0.9


def test_fixed_penalty_and_top_k():
    x = np.arange(10, dtype=float)[:, None]
    model = GainPredictor(alpha=0.1).fit(x, 2.0 * x[:, 0])
    assert top_k(model.predict(x), 3).tolist() == [9, 8, 7]
    with pytest.raises(ValueError, match="Predicted gains must be finite"):
        top_k([0.2, np.nan], 2)
    with pytest.raises(ValueError):
        GainPredictor().fit(x, x[:, 0])


def test_spearman_handles_ties_and_missing_values():
    assert spearman([1, 2, 2, 4], [1, 2, 3, 4]) == pytest.approx(np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3, 4])[0, 1])
    assert spearman([1, np.nan, 3, 4], [1, 5, 3, 4]) == pytest.approx(1.0)
