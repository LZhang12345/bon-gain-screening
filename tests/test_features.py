import math

import numpy as np
import pytest

from bon_screen import (
    compact_features,
    completion_length_variance,
    extract_code_answer,
    extract_math_answer,
    first_correct_sample_position_median,
    majority_size_std,
    per_prompt_answer_entropy_median,
)


def prompt(answers, correct=None, lengths=None):
    completions = [f"{'x' * (lengths[i] if lengths else 0)}\\boxed{{{a}}}" for i, a in enumerate(answers)]
    return {"completions": completions, "correct": correct or [0] * len(answers)}


def test_math_extractor_prefers_the_last_boxed_answer():
    assert extract_math_answer("first \\boxed{3} then \\boxed{4}.") == "4"
    assert extract_math_answer("#### 12") == "12"
    assert extract_math_answer("The answer is 7") == "7"
    assert extract_math_answer("no number here") is None


def test_code_extractor_reads_the_first_signature():
    assert extract_code_answer("def add(a, b):\n    return a + b") == "add(a, b)"
    assert extract_code_answer("print(1)") is None


def test_majority_size_std_is_the_spread_of_modal_fractions():
    prompts = [prompt(["1", "1", "1", "2"]), prompt(["5", "5", "6", "7"])]
    assert majority_size_std(prompts, "math") == pytest.approx(np.std([0.75, 0.5]))


def test_majority_size_std_needs_two_prompts():
    assert math.isnan(majority_size_std([prompt(["1", "1"])], "math"))


def test_entropy_median_uses_natural_log():
    prompts = [prompt(["1", "1", "1", "1"]), prompt(["1", "2", "3", "4"]), prompt(["1", "1", "2", "2"])]
    assert per_prompt_answer_entropy_median(prompts, "math") == pytest.approx(math.log(2))


def test_length_variance_averages_within_prompt_spreads():
    prompts = [prompt(["1", "1"], lengths=[0, 10]), prompt(["1", "1"], lengths=[5, 5])]
    assert completion_length_variance(prompts) == pytest.approx(2.5)


def test_first_correct_position_skips_prompts_without_a_correct_sample():
    prompts = [prompt(["1"] * 4, [0, 0, 1, 0]), prompt(["1"] * 4, [1, 1, 0, 0]), prompt(["1"] * 4, [0, 0, 0, 0])]
    assert first_correct_sample_position_median(prompts) == pytest.approx(1.0)


def test_only_the_first_correct_feature_reads_labels():
    prompts = [prompt(["1", "1", "2", "3"], [1, 1, 0, 0]), prompt(["4", "5", "5", "5"], [0, 0, 1, 1])]
    relabeled = [dict(p, correct=[0, 0, 0, 1]) for p in prompts]
    a, b = compact_features(prompts, "math"), compact_features(relabeled, "math")
    assert a["first_correct_sample_position_median"] != b["first_correct_sample_position_median"]
    for name in ("majority_size_std", "per_prompt_answer_entropy_median", "completion_length_variance"):
        assert a[name] == b[name]
