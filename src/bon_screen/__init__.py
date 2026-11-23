from .answers import extract_answers, extract_code_answer, extract_math_answer
from .features import (
    FEATURES,
    LABEL_FREE_FEATURES,
    compact_features,
    completion_length_variance,
    first_correct_sample_position_median,
    majority_size_std,
    per_prompt_answer_entropy_median,
)
from .gain import best_of_n_accuracy, bon_gain, pass_at_1
from .predictor import (
    ALPHAS,
    REFERENCE_TEMPERATURE,
    GainPredictor,
    design_matrix,
    grouped_bootstrap_interval,
    leave_one_group_out,
    spearman,
    top_k,
)
from .screening import ScreeningResult, screen

__version__ = "0.1.0"

__all__ = [
    "ALPHAS",
    "FEATURES",
    "LABEL_FREE_FEATURES",
    "REFERENCE_TEMPERATURE",
    "GainPredictor",
    "ScreeningResult",
    "best_of_n_accuracy",
    "bon_gain",
    "compact_features",
    "completion_length_variance",
    "design_matrix",
    "extract_answers",
    "extract_code_answer",
    "extract_math_answer",
    "first_correct_sample_position_median",
    "grouped_bootstrap_interval",
    "leave_one_group_out",
    "majority_size_std",
    "pass_at_1",
    "per_prompt_answer_entropy_median",
    "screen",
    "spearman",
    "top_k",
]
