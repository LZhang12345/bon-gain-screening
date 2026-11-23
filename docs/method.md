# Method notes

## The gain we predict

For a configuration (a fine-tuned model with its post-training method and task domain) and a sampling temperature $T$, draw $k$ candidate answers for each of $P$ labeled validation prompts. A reward model scores every candidate. Best-of-$N$ returns the highest-scoring candidate, and its accuracy is $\mathrm{BoN}@k$. Single-sample accuracy, $\mathrm{pass}@1$, is the mean correctness of the same $k$ candidates. The gain is

$$
g(c, T) = \mathrm{BoN}@k(c, T) - \mathrm{pass}@1(c, T).
$$

`bon_screen.gain.bon_gain(scores, correct)` computes it from two arrays of shape `(prompts, candidates)`.

## The four statistics

All four are computed from the same sampled answers, one value per configuration and temperature. Answers are extracted from each completion first (`bon_screen.answers`).

| Name in the code | What it measures | Needs gold answers |
| :--- | :--- | :---: |
| `majority_size_std` | For each prompt, the share of samples that give the most frequent answer. The statistic is the standard deviation of that share across prompts. | no |
| `per_prompt_answer_entropy_median` | Shannon entropy (natural log) of the extracted answers for each prompt, then the median across prompts. | no |
| `completion_length_variance` | Standard deviation of completion lengths in characters within each prompt, averaged across prompts. | no |
| `first_correct_sample_position_median` | Index of the first correct sample for each prompt with at least one correct sample, then the median across prompts. | yes |

The two answer-distribution statistics skip prompts with fewer than two completions or no extractable answers. Completion-length spread uses prompts with at least two completions. Only the first-correct-sample statistic requires correctness labels.

## The predictor

Each observation is one configuration at one temperature. The predictor is a ridge regression with one coefficient per statistic, shared across temperatures, and a temperature term:

$$
g(c, T) = \boldsymbol\beta^\top \mathbf{x}(c, T) + \gamma\,(T - 0.7) + \varepsilon(c, T).
$$

Features are standardized on the training rows only. The ridge penalty is chosen from $\{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$ by cross-validation in which every fold keeps whole groups of configurations together (`bon_screen.predictor.select_alpha`). `GainPredictor` wraps this, and `leave_one_group_out` holds out each group of configurations that share a base checkpoint and task domain in turn.

The NumPy implementation follows the paper's grouped rerun, which reports Spearman correlation of 0.898. On the synthetic example, its predictions match a scikit-learn pipeline of `StandardScaler` and `Ridge` tuned with `GridSearchCV` over `GroupKFold`.

## Screening

1. Generate candidates once for every configuration you want to compare.
2. Score a small support panel with the reward model and compute its gains.
3. Compute the four statistics for every configuration.
4. Fit the predictor on the support panel and rank the remaining configurations.
5. Run the reward model only on the configurations ranked highest.

`bon_screen.screening.screen` performs step 4 and returns the shortlist for scoring in step 5. Predictions are averaged over temperatures for each configuration before ranking.

## What the paper adds

The paper also gives an error bound that separates the linear approximation error, the error in the estimated statistics, the error in the fitted coefficients, and the error in the measured gain, and relates these to ranking errors. That analysis is not code and is not reproduced here.
