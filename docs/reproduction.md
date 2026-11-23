# Reproduction scope

## Included

- The four statistics used by the paper's predictor, with the answer extractors for math and code
- The best-of-$N$ gain computed from reward-model scores and correctness labels
- The ridge predictor with a temperature term, fold-local standardization, and grouped choice of the penalty
- Held-out evaluation by group, Spearman correlation, and a grouped bootstrap interval
- Screening: fit on scored configurations, rank new ones, and select the top $K$
- A synthetic demo, two small example inputs, and tests that run on CPU

## Not included

- Model training, candidate generation, and reward-model scoring. Any sampler and any reward model can produce the inputs.
- Benchmark prompts and model outputs. Obtain the benchmarks and models from their maintainers under their own terms.
- The comparison methods reported in the paper
- The cluster job scripts behind the paper's tables

## Running it on your own models

1. For each configuration and temperature, write one JSON object per prompt with `completions` (a list of strings) and `correct` (a list of 0 or 1, one per completion). See `examples/generations.jsonl`.
2. Compute the statistics: `python -m bon_screen features generations.jsonl --domain math`.
3. For the configurations you score with your reward model, compute the gain with `bon_screen.gain.bon_gain(scores, correct)`.
4. Put one row per configuration and temperature in a CSV with the columns of `examples/screening_table.csv`. Leave `gain` empty for configurations you have not scored.
5. Rank them: `python -m bon_screen screen table.csv --top 5`.

The `group` column decides which configurations are kept together when the penalty is chosen. Use one group per base checkpoint and task domain, as in the paper.

## Regenerating the figures

```bash
python -m pip install -e '.[plots]'
python scripts/build_readme_assets.py
python scripts/build_method_animation.py
```

Both scripts use the synthetic configurations in `bon_screen.toy` and write their values next to the images in `assets/`. `assets/overview.png` is Figure 1 of the paper.
