from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .predictor import design_matrix, grouped_bootstrap_interval, leave_one_group_out, spearman
from .screening import average_by_configuration, screen
from .toy import ToyRow, toy_rows

SUPPORT_GROUPS = ("A-math", "B-code", "C-math", "D-code")
TOP_K = 5


def as_record(row: ToyRow) -> dict:
    return asdict(row)


def run_demo(seed: int = 0) -> dict:
    rows = toy_rows(seed=seed)
    x = design_matrix([r.features for r in rows], [r.temperature for r in rows])
    y = np.asarray([r.gain for r in rows])
    groups = [r.group for r in rows]
    held_out = leave_one_group_out(x, y, groups)
    low, high = grouped_bootstrap_interval(held_out, y, groups, n_bootstrap=500, seed=seed)

    support = [as_record(r) for r in rows if r.group in SUPPORT_GROUPS]
    candidates = [as_record(r) for r in rows if r.group not in SUPPORT_GROUPS]
    result = screen(support, candidates, TOP_K)
    names, measured = average_by_configuration(
        [r["configuration"] for r in candidates], [r["gain"] for r in candidates]
    )
    measured_by_name = dict(zip(names, measured))
    oracle = [names[i] for i in np.argsort(-measured, kind="mergesort")[:TOP_K]]
    selected_gain = float(np.mean([measured_by_name[n] for n in result.selected]))
    oracle_gain = float(np.mean([measured_by_name[n] for n in oracle]))
    random_gain = float(np.mean(measured))
    return {
        "rows": len(rows),
        "configurations": len({r.configuration for r in rows}),
        "groups": len(set(groups)),
        "held_out_spearman": spearman(held_out, y),
        "held_out_interval": [low, high],
        "support_configurations": len({r["configuration"] for r in support}),
        "candidate_configurations": len(names),
        "top_k": TOP_K,
        "selected": result.selected,
        "oracle": oracle,
        "precision_at_k": len(set(result.selected) & set(oracle)) / TOP_K,
        "selected_mean_gain": selected_gain,
        "random_mean_gain": random_gain,
        "oracle_mean_gain": oracle_gain,
        "screen_spearman": spearman(result.predicted_gain, [measured_by_name[n] for n in result.configurations]),
        "scored_candidates": TOP_K,
        "skipped_candidates": len(names) - TOP_K,
        "predicted_gain": dict(zip(result.configurations, map(float, result.predicted_gain))),
        "measured_gain": {n: float(v) for n, v in measured_by_name.items()},
    }


def print_demo(seed: int = 0) -> dict:
    r = run_demo(seed)
    low, high = r["held_out_interval"]
    print("Synthetic configurations, not results from the paper.")
    print()
    print(f"Held-out ranking ({r['configurations']} configurations, {r['groups']} groups, one group held out at a time)")
    print(f"  Spearman between predicted and measured gain: {r['held_out_spearman']:.3f}  (95% interval {low:.3f} to {high:.3f})")
    print()
    print(
        f"Screening: fit on {r['support_configurations']} scored configurations, "
        f"rank {r['candidate_configurations']} new ones, score the top {r['top_k']}"
    )
    print(f"  {'Selection':<28}{'Mean measured gain':>20}")
    print(f"  {'Top 5 by predicted gain':<28}{r['selected_mean_gain']:>20.3f}")
    print(f"  {'Random 5 (expected)':<28}{r['random_mean_gain']:>20.3f}")
    print(f"  {'True top 5':<28}{r['oracle_mean_gain']:>20.3f}")
    print(f"  Overlap with the true top 5: {round(r['precision_at_k'] * r['top_k'])} of {r['top_k']}")
    print(f"  Reward-model scoring runs: {r['scored_candidates']} of {r['candidate_configurations']} candidates ({r['skipped_candidates']} skipped)")
    return r
