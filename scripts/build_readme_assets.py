from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bon_screen.demo import run_demo
from bon_screen.predictor import design_matrix, leave_one_group_out, spearman
from bon_screen.toy import toy_rows

COLORS = {"ink": "#20364e", "muted": "#8a9bb3", "teal": "#149d81", "grid": "#e6ebf2", "random": "#7253c8"}
T_COLORS = {0.3: "#0072B2", 0.7: "#D55E00", 1.0: "#009E73"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b9c3d0")
    ax.tick_params(colors="#44546a", labelsize=9)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def build(output_dir: Path) -> dict:
    rows = toy_rows(seed=0)
    x = design_matrix([r.features for r in rows], [r.temperature for r in rows])
    y = np.asarray([r.gain for r in rows])
    held_out = leave_one_group_out(x, y, [r.group for r in rows])
    rho = spearman(held_out, y)
    demo = run_demo(seed=0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    lo, hi = float(min(y.min(), held_out.min())) - 0.02, float(max(y.max(), held_out.max())) + 0.02
    left.plot([lo, hi], [lo, hi], color="#b9c3d0", lw=1, ls="--")
    for t, color in T_COLORS.items():
        mask = np.asarray([r.temperature == t for r in rows])
        left.scatter(y[mask], held_out[mask], s=26, color=color, alpha=0.85, label=f"T = {t}")
    left.set_xlim(lo, hi)
    left.set_ylim(lo, hi)
    left.set_xlabel("Measured best-of-N gain", color="#44546a")
    left.set_ylabel("Predicted gain (group held out)", color="#44546a")
    left.set_title("Predictions follow measured gains", loc="left", fontsize=12, color=COLORS["ink"], weight="bold")
    left.text(lo + 0.01, hi - 0.03, f"Spearman {rho:.2f}", color=COLORS["ink"], fontsize=10, weight="bold")
    left.legend(frameon=False, fontsize=9, loc="lower right")
    style(left)

    names = sorted(demo["predicted_gain"], key=lambda n: -demo["predicted_gain"][n])
    measured = np.asarray([demo["measured_gain"][n] for n in names])
    selected = set(demo["selected"])
    colors = [COLORS["teal"] if n in selected else "#c9d3df" for n in names]
    pos = np.arange(len(names))
    right.bar(pos, measured, color=colors, width=0.75)
    right.axhline(demo["random_mean_gain"], color=COLORS["random"], lw=1.6, ls="--")
    right.text(len(names) - 0.5, demo["random_mean_gain"] + 0.012, f"random pick, expected {demo['random_mean_gain']:.2f}",
               color=COLORS["random"], fontsize=9.5, ha="right")
    top = float(measured.max())
    right.set_ylim(0, top * 1.2)
    right.text(len(names) - 0.5, top * 1.1, f"scored: top {demo['top_k']}, mean gain {demo['selected_mean_gain']:.2f}",
               color=COLORS["teal"], fontsize=10, weight="bold", ha="right")
    right.set_xticks([])
    right.set_xlim(-0.8, len(names) - 0.2)
    right.set_xlabel(f"{len(names)} new candidates, ordered by predicted gain (left = highest)", color="#44546a")
    right.set_ylabel("Measured best-of-N gain", color="#44546a")
    right.set_title(f"Score only the top {demo['top_k']}, skip the other {demo['skipped_candidates']}", loc="left",
                    fontsize=12, color=COLORS["ink"], weight="bold")
    style(right)

    fig.text(0.008, 0.012, "Synthetic configurations produced by the released code. Not a result from the paper.",
             fontsize=8.5, color="#7a8798")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "toy_demo.png", dpi=150, facecolor="white", metadata={"Software": None})
    fig.savefig(output_dir / "toy_demo.svg", facecolor="white", metadata={"Creator": None, "Date": None, "Format": None, "Type": None})
    plt.close(fig)

    values = {
        "held_out_spearman": round(rho, 6),
        "candidates_by_predicted_gain": names,
        "measured_gain": [round(float(v), 6) for v in measured],
        "selected": demo["selected"],
        "selected_mean_gain": round(demo["selected_mean_gain"], 6),
        "random_mean_gain": round(demo["random_mean_gain"], 6),
        "oracle_mean_gain": round(demo["oracle_mean_gain"], 6),
    }
    (output_dir / "toy_demo.json").write_text(json.dumps(values, indent=2) + "\n")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    build(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
