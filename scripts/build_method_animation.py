from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from bon_screen.demo import run_demo

INK = "#20364e"
MUTED = "#5f7391"
TEAL = "#149d81"
GRAY = "#c9d3df"
PURPLE = "#7253c8"
STAGES = (
    ("1 / 3", "Predict each candidate's gain", "Four statistics of its sampled answers go into the fitted predictor."),
    ("2 / 3", "Rank and keep the top 5", "Candidates below the cut are not scored by the reward model."),
    ("3 / 3", "Score only the top 5", "Dots show measured gains, computed here for every candidate to check the pick."),
)


def card(fig, y, title, value, detail, highlight=False):
    ax = fig.add_axes((0.70, y, 0.27, 0.15))
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.92, boxstyle="round,pad=0.0,rounding_size=0.08",
                                facecolor="#f3f6fa" if highlight else "#f8f9fb",
                                edgecolor=TEAL if highlight else "#e1e6ee", linewidth=2 if highlight else 1,
                                transform=ax.transAxes))
    ax.text(0.07, 0.68, title, fontsize=11.5, color=TEAL if highlight else MUTED, weight="bold", transform=ax.transAxes)
    ax.text(0.93, 0.68, value, fontsize=17, color=INK, weight="bold", ha="right", transform=ax.transAxes)
    ax.text(0.07, 0.26, detail, fontsize=10, color=MUTED, transform=ax.transAxes)


def frame(demo, order, stage, progress):
    names = list(demo["predicted_gain"])
    predicted = np.asarray([demo["predicted_gain"][n] for n in names])
    measured = np.asarray([demo["measured_gain"][n] for n in names])
    selected = set(demo["selected"])
    fig = plt.figure(figsize=(12, 6.2), dpi=100)
    fig.patch.set_facecolor("white")
    label, headline, detail = STAGES[stage]
    fig.text(0.065, 0.93, "Screening before reward-model scoring", fontsize=21, color=INK, weight="bold")
    fig.text(0.065, 0.865, f"{label}   {headline}", fontsize=15, color=PURPLE, weight="bold")
    fig.text(0.065, 0.82, detail, fontsize=11.5, color=MUTED)
    ax = fig.add_axes((0.065, 0.15, 0.58, 0.6))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b9c3d0")
    ax.tick_params(colors="#5f7391", labelsize=9)
    ax.grid(axis="y", color="#e6ebf2", linewidth=0.8)
    ax.set_axisbelow(True)
    idx = np.arange(len(names)) if stage == 0 else order
    heights = predicted[idx] * (progress if stage == 0 else 1.0)
    colors = [TEAL if (stage > 0 and names[i] in selected) else ("#9fb0c4" if stage == 0 else GRAY) for i in idx]
    ax.bar(np.arange(len(idx)), heights, color=colors, width=0.75)
    if stage > 0:
        ax.axvline(len(selected) - 0.5, color=PURPLE, lw=1.6, ls="--")
        ax.text(len(selected) - 0.3, 0.47, "cut", color=PURPLE, fontsize=10)
    if stage == 2:
        shown = round(progress * len(idx))
        ax.scatter(np.arange(shown), measured[idx][:shown], s=30, color=INK, zorder=3, label="measured gain")
        ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_ylim(0, 0.5)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(names)} new candidates" + ("" if stage == 0 else ", ordered by predicted gain"), color="#44546a")
    ax.set_ylabel("Predicted best-of-N gain", color="#44546a")
    fig.text(0.70, 0.78, "Reward-model budget", fontsize=15, color=INK, weight="bold")
    scored = len(selected) if stage > 0 else 0
    card(fig, 0.58, "Scoring runs", f"{scored} of {len(names)}",
         "every candidate is generated once" if stage == 0 else f"{len(names) - scored} candidates skipped", stage > 0)
    if stage == 2:
        card(fig, 0.40, "Top 5, measured", f"{demo['selected_mean_gain']:.2f}", "mean best-of-N gain", True)
        card(fig, 0.22, "Random 5", f"{demo['random_mean_gain']:.2f}", "expected mean gain")
    fig.text(0.065, 0.045, f"Predictor fitted on {demo['support_configurations']} scored configurations.",
             fontsize=11, color="#44546a")
    fig.text(0.065, 0.012, "Synthetic configurations from the released code. Not results from the paper.",
             fontsize=9.5, color="#7a8798")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def build(output_dir: Path) -> dict:
    demo = run_demo(seed=0)
    names = list(demo["predicted_gain"])
    order = np.argsort(-np.asarray([demo["predicted_gain"][n] for n in names]), kind="mergesort")
    frames, durations = [], []
    for stage, steps, hold in ((0, 10, 1400), (1, 1, 2200), (2, 12, 3200)):
        for s in range(1, steps + 1):
            frames.append(frame(demo, order, stage, s / steps))
            durations.append(hold if s == steps else 90)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    palette[0].save(output_dir / "method_animation.gif", save_all=True, append_images=palette[1:],
                    duration=durations, loop=0, optimize=True)
    frames[-1].save(output_dir / "method_animation_poster.png")
    values = {
        "candidates": len(names),
        "support_configurations": demo["support_configurations"],
        "selected": demo["selected"],
        "selected_mean_gain": round(demo["selected_mean_gain"], 6),
        "random_mean_gain": round(demo["random_mean_gain"], 6),
        "frames": len(frames),
    }
    (output_dir / "method_animation.json").write_text(json.dumps(values, indent=2) + "\n")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    build(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
