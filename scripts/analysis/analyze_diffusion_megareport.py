#!/usr/bin/env python3
"""Recall curves for the diffusion-latency test (no plots existed before)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/diffusion_test")
COLORS = {"isolated": "#264653", "naive": "#e9c46a", "pre_exposed": "#2a9d8f"}


def load_curve(name):
    s = json.load(open(RESULTS_DIR / name / "summary.json"))
    return [(m["round"], m["new_silo_morven_recall"]) for m in s["round_metrics"] if "new_silo_morven_recall" in m]


def plot_panel(ax, name_fn, title):
    for cond in ["isolated", "naive", "pre_exposed"]:
        seed_curves = [load_curve(name_fn(cond, s)) for s in (42, 43)]
        rounds = [r for r, _ in seed_curves[0]]
        vals = np.array([[v for _, v in c] for c in seed_curves])
        mean = vals.mean(axis=0)
        ax.plot(rounds, mean, marker="o", color=COLORS[cond], label=cond)
    ax.axvline(15, color="red", ls="--", lw=0.8, label="silo_3 exposed")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("FL round"); ax.set_ylabel("silo_3's own morven recall")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_panel(axes[0], lambda c, s: f"{c}_seed{s}", "Generous (8/round, 3 local epochs)")
    plot_panel(axes[1], lambda c, s: f"{c}_epoch1_thin_seed{s}", "Thin (2/round, 1 local epoch)")
    fig.suptitle("Diffusion-latency test: no separation between conditions in either regime\n"
                "(mean of 2 seeds; all three reach detection the round silo_3 is first exposed)", fontsize=11)
    fig.tight_layout()
    path = RESULTS_DIR / "diffusion_latency_curves.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
