#!/usr/bin/env python3
"""Homogenization check figure: inter-silo centroid distance, federated vs isolated, n=3 seeds."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("results/falsification")


def main():
    data = json.load(open(OUT_DIR / "homogenization_check_n10.json"))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for cond, key, color in [("federated", "federated_per_seed", "#2a9d8f"),
                              ("isolated", "isolated_per_seed", "#e76f51")]:
        runs = data[key]
        rounds = [c["round"] for c in runs[0]["curve"]]
        vals = np.array([[c["centroid_distance"] for c in r["curve"]] for r in runs])
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        ax.plot(rounds, mean, marker="o", color=color, label=cond)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("FL round")
    ax.set_ylabel("Pairwise inter-silo centroid distance\n(velarex/sornathis probes)")
    test = data["exact_permutation_test"]
    ax.set_title(
        "Homogenization check (n=10 seeds, mean±std band)\n"
        "Federated: pre-aggregation local drift, corrected every round\n"
        "Isolated: freely accumulating drift, never corrected\n"
        f"Final round: p={test['p_two_sided']:.2f} (exact permutation, n=3/arm)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "homogenization_check.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
