#!/usr/bin/env python3
"""C3 shuffled-label control figure: ARI curve, normal vs shuffled, n=3 seeds."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("results/falsification")


def load(seed):
    path = OUT_DIR / (f"c3_shuffled_label_control_seed{seed}.json" if seed != 42 else "c3_shuffled_label_control.json")
    return json.load(open(path))


def main():
    runs = [load(s) for s in [42, 43, 44]]
    rounds = [c["round"] for c in runs[0]["normal"]["curve"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in [
        (axes[0], "kmeans_ari_true_label", "Unsupervised KMeans ARI vs. true label\n(0 = random partition, 1 = perfect recovery)"),
        (axes[1], "silhouette_true_label", "Silhouette (true Velarex/Sornathis grouping)"),
    ]:
        for cond, color in [("normal", "#2a9d8f"), ("shuffled", "#e76f51")]:
            vals = np.array([[c[metric] for c in r[cond]["curve"]] for r in runs])
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            ax.plot(rounds, mean, marker="o", color=color, label=cond)
            ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("FL round")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("C3 — shuffled-label control (n=3 seeds, mean±std band)\n"
                "Normal training recovers true disease split unsupervised; shuffled training never does", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "c3_ari_curve_n3.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
