#!/usr/bin/env python3
"""Dirichlet-alpha mini-sweep figure: FL gain (federated - isolated diag_acc) vs alpha, n=3 seeds."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("results/falsification")


def main():
    data = json.load(open(OUT_DIR / "dirichlet_alpha_sweep_n10.json"))
    rows = data["rows"]
    alphas = [r["alpha"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    fed_means = [np.mean(r["federated_diag_acc"]) for r in rows]
    fed_stds = [np.std(r["federated_diag_acc"]) for r in rows]
    iso_means = [np.mean(r["isolated_diag_acc"]) for r in rows]
    iso_stds = [np.std(r["isolated_diag_acc"]) for r in rows]
    x = np.arange(len(alphas))
    ax.errorbar(x, fed_means, yerr=fed_stds, marker="o", color="#2a9d8f", label="federated", capsize=4)
    ax.errorbar(x, iso_means, yerr=iso_stds, marker="o", color="#e76f51", label="isolated", capsize=4)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", label="chance (binary)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel("Dirichlet α (per-silo class-mix draw; low α = more non-IID)")
    ax.set_ylabel("Held-out diag_acc (mean±std, n=3 seeds)")
    ax.set_title("Federated vs isolated accuracy across the heterogeneity axis", fontsize=10)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    gains = [r["fl_gain"] for r in rows]
    ps = [r["exact_permutation_test"]["p_two_sided"] for r in rows]
    colors = ["#e76f51" if g < 0 else "#2a9d8f" for g in gains]
    ax2.bar(x, gains, color=colors)
    for xi, g, p in zip(x, gains, ps):
        ax2.text(xi, g + (0.01 if g >= 0 else -0.02), f"p={p:.2f}", ha="center", fontsize=8)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(a) for a in alphas])
    ax2.set_xlabel("Dirichlet α")
    ax2.set_ylabel("FL gain (federated − isolated diag_acc)")
    ax2.set_title("FL gain peaks at intermediate α; federated\ncatastrophically collapses (2/3 seeds) at α=0.1", fontsize=10)

    fig.suptitle("Dirichlet-α mini-sweep (n=3 seeds): FL gain vs. standard label-distribution heterogeneity", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "dirichlet_alpha_sweep.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
