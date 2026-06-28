#!/usr/bin/env python3
"""Comparison plot for the FedAvg/FedProx/FedSGD/FedAdam x {morven, c7} sweep."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/aggregation_comparison")
OUT_PATH = RESULTS_DIR / "aggregation_comparison.png"
AGGREGATORS = ["fedavg", "fedprox", "fedsgd", "fedadam"]


def load(agg, scenario):
    d = RESULTS_DIR / f"{agg}_{scenario}_seed42"
    return json.load(open(d / "summary.json"))


def main():
    c7_fp = [load(a, "c7")["final_velarex_as_unknown_rate"] for a in AGGREGATORS]
    real_tp = [load(a, "morven")["final_morven_as_unknown_rate"] for a in AGGREGATORS]
    real_fp = [load(a, "morven")["final_velarex_as_unknown_rate"] for a in AGGREGATORS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(AGGREGATORS))

    ax = axes[0]
    ax.bar(x, c7_fp, color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels(AGGREGATORS)
    ax.set_ylabel("rate")
    ax.set_title("C7 false-positive rate\n(Velarex relabeled 'unknown' -> classified 'unknown')\nlower = better")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(c7_fp):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    ax = axes[1]
    width = 0.35
    ax.bar(x - width/2, real_tp, width, color="#2a9d8f", label="real Morven correctly -> 'unknown' (want high)")
    ax.bar(x + width/2, real_fp, width, color="#e76f51", label="real Velarex wrongly -> 'unknown' (want low)")
    ax.set_xticks(x); ax.set_xticklabels(AGGREGATORS)
    ax.set_title("Real-injection scenario\n(genuine Morven novel disease)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    for i, v in enumerate(real_tp):
        ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(real_fp):
        ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle("Aggregation algorithm comparison: false-positive robustness (C7) vs. real detection power",
                fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")

    print("\nSummary table:")
    print(f"{'aggregator':<10}{'C7_FP':>10}{'real_TP':>10}{'real_FP':>10}")
    for a, fp, tp, rfp in zip(AGGREGATORS, c7_fp, real_tp, real_fp):
        print(f"{a:<10}{fp:>10.3f}{tp:>10.3f}{rfp:>10.3f}")


if __name__ == "__main__":
    main()
