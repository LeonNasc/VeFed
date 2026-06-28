#!/usr/bin/env python3
"""Aggregation comparison figure, rebuilt with n=3 seeds and error bars."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/aggregation_comparison")
AGGREGATORS = ["fedavg", "fedprox", "fedsgd", "fedadam"]
SEEDS = [42, 43, 44]


def load(agg, scenario, seed):
    return json.load(open(RESULTS_DIR / f"{agg}_{scenario}_seed{seed}" / "summary.json"))


def main():
    c7_fp_mean, c7_fp_std = [], []
    real_tp_mean, real_tp_std = [], []
    real_fp_mean, real_fp_std = [], []

    for agg in AGGREGATORS:
        c7_vals = [load(agg, "c7", s)["final_velarex_as_unknown_rate"] for s in SEEDS]
        tp_vals = [load(agg, "morven", s)["final_morven_as_unknown_rate"] for s in SEEDS]
        fp_vals = [load(agg, "morven", s)["final_velarex_as_unknown_rate"] for s in SEEDS]
        c7_fp_mean.append(np.mean(c7_vals)); c7_fp_std.append(np.std(c7_vals))
        real_tp_mean.append(np.mean(tp_vals)); real_tp_std.append(np.std(tp_vals))
        real_fp_mean.append(np.mean(fp_vals)); real_fp_std.append(np.std(fp_vals))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(AGGREGATORS))

    ax = axes[0]
    ax.bar(x, c7_fp_mean, yerr=c7_fp_std, capsize=4, color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels(AGGREGATORS)
    ax.set_ylabel("rate")
    ax.set_title("C7 false-positive rate (n=3 seeds, mean±std)\n"
                "(Velarex relabeled 'unknown' -> classified 'unknown')\nlower = better", fontsize=10)
    ax.set_ylim(0, 1.2)
    for i, (m, s) in enumerate(zip(c7_fp_mean, c7_fp_std)):
        ax.text(i, m + s + 0.03, f"{m:.2f}±{s:.2f}", ha="center", fontsize=8)

    ax = axes[1]
    width = 0.35
    ax.bar(x - width/2, real_tp_mean, width, yerr=real_tp_std, capsize=3, color="#2a9d8f",
          label="real Morven correctly -> 'unknown' (want high)")
    ax.bar(x + width/2, real_fp_mean, width, yerr=real_fp_std, capsize=3, color="#e76f51",
          label="real Velarex wrongly -> 'unknown' (want low)")
    ax.set_xticks(x); ax.set_xticklabels(AGGREGATORS)
    ax.set_title("Real-injection scenario (n=3 seeds, mean±std)\n(genuine Morven novel disease)", fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=8)

    fig.suptitle("Aggregation algorithm comparison, n=3 seeds: false-positive robustness (C7) vs. real detection power\n"
                "(FedAdam's large error bars reflect genuine bimodality across seeds, not measurement noise)",
                fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / "aggregation_comparison.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")

    print("\nSummary table (n=3 mean±std):")
    print(f"{'aggregator':<10}{'C7_FP':>16}{'real_TP':>16}{'real_FP':>16}")
    for a, fp, fps, tp, tps, rfp, rfps in zip(AGGREGATORS, c7_fp_mean, c7_fp_std, real_tp_mean, real_tp_std, real_fp_mean, real_fp_std):
        print(f"{a:<10}{fp:>8.3f}±{fps:<6.3f}{tp:>8.3f}±{tps:<6.3f}{rfp:>8.3f}±{rfps:<6.3f}")


if __name__ == "__main__":
    main()
