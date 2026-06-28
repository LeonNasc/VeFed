#!/usr/bin/env python3
"""Hyperparameter sweep plots: FedProx (mu), FedAdam (server_lr), FedSGD (lr)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/aggregation_comparison")
OUT_PATH = RESULTS_DIR / "aggregation_sweep.png"


def load(name):
    return json.load(open(RESULTS_DIR / name / "summary.json"))


def main():
    fedprox_mus = [0.001, 0.01, 0.1, 1.0]
    fedadam_slrs = [0.01, 0.05, 0.1, 0.5]
    fedsgd_lrs = [0.001, 0.01, 0.1]

    fedprox_fp = [load(f"fedprox_mu{m}_c7_seed42")["final_velarex_as_unknown_rate"] for m in fedprox_mus]
    fedprox_tp = [load(f"fedprox_mu{m}_morven_seed42")["final_morven_as_unknown_rate"] for m in fedprox_mus]

    fedadam_fp = [load(f"fedadam_slr{s}_c7_seed42")["final_velarex_as_unknown_rate"] for s in fedadam_slrs]
    fedadam_tp = [load(f"fedadam_slr{s}_morven_seed42")["final_morven_as_unknown_rate"] for s in fedadam_slrs]

    fedsgd_fp = [load(f"fedsgd_lr{l}_c7_seed42")["final_velarex_as_unknown_rate"] for l in fedsgd_lrs]
    fedsgd_tp = [load(f"fedsgd_lr{l}_morven_seed42")["final_morven_as_unknown_rate"] for l in fedsgd_lrs]

    fedavg = load("fedavg_c7_seed42")
    fedavg_fp, fedavg_tp = fedavg["final_velarex_as_unknown_rate"], load("fedavg_morven_seed42")["final_morven_as_unknown_rate"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(fedprox_mus, fedprox_fp, marker="o", color="#e76f51", label="C7 false-positive")
    ax.plot(fedprox_mus, fedprox_tp, marker="s", color="#2a9d8f", label="real Morven detection")
    ax.axhline(fedavg_fp, color="#264653", ls="--", lw=1, label="FedAvg C7-FP (ref)")
    ax.set_xscale("log"); ax.set_xlabel("prox_mu"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("FedProx mu sweep")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(fedadam_slrs, fedadam_fp, marker="o", color="#e76f51", label="C7 false-positive")
    ax.plot(fedadam_slrs, fedadam_tp, marker="s", color="#2a9d8f", label="real Morven detection")
    ax.axhline(fedavg_fp, color="#264653", ls="--", lw=1, label="FedAvg C7-FP (ref)")
    ax.set_xscale("log"); ax.set_xlabel("server_lr"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("FedAdam server_lr sweep")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(fedsgd_lrs, fedsgd_fp, marker="o", color="#e76f51", label="C7 false-positive")
    ax.plot(fedsgd_lrs, fedsgd_tp, marker="s", color="#2a9d8f", label="real Morven detection")
    ax.axhline(fedavg_fp, color="#264653", ls="--", lw=1, label="FedAvg C7-FP (ref)")
    ax.set_xscale("log"); ax.set_xlabel("local lr"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("FedSGD lr sweep (insensitive)")
    ax.legend(fontsize=8)

    fig.suptitle("Hyperparameter sensitivity: C7 false-positive rate vs. real-detection power", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")

    print("\nFedProx:", list(zip(fedprox_mus, fedprox_fp, fedprox_tp)))
    print("FedAdam:", list(zip(fedadam_slrs, fedadam_fp, fedadam_tp)))
    print("FedSGD: ", list(zip(fedsgd_lrs, fedsgd_fp, fedsgd_tp)))


if __name__ == "__main__":
    main()
