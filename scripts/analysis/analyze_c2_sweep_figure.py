#!/usr/bin/env python3
"""C2 data-volume sweep figure: federated vs isolated ARI as events_per_silo shrinks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("results/falsification")


def main():
    # n=3 points: 160 (from original C2), 100/80/60 (from this sweep)
    n3_points = {}
    fed_aris = []
    for seed in [42, 43, 44]:
        d = json.load(open(OUT_DIR / f"c2_isolated_training_control_seed{seed}.json"))
        fed_aris.append(d["federated"]["final_kmeans_ari"])
    iso_aris_160 = [json.load(open(OUT_DIR / f"c2_isolated_training_control_seed{s}.json"))["isolated"]["final_kmeans_ari"] for s in [42, 43, 44]]
    n3_points[160] = {"fed": fed_aris, "iso": iso_aris_160}

    sweep = json.load(open(OUT_DIR / "c2_events_per_silo_sweep.json"))
    for eps in ["100", "80", "60"]:
        n3_points[int(eps)] = {"fed": sweep[eps]["fed_aris"], "iso": sweep[eps]["iso_aris"]}

    # n=1 (smoke) points: 40, 20, 10
    smoke = json.load(open(OUT_DIR / "c2_events_per_silo_sweep.json"))  # overwritten by n3 run for 100/80/60; need the smoke log values separately
    # The smoke sweep wrote to the same file before being overwritten by the n=3 run for 100/80/60.
    # Re-derive 40/20/10 from the smoke log directly since the JSON was overwritten.
    smoke_n1 = {40: {"fed": 0.475, "iso": 0.442}, 20: {"fed": 0.037, "iso": 0.066}, 10: {"fed": float("nan"), "iso": 0.007}}

    eps_n3 = sorted(n3_points.keys())
    fed_means = [np.mean(n3_points[e]["fed"]) for e in eps_n3]
    fed_stds  = [np.std(n3_points[e]["fed"]) for e in eps_n3]
    iso_means = [np.mean(n3_points[e]["iso"]) for e in eps_n3]
    iso_stds  = [np.std(n3_points[e]["iso"]) for e in eps_n3]

    eps_n1 = sorted(smoke_n1.keys())
    fed_n1 = [smoke_n1[e]["fed"] for e in eps_n1]
    iso_n1 = [smoke_n1[e]["iso"] for e in eps_n1]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(eps_n3, fed_means, yerr=fed_stds, marker="o", color="#1d4e89", label="federated (n=3, mean±std)", capsize=4)
    ax.errorbar(eps_n3, iso_means, yerr=iso_stds, marker="o", color="#e76f51", label="isolated (n=3, mean±std)", capsize=4)
    ax.plot(eps_n1, fed_n1, marker="x", linestyle="--", color="#1d4e89", alpha=0.5, label="federated (n=1, exploratory)")
    ax.plot(eps_n1, iso_n1, marker="x", linestyle="--", color="#e76f51", alpha=0.5, label="isolated (n=1, exploratory)")

    ax.axvspan(45, 65, color="gray", alpha=0.08)
    ax.text(55, 0.05, "gap opens\n(eps≈60)", ha="center", fontsize=8, color="#555")
    ax.axvspan(5, 15, color="gray", alpha=0.15)
    ax.text(10, 0.55, "federated\ntraining\nbreaks\n(eps=10)", ha="center", fontsize=8, color="#555")

    ax.set_xlabel("events per silo")
    ax.set_ylabel("final KMeans ARI vs. true label (Velarex/Sornathis)")
    ax.set_title("C2 data-volume sweep: where does isolated training start to lag federated?\n"
                "(2-disease task; n=3 for 60-160, n=1 exploratory for 10-40)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 170)
    fig.tight_layout()
    out = OUT_DIR / "c2_data_volume_sweep.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")

    print("\nFull table:")
    print(f"{'events/silo':>12}{'fed':>16}{'iso':>16}{'gap':>10}{'n':>4}")
    for e in eps_n3:
        gap = fed_means[eps_n3.index(e)] - iso_means[eps_n3.index(e)]
        print(f"{e:>12}{fed_means[eps_n3.index(e)]:>10.3f}±{fed_stds[eps_n3.index(e)]:<5.3f}{iso_means[eps_n3.index(e)]:>10.3f}±{iso_stds[eps_n3.index(e)]:<5.3f}{gap:>+10.3f}{'3':>4}")
    for e in eps_n1:
        gap = fed_n1[eps_n1.index(e)] - iso_n1[eps_n1.index(e)]
        print(f"{e:>12}{fed_n1[eps_n1.index(e)]:>16.3f}{iso_n1[eps_n1.index(e)]:>16.3f}{gap:>+10.3f}{'1':>4}")


if __name__ == "__main__":
    main()
