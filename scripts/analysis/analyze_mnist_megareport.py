#!/usr/bin/env python3
"""Quick comparison plots for the MNIST falsification results (no plots existed before)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/mnist_falsification")
OUT_DIR = RESULTS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.load(open(RESULTS_DIR / name / "summary.json"))


def main():
    novel_e3 = load("mnist_novel_seed42")["final"]
    c7_e3 = load("mnist_c7_seed42")["final"]
    novel_e1 = load("mnist_novel_epoch1_seed42")["final"]
    c7_e1 = load("mnist_c7_epoch1_seed42")["final"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    labels = ["softmax\nknown->unk", "proto\nknown->unk", "proto\nnovel->unk"]
    e3_vals = [c7_e3["softmax_velarex_as_unk"], c7_e3["proto_velarex_as_unk"], novel_e3["proto_morven_as_unk"]]
    e1_vals = [c7_e1["softmax_velarex_as_unk"], c7_e1["proto_velarex_as_unk"], novel_e1["proto_morven_as_unk"]]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, e3_vals, w, label="local_epochs=3", color="#264653")
    ax.bar(x + w/2, e1_vals, w, label="local_epochs=1", color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("MNIST: known-class leakage (want low) vs.\nnovel-class recall (want high)")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    ax = axes[1]
    sil_labels = ["novel scenario\n(real digit injected)", "c7 scenario\n(known digit relabeled)"]
    sil_vals = [novel_e3["silhouette_inject_vs_rest"], c7_e3["silhouette_inject_vs_rest"]]
    ax.bar(sil_labels, sil_vals, color=["#2a9d8f", "#e76f51"])
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("MNIST silhouette is saturated regardless of\nwhether anything novel actually happened")
    ax.set_ylim(0, 1.0)

    fig.suptitle("MNIST cross-domain replication of C1/C7", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / "mnist_c7_comparison.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
