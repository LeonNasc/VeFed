#!/usr/bin/env python3
"""
Sweep events_per_silo to find where C2's null result (isolated == federated)
breaks down -- i.e. how little local data a silo needs before federation's
pooling advantage actually shows up.

Reuses run_falsification_c2.run_condition() directly, looping over a grid of
events_per_silo values, both conditions, multiple seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_falsification_c2 import run_condition

OUT_DIR = Path("results/falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-per-silo-grid", default="160,80,40,20,10")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--n-silos", type=int, default=3)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--training-device", default="cuda")
    args = ap.parse_args()

    grid = [int(x) for x in args.events_per_silo_grid.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    results = {}
    for eps in grid:
        fed_aris, iso_aris = [], []
        for seed in seeds:
            print(f"\n>>> events_per_silo={eps}  seed={seed}")
            fed = run_condition(True, seed, args.n_rounds, args.n_silos, eps, args.local_epochs, args.training_device)
            iso = run_condition(False, seed, args.n_rounds, args.n_silos, eps, args.local_epochs, args.training_device)
            fed_aris.append(fed["final_kmeans_ari"])
            iso_aris.append(iso["final_kmeans_ari"])
            print(f"    eps={eps} seed={seed}: fed_ari={fed['final_kmeans_ari']:.3f}  iso_ari={iso['final_kmeans_ari']:.3f}")

        results[eps] = {
            "fed_ari_mean": float(np.mean(fed_aris)), "fed_ari_std": float(np.std(fed_aris)),
            "iso_ari_mean": float(np.mean(iso_aris)), "iso_ari_std": float(np.std(iso_aris)),
            "fed_aris": fed_aris, "iso_aris": iso_aris,
        }
        gap = results[eps]["fed_ari_mean"] - results[eps]["iso_ari_mean"]
        print(f"\n=== events_per_silo={eps}: fed={results[eps]['fed_ari_mean']:.3f}±{results[eps]['fed_ari_std']:.3f}  "
             f"iso={results[eps]['iso_ari_mean']:.3f}±{results[eps]['iso_ari_std']:.3f}  gap={gap:+.3f} ===\n")

    out_path = OUT_DIR / "c2_events_per_silo_sweep.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n=== SWEEP SUMMARY ===")
    print(f"{'events/silo':>12}{'fed_ari':>16}{'iso_ari':>16}{'gap':>10}")
    for eps in grid:
        r = results[eps]
        gap = r["fed_ari_mean"] - r["iso_ari_mean"]
        print(f"{eps:>12}{r['fed_ari_mean']:>10.3f}±{r['fed_ari_std']:<5.3f}{r['iso_ari_mean']:>10.3f}±{r['iso_ari_std']:<5.3f}{gap:>+10.3f}")


if __name__ == "__main__":
    main()
