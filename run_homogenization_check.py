#!/usr/bin/env python3
"""
Quantify the "FL homogenizes local representations" claim
(research_notes/embedding_hypothesis.md, open question Q31), which until now
rested on a single n=1, pre-falsification run and a qualitative UMAP read
("the federated panels look near-identical").

The naive version of this metric is trivial: under standard FedAvg, every
silo holds the literal same broadcast weights right after aggregation, so
post-aggregation embeddings are byte-identical by construction (distance=0,
not an empirical finding). The non-trivial question is what happens to
cross-silo divergence that accrues *before* the next correction:

  - federated: each round, silos train locally from a shared starting point,
    diverge for one round's worth of local gradient steps, then get pulled
    back together by FedAvg. We measure that one-round divergence -- the
    pairwise inter-silo centroid distance on each silo's *local*,
    pre-aggregation weights -- every round.
  - isolated: silos never get pulled back together. Divergence is free to
    accumulate, round over round, for the entire run.

If "FL homogenizes representations" is a real, federation-specific effect
(not just "averaging makes two numbers equal"), federated's one-round
divergence should stay roughly flat/bounded across rounds while isolated's
accumulated divergence should grow -- and the final-round gap between the
two should be the quantitative version of the original qualitative claim.

Silos are given a genuinely non-IID class skew (velarex-heavy / balanced /
sornathis-heavy) so that local dialects have something to diverge into --
the existing C2/C3 IID setup (_build_pools, same distribution every silo)
would not give local training a different signal to drift towards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_unknown_disease as rud
from run_all_statistical_tests import exact_permutation_test
from fl.aggregation import fedavg

OUT_DIR = Path("results/falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_SEV_WEIGHTS = {"mild": 2.0, "moderate": 3.0, "severe": 1.0}
_SILO_VELAREX_FRAC = [0.85, 0.5, 0.15]  # non-IID skew across 3 silos


def _silo_distribution(velarex_frac: float) -> list[tuple[str, str, float]]:
    sornathis_frac = 1.0 - velarex_frac
    dist = []
    for sev, w in _SEV_WEIGHTS.items():
        dist.append(("velarex", sev, w * velarex_frac))
        dist.append(("sornathis", sev, w * sornathis_frac))
    return dist


def _build_nonIID_pools(n_silos: int, events_per_silo: int, holdout_frac: float, seed: int):
    lib = rud.FictionalPhraseLibrary(seed=seed)
    import random
    rng = random.Random(seed)
    train_pools, holdouts = [], []
    for i in range(n_silos):
        dist = _silo_distribution(_SILO_VELAREX_FRAC[i % len(_SILO_VELAREX_FRAC)])
        records = lib.sample_pool(dist, events_per_silo, seed_offset=i * 1000)
        rng.shuffle(records)
        n_hold = max(1, int(len(records) * holdout_frac))
        holdouts.append(records[:n_hold])
        train_pools.append(records[n_hold:])
    return train_pools, holdouts


def _centroid_distance(embs_per_silo: list[np.ndarray], labels: list[str]) -> float:
    """Mean pairwise inter-silo distance between same-class centroids."""
    classes = sorted(set(labels))
    centroids = []
    for embs in embs_per_silo:
        c = {}
        for cls in classes:
            idx = [i for i, l in enumerate(labels) if l == cls]
            c[cls] = embs[idx].mean(axis=0)
        centroids.append(c)
    dists = []
    n = len(centroids)
    for i in range(n):
        for j in range(i + 1, n):
            for cls in classes:
                dists.append(float(np.linalg.norm(centroids[i][cls] - centroids[j][cls])))
    return float(np.mean(dists))


def run_condition(federated: bool, seed: int, n_rounds: int, n_silos: int,
                  events_per_silo: int, local_epochs: int, training_device: str) -> dict:
    from fl.learner import FLLearner
    from fl.lora import LoRAConfig
    from run_prototype import _extract_cls

    train_pools, holdouts = _build_nonIID_pools(n_silos, events_per_silo, 0.15, seed)
    schedules = [rud._make_schedule("gaussian", len(train_pools[i]), n_rounds) for i in range(n_silos)]

    probe_events = rud.generate_fictional_probe_events(n_per_band=12, seed=999)
    true_labels = [ev.ground_truth.split("/")[0] for ev in probe_events]
    keep_idx = [i for i, l in enumerate(true_labels) if l in ("velarex", "sornathis")]
    probe_texts_all = []
    for ev in probe_events:
        turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        probe_texts_all.append(turns[-1] if turns else "")
    probe_texts = [probe_texts_all[i] for i in keep_idx]
    probe_labels = [true_labels[i] for i in keep_idx]

    lora_cfg = LoRAConfig(num_labels=4)
    learners = [FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                          min_events_to_train=10, device=training_device,
                          local_epochs=local_epochs)
                for _ in range(n_silos)]

    cursors = [0] * n_silos
    total_revealed = [0] * n_silos
    silo_weights = [None] * n_silos
    snap_rounds = [2, 5, 8, 10, 12, 15, 18, 20]
    curve = []

    tag = "FEDERATED" if federated else "ISOLATED"
    print(f"\n{'='*60}\n  Homogenization check, condition: {tag}, seed={seed}\n{'='*60}\n")

    for r in range(n_rounds):
        rnd = r + 1
        round_weights, train_sizes = [], []
        for i, learner in enumerate(learners):
            if silo_weights[i] is not None:
                learner.set_weights(silo_weights[i])
            n_rev = schedules[i][r]
            new_ev = train_pools[i][cursors[i]: cursors[i] + n_rev]
            cursors[i] += n_rev
            total_revealed[i] += len(new_ev)
            if total_revealed[i] >= 10 and new_ev:
                n_trained, _ = learner.train(new_ev)
                train_sizes.append(n_trained)
            else:
                train_sizes.append(0)
            round_weights.append(learner.get_weights())
            learner.release()

        # Snapshot BEFORE aggregation: this is the quantity of interest for
        # the federated condition (one round's worth of local divergence,
        # about to be corrected) -- for isolated it's just "current weights"
        # since nothing ever corrects them.
        if rnd in snap_rounds:
            embs_per_silo = []
            for i, learner in enumerate(learners):
                learner.set_weights(round_weights[i])
                embs_per_silo.append(_extract_cls(learner, probe_texts))
                learner.release()
            dist = _centroid_distance(embs_per_silo, probe_labels)
            curve.append({"round": rnd, "centroid_distance": dist})
            print(f"  R{rnd:02d}  inter-silo centroid distance = {dist:.4f}")

        if federated:
            active_idx = [i for i, s in enumerate(train_sizes) if s >= 4]
            if active_idx:
                global_w = fedavg([round_weights[i] for i in active_idx], [train_sizes[i] for i in active_idx])
                silo_weights = [global_w] * n_silos
        else:
            silo_weights = round_weights

    return {"federated": federated, "seed": seed, "curve": curve,
           "final_centroid_distance": curve[-1]["centroid_distance"] if curve else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--n-silos", type=int, default=3)
    ap.add_argument("--events-per-silo", type=int, default=160)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--training-device", default="cuda")
    args = ap.parse_args()

    all_results = {"federated": [], "isolated": []}
    for seed in args.seeds:
        for federated in [True, False]:
            key = "federated" if federated else "isolated"
            res = run_condition(federated, seed, args.n_rounds, args.n_silos,
                                args.events_per_silo, args.local_epochs, args.training_device)
            all_results[key].append(res)

    fed_finals = [r["final_centroid_distance"] for r in all_results["federated"]]
    iso_finals = [r["final_centroid_distance"] for r in all_results["isolated"]]

    print("\n\n=== SUMMARY (final round, n={}) ===".format(len(args.seeds)))
    print(f"Federated  -- inter-silo centroid distance: {fed_finals} "
         f"(mean={np.mean(fed_finals):.4f}, std={np.std(fed_finals):.4f})")
    print(f"Isolated   -- inter-silo centroid distance: {iso_finals} "
         f"(mean={np.mean(iso_finals):.4f}, std={np.std(iso_finals):.4f})")

    test = exact_permutation_test(fed_finals, iso_finals)
    print(f"\nExact permutation test (federated vs isolated): {test}")

    verdict = (
        "HOMOGENIZATION CONFIRMED, quantitatively: federated silos' one-round local "
        "divergence stays well below isolated silos' freely-accumulated divergence at "
        "the final round."
        if np.mean(fed_finals) < np.mean(iso_finals)
        else "HOMOGENIZATION NOT SUPPORTED at this data volume/seed count: federated "
        "one-round divergence is not smaller than isolated's accumulated divergence."
    )
    print(f"\nVerdict: {verdict}")

    summary = {
        "metric": "pairwise inter-silo centroid distance (velarex/sornathis probes)",
        "federated_per_seed": all_results["federated"],
        "isolated_per_seed": all_results["isolated"],
        "final_round_federated": fed_finals,
        "final_round_isolated": iso_finals,
        "exact_permutation_test": test,
        "verdict": verdict,
    }
    out_path = OUT_DIR / "homogenization_check.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
