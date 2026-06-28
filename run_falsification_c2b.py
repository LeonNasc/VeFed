#!/usr/bin/env python3
"""
Falsification C2b -- isolated vs. federated training, HARDER 3-disease variant.

C2 (run_falsification_c2.py) found federated and isolated training
statistically indistinguishable on the easy 2-disease (Velarex/Sornathis)
clustering task -- a question raised in discussion: is federation's
contribution masked because the task is too easy for a single silo's local
data to already solve on its own?

This variant makes Morven a third KNOWN class (no held-out novelty at all --
a pure 3-way classification task) and reruns the identical federated-vs-
isolated comparison. If a federation advantage was being masked by task
simplicity, it should appear here: a 3-way split gives each silo's local,
non-IID slice more room to be incomplete, where pooling could plausibly help.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import run_unknown_disease as rud
from fl.aggregation import fedavg

OUT_DIR = Path("results/falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_THREE_DIST = rud._KNOWN_DIST + rud._MORVEN_DIST  # velarex + sornathis + morven, all known


def _build_pools_3disease(n_silos, events_per_silo, holdout_frac, seed):
    lib = rud.FictionalPhraseLibrary(seed=seed)
    rng = random.Random(seed)
    train_pools, holdouts = [], []
    for i in range(n_silos):
        records = lib.sample_pool(_THREE_DIST, events_per_silo, seed_offset=i * 1000)
        rng.shuffle(records)
        n_hold = max(1, int(len(records) * holdout_frac))
        holdouts.append(records[:n_hold])
        train_pools.append(records[n_hold:])
    return train_pools, holdouts


def run_condition(federated: bool, seed: int, n_rounds: int, n_silos: int,
                  events_per_silo: int, local_epochs: int, training_device: str) -> dict:
    from fl.learner import FLLearner
    from fl.lora import LoRAConfig
    from run_prototype import _extract_cls
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans

    train_pools, holdouts = _build_pools_3disease(n_silos, events_per_silo, 0.15, seed)
    schedules = [rud._make_schedule("gaussian", len(train_pools[i]), n_rounds) for i in range(n_silos)]

    probe_events = rud.generate_fictional_probe_events(n_per_band=12, seed=999)
    true_labels = [ev.ground_truth.split("/")[0] for ev in probe_events]  # velarex/sornathis/morven, all kept
    probe_texts = []
    for ev in probe_events:
        turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        probe_texts.append(turns[-1] if turns else "")

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
    print(f"\n{'='*60}\n  C2b (3-disease) condition: {tag}\n{'='*60}\n")

    for r in range(n_rounds):
        rnd = r + 1
        new_this_round = []
        for i in range(n_silos):
            n_rev = schedules[i][r]
            new_ev = train_pools[i][cursors[i]: cursors[i] + n_rev]
            cursors[i] += n_rev
            total_revealed[i] += len(new_ev)
            new_this_round.append(list(new_ev))

        round_weights, train_sizes = [], []
        for i, learner in enumerate(learners):
            if silo_weights[i] is not None:
                learner.set_weights(silo_weights[i])
            new_ev = new_this_round[i]
            if total_revealed[i] >= 10 and new_ev:
                n_trained, _ = learner.train(new_ev)
                train_sizes.append(n_trained)
            else:
                train_sizes.append(0)
            round_weights.append(learner.get_weights())
            learner.release()

        if federated:
            active_idx = [i for i, s in enumerate(train_sizes) if s >= 4]
            if active_idx:
                global_w = fedavg([round_weights[i] for i in active_idx], [train_sizes[i] for i in active_idx])
                silo_weights = [global_w] * n_silos
        else:
            silo_weights = round_weights

        if rnd in snap_rounds and any(w is not None for w in silo_weights):
            if federated:
                learners[-1].set_weights(silo_weights[-1])
                embs = _extract_cls(learners[-1], probe_texts)
                sil = float(silhouette_score(rud._project_umap(embs, seed=seed), true_labels))
                km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(embs)
                ari = float(adjusted_rand_score(true_labels, km.labels_))
                learners[-1].release()
                per_silo_sil, per_silo_ari = [sil] * n_silos, [ari] * n_silos
            else:
                per_silo_sil, per_silo_ari = [], []
                for i, learner in enumerate(learners):
                    if silo_weights[i] is None:
                        continue
                    learner.set_weights(silo_weights[i])
                    embs = _extract_cls(learner, probe_texts)
                    sil = float(silhouette_score(rud._project_umap(embs, seed=seed), true_labels))
                    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(embs)
                    ari = float(adjusted_rand_score(true_labels, km.labels_))
                    learner.release()
                    per_silo_sil.append(sil)
                    per_silo_ari.append(ari)

            mean_sil = float(np.nanmean(per_silo_sil))
            mean_ari = float(np.nanmean(per_silo_ari))
            curve.append({"round": rnd, "silhouette_mean": mean_sil, "kmeans_ari_mean": mean_ari,
                         "per_silo_silhouette": per_silo_sil, "per_silo_ari": per_silo_ari})
            print(f"  R{rnd:02d}  mean_sil={mean_sil:.3f}  mean_ari={mean_ari:.3f}  per_silo_ari={[round(a,2) for a in per_silo_ari]}")

    return {"federated": federated, "curve": curve,
           "final_silhouette": curve[-1]["silhouette_mean"] if curve else float("nan"),
           "final_kmeans_ari": curve[-1]["kmeans_ari_mean"] if curve else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--n-silos", type=int, default=3)
    ap.add_argument("--events-per-silo", type=int, default=160)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--training-device", default="cuda")
    args = ap.parse_args()

    results = {}
    for federated in [True, False]:
        key = "federated" if federated else "isolated"
        results[key] = run_condition(federated, args.seed, args.n_rounds, args.n_silos,
                                     args.events_per_silo, args.local_epochs, args.training_device)

    print("\n\n=== SUMMARY (3-disease variant) ===")
    print(f"Federated -- final silhouette={results['federated']['final_silhouette']:.3f}  "
         f"final ARI={results['federated']['final_kmeans_ari']:.3f}")
    print(f"Isolated  -- final silhouette={results['isolated']['final_silhouette']:.3f}  "
         f"final ARI={results['isolated']['final_kmeans_ari']:.3f}")

    fed_ari = results["federated"]["final_kmeans_ari"]
    iso_ari = results["isolated"]["final_kmeans_ari"]
    verdict = (
        f"C2b PASSES (federation adds value on the harder 3-disease task): isolated ARI={iso_ari:.3f} "
        f"meaningfully lower than federated ARI={fed_ari:.3f}."
        if fed_ari - iso_ari > 0.15 else
        f"C2b still WEAKLY FALSIFIED / inconclusive even at 3 diseases: isolated ARI={iso_ari:.3f}, "
        f"federated ARI={fed_ari:.3f} -- task difficulty was not the masking factor."
    )
    print(f"\nVerdict: {verdict}")

    summary = {"control": "falsification.md C2b -- isolated training control, 3-disease variant",
              **results, "verdict": verdict}
    out_path = OUT_DIR / f"c2b_3disease_isolated_training_control_seed{args.seed}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
