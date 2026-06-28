#!/usr/bin/env python3
"""
Direct test of the train/probe leakage concern: with only 5 phrases per
disease x severity bucket, probes and training pools sample with replacement
from the same tiny finite phrase universe, producing verbatim text overlap
(measured at 37-54% of velarex/sornathis probes in spot checks).

Trains the standard federated condition once, and at each snapshot round
computes KMeans ARI two ways: on ALL probes (replicating the original C2/C3
metric) and on ONLY the subset of probes whose exact text never appeared in
any silo's training pool (the genuinely held-out subset). If ARI stays high
on the non-overlapping subset too, the original result reflects real
generalization. If it drops substantially, the "perfect" scores were
inflated by memorization of a small closed vocabulary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_unknown_disease as rud
from fl.aggregation import fedavg

OUT_DIR = Path("results/falsification")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-per-silo", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--isolated", action="store_true", help="no FedAvg -- each silo keeps own weights")
    args = ap.parse_args()

    seed = args.seed
    federated = not args.isolated
    n_silos, events_per_silo, n_rounds, local_epochs = 3, args.events_per_silo, 20, 3

    from fl.learner import FLLearner
    from fl.lora import LoRAConfig
    from run_prototype import _extract_cls
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans

    train_pools, holdouts = rud._build_pools(n_silos, events_per_silo, 0.15, seed)
    train_texts_set = {rec["text"] for pool in train_pools for rec in pool}
    schedules = [rud._make_schedule("gaussian", len(train_pools[i]), n_rounds) for i in range(n_silos)]

    probe_events = rud.generate_fictional_probe_events(n_per_band=12, seed=999)
    true_labels = [ev.ground_truth.split("/")[0] for ev in probe_events]
    keep_idx = [i for i, l in enumerate(true_labels) if l in ("velarex", "sornathis")]
    probe_texts = []
    for ev in probe_events:
        turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        probe_texts.append(turns[-1] if turns else "")

    sub_true = [true_labels[i] for i in keep_idx]
    sub_texts = [probe_texts[i] for i in keep_idx]
    is_leaked = [t in train_texts_set for t in sub_texts]
    n_leaked = sum(is_leaked)
    print(f"Of {len(keep_idx)} velarex/sornathis probes, {n_leaked} ({n_leaked/len(keep_idx)*100:.0f}%) "
         f"are verbatim duplicates of training text.\n")

    clean_idx_within_sub = [i for i, leaked in enumerate(is_leaked) if not leaked]
    print(f"Genuinely held-out (non-overlapping) probe count: {len(clean_idx_within_sub)}\n")

    lora_cfg = LoRAConfig(num_labels=4)
    learners = [FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                          min_events_to_train=10, device="cuda", local_epochs=local_epochs)
                for _ in range(n_silos)]

    cursors = [0] * n_silos
    total_revealed = [0] * n_silos
    silo_weights = [None] * n_silos
    snap_rounds = [10, 12, 15, 18, 20]
    curve = []

    tag = "FEDERATED" if federated else "ISOLATED"
    print(f"condition: {tag}\n")

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
                embs = _extract_cls(learners[-1], sub_texts)
                learners[-1].release()
                all_embs_list, clean_embs_list = [embs], [embs[clean_idx_within_sub]]
            else:
                all_embs_list, clean_embs_list = [], []
                for i, learner in enumerate(learners):
                    if silo_weights[i] is None:
                        continue
                    learner.set_weights(silo_weights[i])
                    embs = _extract_cls(learner, sub_texts)
                    learner.release()
                    all_embs_list.append(embs)
                    clean_embs_list.append(embs[clean_idx_within_sub])

            ari_all_vals, ari_clean_vals = [], []
            for embs in all_embs_list:
                km_all = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embs)
                ari_all_vals.append(float(adjusted_rand_score(sub_true, km_all.labels_)))
            clean_true = [sub_true[i] for i in clean_idx_within_sub]
            for embs in clean_embs_list:
                km_clean = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embs)
                ari_clean_vals.append(float(adjusted_rand_score(clean_true, km_clean.labels_)))

            ari_all = float(np.mean(ari_all_vals))
            ari_clean = float(np.mean(ari_clean_vals))
            curve.append({"round": rnd, "ari_all_probes": ari_all, "ari_clean_probes_only": ari_clean})
            print(f"  R{rnd:02d}  ARI(all probes)={ari_all:.3f}   ARI(non-overlapping probes only)={ari_clean:.3f}")

    out = {
        "n_leaked": n_leaked, "n_total_probes": len(keep_idx), "leak_fraction": n_leaked / len(keep_idx),
        "curve": curve,
    }
    cond_tag = "fed" if federated else "iso"
    out_path = OUT_DIR / f"leakage_check_{cond_tag}_eps{events_per_silo}_seed{seed}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
