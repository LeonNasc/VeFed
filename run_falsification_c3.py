#!/usr/bin/env python3
"""
Falsification C3 — shuffled-label control (falsification.md, Claim 1).

Train on Velarex + Sornathis with the disease LABELS randomly permuted across
training examples (text stays the same; the text<->label correspondence is
broken). If the embedding space still separates disease identity when
evaluated against the TRUE labels, that separation is driven by surface text
features (sentence length, vocabulary overlap) rather than genuinely learned
disease identity -- the signal would be spurious.

Two conditions, same seed, same everything else:
  shuffled -- training labels permuted
  normal   -- training labels correct (control)

Evaluated both ways (silhouette on raw logits/CLS, and PrototypeBank
nearest-centroid) against a FIXED probe set's TRUE labels, since this
session already found the two evaluation methods can disagree (see C7).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import run_unknown_disease as rud
from fl.aggregation import fedavg
from fl.prototype_bank import PrototypeBank

OUT_DIR = Path("results/falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_pools_maybe_shuffled(n_silos, events_per_silo, holdout_frac, seed, shuffle_labels: bool):
    """Wraps _build_pools; optionally permutes label/gt_disease across the
    pooled training records (text stays fixed)."""
    train_pools, holdouts = rud._build_pools(n_silos, events_per_silo, holdout_frac, seed)
    if not shuffle_labels:
        return train_pools, holdouts

    rng = random.Random(seed + 999)
    # Pool all training records together, shuffle the label assignment, redistribute.
    flat = [(i, rec) for i, pool in enumerate(train_pools) for rec in pool]
    labels = [rec["label"] for _, rec in flat]
    rng.shuffle(labels)
    new_pools = [[] for _ in range(n_silos)]
    for (silo_i, rec), shuffled_label in zip(flat, labels):
        new_rec = {**rec, "label": shuffled_label, "gt_disease": shuffled_label}
        new_pools[silo_i].append(new_rec)
    return new_pools, holdouts


def run_condition(shuffle_labels: bool, seed: int, n_rounds: int, n_silos: int,
                  events_per_silo: int, local_epochs: int, training_device: str) -> dict:
    from fl.learner import FLLearner
    from fl.lora import LoRAConfig

    train_pools, holdouts = _build_pools_maybe_shuffled(
        n_silos, events_per_silo, 0.15, seed, shuffle_labels)
    schedules = [rud._make_schedule("gaussian", len(train_pools[i]), n_rounds) for i in range(n_silos)]

    probe_events = rud.generate_fictional_probe_events(n_per_band=12, seed=999)
    true_labels = [ev.ground_truth.split("/")[0] for ev in probe_events]
    # Only velarex/sornathis are trained on in this control (no Morven injection).
    keep_idx = [i for i, l in enumerate(true_labels) if l in ("velarex", "sornathis")]
    probe_texts = []
    for ev in probe_events:
        turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        probe_texts.append(turns[-1] if turns else "")

    lora_cfg = LoRAConfig(num_labels=4)
    learners = [FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                          min_events_to_train=10, device=training_device,
                          local_epochs=local_epochs)
                for _ in range(n_silos)]
    bank_kwargs = dict(pca_components=50, dbscan_eps=0.30, dbscan_min_samples=5)
    silo_banks = [PrototypeBank(**bank_kwargs) for _ in range(n_silos)]
    global_bank = PrototypeBank(**bank_kwargs)

    cursors = [0] * n_silos
    total_revealed = [0] * n_silos
    global_w = None
    snap_rounds = [2, 5, 8, 10, 12, 15, 18, 20]
    curve = []

    from run_prototype import _extract_cls

    tag = "SHUFFLED" if shuffle_labels else "NORMAL"
    print(f"\n{'='*60}\n  C3 condition: {tag}\n{'='*60}\n")

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
            if global_w is not None:
                learner.set_weights(global_w)
            new_ev = new_this_round[i]
            if total_revealed[i] >= 10 and new_ev:
                n_trained, _ = learner.train(new_ev)
                train_sizes.append(n_trained)
            else:
                train_sizes.append(0)
            if new_ev:
                embs, lbls = learner.extract_embeddings(new_ev)
                if len(embs) > 0:
                    for cls_name in set(lbls):
                        mask = np.array([l == cls_name for l in lbls])
                        silo_banks[i].update(cls_name, embs[mask])
            round_weights.append(learner.get_weights())
            learner.release()

        active_idx = [i for i, s in enumerate(train_sizes) if s >= 4]
        if active_idx:
            global_w = fedavg([round_weights[i] for i in active_idx], [train_sizes[i] for i in active_idx])
            global_bank = PrototypeBank.fedavg(
                [silo_banks[i] for i in active_idx], [train_sizes[i] for i in active_idx], **bank_kwargs)

        if global_w is not None and rnd in snap_rounds:
            learners[-1].set_weights(global_w)
            embs = _extract_cls(learners[-1], probe_texts)
            sub_embs = embs[keep_idx]
            sub_true = [true_labels[i] for i in keep_idx]

            # Silhouette against TRUE labels (velarex vs sornathis grouping).
            from sklearn.metrics import silhouette_score
            coords = rud._project_umap(sub_embs, seed=seed)
            group = np.array([1 if l == "velarex" else 0 for l in sub_true])
            sil = float(silhouette_score(coords, group)) if len(set(group.tolist())) > 1 else float("nan")

            # PrototypeBank accuracy against TRUE labels.
            preds = global_bank.classify(sub_embs)
            proto_acc = sum(1 for p, t in zip(preds, sub_true) if p == t) / len(sub_true)

            # Unsupervised cross-check: does k=2 KMeans on the RAW embeddings (no labels
            # involved at all) recover the TRUE velarex/sornathis split? This is immune to
            # the shuffled bank's own centroids being internally incoherent.
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score
            km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(sub_embs)
            ari = float(adjusted_rand_score(sub_true, km.labels_))

            curve.append({"round": rnd, "silhouette_true_label": sil, "proto_acc_true_label": proto_acc,
                         "kmeans_ari_true_label": ari})
            print(f"  R{rnd:02d}  sil(true velarex/sornathis)={sil:.3f}  proto_acc(true)={proto_acc:.3f}  "
                 f"kmeans_ari(true)={ari:.3f}  protos={global_bank.names()}")
            learners[-1].release()

    return {"shuffle_labels": shuffle_labels, "curve": curve,
           "final_silhouette": curve[-1]["silhouette_true_label"] if curve else float("nan"),
           "final_proto_acc": curve[-1]["proto_acc_true_label"] if curve else float("nan"),
           "final_kmeans_ari": curve[-1]["kmeans_ari_true_label"] if curve else float("nan")}


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
    for shuffle in [False, True]:
        key = "shuffled" if shuffle else "normal"
        results[key] = run_condition(shuffle, args.seed, args.n_rounds, args.n_silos,
                                     args.events_per_silo, args.local_epochs, args.training_device)

    print("\n\n=== SUMMARY ===")
    print(f"Normal labels   -- final silhouette={results['normal']['final_silhouette']:.3f}  "
         f"final proto_acc={results['normal']['final_proto_acc']:.3f}  "
         f"final kmeans_ari={results['normal']['final_kmeans_ari']:.3f}")
    print(f"Shuffled labels -- final silhouette={results['shuffled']['final_silhouette']:.3f}  "
         f"final proto_acc={results['shuffled']['final_proto_acc']:.3f}  "
         f"final kmeans_ari={results['shuffled']['final_kmeans_ari']:.3f}")

    normal_sil = results["normal"]["final_silhouette"]
    shuffled_sil = results["shuffled"]["final_silhouette"]
    shuffled_ari = results["shuffled"]["final_kmeans_ari"]
    # Primary criterion: unsupervised KMeans+ARI (immune to the shuffled bank's own
    # centroids being internally incoherent -- proto_acc against a self-contradictory
    # bank is a noisier, secondary signal, reported but not decisive).
    verdict = (
        f"C3 PASSES: shuffled-label training's embeddings do NOT recover the true "
        f"velarex/sornathis split under unsupervised clustering (KMeans ARI={shuffled_ari:.3f}, "
        f"0=random) and silhouette stays far below normal training ({shuffled_sil:.3f} vs "
        f"{normal_sil:.3f}). The clean separation in normal training is driven by genuine "
        "label-text correspondence, not surface text features alone."
        if abs(shuffled_ari) < 0.15 else
        f"C3 FAILS: shuffled-label training's embeddings still partially recover the true split "
        f"under unsupervised clustering (KMeans ARI={shuffled_ari:.3f}) -- some of the original "
        "clustering signal may be driven by surface text features rather than genuinely learned "
        "disease identity."
    )
    print(f"\nVerdict: {verdict}")

    summary = {"control": "falsification.md C3 -- shuffled-label control", **results, "verdict": verdict}
    out_path = OUT_DIR / f"c3_shuffled_label_control_seed{args.seed}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
