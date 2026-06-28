#!/usr/bin/env python3
"""
Falsification C5 -- untrained OOD absorption control (research_notes/falsification.md).

Once PrototypeBank's "unknown" cluster has formed from the REAL Morven
injection (not a relabeled known disease, unlike C4), probe it with text from
a domain never seen anywhere in training: IMDB movie-review snippets (reused
from the sentiment cross-domain sweep). These are genuinely out-of-distribution
for a clinical-text classifier -- not a held-out disease, not a relabeled
known disease, just unrelated English text.

Question: does the "unknown" bucket absorb these OOD probes too, alongside
genuine Morven? If so, "unknown" has become a generic catch-all for "doesn't
match a known disease" rather than a specific representation of Morven --
exactly the contamination risk found on MNIST (Section 7) that motivated
formalizing this control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_unknown_disease as rud
from fl.aggregation import fedavg
from fl.prototype_bank import PrototypeBank

OUT_DIR = Path("results/falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ood_probes(n: int = 30) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    texts = list(ds["test"]["text"])[:n]
    return [t[:300] for t in texts]  # truncate long reviews for tokenizer speed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--n-silos", type=int, default=3)
    ap.add_argument("--events-per-silo", type=int, default=160)
    ap.add_argument("--injection-round", type=int, default=10)
    ap.add_argument("--injection-per-round", type=int, default=8)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--training-device", default="cuda")
    args = ap.parse_args()

    from fl.learner import FLLearner
    from fl.lora import LoRAConfig
    from run_prototype import _extract_cls

    train_pools, holdouts = rud._build_pools(args.n_silos, args.events_per_silo, 0.15, args.seed)
    schedules = [rud._make_schedule("gaussian", len(train_pools[i]), args.n_rounds) for i in range(args.n_silos)]
    morven_pool = rud._build_morven_pool(n=500, seed=args.seed)

    probe_events = rud.generate_fictional_probe_events(n_per_band=12, seed=999)
    probe_labels = [ev.ground_truth.split("/")[0] for ev in probe_events]
    probe_texts = []
    for ev in probe_events:
        turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        probe_texts.append(turns[-1] if turns else "")
    morven_idx = [i for i, l in enumerate(probe_labels) if l == "morven"]

    ood_texts = load_ood_probes(n=30)

    lora_cfg = LoRAConfig(num_labels=4)
    learners = [FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                          min_events_to_train=10, device=args.training_device,
                          local_epochs=args.local_epochs)
                for _ in range(args.n_silos)]
    bank_kwargs = dict(pca_components=50, dbscan_eps=0.30, dbscan_min_samples=5)
    silo_banks = [PrototypeBank(**bank_kwargs) for _ in range(args.n_silos)]
    global_bank = PrototypeBank(**bank_kwargs)

    cursors = [0] * args.n_silos
    total_revealed = [0] * args.n_silos
    morven_cursor = 0
    global_w = None
    snap_rounds = [10, 12, 15, 18, 20]
    curve = []

    print(f"\n{'='*60}\n  C5 -- untrained OOD absorption control\n{'='*60}\n")

    for r in range(args.n_rounds):
        rnd = r + 1
        new_this_round = []
        for i in range(args.n_silos):
            n_rev = schedules[i][r]
            new_ev = train_pools[i][cursors[i]: cursors[i] + n_rev]
            cursors[i] += n_rev
            total_revealed[i] += len(new_ev)

            if rnd >= args.injection_round and i == 0 and morven_cursor < len(morven_pool):
                m_end = min(morven_cursor + args.injection_per_round, len(morven_pool))
                batch = morven_pool[morven_cursor:m_end]
                morven_cursor = m_end
                new_ev = list(new_ev) + [{**rec, "label": "unknown", "gt_disease": "unknown"} for rec in batch]
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
            probe_embs = _extract_cls(learners[-1], probe_texts)
            ood_embs = _extract_cls(learners[-1], ood_texts)
            learners[-1].release()

            probe_preds = global_bank.classify(probe_embs)
            ood_preds = global_bank.classify(ood_embs)

            morven_recall = sum(1 for i in morven_idx if "unknown" in probe_preds[i] or "morven" in probe_preds[i]) / max(len(morven_idx), 1)
            ood_absorption = sum(1 for p in ood_preds if "unknown" in p or "morven" in p) / max(len(ood_preds), 1)

            curve.append({"round": rnd, "morven_recall": morven_recall, "ood_absorption_rate": ood_absorption,
                         "proto_names": list(global_bank.names())})
            print(f"  R{rnd:02d}  morven_recall={morven_recall:.3f}  ood_absorption={ood_absorption:.3f}  "
                 f"protos={global_bank.names()}")

    final = curve[-1] if curve else {}
    final_ood = final.get("ood_absorption_rate", float("nan"))
    final_morven = final.get("morven_recall", float("nan"))
    verdict = (
        f"C5 FAILS / catch-all contamination found: {final_ood:.1%} of genuinely out-of-distribution "
        f"(IMDB review) probes get classified into the 'unknown'/morven bucket by the final round, "
        f"alongside {final_morven:.1%} real Morven recall. The 'unknown' bucket is not specific to Morven."
        if final_ood > 0.15 else
        f"C5 PASSES: only {final_ood:.1%} of out-of-distribution probes get absorbed into 'unknown', "
        f"vs {final_morven:.1%} real Morven recall -- the bucket appears reasonably specific to the "
        "actual novel disease, not a generic catch-all."
    )
    print(f"\nVerdict: {verdict}")

    summary = {"control": "falsification.md C5 -- untrained OOD absorption (disease-text domain)",
              "curve": curve, "final_ood_absorption": final_ood, "final_morven_recall": final_morven,
              "verdict": verdict}
    out_path = OUT_DIR / f"c5_ood_absorption_seed{args.seed}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
