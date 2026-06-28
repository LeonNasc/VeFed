#!/usr/bin/env python3
"""
Diffusion-latency test — the literal content of the abstract's point (iii):
"cross-institutional knowledge diffusion for early detection of novel
conditions on unseen silos."

Setup: 4 silos train on known diseases (Velarex + Sornathis) throughout.
Silo_3 ("the new region") has ZERO Morven exposure until round
--late-injection-round (default 15), at which point it gets its own local
Morven outbreak. Three conditions vary what the rest of the federation knew
beforehand:

  pre_exposed -- silo_0 was injected with real Morven from round 10 onward
                 (standard setup). By round 15, the federation already
                 "knows" Morven before silo_3 ever sees a case.
  naive       -- same federation (4 silos, FedAvg every round), but NOBODY
                 saw Morven before round 15 -- silo_3's outbreak is the
                 federation's first encounter with it.
  isolated    -- silo_3 trains completely alone, no FedAvg, no other silos
                 exist. The absolute floor: learns Morven (if at all) with
                 zero outside help.

Measured per condition: silo_3's OWN local model (its weights right after
its own local update, before that round's FedAvg -- i.e. exactly what it
knows on its own) classified via PrototypeBank against held-out Morven
probes. The round at which this crosses a detection threshold, compared
across conditions, IS the diffusion-latency number the abstract needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_unknown_disease import _build_pools, _build_morven_pool, _make_schedule, generate_fictional_probe_events
from run_prototype import _extract_cls
from fl.aggregation import fedavg
from fl.prototype_bank import PrototypeBank

OUT_DIR = Path("results/diffusion_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["pre_exposed", "naive", "isolated"])
    ap.add_argument("--n-silos", type=int, default=4)
    ap.add_argument("--events-per-silo", type=int, default=160)
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--early-injection-round", type=int, default=10, help="silo_0's injection round (pre_exposed only)")
    ap.add_argument("--early-injection-per-round", type=int, default=8)
    ap.add_argument("--late-injection-round", type=int, default=15, help="silo_3's (the new silo's) injection round")
    ap.add_argument("--late-injection-per-round", type=int, default=8)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--training-device", default="cuda")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    from fl.learner import FLLearner
    from fl.lora import LoRAConfig

    run_name = args.run_name or f"diffusion_{args.condition}_seed{args.seed}"
    out_dir = OUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    new_silo = args.n_silos - 1   # silo_3 when n_silos=4

    if args.condition == "isolated":
        n_active_silos = 1   # only the new silo exists; nothing to federate with
    else:
        n_active_silos = args.n_silos

    train_pools, holdouts = _build_pools(
        n_silos=n_active_silos, events_per_silo=args.events_per_silo, holdout_frac=0.15, seed=args.seed,
    )
    schedules = [_make_schedule("gaussian", len(train_pools[i]), args.n_rounds) for i in range(n_active_silos)]
    morven_pool = _build_morven_pool(n=500, seed=args.seed)

    probe_events = generate_fictional_probe_events(n_per_band=12, seed=999)
    probe_labels = [ev.ground_truth.split("/")[0] for ev in probe_events]

    lora_cfg = LoRAConfig(num_labels=4)
    learners = [
        FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                  min_events_to_train=10, device=args.training_device,
                  local_epochs=args.local_epochs)
        for _ in range(n_active_silos)
    ]
    silo_banks  = [PrototypeBank(pca_components=50, dbscan_eps=0.30, dbscan_min_samples=5)
                   for _ in range(n_active_silos)]
    global_bank = PrototypeBank(pca_components=50, dbscan_eps=0.30, dbscan_min_samples=5)

    new_silo_idx = 0 if args.condition == "isolated" else new_silo
    early_silo_idx = 0   # silo_0 -- only relevant for pre_exposed

    cursors = [0] * n_active_silos
    morven_cursor_early = 0
    morven_cursor_late = 0
    global_w = None
    round_metrics = []
    snap_rounds = sorted(set(range(args.late_injection_round, args.n_rounds + 1)) |
                        {2, 5, 8, args.late_injection_round - 1})

    print(f"\n{'='*60}\n  Diffusion test -- condition={args.condition}\n"
         f"  new silo exposed from round {args.late_injection_round}\n{'='*60}\n")

    for r in range(args.n_rounds):
        rnd = r + 1
        new_this_round = []
        for i in range(n_active_silos):
            n_rev = schedules[i][r]
            new_ev = list(train_pools[i][cursors[i]: cursors[i] + n_rev])
            cursors[i] += n_rev

            if args.condition == "pre_exposed" and i == early_silo_idx and rnd >= args.early_injection_round and morven_pool:
                end = min(morven_cursor_early + args.early_injection_per_round, len(morven_pool))
                batch = morven_pool[morven_cursor_early:end]
                morven_cursor_early = end
                new_ev += [{**rec, "label": "unknown", "gt_disease": "unknown"} for rec in batch]

            if i == new_silo_idx and rnd >= args.late_injection_round and morven_pool:
                end = min(morven_cursor_late + args.late_injection_per_round, len(morven_pool))
                batch = morven_pool[morven_cursor_late:end]
                morven_cursor_late = end
                new_ev += [{**rec, "label": "unknown", "gt_disease": "unknown"} for rec in batch]

            new_this_round.append(new_ev)

        round_weights, train_sizes = [], []
        for i, learner in enumerate(learners):
            if global_w is not None and args.condition != "isolated":
                learner.set_weights(global_w)
            new_ev = new_this_round[i]
            if new_ev:
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

        # New silo's OWN model -- its weights right now, before any fedavg this round.
        new_silo_weights = round_weights[new_silo_idx]

        for learner in learners:
            learner.release()

        if args.condition != "isolated":
            active_idx = [i for i, s in enumerate(train_sizes) if s >= 5]
            if active_idx:
                global_w = fedavg([round_weights[i] for i in active_idx], [train_sizes[i] for i in active_idx])
                global_bank = PrototypeBank.fedavg(
                    [silo_banks[i] for i in active_idx], [train_sizes[i] for i in active_idx],
                    pca_components=50, dbscan_eps=0.30, dbscan_min_samples=5)
        else:
            global_bank = silo_banks[0]

        probe_metrics = {}
        if rnd in snap_rounds:
            learners[new_silo_idx].set_weights(new_silo_weights)
            probe_texts = []
            for ev in probe_events:
                turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
                probe_texts.append(turns[-1] if turns else "")
            embs = _extract_cls(learners[new_silo_idx], probe_texts)   # no label filtering -- 1:1 with probe_events
            morven_idx = [i for i, l in enumerate(probe_labels) if l == "morven"]
            bank_for_new_silo = silo_banks[new_silo_idx]
            preds = bank_for_new_silo.classify(embs)
            morven_recall = (sum(1 for i in morven_idx if "unknown" in preds[i]) / len(morven_idx)
                            if morven_idx else float("nan"))
            probe_metrics = {"new_silo_morven_recall": morven_recall, "new_silo_proto_names": list(bank_for_new_silo.names())}
            print(f"  R{rnd:02d}: new_silo_morven_recall={morven_recall:.3f}  protos={bank_for_new_silo.names()}")
            learners[new_silo_idx].release()

        round_metrics.append({"round": rnd, "train_sizes": train_sizes, **probe_metrics})

    snap_metrics = [m for m in round_metrics if "new_silo_morven_recall" in m]
    detect_round = next((m["round"] for m in snap_metrics if m["new_silo_morven_recall"] >= 0.5), None)
    summary = {
        "condition": args.condition, "late_injection_round": args.late_injection_round,
        "seed": args.seed, "round_metrics": round_metrics,
        "detect_round_recall_0.5": detect_round,
        "final_morven_recall": snap_metrics[-1]["new_silo_morven_recall"] if snap_metrics else float("nan"),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetect round (recall>=0.5): {detect_round}  final recall: {summary['final_morven_recall']:.3f}")
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
