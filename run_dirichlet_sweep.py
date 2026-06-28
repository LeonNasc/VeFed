#!/usr/bin/env python3
"""
Dirichlet-alpha mini-sweep (falsification.md / ablations.md A5 -- never
previously run as an actual sweep; the only prior "evidence" cited for it
was a single n=1, non-Dirichlet, disjoint-disease-subset run).

Standard FL heterogeneity literature parameterizes label-distribution non-IID
via Dirichlet(alpha) over the class list, independently per silo: alpha -> 0
gives each silo a near-single-class draw (maximally non-IID); alpha -> inf
gives every silo the same balanced mix (IID). This script maps FL gain
(federated diag_acc - isolated diag_acc) across that axis directly, using
the same fast in-memory FictionalPhraseLibrary pipeline as the rest of this
session's falsification controls (run_falsification_c2.py /
run_homogenization_check.py) rather than the heavier Ollama+wandb
run_federated_training pipeline, so a multi-point x multi-seed grid is
actually tractable before the deadline.

For each alpha and seed: draw one Dirichlet(alpha, alpha) vector per silo
over {velarex, sornathis}, build that silo's training pool from it, train
federated vs isolated for n_rounds, and evaluate diag_acc on a fixed,
balanced held-out probe set (not used in any silo's training pool).
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


def _dirichlet_distribution(velarex_frac: float) -> list[tuple[str, str, float]]:
    sornathis_frac = 1.0 - velarex_frac
    dist = []
    for sev, w in _SEV_WEIGHTS.items():
        dist.append(("velarex", sev, w * velarex_frac))
        dist.append(("sornathis", sev, w * sornathis_frac))
    return dist


def _build_dirichlet_pools(n_silos: int, events_per_silo: int, holdout_frac: float,
                           alpha: float, seed: int):
    lib = rud.FictionalPhraseLibrary(seed=seed)
    rng_np = np.random.default_rng(seed)
    import random
    rng = random.Random(seed)
    train_pools, holdouts, silo_fracs = [], [], []
    for i in range(n_silos):
        velarex_frac = float(rng_np.dirichlet([alpha, alpha])[0])
        silo_fracs.append(velarex_frac)
        dist = _dirichlet_distribution(velarex_frac)
        records = lib.sample_pool(dist, events_per_silo, seed_offset=i * 1000)
        rng.shuffle(records)
        n_hold = max(1, int(len(records) * holdout_frac))
        holdouts.append(records[:n_hold])
        train_pools.append(records[n_hold:])
    return train_pools, holdouts, silo_fracs


def _build_fixed_probe_pool(n_per_class: int, seed: int = 999) -> list[dict]:
    lib = rud.FictionalPhraseLibrary(seed=seed)
    dist = _dirichlet_distribution(0.5)  # balanced, held out, never used in training
    return lib.sample_pool(dist, n_per_class * 2, seed_offset=777777)


def run_condition(federated: bool, alpha: float, seed: int, n_rounds: int, n_silos: int,
                  events_per_silo: int, local_epochs: int, training_device: str) -> dict:
    from fl.learner import FLLearner
    from fl.lora import LoRAConfig

    train_pools, holdouts, silo_fracs = _build_dirichlet_pools(
        n_silos, events_per_silo, 0.15, alpha, seed)
    schedules = [rud._make_schedule("gaussian", len(train_pools[i]), n_rounds) for i in range(n_silos)]
    probe_pool = _build_fixed_probe_pool(40, seed=999)

    lora_cfg = LoRAConfig(num_labels=4)
    learners = [FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                          min_events_to_train=10, device=training_device,
                          local_epochs=local_epochs)
                for _ in range(n_silos)]

    cursors = [0] * n_silos
    total_revealed = [0] * n_silos
    silo_weights = [None] * n_silos

    tag = "FEDERATED" if federated else "ISOLATED"
    print(f"\n{'='*60}\n  Dirichlet sweep alpha={alpha} condition={tag} seed={seed}\n"
         f"  silo velarex fractions: {[round(f, 2) for f in silo_fracs]}\n{'='*60}\n")

    for r in range(n_rounds):
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

    if federated:
        learners[-1].set_weights(silo_weights[-1])
        diag_acc = learners[-1].evaluate(probe_pool)["diag_acc"]
        learners[-1].release()
    else:
        accs = []
        for i, learner in enumerate(learners):
            if silo_weights[i] is None:
                continue
            learner.set_weights(silo_weights[i])
            accs.append(learner.evaluate(probe_pool)["diag_acc"])
            learner.release()
        diag_acc = float(np.mean(accs))

    print(f"  final diag_acc = {diag_acc:.3f}")
    return {"federated": federated, "alpha": alpha, "seed": seed,
           "silo_velarex_fracs": silo_fracs, "diag_acc": diag_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 5.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--n-silos", type=int, default=3)
    ap.add_argument("--events-per-silo", type=int, default=160)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--training-device", default="cuda")
    args = ap.parse_args()

    results = {}
    for alpha in args.alphas:
        results[alpha] = {"federated": [], "isolated": []}
        for seed in args.seeds:
            for federated in [True, False]:
                key = "federated" if federated else "isolated"
                res = run_condition(federated, alpha, seed, args.n_rounds, args.n_silos,
                                    args.events_per_silo, args.local_epochs, args.training_device)
                results[alpha][key].append(res)

    print("\n\n=== SUMMARY: FL gain vs Dirichlet alpha ===")
    summary_rows = []
    for alpha in args.alphas:
        fed_accs = [r["diag_acc"] for r in results[alpha]["federated"]]
        iso_accs = [r["diag_acc"] for r in results[alpha]["isolated"]]
        gain = float(np.mean(fed_accs) - np.mean(iso_accs))
        test = exact_permutation_test(fed_accs, iso_accs)
        print(f"alpha={alpha:>4}  federated={np.mean(fed_accs):.3f}±{np.std(fed_accs):.3f}  "
             f"isolated={np.mean(iso_accs):.3f}±{np.std(iso_accs):.3f}  "
             f"FL_gain={gain:+.3f}  p={test['p_two_sided']:.2f}")
        summary_rows.append({"alpha": alpha, "federated_diag_acc": fed_accs, "isolated_diag_acc": iso_accs,
                            "fl_gain": gain, "exact_permutation_test": test})

    out_path = OUT_DIR / "dirichlet_alpha_sweep.json"
    with out_path.open("w") as f:
        json.dump({"alphas": args.alphas, "seeds": args.seeds, "rows": summary_rows,
                   "raw": {str(a): results[a] for a in args.alphas}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
