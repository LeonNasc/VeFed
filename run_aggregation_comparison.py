#!/usr/bin/env python3
"""
Aggregation-algorithm comparison: FedAvg / FedProx / FedSGD / FedAdam, evaluated
via PrototypeBank (nearest-centroid on CLS embeddings; validated as robust to
the C7 false-positive in falsification.md, unlike the softmax/silhouette path).

Two scenarios, same as the falsification checks:
  morven   -- real novel-disease injection (Morven into silo_0 from round 10)
  c7       -- known-disease control: Velarex text relabeled "unknown" at
              injection (no genuine novel content) -- the false-positive
              stress test from run_falsification_c7*.py

Prototype centroids are aggregated via PrototypeBank.fedavg() (weighted mean
of centroid positions) in every condition -- this is unaffected by which
backbone aggregator (FedAvg/FedProx/FedSGD/FedAdam) is in use; what changes
across conditions is only how the BACKBONE weights (and therefore the CLS
embeddings the centroids are computed from) are aggregated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_unknown_disease as rud
from run_prototype import PrototypeConfig, _eval_probes, _load_probe_events
from fl.aggregation import fedavg, FedAdamServer

OUT_DIR = Path("results/aggregation_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_FAKE_NOVEL_DIST = [("velarex", "mild", 2.0), ("velarex", "moderate", 3.0), ("velarex", "severe", 1.0)]


def _build_fake_novel_pool(n: int, seed: int) -> list[dict]:
    """C7 stress test: genuine Velarex text, no novel content."""
    lib = rud.FictionalPhraseLibrary(seed=seed + 77777)
    return lib.sample_pool(_FAKE_NOVEL_DIST, n, seed_offset=0)


# Per-aggregator FLLearner knobs. Backbone aggregation formula is identical
# (plain fedavg) for fedavg/fedprox/fedsgd -- the distinguishing mechanism is
# local (see fl/aggregation.py docstring). fedadam uses a genuinely different
# server-side rule (FedAdamServer).
AGGREGATOR_LEARNER_KWARGS = {
    "fedavg":  dict(prox_mu=0.0,  optimizer="adamw", local_epochs=3, max_local_batches=None),
    "fedprox": dict(prox_mu=0.1,  optimizer="adamw", local_epochs=3, max_local_batches=None),
    "fedsgd":  dict(prox_mu=0.0,  optimizer="sgd",   local_epochs=1, max_local_batches=1),
    "fedadam": dict(prox_mu=0.0,  optimizer="adamw", local_epochs=3, max_local_batches=None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregator", required=True, choices=list(AGGREGATOR_LEARNER_KWARGS))
    ap.add_argument("--scenario",   required=True, choices=["morven", "c7"])
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--n-silos",    type=int, default=3)
    ap.add_argument("--events-per-silo", type=int, default=160)
    ap.add_argument("--n-rounds",   type=int, default=20)
    ap.add_argument("--injection-round", type=int, default=10)
    ap.add_argument("--injection-per-round", type=int, default=8)
    ap.add_argument("--prox-mu",    type=float, default=None, help="override default prox_mu for fedprox")
    ap.add_argument("--server-lr",  type=float, default=1.0, help="FedAdam server learning rate")
    ap.add_argument("--lr",         type=float, default=None, help="override local learning rate (e.g. fedsgd sweep)")
    ap.add_argument("--local-epochs", type=int, default=None, help="override local_epochs for every aggregator")
    ap.add_argument("--training-device", default="cuda")
    ap.add_argument("--run-name",   default="")
    ap.add_argument("--results-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    if args.scenario == "c7":
        rud._build_morven_pool = _build_fake_novel_pool

    from fl.learner import FLLearner
    from fl.lora import LoRAConfig
    from fl.prototype_bank import PrototypeBank
    from run_unknown_disease import _make_schedule, _build_pools

    learner_kwargs = dict(AGGREGATOR_LEARNER_KWARGS[args.aggregator])
    if args.prox_mu is not None:
        learner_kwargs["prox_mu"] = args.prox_mu
    if args.local_epochs is not None:
        learner_kwargs["local_epochs"] = args.local_epochs
    if args.lr is not None:
        learner_kwargs["lr"] = args.lr

    cfg = PrototypeConfig(
        n_silos=args.n_silos, events_per_silo=args.events_per_silo, n_rounds=args.n_rounds,
        seed=args.seed, schedule="gaussian", injection_round=args.injection_round,
        injection_per_round=args.injection_per_round, do_inject=True,
        training_device=args.training_device,
        run_name=args.run_name or f"{args.aggregator}_{args.scenario}_seed{args.seed}",
        results_dir=args.results_dir,
    )
    out_dir = Path(cfg.results_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_events, probe_labels = _load_probe_events(seed=cfg.seed)
    velarex_idx = [i for i, l in enumerate(probe_labels) if l == "velarex"]
    morven_idx  = [i for i, l in enumerate(probe_labels) if l == "morven"]

    train_pools, holdouts = _build_pools(
        n_silos=cfg.n_silos, events_per_silo=cfg.events_per_silo, holdout_frac=0.15, seed=cfg.seed,
    )
    schedules = [_make_schedule(cfg.schedule, len(train_pools[i]), cfg.n_rounds) for i in range(cfg.n_silos)]
    novel_pool = rud._build_morven_pool(n=500, seed=cfg.seed)

    lora_cfg = LoRAConfig(num_labels=4)
    learners = [
        FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                  min_events_to_train=10, device=cfg.training_device, **learner_kwargs)
        for _ in range(cfg.n_silos)
    ]
    bank_kwargs = dict(pca_components=cfg.pca_components, dbscan_eps=cfg.dbscan_eps,
                       dbscan_min_samples=cfg.dbscan_min_samples)
    silo_banks  = [PrototypeBank(**bank_kwargs) for _ in range(cfg.n_silos)]
    global_bank = PrototypeBank(**bank_kwargs)

    server_adam = FedAdamServer(server_lr=args.server_lr) if args.aggregator == "fedadam" else None

    cursors        = [0] * cfg.n_silos
    total_revealed = [0] * cfg.n_silos
    novel_cursor   = 0
    global_w: list | None = None
    round_metrics: list[dict] = []

    print(f"\n{'='*60}\n  Aggregation comparison: {args.aggregator} / {args.scenario}\n"
         f"  silos={cfg.n_silos}  rounds={cfg.n_rounds}  learner_kwargs={learner_kwargs}\n{'='*60}\n")

    for r in range(cfg.n_rounds):
        rnd = r + 1
        new_this_round = []
        for i in range(cfg.n_silos):
            n_rev = schedules[i][r]
            new_ev = train_pools[i][cursors[i]: cursors[i] + n_rev]
            cursors[i] += n_rev
            total_revealed[i] += len(new_ev)

            if cfg.do_inject and rnd >= cfg.injection_round and i == 0 and novel_pool:
                m_end = min(novel_cursor + cfg.injection_per_round, len(novel_pool))
                batch = novel_pool[novel_cursor:m_end]
                novel_cursor = m_end
                new_ev = list(new_ev) + [{**rec, "label": "unknown", "gt_disease": "unknown"} for rec in batch]
            new_this_round.append(list(new_ev))

        round_weights, train_sizes = [], []
        for i, learner in enumerate(learners):
            if global_w is not None:
                learner.set_weights(global_w)   # also anchors FedProx reference
            new_ev = new_this_round[i]
            if total_revealed[i] >= cfg.min_events_to_train and new_ev:
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

        active_idx = [i for i, s in enumerate(train_sizes) if s >= cfg.fedavg_min_examples]
        if active_idx:
            w_list = [round_weights[i] for i in active_idx]
            n_list = [train_sizes[i]   for i in active_idx]
            if args.aggregator == "fedadam":
                global_w = (server_adam.step(global_w, w_list, n_list)
                           if global_w is not None else fedavg(w_list, n_list))
            else:
                global_w = fedavg(w_list, n_list)
            global_bank = PrototypeBank.fedavg(
                [silo_banks[i] for i in active_idx], n_list, **bank_kwargs)

        probe_metrics: dict = {}
        if global_w is not None and rnd in cfg.snap_rounds:
            learners[-1].set_weights(global_w)
            em = _eval_probes(learners[-1], probe_events, probe_labels, global_bank)
            preds = em["proto_preds"]
            velarex_as_unknown = sum(1 for i in velarex_idx if "unknown" in preds[i]) / max(len(velarex_idx), 1)
            morven_as_unknown   = sum(1 for i in morven_idx  if "unknown" in preds[i]) / max(len(morven_idx), 1)
            probe_metrics = {
                "proto_names": list(global_bank.names()),
                "velarex_as_unknown_rate": velarex_as_unknown,
                "morven_as_unknown_rate":  morven_as_unknown,
                "n_unknown_clusters": em["n_unknown_clusters"],
            }
            print(f"  R{rnd:02d}: velarex_as_unk={velarex_as_unknown:.3f}  morven_as_unk={morven_as_unknown:.3f}  "
                 f"protos={global_bank.names()}")
            learners[-1].release()

        round_metrics.append({"round": rnd, "train_sizes": train_sizes, **probe_metrics})

    snap_metrics = [m for m in round_metrics if "proto_names" in m]
    summary = {
        "aggregator": args.aggregator, "scenario": args.scenario, "seed": args.seed,
        "learner_kwargs": learner_kwargs,
        "round_metrics": round_metrics,
        "final_velarex_as_unknown_rate": snap_metrics[-1]["velarex_as_unknown_rate"] if snap_metrics else float("nan"),
        "final_morven_as_unknown_rate":  snap_metrics[-1]["morven_as_unknown_rate"] if snap_metrics else float("nan"),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
