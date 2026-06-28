#!/usr/bin/env python3
"""
Falsification C4, PrototypeBank variant — does the known-disease-relabeled-
"unknown" false-positive (run_falsification_c4.py, silhouette-based: FAILED,
final silhouette 0.904 vs control 0.718) also occur under PrototypeBank
nearest-centroid evaluation (run_prototype.py's architecture)?

Same stress test: inject genuine Velarex text, relabeled "unknown" at training
time, same mechanism/timing/seed-offset as the real Morven injection. The
direct, symmetric check: among real (probe) Velarex events, what fraction does
the global PrototypeBank nearest-centroid-classify as "unknown" instead of
"velarex"? A high rate is the prototype-level analogue of the silhouette
false cluster -- the bank's "unknown" centroid absorbing genuine Velarex.

Forks run_prototype.py's round loop directly (not a black-box call) because
the per-probe predictions needed for this check aren't in its saved metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_unknown_disease as rud
from run_prototype import PrototypeConfig, _eval_probes, _fedavg, _load_probe_events

OUT_DIR = Path("results/falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_FAKE_NOVEL_DIST = [("velarex", "mild", 2.0), ("velarex", "moderate", 3.0), ("velarex", "severe", 1.0)]


def _build_fake_novel_pool(n: int, seed: int) -> list[dict]:
    lib = rud.FictionalPhraseLibrary(seed=seed + 77777)
    return lib.sample_pool(_FAKE_NOVEL_DIST, n, seed_offset=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default="falsification_c4_proto_fake_velarex_unknown")
    ap.add_argument("--training-device", default="cpu")
    args = ap.parse_args()

    rud._build_morven_pool = _build_fake_novel_pool  # same monkeypatch as the silhouette C4 test

    from fl.learner import FLLearner
    from fl.lora import LoRAConfig
    from fl.prototype_bank import PrototypeBank
    from run_unknown_disease import _make_schedule, _build_pools

    cfg = PrototypeConfig(
        n_silos=3, events_per_silo=160, n_rounds=20, seed=args.seed, schedule="gaussian",
        injection_round=10, injection_per_round=8, do_inject=True,
        training_device=args.training_device, run_name=args.run_name,
        results_dir=str(OUT_DIR),
    )
    out_dir = Path(cfg.results_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_events, probe_labels = _load_probe_events(seed=cfg.seed)
    velarex_idx = [i for i, l in enumerate(probe_labels) if l == "velarex"]

    train_pools, holdouts = _build_pools(
        n_silos=cfg.n_silos, events_per_silo=cfg.events_per_silo, holdout_frac=0.15, seed=cfg.seed,
    )
    schedules = [_make_schedule(cfg.schedule, len(train_pools[i]), cfg.n_rounds) for i in range(cfg.n_silos)]
    fake_novel_pool = rud._build_morven_pool(n=500, seed=cfg.seed)

    lora_cfg = LoRAConfig(num_labels=4)
    learners = [FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                          min_events_to_train=10, device=cfg.training_device,
                          local_epochs=args.local_epochs)
                for _ in range(cfg.n_silos)]
    bank_kwargs = dict(pca_components=cfg.pca_components, dbscan_eps=cfg.dbscan_eps,
                       dbscan_min_samples=cfg.dbscan_min_samples)
    silo_banks = [PrototypeBank(**bank_kwargs) for _ in range(cfg.n_silos)]
    global_bank = PrototypeBank(**bank_kwargs)

    cursors = [0] * cfg.n_silos
    total_revealed = [0] * cfg.n_silos
    fake_cursor = 0
    global_w = None
    velarex_as_unknown_curve = []

    print(f"\n{'='*60}\n  C4 PrototypeBank stress test -- Velarex relabeled 'unknown'\n{'='*60}\n")

    for r in range(cfg.n_rounds):
        rnd = r + 1
        new_this_round = []
        for i in range(cfg.n_silos):
            n_rev = schedules[i][r]
            new_ev = train_pools[i][cursors[i]: cursors[i] + n_rev]
            cursors[i] += n_rev
            total_revealed[i] += len(new_ev)

            if cfg.do_inject and rnd >= cfg.injection_round and i == 0 and fake_novel_pool:
                m_end = min(fake_cursor + cfg.injection_per_round, len(fake_novel_pool))
                batch = fake_novel_pool[fake_cursor:m_end]
                fake_cursor = m_end
                new_ev = list(new_ev) + [{**rec, "label": "unknown", "gt_disease": "unknown"} for rec in batch]
            new_this_round.append(list(new_ev))

        round_weights, train_sizes, silo_bank_dicts = [], [], []
        for i, learner in enumerate(learners):
            if global_w is not None:
                learner.set_weights(global_w)
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
            silo_bank_dicts.append(silo_banks[i].to_dict())
            learner.release()

        active_idx = [i for i, s in enumerate(train_sizes) if s >= cfg.fedavg_min_examples]
        if active_idx:
            global_w = _fedavg([round_weights[i] for i in active_idx], [train_sizes[i] for i in active_idx])
        active_banks = [silo_banks[i] for i in active_idx]
        active_weights = [train_sizes[i] for i in active_idx]
        if active_banks:
            global_bank = PrototypeBank.fedavg(active_banks, active_weights, **bank_kwargs)

        if global_w is not None and rnd in cfg.snap_rounds:
            learners[-1].set_weights(global_w)
            em = _eval_probes(learners[-1], probe_events, probe_labels, global_bank)
            preds = em["proto_preds"]
            velarex_as_unknown = sum(1 for i in velarex_idx if "unknown" in preds[i]) / max(len(velarex_idx), 1)
            velarex_as_unknown_curve.append({"round": rnd, "velarex_as_unknown_rate": velarex_as_unknown,
                                            "proto_names": list(global_bank.names())})
            print(f"  R{rnd:02d}: velarex probes classified 'unknown' = {velarex_as_unknown:.3f}  "
                 f"protos={global_bank.names()}")
            learners[-1].release()

    final = velarex_as_unknown_curve[-1] if velarex_as_unknown_curve else {}
    final_rate = final.get("velarex_as_unknown_rate", float("nan"))
    verdict = (
        f"C4 (PrototypeBank) FAILS: {final_rate:.1%} of genuine Velarex probes get nearest-"
        "centroid-classified as 'unknown' by the final round, purely because some Velarex "
        "text was relabeled 'unknown' during training. The bank's 'unknown' centroid partly "
        "absorbed genuine Velarex content -- same false-positive risk as the silhouette "
        "evaluation, via a different mechanism (centroid contamination, not a softmax "
        "decision-region artifact)."
        if final_rate > 0.15 else
        f"C4 (PrototypeBank) PASSES/borderline: only {final_rate:.1%} of genuine Velarex probes "
        "get classified 'unknown' -- PrototypeBank appears more robust to this specific "
        "false-positive than the softmax/silhouette pipeline."
    )
    print(f"\nVerdict: {verdict}")

    summary = {
        "control": "falsification.md C4 -- known-disease control injection (PrototypeBank variant)",
        "injected_content": "Velarex text, relabeled 'unknown' at injection",
        "velarex_as_unknown_curve": velarex_as_unknown_curve,
        "final_velarex_as_unknown_rate": final_rate,
        "verdict": verdict,
    }
    out_path = OUT_DIR / f"c4_prototype_known_disease_control_seed{args.seed}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
