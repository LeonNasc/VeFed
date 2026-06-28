#!/usr/bin/env python3
"""
Unknown-disease detection experiment using REAL MIMIC-IV-ED clinical text.

Real-data analog of run_unknown_disease.py's fictional experiment. Cohort
sizes are determined by which 3 disease buckets scripts/preprocess_mimicel.py
was run with (see --cohorts there); whichever cohort is passed as
--novel-group here is injected from --injection-round onward, labelled
"unknown" -- playing Morven's role. Everything else is a known disease,
present from round 1.

Text is the real `chiefcomplaint` field from MIMIC-IV-ED triage (e.g.
"FEVER/TRAVEL", "Chest pain") -- not a synthetic phrase bank or LLM
paraphrase. This checks whether the embedding-separability finding from the
fictional-disease sweep (results/unknown_disease/sweep_*) holds on authentic
clinical text.

CAVEAT: if --novel-group has a small cohort (e.g. malaria, 19 stays), every
injection round resamples with replacement from the same small text pool --
treat results as a small-N pilot, not a robust estimate. pneumonia/sepsis/uti
cohorts (2,330-9,332 stays) don't have this problem.

Data: build the pool first, e.g.
    python scripts/preprocess_mimicel.py --csv <mimicel.csv> \\
        --out MIMIC/mimic_unknown_disease_pool_pneu_sepsis_uti.csv \\
        --cohorts pneumonia,sepsis,uti
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_unknown_disease import _make_probe_event, _make_schedule, _project_umap, _extract_embeddings


def load_pool(csv_path: str) -> dict[str, list[str]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("chiefcomplaint") or "").strip()
            if text:
                by_group[row["diagnosis_group"]].append(text)
    return by_group


def _silhouette_unknown(coords: np.ndarray, labels: list[str]) -> float:
    from sklearn.metrics import silhouette_samples
    group = np.array([1 if l == "unknown" else 0 for l in labels])
    if group.sum() < 2 or (group == 0).sum() < 2:
        return float("nan")
    try:
        scores = silhouette_samples(coords, group)
        return float(np.mean(scores[group == 1]))
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-csv",          default="MIMIC/mimic_unknown_disease_pool.csv")
    ap.add_argument("--schedule",          default="gaussian")
    ap.add_argument("--n-silos",           type=int, default=3)
    ap.add_argument("--n-rounds",          type=int, default=20)
    ap.add_argument("--events-per-silo",   type=int, default=200)
    ap.add_argument("--injection-round",   type=int, default=10)
    ap.add_argument("--injection-per-round", type=int, default=4)
    ap.add_argument("--replay-buffer-size", type=int, default=2048,
                    help="FLLearner in-memory training buffer cap per silo; raise this if "
                         "--events-per-silo x --n-rounds would exceed it (events would "
                         "silently age out of training otherwise).")
    ap.add_argument("--no-injection",      action="store_true")
    ap.add_argument("--holdout-frac",      type=float, default=0.15)
    ap.add_argument("--seed",              type=int, default=42)
    ap.add_argument("--results-dir",       default="results/unknown_disease")
    ap.add_argument("--run-name",          default="")
    ap.add_argument("--training-device",   default="cuda")
    ap.add_argument("--probe-n",           type=int, default=12)
    ap.add_argument("--novel-group",       default="malaria",
                    help="Cohort name (as written in --pool-csv's diagnosis_group column) "
                         "injected from --injection-round onward, labelled 'unknown'. "
                         "All other cohorts in the pool are known diseases from round 1.")
    args = ap.parse_args()

    do_inject = not args.no_injection
    rng = random.Random(args.seed)

    by_group = load_pool(args.pool_csv)
    print("Cohort sizes:", {g: len(v) for g, v in by_group.items()})
    if args.novel_group not in by_group:
        raise ValueError(f"--novel-group {args.novel_group!r} not found in pool; "
                         f"available cohorts: {list(by_group)}")
    label_map = {g: ("unknown" if g == args.novel_group else g) for g in by_group}

    # Held-out probe texts (fixed; independent of training pools).
    probe_texts, remaining = {}, {}
    for g, texts in by_group.items():
        texts = texts[:]
        rng.shuffle(texts)
        n_probe = min(args.probe_n, max(1, len(texts) // 4))
        probe_texts[g] = texts[:n_probe]
        remaining[g]   = texts[n_probe:] or texts  # reuse if too few to split

    probe_events = []
    for g, texts in probe_texts.items():
        lbl = label_map[g]
        probe_events.extend(_make_probe_event(lbl, t) for t in texts)
    probe_labels = [ev.ground_truth for ev in probe_events]

    # Known-disease training pools, split round-robin across silos.
    known_groups = [g for g in by_group if g != args.novel_group]
    pooled_known = [
        {"text": t, "label": label_map[g], "gt_disease": label_map[g]}
        for g in known_groups for t in remaining[g]
    ]
    rng.shuffle(pooled_known)
    n_hold     = max(1, int(len(pooled_known) * args.holdout_frac))
    holdout_all = pooled_known[:n_hold]
    train_all   = pooled_known[n_hold:]
    train_pools = [train_all[i::args.n_silos]   for i in range(args.n_silos)]
    holdouts    = [holdout_all[i::args.n_silos] for i in range(args.n_silos)]

    morven_pool = [
        {"text": t, "label": "unknown", "gt_disease": "unknown"}
        for t in remaining.get(args.novel_group, [])
    ]
    print(f"Known pool sizes per silo: {[len(p) for p in train_pools]}  "
         f"{args.novel_group} pool: {len(morven_pool)}")

    schedule = _make_schedule(args.schedule, args.events_per_silo, args.n_rounds)

    from fl.lora import LoRAConfig
    from fl.learner import FLLearner
    from fl.train import _fedavg

    lora_cfg = LoRAConfig(model_name_or_path="distilbert-base-uncased", num_labels=None,
                          rank=8, lora_alpha=16.0, lora_dropout=0.05)
    learners = [
        FLLearner(lora_config=lora_cfg, label_space="disease", min_events_to_train=10,
                  local_epochs=3, batch_size=8, lr=1e-4, train_sample_cap=180,
                  replay_buffer_size=args.replay_buffer_size,
                  device=args.training_device)
        for _ in range(args.n_silos)
    ]

    run_name = args.run_name or f"mimic_unk_{args.schedule}_{int(time.time())}"
    out_dir  = Path(args.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cursors       = [0] * args.n_silos
    morven_cursor = 0
    global_w: list | None = None
    round_metrics: list[dict]   = []
    silhouette_curve: list[dict] = []
    embedding_snapshots: dict[int, dict] = {}
    snap_rounds = [2, 5, 8, 10, 12, 15, 20]
    t_start = time.time()

    print(f"\n{'═'*60}\n  MIMIC Real-Data Unknown Disease Experiment — {args.schedule.upper()}\n"
          f"  silos={args.n_silos}  rounds={args.n_rounds}\n"
          f"  inject={'YES (round '+str(args.injection_round)+')' if do_inject else 'NO'}\n{'═'*60}\n")

    for r in range(args.n_rounds):
        rnd = r + 1
        new_this_round: list[list[dict]] = []
        for i in range(args.n_silos):
            n_rev = schedule[r]
            pool  = train_pools[i]
            batch = [pool[(cursors[i] + k) % len(pool)] for k in range(n_rev)] if pool else []
            cursors[i] += n_rev

            inject = do_inject and rnd >= args.injection_round and morven_pool and i == 0
            if inject:
                m_batch = [morven_pool[(morven_cursor + k) % len(morven_pool)]
                          for k in range(args.injection_per_round)]
                morven_cursor += args.injection_per_round
                batch = batch + m_batch
            new_this_round.append(batch)

        eval_metrics, round_weights, train_sizes, losses = [], [], [], []
        for i, learner in enumerate(learners):
            if global_w is not None:
                learner.set_weights(global_w)
            m = learner.evaluate(holdouts[i]) if holdouts[i] else {}
            eval_metrics.append(m)

            new_ev = new_this_round[i]
            if new_ev:
                n_trained, epoch_losses = learner.train(new_ev)
                train_sizes.append(n_trained)
                losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
            else:
                train_sizes.append(0)
                losses.append(float("nan"))

            round_weights.append(learner.get_weights())
            learner.release()

        active_idx = [i for i, s in enumerate(train_sizes) if s > 0 and s >= 4]
        if active_idx:
            global_w = _fedavg([round_weights[i] for i in active_idx],
                               [train_sizes[i] for i in active_idx])

        if rnd in snap_rounds and global_w is not None:
            learners[-1].set_weights(global_w)
            cls, logits = _extract_embeddings(learners[-1], probe_events)
            embedding_snapshots[rnd] = {"cls": cls, "logits": logits}
            coords = _project_umap(logits, seed=args.seed)
            sil    = _silhouette_unknown(coords, probe_labels)
            silhouette_curve.append({"round": rnd, "silhouette": sil})
            print(f"    [embed] round {rnd}  sil={sil:.3f}" if sil == sil else f"    [embed] round {rnd}  sil=n/a")
            learners[-1].release()

        agg_diag = float(np.mean([m.get("diag_acc", float("nan")) for m in eval_metrics]))
        n_events = sum(len(b) for b in new_this_round)
        print(f"  R{rnd:02d}  diag={agg_diag:.3f}  events={n_events}"
             + (f"  [{args.novel_group} injected]" if do_inject and rnd >= args.injection_round else ""))
        round_metrics.append({"round": rnd, "agg_diag_acc": agg_diag, "n_events": n_events})

    final_diag_list = []
    for i, learner in enumerate(learners):
        if global_w is not None:
            learner.set_weights(global_w)
        m = learner.evaluate(holdouts[i]) if holdouts[i] else {}
        learner.release()
        final_diag_list.append(m.get("diag_acc", float("nan")))
    final_diag = float(np.nanmean(final_diag_list))
    elapsed    = time.time() - t_start

    with (out_dir / "round_metrics.json").open("w") as f:
        json.dump(round_metrics, f, indent=2)
    with (out_dir / "silhouette.json").open("w") as f:
        json.dump(silhouette_curve, f, indent=2)
    for rnd, snap in embedding_snapshots.items():
        np.savez_compressed(out_dir / f"logits_r{rnd:02d}.npz", logits=snap["logits"], cls=snap["cls"])

    summary = {
        "mode": "mimic_real", "schedule": args.schedule,
        "n_silos": args.n_silos, "n_rounds": args.n_rounds,
        "injection_round": args.injection_round, "do_inject": do_inject,
        "novel_group": args.novel_group,
        "known_groups": known_groups,
        "cohort_sizes": {g: len(v) for g, v in by_group.items()},
        "final_diag_acc": final_diag, "wall_seconds": elapsed,
        "silhouette_curve": silhouette_curve,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Final holdout diag={final_diag:.3f}  wall={elapsed:.0f}s\n  Results written to {out_dir}/")


if __name__ == "__main__":
    main()
