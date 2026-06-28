#!/usr/bin/env python3
"""
Prototype-bank unknown-disease detection on REAL MIMIC-IV-ED clinical text.

Combines two fixes identified after the fixed-class FLLearner softmax evaluation
was found to collapse under class imbalance (see CASE_SUMMARY_SWEEP_REPORT.md
section 8 discussion):

  1. Evaluation: PrototypeBank (fl/prototype_bank.py) nearest-centroid
     classification on CLS embeddings, instead of trusting argmax(softmax head).
     Mirrors run_prototype.py's architecture exactly (backbone still trained
     with a softmax head for representation learning; the head's own argmax
     decision is not used for the reported metrics).

  2. Text generation: 3 mechanisms over the same real MIMIC row (chiefcomplaint
     + vitals), via --text-type:
       raw    -- bare chiefcomplaint field (the original, terse mechanism)
       phrase -- chiefcomplaint + real vitals, deterministic template
       ollama -- phi3:mini writes a naturalistic statement from the same
                 real chiefcomplaint + vitals (not a canned phrase bank)

Data: build the pool first with scripts/preprocess_mimicel.py --cohorts ...
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

from run_unknown_disease import _make_probe_event, _make_schedule, _project_umap
from simulation.mimic_text import mimic_raw, mimic_phrase_library, mimic_guided_ollama

OUT_DIR = Path("results/mimic_prototype")


def load_pool_rows(csv_path: str) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("chiefcomplaint") or "").strip():
                by_group[row["diagnosis_group"]].append(row)
    return by_group


def make_text_fn(text_type: str, seed: int):
    """Return fn(row) -> str, with caching by stay_id for the LLM mechanism."""
    if text_type == "raw":
        return lambda row: mimic_raw(row)
    if text_type == "phrase":
        return lambda row: mimic_phrase_library(row)
    if text_type == "ollama":
        from simulation.patient_llm import PatientLLMClient
        from simulation.symptom_language import Personality
        client = PatientLLMClient()
        rng = random.Random(seed)
        cache: dict[str, str] = {}

        def fn(row):
            sid = row.get("stay_id", "")
            if sid in cache:
                return cache[sid]
            personality = rng.choice(list(Personality))
            text = mimic_guided_ollama(row, client, personality, rng)
            cache[sid] = text
            return text
        return fn
    raise ValueError(f"unknown text_type {text_type!r}")


def _silhouette_unknown(coords: np.ndarray, labels: list[str]) -> float:
    from sklearn.metrics import silhouette_samples
    group = np.array([1 if l == "unknown" else 0 for l in labels])
    if group.sum() < 2 or (group == 0).sum() < 2:
        return float("nan")
    try:
        return float(np.mean(silhouette_samples(coords, group)[group == 1]))
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-csv",          required=True)
    ap.add_argument("--novel-group",       required=True)
    ap.add_argument("--text-type",         default="raw", choices=["raw", "phrase", "ollama"])
    ap.add_argument("--schedule",          default="gaussian")
    ap.add_argument("--n-silos",           type=int, default=3)
    ap.add_argument("--n-rounds",          type=int, default=20)
    ap.add_argument("--events-per-silo",   type=int, default=200)
    ap.add_argument("--injection-round",   type=int, default=10)
    ap.add_argument("--injection-per-round", type=int, default=16)
    ap.add_argument("--no-injection",      action="store_true")
    ap.add_argument("--holdout-frac",      type=float, default=0.15)
    ap.add_argument("--seed",              type=int, default=42)
    ap.add_argument("--results-dir",       default=str(OUT_DIR))
    ap.add_argument("--run-name",          default="")
    ap.add_argument("--training-device",   default="cuda")
    ap.add_argument("--probe-n",           type=int, default=30)
    ap.add_argument("--replay-buffer-size", type=int, default=4096)
    ap.add_argument("--pca-components",    type=int, default=50)
    ap.add_argument("--dbscan-eps",        type=float, default=0.30)
    ap.add_argument("--dbscan-min-samples", type=int, default=5)
    args = ap.parse_args()

    do_inject = not args.no_injection
    rng = random.Random(args.seed)
    text_fn = make_text_fn(args.text_type, args.seed)

    by_group = load_pool_rows(args.pool_csv)
    print("Cohort sizes:", {g: len(v) for g, v in by_group.items()})
    if args.novel_group not in by_group:
        raise ValueError(f"--novel-group {args.novel_group!r} not in pool; available: {list(by_group)}")
    label_map = {g: ("unknown" if g == args.novel_group else g) for g in by_group}

    # Held-out probe rows (fixed; independent of training pools).
    probe_rows, remaining = {}, {}
    for g, rows in by_group.items():
        rows = rows[:]
        rng.shuffle(rows)
        n_probe = min(args.probe_n, max(1, len(rows) // 4))
        probe_rows[g] = rows[:n_probe]
        remaining[g]  = rows[n_probe:] or rows

    print(f"Generating probe text ({args.text_type})...")
    probe_events = []
    for g, rows in probe_rows.items():
        lbl = label_map[g]
        for row in rows:
            text = text_fn(row)
            if text:
                probe_events.append(_make_probe_event(lbl, text))
    probe_labels = [ev.ground_truth for ev in probe_events]

    # Known/novel pools kept as RAW ROWS — text is generated lazily, only for
    # rows actually drawn in a round (critical for --text-type ollama: eagerly
    # generating text for the whole pool, e.g. 5,765 known rows, would mean
    # thousands of unnecessary LLM calls before training even starts).
    known_groups = [g for g in by_group if g != args.novel_group]
    pooled_known_rows = [(row, label_map[g]) for g in known_groups for row in remaining[g]]
    rng.shuffle(pooled_known_rows)
    train_all_rows = pooled_known_rows  # no holdout split — proto_acc_novel (probes) is the metric
    train_pools = [train_all_rows[i::args.n_silos] for i in range(args.n_silos)]

    novel_pool_rows = [(row, "unknown") for row in remaining.get(args.novel_group, [])]
    print(f"Known pool sizes per silo: {[len(p) for p in train_pools]}  {args.novel_group} pool: {len(novel_pool_rows)}")

    _text_cache: dict[int, str] = {}

    def row_to_record(row: dict, label: str) -> dict | None:
        key = id(row)
        text = _text_cache.get(key)
        if text is None:
            text = text_fn(row)
            _text_cache[key] = text
        if not text:
            return None
        return {"text": text, "label": label, "gt_disease": label}

    schedule = _make_schedule(args.schedule, args.events_per_silo, args.n_rounds)

    from fl.lora import LoRAConfig
    from fl.learner import FLLearner
    from fl.train import _fedavg
    from fl.prototype_bank import PrototypeBank

    lora_cfg = LoRAConfig(model_name_or_path="distilbert-base-uncased", num_labels=None,
                          rank=8, lora_alpha=16.0, lora_dropout=0.05)
    learners = [
        FLLearner(lora_config=lora_cfg, label_space="disease", min_events_to_train=10,
                  local_epochs=3, batch_size=8, lr=1e-4, train_sample_cap=180,
                  replay_buffer_size=args.replay_buffer_size, device=args.training_device)
        for _ in range(args.n_silos)
    ]
    bank_kwargs = dict(pca_components=args.pca_components, dbscan_eps=args.dbscan_eps,
                       dbscan_min_samples=args.dbscan_min_samples)
    silo_banks  = [PrototypeBank(**bank_kwargs) for _ in range(args.n_silos)]
    global_bank = PrototypeBank(**bank_kwargs)

    run_name = args.run_name or f"mimic_proto_{args.text_type}_{int(time.time())}"
    out_dir  = Path(args.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cursors      = [0] * args.n_silos
    novel_cursor = 0
    global_w: list | None = None
    round_metrics: list[dict] = []
    snap_rounds = [2, 5, 8, 10, 12, 15, 20]
    t_start = time.time()

    print(f"\n{'═'*60}\n  MIMIC Prototype-Bank Experiment — {args.text_type} text, {args.schedule} schedule\n"
          f"  silos={args.n_silos}  rounds={args.n_rounds}  novel={args.novel_group}\n"
          f"  inject={'YES (round '+str(args.injection_round)+')' if do_inject else 'NO'}\n{'═'*60}\n")

    for r in range(args.n_rounds):
        rnd = r + 1
        new_this_round: list[list[dict]] = []
        for i in range(args.n_silos):
            n_rev = schedule[r]
            pool  = train_pools[i]
            row_batch = [pool[(cursors[i] + k) % len(pool)] for k in range(n_rev)] if pool else []
            cursors[i] += n_rev

            if do_inject and rnd >= args.injection_round and novel_pool_rows and i == 0:
                n_batch = [novel_pool_rows[(novel_cursor + k) % len(novel_pool_rows)]
                          for k in range(args.injection_per_round)]
                novel_cursor += args.injection_per_round
                row_batch = row_batch + n_batch

            batch = [rec for row, lbl in row_batch if (rec := row_to_record(row, lbl)) is not None]
            new_this_round.append(batch)

        round_weights, train_sizes = [], []
        for i, learner in enumerate(learners):
            if global_w is not None:
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
            learner.release()

        active_idx = [i for i, s in enumerate(train_sizes) if s >= 4]
        if active_idx:
            global_w = _fedavg([round_weights[i] for i in active_idx],
                               [train_sizes[i]   for i in active_idx])
            global_bank = PrototypeBank.fedavg(
                [silo_banks[i] for i in active_idx], [train_sizes[i] for i in active_idx], **bank_kwargs)

        probe_metrics: dict = {}
        if global_w is not None and rnd in snap_rounds:
            learners[-1].set_weights(global_w)
            texts = [ev.conversation[0]["text"] for ev in probe_events]
            embs, aligned_labels = learners[-1].extract_embeddings(
                [{"text": t, "label": l, "gt_disease": l} for t, l in zip(texts, probe_labels)]
            )
            if len(aligned_labels) != len(probe_labels):
                print(f"    [proto] WARNING R{rnd}: {len(probe_labels) - len(aligned_labels)} "
                     f"probe(s) dropped (empty text or unmapped label) — using aligned labels")
            proto_preds = global_bank.classify(embs)
            novel_idx = [i for i, l in enumerate(aligned_labels) if l == "unknown"]
            proto_acc_novel = (
                sum(proto_preds[i] == "unknown" for i in novel_idx) / max(len(novel_idx), 1)
                if novel_idx else float("nan")
            )
            proto_acc_all = sum(p == t for p, t in zip(proto_preds, aligned_labels)) / max(len(aligned_labels), 1)

            unknown_names = {n for n in global_bank.names() if "unknown" in n}
            unk_mask = np.array([p in unknown_names for p in proto_preds])
            n_unk_clusters = (
                global_bank.dbscan_cluster_count(embs[unk_mask])
                if unk_mask.sum() >= 2 * args.dbscan_min_samples else (1 if unk_mask.sum() > 0 else 0)
            )

            coords = _project_umap(embs, seed=args.seed)
            sil = _silhouette_unknown(coords, aligned_labels)

            probe_metrics = {
                "proto_acc_all": proto_acc_all, "proto_acc_novel": proto_acc_novel,
                "n_unknown_clusters": n_unk_clusters, "n_named_protos": len(global_bank.names()),
                "proto_names": list(global_bank.names()), "silhouette": sil,
            }
            np.savez_compressed(out_dir / f"embs_r{rnd:02d}.npz", embs=embs)
            learners[-1].release()
            print(f"    [proto] R{rnd}: acc_novel={proto_acc_novel:.3f} sil={sil:.3f} "
                 f"unk_clusters={n_unk_clusters} protos={global_bank.names()}")

        round_metrics.append({"round": rnd, "train_sizes": train_sizes,
                             "novel_injected": do_inject and rnd >= args.injection_round, **probe_metrics})
        print(f"  R{rnd:02d}  trained={train_sizes}" + (f"  [{args.novel_group} injected]"
             if do_inject and rnd >= args.injection_round else ""))

    elapsed = time.time() - t_start
    with (out_dir / "round_metrics.json").open("w") as f:
        json.dump(round_metrics, f, indent=2)

    final_metrics = [m for m in round_metrics if "proto_acc_novel" in m]
    summary = {
        "mode": "mimic_prototype", "text_type": args.text_type, "schedule": args.schedule,
        "novel_group": args.novel_group, "known_groups": known_groups,
        "cohort_sizes": {g: len(v) for g, v in by_group.items()},
        "wall_seconds": elapsed,
        "final_proto_acc_novel": final_metrics[-1]["proto_acc_novel"] if final_metrics else float("nan"),
        "final_silhouette": final_metrics[-1]["silhouette"] if final_metrics else float("nan"),
        "silhouette_curve": [{"round": m["round"], "silhouette": m["silhouette"]} for m in final_metrics],
        "proto_acc_novel_curve": [{"round": m["round"], "proto_acc_novel": m["proto_acc_novel"]} for m in final_metrics],
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Final proto_acc_novel={summary['final_proto_acc_novel']:.3f}  wall={elapsed:.0f}s")
    print(f"  Results written to {out_dir}/")


if __name__ == "__main__":
    main()
