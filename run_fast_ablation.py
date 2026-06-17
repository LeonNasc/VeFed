"""
Fast comprehensive ablation — templates then phrase bank, no Ollama.

Runs the full 5-schedule × 2-distribution (IID / non-IID) × N-seeds matrix
for each generator tier, then produces:
  - Summary statistics table  (mean ± std across seeds)
  - Per-condition confusion matrices
  - t-SNE embedding plots coloured by disease and by condition

Usage
-----
    # Full run: templates then phrase bank, 3 seeds each
    python run_fast_ablation.py

    # Single generator, 1 seed, quick smoke test
    python run_fast_ablation.py --generators template --seeds 42 --rounds 10

    # Skip plotting (just collect numbers)
    python run_fast_ablation.py --no-plots

    # Resume from existing results (skip completed runs)
    python run_fast_ablation.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WANDB_MODE", "disabled")

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_GENERATORS    = ["template", "phrase_library"]
DEFAULT_SEEDS         = [42, 43, 44]
DEFAULT_SCHEDULES     = ["flat", "gaussian", "burst", "ramp", "parabola"]
DEFAULT_ROUNDS        = 20
DEFAULT_CONFUSION     = 0.10   # ~10% atypical presentations → ≈90% accuracy ceiling
EVENTS_PER_SILO       = 200
N_SILOS               = 3
RESULTS_ROOT          = Path("results/fast_ablation")

# Phase-mismatch presets: each tuple is one (silo_0, silo_1, silo_2) schedule combo.
# Simulates hospitals at different epidemic stages within the same FL round.
MIXED_PRESETS: dict[str, list[str]] = {
    "early+mid+late":  ["burst",        "gaussian",     "ramp"         ],  # peak early / mid / late
    "late+mid+early":  ["ramp",         "gaussian",     "burst"        ],  # reversed phase order
    "flat+peak+flat":  ["flat",         "gaussian",     "flat"         ],  # one silo has a surge, others constant
    "burst+flat+ramp": ["burst",        "flat",         "ramp"         ],  # boom-bust vs steady vs slow-ramp
    # Same shape (Gaussian), silos just peak at different rounds
    # mu ≈ 20 × {1/6, 1/2, 5/6} → rounds 3, 10, 17 (0-indexed out of 20)
    "gauss@3+10+17":   ["gaussian@3",   "gaussian@10",  "gaussian@17"  ],
}


# ── Non-IID silo distribution ─────────────────────────────────────────────────

def _noniid_dists(n_silos: int) -> list:
    base = [
        ("influenza",      "mild",     "neutral", 1.0),
        ("influenza",      "moderate", "neutral", 1.0),
        ("influenza",      "severe",   "neutral", 0.5),
        ("pneumonia",      "mild",     "neutral", 1.0),
        ("pneumonia",      "moderate", "neutral", 1.0),
        ("pneumonia",      "severe",   "neutral", 0.5),
        ("non-infectious", "mild",     "anxious", 0.5),
    ]
    dists = []
    for i in range(n_silos):
        flu_w   = 3.0 if i % 2 == 0 else 1.0
        pneu_w  = 1.0 if i % 2 == 0 else 3.0
        d = [
            (dis, sev, per, w * (flu_w if dis == "influenza" else
                                 pneu_w if dis == "pneumonia" else 1.0))
            for dis, sev, per, w in base
        ]
        dists.append(d)
    return dists


# ── Single experiment run ─────────────────────────────────────────────────────

def run_one(
    generator:           str,
    schedule:            str,
    dist:                str,        # "iid" | "noniid"
    seed:                int,
    n_rounds:            int,
    resume:              bool,
    device:              str        = "cpu",
    confusion_rate:      float      = DEFAULT_CONFUSION,
    silo_schedule_names: list[str] | None = None,
) -> dict:
    from fl.scheduled_runner import ScheduledFLConfig, run_scheduled_fl

    tag     = f"{generator}/{schedule}/{dist}/seed{seed}"
    out_dir = RESULTS_ROOT / generator / schedule / dist / f"seed{seed}"

    if resume and (out_dir / "summary.json").exists():
        with (out_dir / "summary.json").open() as f:
            summary = json.load(f)
        print(f"  [skip] {tag}  diag={summary['final_diag_acc']:.3f}")
        return summary

    silo_dists = _noniid_dists(N_SILOS) if dist == "noniid" else None

    cfg = ScheduledFLConfig(
        mode                 = "generate",
        generator            = generator,
        n_silos              = N_SILOS,
        events_per_silo      = EVENTS_PER_SILO,
        silo_distributions   = silo_dists,
        schedule_name        = schedule,
        silo_schedule_names  = silo_schedule_names,
        n_rounds             = n_rounds,
        seed                 = seed,
        gen_seed             = seed,
        training_device      = device,
        confusion_rate       = confusion_rate,
        run_name             = tag.replace("/", "_"),
        results_dir          = str(RESULTS_ROOT / generator / schedule / dist),
    )

    result = run_scheduled_fl(cfg)
    return {
        "final_diag_acc":   result["final_diag"],
        "final_triage_acc": result["final_triage"],
        "schedule":         schedule,
        "generator":        generator,
        "dist":             dist,
        "seed":             seed,
        "rounds":           result["rounds"],
    }


# ── Confusion matrix ──────────────────────────────────────────────────────────

def compute_confusion(
    generator_name: str,
    schedule: str,
    dist: str,
    seeds: list[int],
    n_rounds: int,
) -> None:
    """Re-evaluate saved generator pools on final model weights and plot confusion."""
    import torch
    from simulation.phrase_sampler import PhraseLibrary, TemplateSampler
    from fl.scheduled_runner import ScheduledFLConfig, _build_generated_pools, _make_generator
    from fl.lora import LoRAConfig
    from fl.learner import FLLearner, build_label_map, _split_label
    import random

    label2id, id2label = build_label_map()

    all_preds, all_trues = [], []

    for seed in seeds:
        summary_path = (
            RESULTS_ROOT / generator_name / schedule / dist / f"seed{seed}" / "summary.json"
        )
        if not summary_path.exists():
            continue

        # Re-run a single round with full data to get final model predictions on holdout
        cfg = ScheduledFLConfig(
            mode="generate", generator=generator_name,
            n_silos=N_SILOS, events_per_silo=EVENTS_PER_SILO,
            silo_distributions=_noniid_dists(N_SILOS) if dist == "noniid" else None,
            schedule_name=schedule, n_rounds=n_rounds,
            seed=seed, gen_seed=seed,
        )
        rng = random.Random(seed)
        _, holdouts = _build_generated_pools(cfg, rng)

        lora_cfg = LoRAConfig(
            model_name_or_path="distilbert-base-uncased",
            num_labels=None, rank=8, lora_alpha=16.0, lora_dropout=0.05,
        )
        from fl.lora import build_model
        from fl.train import _fedavg

        # Load saved weights
        import numpy as np
        weights_path = (
            RESULTS_ROOT / generator_name / schedule / dist
            / f"seed{seed}" / "fl_final.npz"
        )
        if not weights_path.exists():
            continue

        model = build_model(lora_cfg)
        w = np.load(weights_path)
        from fl.lora import set_lora_weights
        set_lora_weights(model, [w[k] for k in sorted(w.keys())])
        model.eval()

        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        for hold in holdouts:
            texts, labels = [], []
            for r in hold:
                lid = label2id.get(r.get("label") or "")
                if lid is None or not r.get("text"):
                    continue
                texts.append(r["text"]); labels.append(lid)
            if not texts:
                continue
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"])
            preds = out.logits.argmax(dim=-1).tolist()
            all_preds.extend(preds); all_trues.extend(labels)

    if not all_preds:
        return

    # Print confusion
    from collections import Counter
    disease_preds = [_split_label(id2label[p])[0] for p in all_preds]
    disease_trues = [_split_label(id2label[t])[0] for t in all_trues]
    classes = sorted(set(disease_trues))
    idx = {c: i for i, c in enumerate(classes)}
    n   = len(classes)
    mat = [[0]*n for _ in range(n)]
    for p, t in zip(disease_preds, disease_trues):
        if p in idx and t in idx:
            mat[idx[t]][idx[p]] += 1

    title = f"{generator_name} / {schedule} / {dist}"
    cw = max(len(c) for c in classes) + 2
    print(f"\n  Confusion — {title}  (n={len(all_preds)})")
    header = f"  {'':>{cw}}" + "".join(f"{c:>{cw}}" for c in classes)
    print(header)
    for ri, rc in enumerate(classes):
        row = "".join(f"{mat[ri][ci]:>{cw}}" for ci in range(n))
        diag_rate = mat[ri][idx[rc]] / max(sum(mat[ri]), 1)
        print(f"  {rc:>{cw}}{row}  ({diag_rate:.2f})")

    diag_ok = sum(1 for p, t in zip(disease_preds, disease_trues) if p == t)
    print(f"  diag_acc={diag_ok/len(all_preds):.3f}")


# ── Embedding plot ────────────────────────────────────────────────────────────

def plot_embeddings(
    all_results: dict,
    label_records: dict,   # generator → list[dict] with "text", "gt_disease"
) -> None:
    """t-SNE of CLS embeddings coloured by disease, faceted by generator."""
    try:
        import torch
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        print(f"  [embed] skipping — missing dependency: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    base_model = AutoModel.from_pretrained("distilbert-base-uncased")
    base_model.eval()

    DISEASE_COLORS = {
        "influenza":      "#e05c5c",
        "pneumonia":      "#5c8fe0",
        "non-infectious": "#5cc87a",
    }

    fig, axes = plt.subplots(1, len(label_records), figsize=(6 * len(label_records), 5))
    if len(label_records) == 1:
        axes = [axes]

    for ax, (gen_name, records) in zip(axes, label_records.items()):
        texts   = [r["text"]       for r in records[:300]]
        labels  = [r["gt_disease"] for r in records[:300]]

        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = base_model(**enc)
        cls_emb = out.last_hidden_state[:, 0, :].numpy()

        coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(cls_emb)
        for disease, color in DISEASE_COLORS.items():
            mask = [i for i, l in enumerate(labels) if l == disease]
            if mask:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=color, label=disease, alpha=0.7, s=20)
        ax.set_title(gen_name, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("CLS embedding t-SNE by disease (base DistilBERT, no fine-tuning)",
                 fontsize=12)
    plt.tight_layout()
    out_path = RESULTS_ROOT / "embeddings_by_generator.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  [embed] saved → {out_path}")


# ── Statistics table ──────────────────────────────────────────────────────────

def print_stats(collected: list[dict]) -> None:
    from collections import defaultdict
    # Group by (generator, schedule, dist)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in collected:
        key = (r["generator"], r["schedule"], r["dist"])
        groups[key].append(r.get("final_diag_acc", float("nan")))

    print(f"\n{'═'*70}")
    print("  SUMMARY  (diag_acc mean ± std across seeds)")
    print(f"{'═'*70}")
    print(f"  {'generator':15s}  {'schedule':10s}  {'dist':6s}   mean    std   n")
    print(f"  {'─'*65}")
    for (gen, sched, dist), vals in sorted(groups.items()):
        vals = [v for v in vals if v == v]  # drop NaN
        if not vals:
            continue
        mean = np.mean(vals)
        std  = np.std(vals)
        print(f"  {gen:15s}  {sched:10s}  {dist:6s}   {mean:.3f}   {std:.3f}   {len(vals)}")

    # Save as JSON
    summary_out = RESULTS_ROOT / "statistics.json"
    stats = {
        f"{gen}/{sched}/{dist}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for (gen, sched, dist), v in groups.items()
        if any(x == x for x in v)
    }
    with summary_out.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Statistics written to {summary_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="+", default=DEFAULT_GENERATORS,
                    choices=["template", "phrase_library"])
    ap.add_argument("--seeds",      nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--schedules",  nargs="+", default=DEFAULT_SCHEDULES,
                    choices=DEFAULT_SCHEDULES)
    ap.add_argument("--dists",      nargs="+", default=["iid", "noniid"],
                    choices=["iid", "noniid"])
    ap.add_argument("--rounds",     type=int,  default=DEFAULT_ROUNDS)
    ap.add_argument("--device",         default="cpu", choices=["cpu", "cuda"],
                    help="Training device")
    ap.add_argument("--confusion-rate", type=float, default=DEFAULT_CONFUSION,
                    help="Fraction of records with text from wrong disease bucket "
                         "(0.0 = clean, 0.10 = ~90%% ceiling)")
    ap.add_argument("--mixed",      action="store_true",
                    help="Run phase-mismatch presets (different schedule per silo) "
                         "instead of homogeneous schedules")
    ap.add_argument("--no-plots",    action="store_true")
    ap.add_argument("--resume",      action="store_true",
                    help="Skip runs whose summary.json already exists")
    ap.add_argument("--results-root", default=None,
                    help="Override output directory root "
                         "(default: results/fast_ablation)")
    args = ap.parse_args()

    global RESULTS_ROOT
    if args.results_root is not None:
        RESULTS_ROOT = Path(args.results_root)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    collected: list[dict] = []

    total = len(args.generators) * len(args.schedules) * len(args.dists) * len(args.seeds)
    done  = 0
    t0    = time.time()

    if args.mixed:
        # ── Phase-mismatch presets ────────────────────────────────────────────
        mixed_conditions = [
            (gen, preset_name, dist, seed)
            for gen in args.generators
            for preset_name in MIXED_PRESETS
            for dist in args.dists
            for seed in args.seeds
        ]
        total = len(mixed_conditions)
        print(f"\n{'━'*70}")
        print(f"  PHASE-MISMATCH RUN  ({total} conditions)")
        print(f"{'━'*70}")
        for gen, preset_name, dist, seed in mixed_conditions:
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 1 else 0
            silo_scheds = MIXED_PRESETS[preset_name]
            print(f"\n  [{done}/{total}  eta {eta/60:.1f}min] "
                  f"{gen} / {preset_name} / {dist} / seed={seed}")
            result = run_one(
                gen, preset_name, dist, seed, args.rounds, args.resume,
                device=args.device,
                confusion_rate=args.confusion_rate,
                silo_schedule_names=silo_scheds,
            )
            collected.append(result)

    else:
        # ── Homogeneous schedules ─────────────────────────────────────────────
        total = len(args.generators) * len(args.schedules) * len(args.dists) * len(args.seeds)

    for generator in args.generators:
        if args.mixed:
            break   # embedding/confusion plots still run below
        print(f"\n{'━'*70}")
        print(f"  GENERATOR: {generator.upper()}")
        print(f"{'━'*70}")

        gen_records: list[dict] = []   # for embedding plot

        for dist in args.dists:
            for schedule in args.schedules:
                for seed in args.seeds:
                    done += 1
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done) if done > 1 else 0
                    print(f"\n  [{done}/{total}  eta {eta/60:.1f}min] "
                          f"{generator} / {schedule} / {dist} / seed={seed}")
                    result = run_one(
                        generator, schedule, dist, seed, args.rounds, args.resume,
                        device=args.device,
                        confusion_rate=args.confusion_rate,
                    )
                    collected.append(result)

        # Collect sample texts for embedding
        if not args.no_plots:
            from simulation.phrase_sampler import PhraseLibrary, TemplateSampler
            lib = TemplateSampler(seed=42) if generator == "template" \
                  else PhraseLibrary(seed=42)
            import random
            rng = random.Random(42)
            for _ in range(300):
                disease  = rng.choice(["influenza", "pneumonia", "non-infectious"])
                severity = rng.choice(["mild", "moderate", "severe"])
                rec = lib.sample(disease, severity, "neutral", days=rng.randint(1,10))
                gen_records.append(rec)

        if not args.no_plots and gen_records:
            plot_embeddings({}, {generator: gen_records})

    # Confusion matrices for all completed conditions
    if not args.no_plots:
        for generator in args.generators:
            for dist in args.dists:
                for schedule in args.schedules:
                    compute_confusion(
                        generator, schedule, dist, args.seeds, args.rounds
                    )

    print_stats(collected)

    total_time = time.time() - t0
    print(f"\n  Total wall time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
