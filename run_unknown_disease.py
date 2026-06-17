#!/usr/bin/env python3
"""
Unknown disease detection experiment — embedding-space visualization.

Protocol
--------
Three silos train on Velarex + Sornathis (known fictional diseases) for
n_rounds of federated learning.

From round `injection_round` onward, Morven Syndrome cases are injected into
silo_0's training batch, labelled "unknown" (clinicians see novel patients
but don't know the disease name).

A fixed probe set — Velarex, Sornathis, AND Morven events — is passed through
the global model every round to extract CLS embeddings.

Primary outputs (written to results/unknown_disease/<run>/):
  umap_evolution.png         — Morven cluster emergence across rounds
  umap_panel_round_N.png     — full UMAP for selected rounds
  silhouette_curve.png       — silhouette(Morven vs known) per round
  round_metrics.json
  summary.json

Usage
-----
    python run_unknown_disease.py                      # gaussian schedule
    python run_unknown_disease.py --schedule uniform   # uniform control
    python run_unknown_disease.py --injection-round 5  # inject earlier
    python run_unknown_disease.py --no-injection       # ablation: no morven
    python run_unknown_disease.py --training-device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── Fictional phrase banks ────────────────────────────────────────────────────

def _load_fictional_banks() -> dict[str, list[list[str]]]:
    """Return {disease_name: [band0_phrases, band1_phrases, band2_phrases]}."""
    from simulation.fictional_diseases import FICTIONAL_DISEASES
    return {
        name: info["phrase_banks"]
        for name, info in FICTIONAL_DISEASES.items()
        if "phrase_banks" in info
    }


class FictionalPhraseLibrary:
    """Phrase sampler for fictional diseases (Velarex, Sornathis, Morven)."""

    _BAND_TO_SEV = {0: "mild", 1: "moderate", 2: "severe"}
    _SEV_TO_BAND = {"mild": 0, "moderate": 1, "severe": 2}

    def __init__(self, seed: int = 42):
        self._rng   = random.Random(seed)
        self._banks = _load_fictional_banks()

    def sample(self, disease: str, severity: str, days: int | None = None) -> dict:
        if days is None:
            days = self._rng.randint(1, 14)

        bands = self._banks.get(disease)
        if bands is None:
            raise ValueError(f"Unknown disease: {disease!r}")

        band_idx = self._SEV_TO_BAND.get(severity, 0)
        phrases  = bands[band_idx]
        text     = self._rng.choice(phrases).format(days=days)
        return {
            "text":        text,
            "label":       disease,   # disease-only for fictional_disease label space
            "gt_disease":  disease,
            "gt_severity": severity,
            "days":        days,
        }

    def sample_pool(
        self,
        distribution: list[tuple[str, str, float]],  # (disease, severity, weight)
        n: int,
        seed_offset: int = 0,
    ) -> list[dict]:
        rng     = random.Random(self._rng.randint(0, 999999) + seed_offset)
        total   = sum(w for *_, w in distribution)
        records = []
        for _ in range(n):
            r, cumulative = rng.random() * total, 0.0
            for disease, severity, w in distribution:
                cumulative += w
                if r <= cumulative:
                    records.append(self.sample(disease, severity))
                    break
            else:
                d, s, _ = distribution[-1]
                records.append(self.sample(d, s))
        return records


# ── Probe events (for embedding snapshots, not training) ─────────────────────

def _make_probe_event(gt: str, text: str):
    """Minimal duck-typed probe event for EmbeddingTracker."""
    ev = types.SimpleNamespace()
    ev.ground_truth  = gt
    ev.conversation  = [{"role": "patient", "text": text}]
    return ev


def generate_fictional_probe_events(
    n_per_band: int = 12,
    seed: int = 999,
) -> list:
    """
    Build a fixed probe set: Velarex + Sornathis + Morven events.
    No WorldEngine needed — uses phrase banks directly.
    """
    lib    = FictionalPhraseLibrary(seed=seed)
    events = []
    for disease in ("velarex", "sornathis", "morven"):
        for severity in ("mild", "moderate", "severe"):
            for _ in range(n_per_band):
                rec = lib.sample(disease, severity)
                gt  = f"{disease}/{severity}"
                events.append(_make_probe_event(gt, rec["text"]))
    return events


# ── Data pools ────────────────────────────────────────────────────────────────

_KNOWN_DIST: list[tuple[str, str, float]] = [
    ("velarex",   "mild",     2.0),
    ("velarex",   "moderate", 3.0),
    ("velarex",   "severe",   1.0),
    ("sornathis", "mild",     1.5),
    ("sornathis", "moderate", 3.0),
    ("sornathis", "severe",   1.5),
]

_MORVEN_DIST: list[tuple[str, str, float]] = [
    ("morven", "mild",     3.0),
    ("morven", "moderate", 2.0),
    ("morven", "severe",   1.0),
]


def _build_pools(
    n_silos: int,
    events_per_silo: int,
    holdout_frac: float,
    seed: int,
) -> tuple[list[list[dict]], list[list[dict]]]:
    lib = FictionalPhraseLibrary(seed=seed)
    rng = random.Random(seed)
    train_pools, holdouts = [], []
    for i in range(n_silos):
        records = lib.sample_pool(_KNOWN_DIST, events_per_silo, seed_offset=i * 1000)
        rng.shuffle(records)
        n_hold = max(1, int(len(records) * holdout_frac))
        holdouts.append(records[:n_hold])
        train_pools.append(records[n_hold:])
    return train_pools, holdouts


def _build_morven_pool(n: int, seed: int) -> list[dict]:
    lib = FictionalPhraseLibrary(seed=seed + 77777)
    return lib.sample_pool(_MORVEN_DIST, n, seed_offset=0)


# ── FL schedule helpers ───────────────────────────────────────────────────────

def _make_schedule(name: str, total: int, n_rounds: int, **kw) -> list[int]:
    from fl.schedules import make_schedule
    return make_schedule(name, total, n_rounds, **kw)


# ── Silhouette coefficient ────────────────────────────────────────────────────

def _silhouette_morven(coords: np.ndarray, labels: list[str]) -> float:
    """
    Mean silhouette of Morven points vs. the known-disease (velarex+sornathis)
    backdrop. Returns NaN if fewer than 2 Morven points exist.

    Positive = Morven is well-separated from known diseases.
    """
    from sklearn.metrics import silhouette_samples

    group = np.array(
        [1 if "morven" in lbl else 0 for lbl in labels], dtype=int
    )
    if group.sum() < 2 or (group == 0).sum() < 2:
        return float("nan")

    # Compute pairwise-silhouette for both classes; report mean Morven score
    try:
        scores = silhouette_samples(coords, group)
        morven_mask = group == 1
        return float(np.mean(scores[morven_mask]))
    except Exception:
        return float("nan")


# ── UMAP wrapper ──────────────────────────────────────────────────────────────

def _project_umap(data: np.ndarray, seed: int = 42) -> np.ndarray:
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(
            n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1
        )
        return reducer.fit_transform(data)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(data)


# ── Color helpers ─────────────────────────────────────────────────────────────

_DISEASE_COLORS = {
    "velarex":   "#e63946",   # vivid red
    "sornathis": "#4361ee",   # indigo blue
    "morven":    "#2a9d8f",   # teal — visually distinct
}
_SEV_ALPHA = {"mild": 0.45, "moderate": 0.75, "severe": 1.00}


# ── Embedding snapshot (single forward pass) ──────────────────────────────────

def _extract_embeddings(
    learner,          # FLLearner — model must have weights set before calling
    probe_events: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cls (N,H), logits (N,C)) for all probe events."""
    import torch

    tokenizer = learner._tokenizer   # always built in __init__

    texts = []
    for ev in probe_events:
        turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        texts.append(turns[-1] if turns else "")

    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=128, return_tensors="pt",
    )
    model = learner.model   # property: builds lazily if needed
    dev   = next(model.parameters()).device
    enc   = {k: v.to(dev) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids       = enc["input_ids"],
            attention_mask  = enc["attention_mask"],
            output_hidden_states = True,
        )
    cls    = out.hidden_states[-1][:, 0, :].cpu().numpy()
    logits = out.logits.cpu().numpy()
    return cls, logits


# ── Panel plot ────────────────────────────────────────────────────────────────

def _plot_umap_panel(
    ax,
    coords: np.ndarray,
    labels: list[str],
    title: str,
    show_morven: bool = True,
) -> None:
    import matplotlib.patches as mpatches

    plotted = set()
    for i, lbl in enumerate(labels):
        disease = lbl.split("/")[0]
        sev     = lbl.split("/")[1] if "/" in lbl else "mild"
        color   = _DISEASE_COLORS.get(disease, "#888888")
        alpha   = _SEV_ALPHA.get(sev, 0.7)
        if disease == "morven" and not show_morven:
            continue
        size   = 55 if disease == "morven" else 25
        marker = "D" if disease == "morven" else "o"
        zorder = 3 if disease == "morven" else 2
        ax.scatter(
            coords[i, 0], coords[i, 1],
            color=color, alpha=alpha, s=size,
            marker=marker, linewidths=0.5 if disease == "morven" else 0,
            edgecolors="black" if disease == "morven" else "none",
            zorder=zorder,
        )
        plotted.add(disease)

    ax.set_title(title, fontsize=9, color="#212529")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values():
        sp.set_color("#ced4da")
        sp.set_linewidth(0.5)


# ── Main experiment ───────────────────────────────────────────────────────────

@dataclass
class UnknownDiseaseConfig:
    schedule:          str   = "gaussian"   # "gaussian" | "uniform" | "burst"
    n_silos:           int   = 3
    events_per_silo:   int   = 200
    n_rounds:          int   = 20
    injection_round:   int   = 10           # first round Morven appears in silo_0
    injection_per_round: int = 8            # Morven events injected per round
    do_inject:         bool  = True         # False = control (no Morven)

    holdout_frac:      float = 0.15
    lora_rank:         int   = 8
    lora_alpha:        float = 16.0
    lora_dropout:      float = 0.05
    local_epochs:      int   = 3
    batch_size:        int   = 8
    lr:                float = 1e-4
    min_events_to_train: int = 10
    fedavg_min_examples: int = 4
    train_sample_cap:  int   = 180

    gaussian_mu:       float = 10.0
    gaussian_sigma:    float | None = None

    probe_per_band:    int   = 12    # probe events per disease × severity band
    probe_seed:        int   = 999
    snap_rounds: list[int]   = field(default_factory=lambda: [2, 5, 8, 10, 12, 15, 20])

    # Hypothesis test flags
    local_only:        bool  = False  # skip FedAvg — each silo keeps own weights
    per_silo_snaps:    bool  = False  # also save per-silo logits at snap rounds
    n_exposed_silos:   int   = 1      # how many silos (0..k-1) receive Morven injection
    confound_silos:    bool  = False  # all silos see Morven, but only silo_0 labels it "unknown";
                                      # silos 1+ mislabel it as a known disease (velarex/sornathis)

    # Attribution experiment (Phase 2)
    # all_silos_unknown: all silos label Morven "unknown" from injection_round onward
    # attribution_round: from this round, silo_0 relabels Morven as "morven" (named class)
    # label_space: use "fictional_disease_5" to add the morven output slot
    all_silos_unknown: bool  = False  # all silos label Morven as "unknown" (Phase 1 detection)
    attribution_round: int   = 0      # 0 = disabled; >0 = round silo_0 switches to "morven" label
    label_space:       str   = "fictional_disease"  # "fictional_disease" (4-cls) | "fictional_disease_5" (5-cls)

    training_device:   str   = "cuda"
    seed:              int   = 42
    results_dir:       str   = "results/unknown_disease"
    run_name:          str   = ""


def run_unknown_disease(cfg: UnknownDiseaseConfig) -> dict:
    from fl.lora import LoRAConfig
    from fl.learner import FLLearner
    from fl.train import _fedavg

    t_start = time.time()
    rng     = random.Random(cfg.seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_pools, holdouts = _build_pools(
        cfg.n_silos, cfg.events_per_silo, cfg.holdout_frac, cfg.seed,
    )
    morven_pool = _build_morven_pool(
        cfg.injection_per_round * cfg.n_rounds, cfg.seed,
    ) if cfg.do_inject else []

    # ── Schedule ──────────────────────────────────────────────────────────────
    min_pool      = min(len(p) for p in train_pools)
    sched_kw: dict = {}
    if cfg.schedule == "gaussian":
        sched_kw["mu"]    = cfg.gaussian_mu
        if cfg.gaussian_sigma is not None:
            sched_kw["sigma"] = cfg.gaussian_sigma
    schedule = _make_schedule(cfg.schedule, min_pool, cfg.n_rounds, **sched_kw)
    schedules = [schedule] * cfg.n_silos

    # ── Learners ──────────────────────────────────────────────────────────────
    lora_cfg = LoRAConfig(
        model_name_or_path = "distilbert-base-uncased",
        num_labels         = None,
        rank               = cfg.lora_rank,
        lora_alpha         = cfg.lora_alpha,
        lora_dropout       = cfg.lora_dropout,
    )
    learners = [
        FLLearner(
            lora_config          = lora_cfg,
            label_space          = cfg.label_space,
            min_events_to_train  = cfg.min_events_to_train,
            local_epochs         = cfg.local_epochs,
            batch_size           = cfg.batch_size,
            lr                   = cfg.lr,
            train_sample_cap     = cfg.train_sample_cap,
            device               = cfg.training_device,
        )
        for _ in range(cfg.n_silos)
    ]

    # ── Fixed probe events (Velarex + Sornathis + Morven) ────────────────────
    probe_events = generate_fictional_probe_events(cfg.probe_per_band, cfg.probe_seed)
    probe_labels = [ev.ground_truth for ev in probe_events]

    # ── Output directory (needed during loop for per-silo logit saving) ─────────
    run_name = cfg.run_name or f"unk_{cfg.schedule}_{int(time.time())}"
    out_dir  = Path(cfg.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── State ─────────────────────────────────────────────────────────────────
    cursors         = [0] * cfg.n_silos
    total_revealed  = [0] * cfg.n_silos
    morven_cursors  = [0] * cfg.n_silos   # one cursor per silo (confound uses all)
    global_w: list | None = None

    round_metrics:     list[dict]          = []
    silhouette_curve:    list[dict]                = []
    embedding_snapshots: dict[int, dict[str, np.ndarray]] = {}  # round → {cls, logits}

    print(f"\n{'═'*60}")
    print(f"  Unknown Disease Experiment — {cfg.schedule.upper()} schedule")
    print(f"  silos={cfg.n_silos}  rounds={cfg.n_rounds}")
    print(f"  inject={'YES (round '+str(cfg.injection_round)+')' if cfg.do_inject else 'NO (control)'}")
    print(f"  probe events: {len(probe_events)}")
    print(f"{'═'*60}\n")

    for r in range(cfg.n_rounds):
        rnd = r + 1

        # ── Reveal new events ────────────────────────────────────────────────
        new_this_round: list[list[dict]] = []
        for i in range(cfg.n_silos):
            n_rev   = schedules[i][r]
            new_ev  = train_pools[i][cursors[i]: cursors[i] + n_rev]
            cursors[i]         = min(cursors[i] + n_rev, len(train_pools[i]))
            total_revealed[i] += len(new_ev)
            # Morven injection
            inject_this_silo = (
                cfg.do_inject and rnd >= cfg.injection_round and morven_pool
                and (i < cfg.n_exposed_silos or cfg.confound_silos or cfg.all_silos_unknown)
            )
            if inject_this_silo:
                m_end  = min(morven_cursors[i] + cfg.injection_per_round, len(morven_pool))
                morven_batch      = morven_pool[morven_cursors[i]:m_end]
                morven_cursors[i] = m_end

                attr_active = cfg.attribution_round > 0 and rnd >= cfg.attribution_round
                if cfg.all_silos_unknown or cfg.confound_silos:
                    if i == 0 and attr_active:
                        # Phase 2: silo_0 has confirmed the disease identity
                        label = "morven"
                    elif cfg.confound_silos and i > 0:
                        # Confound: silos 1+ actively mislabel as known disease
                        label = rng.choice(["velarex", "sornathis"])
                    else:
                        # Phase 1 or all_silos_unknown: everyone flags as novel/unknown
                        label = "unknown"
                else:
                    # Original single-silo inject: only silo_0, always "unknown"
                    label = "unknown"

                morven_labeled = [{**rec, "label": label, "gt_disease": label}
                                  for rec in morven_batch]
                new_ev = new_ev + morven_labeled
            new_this_round.append(new_ev)

        # ── Per-silo: train → collect weights ────────────────────────────────
        eval_metrics:  list[dict]  = []
        round_weights: list[list]  = []
        train_sizes:   list[int]   = []
        losses:        list[float] = []

        attr_phase = cfg.attribution_round > 0 and rnd >= cfg.attribution_round
        for i, learner in enumerate(learners):
            # In Phase 2: silo_0 keeps its local weights — don't overwrite with
            # the FedAvg global model (which has no "morven" signal from other silos).
            if global_w is not None and not cfg.local_only and not (i == 0 and attr_phase):
                learner.set_weights(global_w)

            m = learner.evaluate(holdouts[i]) if holdouts[i] else {}
            eval_metrics.append(m)

            new_ev = new_this_round[i]
            attr_active_silo0 = (
                i == 0 and cfg.attribution_round > 0 and rnd >= cfg.attribution_round
            )
            if total_revealed[i] >= cfg.min_events_to_train and new_ev:
                if attr_active_silo0:
                    # Phase 2: silo_0 only updates the classifier head (backbone frozen)
                    if rnd == cfg.attribution_round:
                        # Warm-start: copy unknown head → morven head before first update
                        learner.init_attribution_class(
                            _IDX.get("unknown", 3), _IDX.get("morven", 4)
                        )
                    n_trained, epoch_losses = learner.train_head_only(new_ev)
                else:
                    n_trained, epoch_losses = learner.train(new_ev)
                train_sizes.append(n_trained)
                losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
            else:
                train_sizes.append(0)
                losses.append(float("nan"))

            round_weights.append(learner.get_weights())
            learner.release()   # release all — re-loaded lazily for snapshots

        # ── FedAvg (skipped in local_only mode) ──────────────────────────────
        if not cfg.local_only:
            active_idx = [
                i for i, s in enumerate(train_sizes)
                if s > 0 and s >= cfg.fedavg_min_examples
                # In Phase 2: exclude silo_0 from FedAvg — it trains head-only
                # locally and must not let other silos' random "morven" rows
                # overwrite its warm-started classifier.
                and not (i == 0 and attr_phase)
            ]
            if active_idx:
                wts      = [round_weights[i] for i in active_idx]
                sizes    = [train_sizes[i]   for i in active_idx]
                global_w = _fedavg(wts, sizes)

        # ── Embedding snapshots ───────────────────────────────────────────────
        if rnd in cfg.snap_rounds and round_weights:
            # Per-silo local snapshots (for local_only mode or hypothesis test)
            if cfg.per_silo_snaps or cfg.local_only:
                for i, learner in enumerate(learners):
                    learner.set_weights(round_weights[i])
                    cls_i, logits_i = _extract_embeddings(learner, probe_events)
                    np.savez_compressed(
                        out_dir / f"logits_r{rnd:02d}_silo{i}.npz",
                        logits=logits_i, cls=cls_i,
                    )
                    learner.release()

            # In Phase 2: also snapshot silo_0's local model so the attribution
            # curve can show P(morven|Morven) from the model that actually trains on it.
            if attr_phase and round_weights[0] is not None:
                learners[0].set_weights(round_weights[0])
                cls0, logits0 = _extract_embeddings(learners[0], probe_events)
                np.savez_compressed(
                    out_dir / f"logits_r{rnd:02d}_silo0.npz",
                    logits=logits0, cls=cls0,
                )
                learners[0].release()

            # Global model snapshot (federated mode only)
            if not cfg.local_only and global_w is not None:
                learners[-1].set_weights(global_w)
                cls, logits = _extract_embeddings(learners[-1], probe_events)
                embedding_snapshots[rnd] = {"cls": cls, "logits": logits}

                coords_2d = _project_umap(logits, seed=cfg.seed)
                sil       = _silhouette_morven(coords_2d, probe_labels)
                silhouette_curve.append({"round": rnd, "silhouette": sil})
                sil_msg = f"  morven_sil={sil:.3f}" if not np.isnan(sil) else ""
                print(f"    [embed] round {rnd} snapshotted{sil_msg}")
                learners[-1].release()

        # ── Metrics ──────────────────────────────────────────────────────────
        agg_diag   = float(np.mean([m.get("diag_acc",   float("nan")) for m in eval_metrics]))
        agg_triage = float(np.mean([m.get("triage_acc", float("nan")) for m in eval_metrics]))
        loss_str   = f"{float(np.nanmean(losses)):.3f}" if any(l == l for l in losses) else "n/a"

        sil_now = silhouette_curve[-1]["silhouette"] if silhouette_curve and rnd in cfg.snap_rounds else float("nan")
        sil_str = f"{sil_now:.3f}" if not np.isnan(sil_now) else ""

        print(
            f"  R{rnd:02d}  diag={agg_diag:.3f}  loss={loss_str}"
            + (f"  sil={sil_str}" if sil_str else "")
            + (f"  morven_injected={'yes' if rnd>=cfg.injection_round and cfg.do_inject else 'no'}")
        )

        rm = {
            "round":           rnd,
            "agg_diag_acc":    agg_diag,
            "agg_triage_acc":  agg_triage,
            "mean_loss":       float(np.nanmean(losses)) if any(l == l for l in losses) else float("nan"),
            "silo_diag":       [m.get("diag_acc", float("nan")) for m in eval_metrics],
            "train_sizes":     train_sizes,
            "morven_injected": cfg.do_inject and rnd >= cfg.injection_round,
            "morven_cursors":  list(morven_cursors),
            "cursors":         list(cursors),
        }
        round_metrics.append(rm)

    # ── Final eval ────────────────────────────────────────────────────────────
    final_diag_list = []
    for i, learner in enumerate(learners):
        if global_w is not None:
            learner.set_weights(global_w)
        m = learner.evaluate(holdouts[i]) if holdouts[i] else {}
        learner.release()
        final_diag_list.append(m.get("diag_acc", float("nan")))

    final_diag = float(np.nanmean(final_diag_list))
    elapsed    = time.time() - t_start
    print(f"\n  Final holdout diag={final_diag:.3f}  wall={elapsed:.0f}s")

    # ── Save metrics ──────────────────────────────────────────────────────────
    with (out_dir / "round_metrics.json").open("w") as f:
        json.dump(round_metrics, f, indent=2)

    with (out_dir / "silhouette.json").open("w") as f:
        json.dump(silhouette_curve, f, indent=2)

    # Save raw logits for post-hoc analysis / alternative plots
    for rnd, snap in embedding_snapshots.items():
        np.savez_compressed(
            out_dir / f"logits_r{rnd:02d}.npz",
            logits = snap["logits"],
            cls    = snap["cls"],
        )

    summary = {
        "schedule":          cfg.schedule,
        "n_silos":           cfg.n_silos,
        "n_rounds":          cfg.n_rounds,
        "injection_round":   cfg.injection_round,
        "do_inject":         cfg.do_inject,
        "final_diag_acc":    final_diag,
        "wall_seconds":      elapsed,
        "silhouette_curve":  silhouette_curve,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if embedding_snapshots:
        _make_umap_plots(
            embedding_snapshots, probe_labels, cfg, out_dir, silhouette_curve
        )

    print(f"\n  Results written to {out_dir}/")
    return {"summary": summary, "round_metrics": round_metrics}


# ── Visualization ─────────────────────────────────────────────────────────────

def _make_umap_plots(
    snapshots: dict[int, np.ndarray],
    labels: list[str],
    cfg: UnknownDiseaseConfig,
    out_dir: Path,
    sil_curve: list[dict],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rounds_sorted = sorted(snapshots)
    n = len(rounds_sorted)

    # ── Figure 1: UMAP evolution grid ────────────────────────────────────────
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0),
                             constrained_layout=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax_i, rnd in enumerate(rounds_sorted):
        ax     = axes_flat[ax_i]
        coords = _project_umap(snapshots[rnd]["logits"], seed=cfg.seed)
        _plot_umap_panel(ax, coords, labels,
                         title=f"Round {rnd}" + (" ★" if rnd == cfg.injection_round else ""),
                         show_morven=True)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(color=_DISEASE_COLORS["velarex"],   label="Velarex"),
        mpatches.Patch(color=_DISEASE_COLORS["sornathis"], label="Sornathis"),
        mpatches.Patch(color=_DISEASE_COLORS["morven"],    label="Morven (novel ◆)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.95, facecolor="white", edgecolor="#ced4da",
               bbox_to_anchor=(0.5, -0.03))
    title_inj = f"Morven injected at round {cfg.injection_round}" if cfg.do_inject else "Control (no Morven)"
    fig.suptitle(f"Embedding evolution — {cfg.schedule.upper()} schedule\n{title_inj}",
                 fontsize=11, color="#212529")

    path = out_dir / "umap_evolution.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Figure 2: Silhouette curve ────────────────────────────────────────────
    if sil_curve:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fa")

        xs  = [d["round"] for d in sil_curve]
        ys  = [d["silhouette"] for d in sil_curve]
        ax.plot(xs, ys, marker="o", color=_DISEASE_COLORS["morven"], linewidth=1.8, markersize=6)
        ax.axhline(0, color="#6c757d", linewidth=0.8, linestyle="--", label="detection threshold")
        if cfg.do_inject:
            ax.axvline(cfg.injection_round, color=_DISEASE_COLORS["morven"],
                       linewidth=1.0, linestyle=":", label=f"injection (round {cfg.injection_round})")

        ax.set_xlabel("FL Round", fontsize=10)
        ax.set_ylabel("Morven silhouette score", fontsize=10)
        ax.set_title("Novel disease cluster detectability over rounds\n"
                     "(positive = Morven separates from known diseases in embedding space)",
                     fontsize=10, color="#212529")
        ax.legend(fontsize=9)
        for sp in ax.spines.values():
            sp.set_color("#ced4da")
            sp.set_linewidth(0.5)

        path = out_dir / "silhouette_curve.png"
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Figure 3: Side-by-side before/after injection ─────────────────────────
    snap_rounds = sorted(snapshots)
    pre  = [r for r in snap_rounds if r < cfg.injection_round]
    post = [r for r in snap_rounds if r >= cfg.injection_round]
    if pre and post and cfg.do_inject:
        r_pre  = max(pre)
        r_post = min(post)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        fig.patch.set_facecolor("white")

        for ax, rnd, label in [
            (axes[0], r_pre,  f"Round {r_pre} (before injection)"),
            (axes[1], r_post, f"Round {r_post} (after injection ★)"),
        ]:
            coords = _project_umap(snapshots[rnd]["logits"], seed=cfg.seed)
            _plot_umap_panel(ax, coords, labels, title=label, show_morven=True)

        patches = [
            mpatches.Patch(color=_DISEASE_COLORS["velarex"],   label="Velarex"),
            mpatches.Patch(color=_DISEASE_COLORS["sornathis"], label="Sornathis"),
            mpatches.Patch(color=_DISEASE_COLORS["morven"],    label="Morven (novel ◆)"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
                   framealpha=0.95, facecolor="white", edgecolor="#ced4da",
                   bbox_to_anchor=(0.5, -0.04))
        fig.suptitle("Embedding space before vs. after novel disease injection",
                     fontsize=11, color="#212529")

        path = out_dir / "before_after_injection.png"
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Figure 4: P(unknown) boxplot panel ───────────────────────────────────
    plot_softmax_panel(snapshots, labels, cfg, out_dir)

    # ── Figure 5: Logit scatter (known-disease vs unknown score) ─────────────
    key_rounds = [r for r in [5, 8, 10, 15, 20] if r in snapshots]
    plot_logit_scatter(snapshots, labels, cfg, out_dir, show_rounds=key_rounds)


# ── Alternative visualizations ────────────────────────────────────────────────

# Label indices — 4-class (fictional_disease) and 5-class (fictional_disease_5)
_IDX = {"non-infectious": 0, "velarex": 1, "sornathis": 2, "unknown": 3, "morven": 4}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def plot_softmax_panel(
    snapshots:  dict[int, dict],
    labels:     list[str],
    cfg:        "UnknownDiseaseConfig",
    out_dir:    Path,
) -> Path:
    """
    Line plot of mean P(unknown) per disease across snapshot rounds.

    Three lines: Velarex, Sornathis, Morven.
    After injection, the Morven line should rise while the others stay flat.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds        = sorted(snapshots)
    disease_order = ["velarex", "sornathis", "morven"]
    colors        = [_DISEASE_COLORS[d] for d in disease_order]

    # Mean P(unknown) per disease per round
    means: dict[str, list[float]] = {d: [] for d in disease_order}
    bands: dict[str, list[float]] = {d: [] for d in disease_order}   # ±1 std
    for rnd in rounds:
        probs = _softmax(snapshots[rnd]["logits"])   # (N, 4)
        p_unk = probs[:, _IDX["unknown"]]
        for dis in disease_order:
            idx = [i for i, l in enumerate(labels) if l.split("/")[0] == dis]
            v   = p_unk[idx]
            means[dis].append(float(v.mean()))
            bands[dis].append(float(v.std()))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    xs = rounds
    for dis, color in zip(disease_order, colors):
        lw    = 2.2 if dis == "morven" else 1.4
        ms    = 7   if dis == "morven" else 5
        mk    = "D" if dis == "morven" else "o"
        ys    = means[dis]
        stds  = bands[dis]
        ax.plot(xs, ys, marker=mk, color=color, linewidth=lw,
                markersize=ms, label=dis.capitalize(), zorder=3)
        ax.fill_between(xs,
                        [y - s for y, s in zip(ys, stds)],
                        [y + s for y, s in zip(ys, stds)],
                        color=color, alpha=0.12, zorder=2)

    ax.set_xlabel("FL Round", fontsize=10)
    ax.set_ylabel("Mean P(unknown)", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(xs)

    if cfg.do_inject:
        ax.axvline(cfg.injection_round, color=_DISEASE_COLORS["morven"],
                   linewidth=1.0, linestyle=":",
                   label=f"injection (R{cfg.injection_round})", zorder=1)

    ax.legend(fontsize=9, framealpha=0.9)
    title_sfx = f"injection at R{cfg.injection_round}" if cfg.do_inject else "control — no Morven"
    ax.set_title(f"Probability of 'unknown' label per disease probe  [{title_sfx}]",
                 fontsize=10, color="#212529")
    for sp in ax.spines.values():
        sp.set_color("#ced4da"); sp.set_linewidth(0.5)

    path = out_dir / "prob_unknown_panel.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_logit_scatter(
    snapshots:  dict[int, dict],
    labels:     list[str],
    cfg:        "UnknownDiseaseConfig",
    out_dir:    Path,
    show_rounds: list[int] | None = None,
) -> Path:
    """
    2D logit scatter across key rounds: x = known-disease score, y = unknown score.

    x = (logit[velarex] + logit[sornathis]) / 2   "model thinks it's a known disease"
    y = logit[unknown]                              "model thinks it's something new"

    Morven pre-injection: scattered in known-disease territory (high x, low y).
    Morven post-injection: pushed into unknown territory (low x, high y).
    Velarex + Sornathis act as anchors — should stay in their respective corners.
    All panels use the same axis limits so evolution is visually comparable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = sorted(snapshots)
    if show_rounds is not None:
        rounds = [r for r in show_rounds if r in snapshots]

    # Compute global axis limits across all rounds for comparability
    all_x, all_y = [], []
    for rnd in sorted(snapshots):
        lg = snapshots[rnd]["logits"]
        all_x.append((lg[:, _IDX["velarex"]] + lg[:, _IDX["sornathis"]]) / 2)
        all_y.append(lg[:, _IDX["unknown"]])
    x_all = np.concatenate(all_x); y_all = np.concatenate(all_y)
    pad   = 0.5
    xlim  = (x_all.min() - pad, x_all.max() + pad)
    ylim  = (y_all.min() - pad, y_all.max() + pad)

    n = len(rounds)
    if n == 0:
        return out_dir / "logit_scatter.png"
    fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 3.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    disease_order = ["velarex", "sornathis", "morven"]

    for ax, rnd in zip(axes, rounds):
        lg     = snapshots[rnd]["logits"]
        x_vals = (lg[:, _IDX["velarex"]] + lg[:, _IDX["sornathis"]]) / 2
        y_vals = lg[:, _IDX["unknown"]]

        for dis in disease_order:
            idx = [i for i, l in enumerate(labels) if l.split("/")[0] == dis]
            if not idx:
                continue
            color  = _DISEASE_COLORS[dis]
            marker = "D" if dis == "morven" else "o"
            size   = 55 if dis == "morven" else 22
            ax.scatter(x_vals[idx], y_vals[idx],
                       c=color, marker=marker, s=size, alpha=0.75,
                       linewidths=0.5 if dis == "morven" else 0,
                       edgecolors="black" if dis == "morven" else "none",
                       zorder=3 if dis == "morven" else 2,
                       label=dis.capitalize())

        # Diagonal guide: x == y → model is indifferent
        lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], color="#ced4da", linewidth=0.7,
                linestyle="--", zorder=1)

        star = " ★" if rnd == cfg.injection_round else ""
        inj_tag = " (post-inj)" if cfg.do_inject and rnd >= cfg.injection_round else ""
        ax.set_title(f"Round {rnd}{star}{inj_tag}", fontsize=9, color="#212529")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da"); sp.set_linewidth(0.5)
        ax.tick_params(labelsize=7)
        if ax is axes[0]:
            ax.set_xlabel("known-disease logit  (velarex+sornathis)/2", fontsize=8)
            ax.set_ylabel("unknown logit", fontsize=8)

    axes[0].legend(fontsize=8, framealpha=0.9, markerscale=0.9)

    inj_info = f"injection at R{cfg.injection_round}" if cfg.do_inject else "control — no Morven"
    fig.suptitle(f"Logit space: known-disease vs unknown score  [{inj_info}]",
                 fontsize=10, color="#212529")

    path = out_dir / "logit_scatter.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── SIR-driven variant ────────────────────────────────────────────────────────

def _build_sir_world(progressions: list[str], n_agents: int,
                     seed: int, n_initial_seeds: int = 2,
                     beta_scale: float = 2.0,
                     use_ollama: bool = False,
                     case_summarizer=None):
    """Return a WorldEngine configured for fictional diseases."""
    from simulation.world import WorldEngine
    from simulation.world_config import WorldConfig, AgentConfig, EpidemicConfig
    if use_ollama:
        from simulation.data_sources import OllamaFictionalDataSource
        data_source = OllamaFictionalDataSource(seed=seed)
    else:
        from simulation.data_sources import FictionalDataSource
        data_source = FictionalDataSource(seed=seed)

    cfg = WorldConfig(
        agents   = AgentConfig(
            num_agents      = n_agents,
            data_source     = data_source,
            case_summarizer = case_summarizer,
        ),
        epidemic = EpidemicConfig(
            progressions    = progressions,
            disease_strategy= progressions[0],
            initial_seeds   = n_initial_seeds,
            beta_scale      = beta_scale,
        ),
        seed_offset = 0,
    )
    return WorldEngine(cfg, seed=seed)


def _event_to_record(event, label: str | None = None) -> dict:
    """Convert a DiagnosticEvent from WorldEngine to the dict format used in FL."""
    case_summary = getattr(event, "case_summary", None)
    if case_summary:
        text = case_summary
    else:
        turns = [t["text"] for t in event.conversation if t["role"] == "patient"]
        text  = " ".join(turns) if turns else ""
    return {
        "text":         text,
        "label":        label if label is not None else (event.gt_disease or "unknown"),
        "gt_disease":   event.gt_disease or "unknown",
        "case_summary": case_summary or "",
    }


def run_sir_unknown_disease(cfg: "UnknownDiseaseConfig",
                            days_per_round: int = 3,
                            n_agents: int = 60,
                            beta_scale: float = 2.0,
                            use_ollama: bool = False,
                            case_summarizer=None) -> dict:
    """
    Unknown disease detection experiment driven by live SIR dynamics.

    Each FL round advances all WorldEngines by `days_per_round` days.
    Known-disease worlds (Velarex + Sornathis) run for all rounds.
    Morven worlds run in parallel from round 1 but only contribute events
    from `cfg.injection_round` onward (simulating undetected spread before
    the first clinical alert).

    Holdout evaluation still uses phrase-bank records for a stable benchmark.
    """
    from fl.lora import LoRAConfig
    from fl.learner import FLLearner
    from fl.train import _fedavg

    t_start = time.time()
    rng     = random.Random(cfg.seed)

    # ── Static holdout for evaluation (phrase-bank, unchanged) ───────────────
    _, holdouts = _build_pools(
        cfg.n_silos, cfg.events_per_silo, cfg.holdout_frac, cfg.seed,
    )

    # ── WorldEngines — one known-disease world + one morven world per silo ───
    known_worlds = [
        _build_sir_world(
            ["Velarex", "Sornathis"], n_agents,
            seed=cfg.seed + i * 1000,
            n_initial_seeds=3,
            beta_scale=beta_scale,
            use_ollama=use_ollama,
            case_summarizer=case_summarizer,
        )
        for i in range(cfg.n_silos)
    ]
    morven_worlds = [
        _build_sir_world(
            ["Morven"], max(n_agents // 3, 10),
            seed=cfg.seed + i * 1000 + 77777,
            n_initial_seeds=1,
            beta_scale=beta_scale,
            use_ollama=use_ollama,
            case_summarizer=case_summarizer,
        )
        for i in range(cfg.n_silos)
    ] if cfg.do_inject else []

    # ── Learners ──────────────────────────────────────────────────────────────
    lora_cfg = LoRAConfig(
        model_name_or_path = "distilbert-base-uncased",
        num_labels         = None,
        rank               = cfg.lora_rank,
        lora_alpha         = cfg.lora_alpha,
        lora_dropout       = cfg.lora_dropout,
    )
    learners = [
        FLLearner(
            lora_config         = lora_cfg,
            label_space         = cfg.label_space,
            min_events_to_train = cfg.min_events_to_train,
            local_epochs        = cfg.local_epochs,
            batch_size          = cfg.batch_size,
            lr                  = cfg.lr,
            train_sample_cap    = cfg.train_sample_cap,
            device              = cfg.training_device,
        )
        for _ in range(cfg.n_silos)
    ]

    # ── Fixed probe events (phrase-bank — same as non-SIR mode) ──────────────
    probe_events = generate_fictional_probe_events(cfg.probe_per_band, cfg.probe_seed)
    probe_labels = [ev.ground_truth for ev in probe_events]

    run_name = cfg.run_name or f"sir_unk_{cfg.schedule}_{int(time.time())}"
    out_dir  = Path(cfg.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total_seen = [0] * cfg.n_silos
    global_w: list | None = None

    round_metrics:       list[dict]          = []
    silhouette_curve:    list[dict]          = []
    embedding_snapshots: dict[int, dict]     = {}

    summarizer_tag = (
        type(case_summarizer).__name__ if case_summarizer is not None else "none"
    )
    print(f"\n{'═'*60}")
    print(f"  Unknown Disease Experiment (SIR) — {cfg.n_silos} silos")
    print(f"  rounds={cfg.n_rounds}  days_per_round={days_per_round}  n_agents={n_agents}")
    print(f"  inject={'YES (round '+str(cfg.injection_round)+')' if cfg.do_inject else 'NO'}")
    print(f"  case_summarizer={summarizer_tag}")
    print(f"{'═'*60}\n")

    for r in range(cfg.n_rounds):
        rnd = r + 1

        # ── Advance all worlds ────────────────────────────────────────────────
        new_this_round: list[list[dict]] = []
        for i in range(cfg.n_silos):
            known_events = known_worlds[i].run_sim_days(days_per_round)
            records      = [
                _event_to_record(ev)
                for ev in known_events
                if not getattr(ev, "is_background", False)
            ]

            inject_this_silo = (
                cfg.do_inject
                and rnd >= cfg.injection_round
                and i < cfg.n_exposed_silos
                and morven_worlds
            )
            if inject_this_silo:
                morven_events = morven_worlds[i].run_sim_days(days_per_round)
                morven_recs   = [
                    _event_to_record(ev, label="unknown")
                    for ev in morven_events
                    if not getattr(ev, "is_background", False)
                ]
                records = records + morven_recs
            elif morven_worlds and i < cfg.n_exposed_silos:
                # Advance morven world even before injection (silent spread)
                morven_worlds[i].run_sim_days(days_per_round)

            total_seen[i] += len(records)
            new_this_round.append(records)

        # ── Per-silo train → collect weights ──────────────────────────────────
        eval_metrics:  list[dict]  = []
        round_weights: list[list]  = []
        train_sizes:   list[int]   = []
        losses:        list[float] = []

        for i, learner in enumerate(learners):
            if global_w is not None and not cfg.local_only:
                learner.set_weights(global_w)

            m = learner.evaluate(holdouts[i]) if holdouts[i] else {}
            eval_metrics.append(m)

            new_ev = new_this_round[i]
            if total_seen[i] >= cfg.min_events_to_train and new_ev:
                n_trained, epoch_losses = learner.train(new_ev)
                train_sizes.append(n_trained)
                losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
            else:
                train_sizes.append(0)
                losses.append(float("nan"))

            round_weights.append(learner.get_weights())
            learner.release()

        # ── FedAvg ────────────────────────────────────────────────────────────
        if not cfg.local_only:
            active_idx = [
                i for i, s in enumerate(train_sizes)
                if s > 0 and s >= cfg.fedavg_min_examples
            ]
            if active_idx:
                wts      = [round_weights[i] for i in active_idx]
                sizes    = [train_sizes[i]   for i in active_idx]
                global_w = _fedavg(wts, sizes)

        # ── Embedding snapshots ───────────────────────────────────────────────
        if rnd in cfg.snap_rounds and global_w is not None:
            learners[-1].set_weights(global_w)
            cls, logits = _extract_embeddings(learners[-1], probe_events)
            embedding_snapshots[rnd] = {"cls": cls, "logits": logits}
            coords_2d = _project_umap(logits, seed=cfg.seed)
            sil       = _silhouette_morven(coords_2d, probe_labels)
            silhouette_curve.append({"round": rnd, "silhouette": sil})
            sil_msg = f"  morven_sil={sil:.3f}" if not np.isnan(sil) else ""
            print(f"    [embed] round {rnd} snapshotted{sil_msg}")
            learners[-1].release()

        # ── Metrics ───────────────────────────────────────────────────────────
        agg_diag  = float(np.mean([m.get("diag_acc",   float("nan")) for m in eval_metrics]))
        _finite = [l for l in losses if l == l]  # filter nan
        loss_str  = f"{float(np.mean(_finite)):.3f}" if _finite else "n/a"
        n_known   = sum(len(new_this_round[i]) for i in range(cfg.n_silos))
        sil_now   = silhouette_curve[-1]["silhouette"] if silhouette_curve and rnd in cfg.snap_rounds else float("nan")
        sil_str   = f"  sil={sil_now:.3f}" if not np.isnan(sil_now) else ""
        print(
            f"  R{rnd:02d}  diag={agg_diag:.3f}  loss={loss_str}"
            f"  events={n_known}"
            + sil_str
            + (f"  [morven active]" if cfg.do_inject and rnd >= cfg.injection_round else "")
        )

        round_metrics.append({
            "round":           rnd,
            "agg_diag_acc":    agg_diag,
            "mean_loss":       float(np.mean(_finite)) if _finite else float("nan"),
            "train_sizes":     train_sizes,
            "n_events":        n_known,
            "morven_injected": cfg.do_inject and rnd >= cfg.injection_round,
        })

    # ── Final eval ────────────────────────────────────────────────────────────
    final_diag_list = []
    for i, learner in enumerate(learners):
        if global_w is not None:
            learner.set_weights(global_w)
        m = learner.evaluate(holdouts[i]) if holdouts[i] else {}
        learner.release()
        final_diag_list.append(m.get("diag_acc", float("nan")))

    final_diag = float(np.nanmean(final_diag_list))
    elapsed    = time.time() - t_start
    print(f"\n  Final holdout diag={final_diag:.3f}  wall={elapsed:.0f}s")

    # ── Save outputs ──────────────────────────────────────────────────────────
    with (out_dir / "round_metrics.json").open("w") as f:
        json.dump(round_metrics, f, indent=2)
    with (out_dir / "silhouette.json").open("w") as f:
        json.dump(silhouette_curve, f, indent=2)
    for rnd, snap in embedding_snapshots.items():
        np.savez_compressed(
            out_dir / f"logits_r{rnd:02d}.npz",
            logits=snap["logits"], cls=snap["cls"],
        )

    summary = {
        "mode":            "sir",
        "days_per_round":  days_per_round,
        "n_agents":        n_agents,
        "n_silos":         cfg.n_silos,
        "n_rounds":        cfg.n_rounds,
        "injection_round": cfg.injection_round,
        "do_inject":       cfg.do_inject,
        "final_diag_acc":  final_diag,
        "wall_seconds":    elapsed,
        "silhouette_curve": silhouette_curve,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if embedding_snapshots:
        _make_umap_plots(embedding_snapshots, probe_labels, cfg, out_dir, silhouette_curve)

    print(f"\n  Results written to {out_dir}/")
    return {"summary": summary, "round_metrics": round_metrics}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> UnknownDiseaseConfig:
    p = argparse.ArgumentParser(description="Unknown disease embedding experiment")
    p.add_argument("--schedule",          default="gaussian")
    p.add_argument("--n-silos",           type=int,   default=3)
    p.add_argument("--n-rounds",          type=int,   default=20)
    p.add_argument("--events-per-silo",   type=int,   default=200)
    p.add_argument("--injection-round",   type=int,   default=10)
    p.add_argument("--injection-per-round", type=int, default=8)
    p.add_argument("--no-injection",      action="store_true")
    p.add_argument("--n-exposed-silos",   type=int,   default=1)
    p.add_argument("--gaussian-mu",       type=float, default=10.0)
    p.add_argument("--probe-per-band",    type=int,   default=12)
    p.add_argument("--training-device",   default="cuda")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--results-dir",       default="results/unknown_disease")
    p.add_argument("--run-name",          default="")
    p.add_argument("--snap-rounds",       default="2,5,8,10,12,15,20",
                   help="Comma-separated round numbers to snapshot embeddings")
    p.add_argument("--sir",               action="store_true",
                   help="Drive patient volume via live SIR epidemic instead of phrase-bank schedule")
    p.add_argument("--sir-days-per-round", type=int,   default=3,
                   help="SIR mode: simulation days advanced per FL round (default 3)")
    p.add_argument("--sir-n-agents",      type=int,   default=60,
                   help="SIR mode: agents per silo in the known-disease world (default 60)")
    p.add_argument("--sir-beta-scale",    type=float, default=2.0,
                   help="SIR mode: beta multiplier (default 2.0, matching sir-cal-2x)")
    p.add_argument("--ollama",            action="store_true",
                   help="SIR mode: use OllamaFictionalDataSource (phi3:mini) instead of phrase banks")
    p.add_argument("--case-summary",      action="store_true",
                   help="SIR mode: compile a structured case summary after each conversation; "
                        "DistilBERT classifies the summary instead of raw patient turns")
    p.add_argument("--ollama-summary",    action="store_true",
                   help="SIR mode: use phi3:mini to write the case summary (requires Ollama); "
                        "falls back to template summarizer if unavailable")
    args = p.parse_args()

    snap_rounds = [int(x) for x in args.snap_rounds.split(",")]

    return UnknownDiseaseConfig(
        schedule            = args.schedule,
        n_silos             = args.n_silos,
        n_rounds            = args.n_rounds,
        events_per_silo     = args.events_per_silo,
        injection_round     = args.injection_round,
        injection_per_round = args.injection_per_round,
        do_inject           = not args.no_injection,
        n_exposed_silos     = args.n_exposed_silos,
        gaussian_mu         = args.gaussian_mu,
        probe_per_band      = args.probe_per_band,
        training_device     = args.training_device,
        seed                = args.seed,
        results_dir         = args.results_dir,
        run_name            = args.run_name,
        snap_rounds         = snap_rounds,
    ), args


if __name__ == "__main__":
    cfg, args = _parse_args()
    if args.sir:
        case_summarizer = None
        if getattr(args, "case_summary", False) or getattr(args, "ollama_summary", False):
            if getattr(args, "ollama_summary", False):
                from simulation.case_summary import OllamaCaseSummarizer
                case_summarizer = OllamaCaseSummarizer()
            else:
                from simulation.case_summary import TemplateCaseSummarizer
                case_summarizer = TemplateCaseSummarizer()
        run_sir_unknown_disease(cfg,
                                days_per_round=args.sir_days_per_round,
                                n_agents=args.sir_n_agents,
                                beta_scale=args.sir_beta_scale,
                                use_ollama=args.ollama,
                                case_summarizer=case_summarizer)
    else:
        run_unknown_disease(cfg)
