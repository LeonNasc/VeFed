"""
Scheduled FL runner — isolates SIR dynamics from FL convergence.

Two data modes
--------------
"replay"   : Load an existing train.jsonl pool (Ollama-generated or phrase-
             library-generated) and reveal events to each silo according to
             the schedule. Holdout is the existing holdout.jsonl. Identical
             data, varying temporal structure.

"generate" : Use PhraseLibrary to synthesise records on the fly. No on-disk
             dataset needed. Fastest mode; good for quick FL convergence tests.

In both modes the FL training loop is identical to the main ablation:
  evaluate_holdout() → train on newly-revealed events → FedAvg → repeat.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ScheduledFLConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    mode: str = "generate"          # "replay" | "generate"

    # replay mode: point at an existing dataset directory
    # (expects silo_0/train.jsonl, silo_0/holdout.jsonl, …)
    dataset_dir: str = ""

    # generate mode: synthesise via a text generator
    # "phrase_library" | "template" — or pass a generator object via run_scheduled_fl()
    generator: str = "phrase_library"
    n_silos: int = 3
    events_per_silo: int = 200      # total pool size per silo
    # disease/severity distributions per silo for non-IID
    # list[list[tuple[disease,severity,personality,weight]]]
    # if None, uses a balanced IID default
    silo_distributions: Optional[list[list[tuple[str, str, str, float]]]] = None
    holdout_fraction: float = 0.2
    gen_seed: int = 42

    # ── Schedule ──────────────────────────────────────────────────────────────
    schedule_name: str = "gaussian"   # flat|ramp|gaussian|burst|parabola
    # Per-silo overrides — models epidemic phase asynchrony across hospitals.
    # e.g. ["burst", "gaussian", "ramp"] → silo 0 peaks early, 1 mid, 2 late.
    # Silos not covered fall back to schedule_name.
    silo_schedule_names: Optional[list[str]] = None
    n_rounds: int = 20
    # override total events revealed (defaults to pool size if None)
    total_events_override: Optional[int] = None
    # gaussian-specific
    gaussian_mu: Optional[float] = None
    gaussian_sigma: Optional[float] = None
    # burst-specific
    burst_tau: Optional[float] = None

    # ── Data difficulty ───────────────────────────────────────────────────────
    # Fraction of records whose text is drawn from a different disease bucket
    # (atypical presentation). Ground-truth label is always correct.
    # 0.0 = perfectly clean; 0.10 → ~90% accuracy ceiling.
    confusion_rate: float = 0.0

    # ── FL hypers ─────────────────────────────────────────────────────────────
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    local_epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-4
    min_events_to_train: int = 5
    # Silos training fewer examples than this are excluded from the FedAvg aggregate.
    # 0 = include all silos that ran local training (original behaviour).
    fedavg_min_examples: int = 0
    train_sample_cap: int = 512
    training_device: str = "cpu"

    # ── Output ────────────────────────────────────────────────────────────────
    run_name: str = ""
    results_dir: str = "results/scheduled"
    seed: int = 42


# ── IID default distribution ──────────────────────────────────────────────────

_IID_DISTRIBUTION: list[tuple[str, str, str, float]] = [
    # (disease, severity, personality, weight)
    ("influenza",       "mild",     "stoic",   1.0),
    ("influenza",       "mild",     "neutral", 2.0),
    ("influenza",       "mild",     "anxious", 1.0),
    ("influenza",       "moderate", "stoic",   1.0),
    ("influenza",       "moderate", "neutral", 2.0),
    ("influenza",       "moderate", "anxious", 1.0),
    ("influenza",       "severe",   "neutral", 1.0),
    ("pneumonia",       "mild",     "stoic",   1.0),
    ("pneumonia",       "mild",     "neutral", 2.0),
    ("pneumonia",       "mild",     "anxious", 1.0),
    ("pneumonia",       "moderate", "stoic",   1.0),
    ("pneumonia",       "moderate", "neutral", 2.0),
    ("pneumonia",       "moderate", "anxious", 1.0),
    ("pneumonia",       "severe",   "neutral", 1.0),
    ("non-infectious",  "mild",     "neutral", 1.0),
    ("non-infectious",  "mild",     "anxious", 2.0),
    ("non-infectious",  "discharge","neutral", 1.0),
]


def _weighted_sample(dist: list[tuple[str, str, str, float]], rng: random.Random):
    total = sum(w for *_, w in dist)
    r = rng.random() * total
    cumulative = 0.0
    for disease, severity, personality, w in dist:
        cumulative += w
        if r <= cumulative:
            return disease, severity, personality
    return dist[-1][0], dist[-1][1], dist[-1][2]


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _build_replay_pools(
    dataset_dir: Path,
    n_silos: int,
    rng: random.Random,
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Return (train_pools, holdouts) for each silo from existing JSONL."""
    train_pools, holdouts = [], []
    for i in range(n_silos):
        silo_dir = dataset_dir / f"silo_{i}"
        pool  = _load_jsonl(silo_dir / "train.jsonl")
        hold  = _load_jsonl(silo_dir / "holdout.jsonl")
        if not pool:
            raise FileNotFoundError(f"No train.jsonl found at {silo_dir}")
        rng.shuffle(pool)           # randomise reveal order
        train_pools.append(pool)
        holdouts.append(hold)
    return train_pools, holdouts


def _make_generator(cfg: ScheduledFLConfig):
    """Instantiate the text generator from cfg.generator string or pass-through object."""
    if hasattr(cfg.generator, "sample"):
        return cfg.generator   # already a generator object
    cr = getattr(cfg, "confusion_rate", 0.0)
    if cfg.generator == "template":
        from simulation.phrase_sampler import TemplateSampler
        return TemplateSampler(seed=cfg.gen_seed, confusion_rate=cr)
    from simulation.phrase_sampler import PhraseLibrary
    return PhraseLibrary(seed=cfg.gen_seed, confusion_rate=cr)


def _build_generated_pools(
    cfg: ScheduledFLConfig,
    rng: random.Random,
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Generate fresh records via the configured text generator."""
    lib = _make_generator(cfg)

    if cfg.silo_distributions is None:
        dists = [_IID_DISTRIBUTION] * cfg.n_silos
    else:
        dists = cfg.silo_distributions

    holdout_frac = cfg.holdout_fraction
    train_pools, holdouts = [], []

    for i, dist in enumerate(dists):
        all_recs = []
        days_rng = random.Random(cfg.gen_seed + i * 1000)
        for _ in range(cfg.events_per_silo):
            disease, severity, personality = _weighted_sample(dist, rng)
            days = days_rng.randint(1, 14)
            rec  = lib.sample(disease, severity, personality, days=days)
            all_recs.append(rec)

        rng.shuffle(all_recs)
        n_hold  = max(1, int(len(all_recs) * holdout_frac))
        holdouts.append(all_recs[:n_hold])
        train_pools.append(all_recs[n_hold:])

    return train_pools, holdouts


# ── Main runner ───────────────────────────────────────────────────────────────

def run_scheduled_fl(cfg: ScheduledFLConfig) -> dict:
    """
    Run the scheduled FL experiment.

    Returns
    -------
    dict with keys:
      "config"       : cfg
      "schedule"     : list[int] — events-per-round schedule used
      "rounds"       : list[dict] — per-round metrics
      "final_diag"   : float — holdout diag acc of final global model
      "final_triage" : float — holdout triage acc of final global model
    """
    from fl.lora import LoRAConfig
    from fl.learner import FLLearner
    from fl.schedules import make_schedule
    from fl.train import _fedavg

    t_start = time.time()
    rng = random.Random(cfg.seed)

    # ── Build data pools ─────────────────────────────────────────────────────
    if cfg.mode == "replay":
        if not cfg.dataset_dir:
            raise ValueError("dataset_dir must be set for replay mode")
        ds_dir = Path(cfg.dataset_dir)
        n_silos = sum(1 for p in ds_dir.iterdir()
                      if p.is_dir() and p.name.startswith("silo_"))
        train_pools, holdouts = _build_replay_pools(ds_dir, n_silos, rng)
    else:
        n_silos = cfg.n_silos
        train_pools, holdouts = _build_generated_pools(cfg, rng)

    # ── Build per-silo schedules ──────────────────────────────────────────────
    pool_sizes   = [len(p) for p in train_pools]
    min_pool     = min(pool_sizes)
    total_events = cfg.total_events_override or min_pool

    def _build_one_schedule(name: str) -> list[int]:
        kwargs: dict = {}
        # "gaussian@10" → base="gaussian", mu=10.0
        base_name = name.split("@")[0]
        if "@" in name:
            kwargs["mu"] = float(name.split("@", 1)[1])
        if base_name == "gaussian":
            if "mu" not in kwargs and cfg.gaussian_mu    is not None: kwargs["mu"]    = cfg.gaussian_mu
            if cfg.gaussian_sigma is not None: kwargs["sigma"] = cfg.gaussian_sigma
        elif base_name == "burst":
            if cfg.burst_tau is not None: kwargs["tau"] = cfg.burst_tau
        return make_schedule(base_name, total_events, cfg.n_rounds, **kwargs)

    if cfg.silo_schedule_names:
        schedules = [
            _build_one_schedule(
                cfg.silo_schedule_names[i]
                if i < len(cfg.silo_schedule_names)
                else cfg.schedule_name
            )
            for i in range(n_silos)
        ]
        sched_label = "+".join(
            cfg.silo_schedule_names[i] if i < len(cfg.silo_schedule_names)
            else cfg.schedule_name
            for i in range(n_silos)
        )
    else:
        s = _build_one_schedule(cfg.schedule_name)
        schedules  = [s] * n_silos
        sched_label = cfg.schedule_name

    # ── Build learners ────────────────────────────────────────────────────────
    lora_cfg = LoRAConfig(
        model_name_or_path = "distilbert-base-uncased",
        num_labels         = None,   # filled by FLLearner
        rank               = cfg.lora_rank,
        lora_alpha         = cfg.lora_alpha,
        lora_dropout       = cfg.lora_dropout,
    )
    learners = [
        FLLearner(
            lora_config         = lora_cfg,
            label_space         = "disease",
            min_events_to_train = cfg.min_events_to_train,
            local_epochs        = cfg.local_epochs,
            batch_size          = cfg.batch_size,
            lr                  = cfg.lr,
            train_sample_cap    = cfg.train_sample_cap,
            device              = cfg.training_device,
        )
        for _ in range(n_silos)
    ]

    # ── Cursors into each pool ────────────────────────────────────────────────
    cursors = [0] * n_silos

    # Count of events revealed per silo (used for min_events_to_train check).
    # FLLearner accumulates in its own replay buffer; we don't re-pass history.
    total_revealed: list[int] = [0] * n_silos

    # FedAvg weights persist as a Python variable across rounds (same pattern
    # as fl/train.py).  None = random init for round 1.
    global_w: list | None = None

    round_metrics: list[dict] = []

    print(f"\n{'═'*60}")
    print(f"  Scheduled FL — {sched_label.upper()} schedule")
    print(f"  mode={cfg.mode}  silos={n_silos}  rounds={cfg.n_rounds}")
    print(f"  total_events_per_silo={total_events}")
    for i, s in enumerate(schedules):
        print(f"  silo_{i}: {s}")
    print(f"{'═'*60}")

    for r in range(cfg.n_rounds):
        # ── Reveal new events this round (per-silo schedule) ─────────────────
        new_this_round: list[list[dict]] = []
        for i in range(n_silos):
            n_reveal_i = schedules[i][r]
            new_events = train_pools[i][cursors[i]: cursors[i] + n_reveal_i]
            cursors[i]         = min(cursors[i] + n_reveal_i, len(train_pools[i]))
            total_revealed[i] += len(new_events)
            new_this_round.append(new_events)

        # ── Per-silo: apply weights → evaluate → train → collect → release ───
        #
        # Mirror the pattern in fl/train.py:
        #   set_weights(global_w) → evaluate → train → get_weights → release
        #
        # Crucially, get_weights() is called BEFORE release() so we capture
        # the trained weights, not a freshly re-initialised random model.
        eval_metrics:  list[dict]        = []
        round_weights: list[list]        = []
        train_sizes:   list[int]         = []
        losses:        list[float]       = []

        for i, learner in enumerate(learners):
            # Apply last round's FedAvg result (rebuilds model if needed)
            if global_w is not None:
                learner.set_weights(global_w)

            # Prequential evaluate (before seeing this round's new data)
            m = learner.evaluate(holdouts[i]) if holdouts[i] else {}
            eval_metrics.append(m)

            # Train on NEW events only — FLLearner's replay buffer accumulates
            new_ev = new_this_round[i]
            if total_revealed[i] >= cfg.min_events_to_train and new_ev:
                n_trained, epoch_losses = learner.train(new_ev)
                train_sizes.append(n_trained)
                losses.append(
                    float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                )
            else:
                train_sizes.append(0)
                losses.append(float("nan"))

            # Collect weights while model is still loaded
            round_weights.append(learner.get_weights())

            # Free VRAM
            learner.release()

        # ── FedAvg ────────────────────────────────────────────────────────────
        active_idx = [
            i for i, s in enumerate(train_sizes)
            if s > 0 and s >= cfg.fedavg_min_examples
        ]
        if active_idx:
            wts   = [round_weights[i] for i in active_idx]
            sizes = [train_sizes[i]   for i in active_idx]
            global_w = _fedavg(wts, sizes)

        agg_diag   = float(np.mean([m.get("diag_acc",   float("nan")) for m in eval_metrics]))
        agg_triage = float(np.mean([m.get("triage_acc", float("nan")) for m in eval_metrics]))

        rm = {
            "round":             r + 1,
            "n_reveal":          [schedules[i][r] for i in range(n_silos)],
            "agg_diag_acc":      agg_diag,
            "agg_triage_acc":    agg_triage,
            "mean_loss":         float(np.nanmean(losses)) if any(l==l for l in losses) else float("nan"),
            "silo_diag":         [m.get("diag_acc",   float("nan")) for m in eval_metrics],
            "silo_triage":       [m.get("triage_acc", float("nan")) for m in eval_metrics],
            "train_sizes":       train_sizes,
            "silos_in_fedavg":   len(active_idx),
            "cursors":           list(cursors),
        }
        round_metrics.append(rm)

        loss_str    = f"{rm['mean_loss']:.3f}" if not np.isnan(rm["mean_loss"]) else "n/a"
        reveal_str  = "+".join(str(schedules[i][r]) for i in range(n_silos))
        print(
            f"  R{r+1:02d} reveal=[{reveal_str}]  "
            f"diag={agg_diag:.3f}  triage={agg_triage:.3f}  loss={loss_str}"
        )

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_evals = []
    for i, learner in enumerate(learners):
        if global_w is not None:
            learner.set_weights(global_w)
        m = learner.evaluate(holdouts[i]) if holdouts[i] else {}
        learner.release()
        final_evals.append(m)

    final_diag   = float(np.mean([m.get("diag_acc",   float("nan")) for m in final_evals]))
    final_triage = float(np.mean([m.get("triage_acc", float("nan")) for m in final_evals]))

    elapsed = time.time() - t_start
    print(f"\n  Final holdout — diag={final_diag:.3f}  triage={final_triage:.3f}")
    print(f"  Wall time: {elapsed:.0f}s\n")

    # ── Save results ──────────────────────────────────────────────────────────
    run_name = cfg.run_name or f"{cfg.schedule_name}_{int(time.time())}"
    out_dir  = Path(cfg.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "round_metrics.json").open("w") as f:
        json.dump(round_metrics, f, indent=2)
    with (out_dir / "summary.json").open("w") as f:
        json.dump({
            "schedule":        sched_label,
            "mode":            cfg.mode,
            "n_silos":         n_silos,
            "n_rounds":        cfg.n_rounds,
            "schedule_values": schedules,
            "final_diag_acc":   final_diag,
            "final_triage_acc": final_triage,
            "wall_seconds":     elapsed,
        }, f, indent=2)

    return {
        "config":       cfg,
        "schedule":     schedules,
        "rounds":       round_metrics,
        "final_diag":   final_diag,
        "final_triage": final_triage,
    }

