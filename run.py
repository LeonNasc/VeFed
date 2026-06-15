#!/usr/bin/env python3
"""
FedWorld interactive launcher.

Run with no arguments for the guided wizard:
    python run.py

Run a named preset directly:
    python run.py --preset standard
    python run.py --preset non-iid --offline
    python run.py --preset smoke --no-ollama

Or pass --help for the full non-interactive CLI:
    python run.py --help

Modes
-----
  tui        Analytics dashboard (SIR chart, clinic log, agent panels)
  rogue      Roguelike ncurses map view
  fl         Federated LoRA training across N silos (logs to W&B)

Presets (--preset <name>)
--------------------------
  smoke        2 silos · 15 agents · Influenza · fast smoke test (offline W&B)
  standard     5 silos · 300 agents · non-IID gradient · sim_days=2 · epidemic active ≥20 rounds
  multi-disease 5 silos · 100 agents · Influenza + Bacterial Pneumonia non-IID
  non-iid      5 silos · asymmetric disease mix, beta scale, and population per silo
  long-noniid-5s  5 silos · 300 agents · sim_days=2 · epidemic active ≥20 rounds · Ollama-ready
  hard-triage  3 silos · Bacterial Pneumonia only · maximum triage difficulty
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional


# ── Config object shared by all modes ─────────────────────────────────────────

@dataclass
class RunConfig:
    mode:               str        = "tui"
    num_agents:         int        = 30
    seed:               int        = 42
    progressions:       list[str]  = None   # set in __post_init__
    disease_strategy:   str        = "Influenza"
    end_condition:      str        = "extinction"
    end_condition_param: Optional[int] = None
    # FL-only
    num_silos:          int        = 3
    max_rounds:         int        = 100
    sim_days:           int        = 7
    min_events_to_train: int       = 10
    fedavg_min_examples: int       = 0
    local_epochs:       int        = 3
    wandb_project:      str        = "fedworld"
    wandb_run_name:     Optional[str] = None
    wandb_offline:      bool       = False
    use_ollama:         bool       = True
    # Non-IID: list of WorldConfig dicts (set by presets; None = all silos identical)
    world_configs:          Optional[list] = None
    reveal_incubating_icd:  bool           = True
    beta_scale:             float          = 1.0
    initial_seeds:          int            = 3
    contact_rate_sigma:     float          = 0.0
    training_device:        str            = "cpu"
    fl_backend:             str            = "inprocess"   # "inprocess" or "flower"
    # Fictional disease experiment — set to activate fictional prompts + labels
    disease_glossary:               Optional[list] = None
    doctor_label_space:             str            = "disease"
    # Ablation: explicitly tell all LLMs that influenza/pneumonia don't exist.
    # Only meaningful when disease_glossary is also set.
    explicit_real_disease_exclusion: bool          = False
    # Shared-world mode: one SIR world with N hospital districts as FL silos.
    # Only used when mode == "shared_fl".
    n_hospitals:              int            = 2
    hospital_disease_weights: Optional[list] = None  # [[p_flu, p_pneumo], ...] per hospital

    def __post_init__(self):
        if self.progressions is None:
            self.progressions = ["Influenza"]


# ── Presets ───────────────────────────────────────────────────────────────────

def _make_presets() -> dict[str, RunConfig]:
    """
    Named experiment presets. Each returns a fully configured RunConfig.
    CLI flags applied after --preset override individual fields.
    """
    from fl.train import WorldConfig

    return {
        # ══ Three-way ablation: IID → non-IID → extreme non-IID ══════════════
        # All three use identical infrastructure (2 silos, 150 agents each,
        # 7 sim-days/round, epochs=3, extinction end condition).
        # Only the disease distribution per silo changes.
        # Run each with --mode local_only / fl / centralized for the full comparison.

        # ── iid ───────────────────────────────────────────────────────────────
        # Both silos see the same 50/50 flu/pneumonia mix.
        # Establishes that FL works in the easy case.
        "iid": RunConfig(
            mode                = "fl",
            num_silos           = 2,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 200,
            min_events_to_train = 10,
            local_epochs        = 3,
            sim_days            = 4,
            max_rounds          = 50,
            wandb_offline       = True,
            use_ollama          = False,
            world_configs       = [
                WorldConfig(num_agents=400,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=10,
                            end_condition="horizon", end_condition_param=200),
                WorldConfig(num_agents=400,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=10,
                            end_condition="horizon", end_condition_param=200),
            ],
        ),

        # ── non-iid ───────────────────────────────────────────────────────────
        # Silos see the same two diseases but at different ratios (75/25 vs 25/75).
        # Tests FL robustness under partial heterogeneity.
        "non-iid": RunConfig(
            mode                = "fl",
            num_silos           = 2,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 200,
            min_events_to_train = 10,
            local_epochs        = 3,
            sim_days            = 4,
            max_rounds          = 50,
            wandb_offline       = True,
            use_ollama          = False,
            world_configs       = [
                WorldConfig(num_agents=400,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.75, 0.25],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=10,
                            end_condition="horizon", end_condition_param=200),
                WorldConfig(num_agents=400,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.25, 0.75],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=10,
                            end_condition="horizon", end_condition_param=200),
            ],
        ),

        # ── extreme-noniid ────────────────────────────────────────────────────
        # Each silo sees exactly one disease — maximum heterogeneity.
        # Horizon end condition (200 sim-days = 50 rounds × 4 days) chosen because
        # stochastic SIR with beta=2.0 is bimodal — extinction timing is unreliable
        # across seeds (see Q41). Horizon gives identical run length for fair comparison.
        # Tests whether FL can transfer knowledge across completely disjoint populations.
        "extreme-noniid": RunConfig(
            mode                = "fl",
            num_silos           = 2,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 200,
            min_events_to_train = 10,
            local_epochs        = 3,
            sim_days            = 4,
            max_rounds          = 50,
            wandb_offline       = True,
            use_ollama          = False,
            world_configs       = [
                WorldConfig(num_agents=400, progressions=["Influenza"],
                            disease_strategy="Influenza", beta_scale=2.5,
                            initial_seeds=10,
                            end_condition="horizon", end_condition_param=200),
                WorldConfig(num_agents=400, progressions=["Bacterial Pneumonia"],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=10,
                            end_condition="horizon", end_condition_param=200),
            ],
        ),

        # ── smoke ─────────────────────────────────────────────────────────────
        # Minimal run for quick pipeline testing — no Ollama, offline W&B, tiny world.
        "smoke": RunConfig(
            mode                = "fl",
            num_agents          = 15,
            num_silos           = 2,
            progressions        = ["Influenza"],
            disease_strategy    = "Influenza",
            end_condition       = "extinction",
            min_events_to_train = 3,
            local_epochs        = 1,
            wandb_offline       = True,
            use_ollama          = False,
        ),

        # ── standard ──────────────────────────────────────────────────────────
        # Default research run: 5-silo non-IID gradient, Ollama enabled.
        # sim_days=2 keeps epidemic active for ~25-30 FL rounds (β=2.0 + 300 agents
        # gives a ~50-60 sim-day arc; sim_days=2 maps that onto rounds 1:1).
        "standard": RunConfig(
            mode                = "fl",
            num_silos           = 5,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 60,
            min_events_to_train = 8,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 30,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=300,
                            progressions=["Influenza"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=300,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.75, 0.25],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=300,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=300,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.25, 0.75],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=300,
                            progressions=["Bacterial Pneumonia"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
            ],
        ),

        # ── multi-disease ─────────────────────────────────────────────────────
        # Three archetypes in circulation (Dirichlet α=0.3 assigns weights per silo).
        # Tests non-IID disease identity under mixed spread without explicit WorldConfig.
        "multi-disease": RunConfig(
            mode                = "fl",
            num_agents          = 100,
            num_silos           = 5,
            progressions        = ["Influenza", "Bacterial Pneumonia"],
            disease_strategy    = "Influenza",
            end_condition       = "extinction",
            min_events_to_train = 10,
            local_epochs        = 10,
        ),

        # ── non-iid ───────────────────────────────────────────────────────────
        # Two silos with maximally different disease profiles — the cleanest
        # non-IID design for the three-way comparison (local / FL / centralized).
        # ── static-noniid ─────────────────────────────────────────────────────
        # Non-IID without SIR dynamics. Each silo generates a fixed number of
        # clinic visits per round; `infectious_fraction` is the "slider" that
        # controls the proportion that are infectious disease cases.
        # Removes epidemic temporal distribution shift — clean FL baseline.
        "static-noniid": RunConfig(
            mode                = "fl",
            num_agents          = 1,    # unused in static mode
            num_silos           = 5,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 30,   # fixed 30 rounds
            min_events_to_train = 8,
            local_epochs        = 3,
            sim_days            = 5,    # 200 events/silo/round (40/day × 5 days)
            world_configs       = [
                WorldConfig(num_agents=1,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.70, 0.30],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=40,
                            end_condition="horizon", end_condition_param=30),
                WorldConfig(num_agents=1,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.30, 0.70],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=40,
                            end_condition="horizon", end_condition_param=30),
                WorldConfig(num_agents=1,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.50, 0.50],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=40,
                            end_condition="horizon", end_condition_param=30),
                WorldConfig(num_agents=1,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.80, 0.20],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=40,
                            end_condition="horizon", end_condition_param=30),
                WorldConfig(num_agents=1,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.20, 0.80],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=40,
                            end_condition="horizon", end_condition_param=30),
            ],
        ),

        # ── non-iid-10 ───────────────────────────────────────────────────────
        # 10 silos with deterministic disease assignments across 3 archetypes.
        # 4 Flu-dominant · 3 Corona-dominant · 3 Sepsis-dominant silos.
        # Each silo sees only one disease — maximally non-IID deterministic.
        # Designed to test whether clean convergence from the 3-silo non-iid
        # run scales to 10 silos when disease assignment is unambiguous.
        "non-iid-10": RunConfig(
            mode                = "fl",
            num_agents          = 80,
            num_silos           = 10,
            disease_strategy    = "Influenza",
            end_condition       = "extinction",
            min_events_to_train = 8,
            local_epochs        = 3,
            world_configs       = [
                # Influenza-dominant silos (0-4)
                WorldConfig(num_agents=100, progressions=["Influenza"],
                            disease_strategy="Influenza", beta_scale=1.2),
                WorldConfig(num_agents=80,  progressions=["Influenza"],
                            disease_strategy="Influenza", beta_scale=1.0),
                WorldConfig(num_agents=120, progressions=["Influenza"],
                            disease_strategy="Influenza", beta_scale=1.3),
                WorldConfig(num_agents=60,  progressions=["Influenza"],
                            disease_strategy="Influenza", beta_scale=0.9),
                WorldConfig(num_agents=80,  progressions=["Influenza"],
                            disease_strategy="Influenza", beta_scale=1.1),
                # Bacterial Pneumonia-dominant silos (5-9)
                WorldConfig(num_agents=100, progressions=["Bacterial Pneumonia"],
                            disease_strategy="Influenza", beta_scale=0.8),
                WorldConfig(num_agents=80,  progressions=["Bacterial Pneumonia"],
                            disease_strategy="Influenza", beta_scale=0.7),
                WorldConfig(num_agents=60,  progressions=["Bacterial Pneumonia"],
                            disease_strategy="Influenza", beta_scale=0.6),
                WorldConfig(num_agents=50,  progressions=["Bacterial Pneumonia"],
                            disease_strategy="Influenza", beta_scale=0.7),
                WorldConfig(num_agents=40,  progressions=["Bacterial Pneumonia"],
                            disease_strategy="Influenza", beta_scale=0.6),
            ],
        ),

        # ── hard-triage ───────────────────────────────────────────────────────
        # Bacterial Pneumonia only — higher severity, respiratory dominance,
        # slower onset. Designed to test triage of the harder disease.
        "hard-triage": RunConfig(
            mode                = "fl",
            num_agents          = 80,
            num_silos           = 3,
            progressions        = ["Bacterial Pneumonia"],
            disease_strategy    = "Influenza",
            end_condition       = "extinction",
            min_events_to_train = 8,
            local_epochs        = 10,
        ),

        # ── long-noniid-3s ────────────────────────────────────────────────────
        # 3-silo non-IID run with the long-run philosophy: sim_days=2 maps the
        # ~50-60 sim-day epidemic arc onto 25-30 FL rounds. Maximum heterogeneity:
        # pure flu / mixed / pure pneumonia. Ollama-ready (use_ollama not overridden).
        "long-noniid-3s": RunConfig(
            mode                = "fl",
            num_silos           = 3,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 60,
            min_events_to_train = 8,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 30,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=150,
                            progressions=["Influenza"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=150,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=150,
                            progressions=["Bacterial Pneumonia"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
            ],
        ),

        # ── long-burn ─────────────────────────────────────────────────────────
        # Designed for embedding evolution studies: large populations + slower
        # spread keep all silos alive for 30+ rounds at 2 sim-days/round.
        "long-burn": RunConfig(
            mode                = "fl",
            num_agents          = 300,
            num_silos           = 3,
            progressions        = ["Influenza", "Bacterial Pneumonia"],
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 80,
            min_events_to_train = 5,
            local_epochs        = 10,
            sim_days            = 2,
            beta_scale          = 0.7,
            initial_seeds       = 12,
        ),

        # ══ 5-silo overnight ablation ════════════════════════════════════════
        # Shared world config used across all 4 ablation modes
        # (centralized / local_only / fl with IID / fl with non-IID).
        # Horizon=200 sim-days = 50 rounds × 4 days — fixed run length for
        # fair metric comparison regardless of epidemic stochasticity.

        # ── ablation-iid-5s ───────────────────────────────────────────────────
        # 3 silos, all see the same 50/50 flu+pneumonia mix.
        # Easy FL case — establishes the IID ceiling for federation.
        "ablation-iid-5s": RunConfig(
            mode                = "fl",
            num_silos           = 3,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 40,
            min_events_to_train = 4,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 20,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=75,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=75,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=75,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
            ],
        ),

        # ── sir-cal-2x ───────────────────────────────────────────────────────────
        # Volume-scaling experiment: 2× agents (150/silo) to test whether doubling
        # the population linearly doubles clinic events, bringing SIR volume closer
        # to the controlled ablation target (160 events/silo).  All other parameters
        # identical to ablation-iid-5s / SIR-cal2.
        "sir-cal-2x": RunConfig(
            mode                = "fl",
            num_silos           = 3,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 40,
            min_events_to_train = 3,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 20,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=150,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            confusion_rate=0.10,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=150,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            confusion_rate=0.10,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=150,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            confusion_rate=0.10,
                            end_condition="horizon", end_condition_param=40),
            ],
        ),

        # ── sir-cal-3x ───────────────────────────────────────────────────────────
        # Volume-scaling experiment: 3× agents (225/silo) — expected to deliver
        # ~162 events/silo, matching the controlled ablation's 160-event target.
        # Confirms (or refutes) linear scaling of clinic visits with population size.
        "sir-cal-3x": RunConfig(
            mode                = "fl",
            num_silos           = 3,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 40,
            min_events_to_train = 3,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 20,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=225,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            confusion_rate=0.10,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=225,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            confusion_rate=0.10,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=225,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            confusion_rate=0.10,
                            end_condition="horizon", end_condition_param=40),
            ],
        ),

        # ── fictional-noniid ─────────────────────────────────────────────────────
        # Non-IID experiment with invented diseases (Velarex + Sornathis) to
        # eliminate phi3:mini's pretraining prior.  Silo 0 sees only Velarex;
        # silo 1 sees only Sornathis — maximum heterogeneity.
        # Use --mode fl / local_only / centralized for the three-way comparison.
        "fictional-noniid": RunConfig(
            mode                = "fl",
            num_silos           = 2,
            disease_glossary    = ["velarex", "sornathis"],
            doctor_label_space  = "fictional_disease",
            end_condition       = "horizon",
            end_condition_param = 40,
            min_events_to_train = 4,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 20,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=75,
                            progressions=["Velarex"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=75,
                            progressions=["Sornathis"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
            ],
        ),

        # ── fictional-iid ─────────────────────────────────────────────────────
        # IID baseline for fictional disease experiment: both silos see a 50/50
        # mix of Velarex and Sornathis — establishes the IID ceiling.
        "fictional-iid": RunConfig(
            mode                = "fl",
            num_silos           = 2,
            disease_glossary    = ["velarex", "sornathis"],
            doctor_label_space  = "fictional_disease",
            end_condition       = "horizon",
            end_condition_param = 40,
            min_events_to_train = 4,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 20,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=75,
                            progressions=["Velarex", "Sornathis"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=75,
                            progressions=["Velarex", "Sornathis"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
            ],
        ),

        # ── fictional-noniid-explicit ────────────────────────────────────────────
        # Same as fictional-noniid but with an explicit disclaimer injected into
        # every LLM prompt stating that influenza and pneumonia don't exist in
        # this world.  Ablation: tests whether naming-based novelty (Velarex/
        # Sornathis) does more work than explicit exclusion of real disease priors.
        "fictional-noniid-explicit": RunConfig(
            mode                             = "fl",
            num_silos                        = 2,
            disease_glossary                 = ["velarex", "sornathis"],
            doctor_label_space               = "fictional_disease",
            explicit_real_disease_exclusion  = True,
            end_condition                    = "horizon",
            end_condition_param              = 40,
            min_events_to_train              = 4,
            local_epochs                     = 3,
            sim_days                         = 2,
            max_rounds                       = 20,
            wandb_offline                    = True,
            world_configs                    = [
                WorldConfig(num_agents=75,
                            progressions=["Velarex"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=75,
                            progressions=["Sornathis"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
            ],
        ),

        # ── fictional-iid-explicit ───────────────────────────────────────────────
        # Same as fictional-iid but with explicit real-disease exclusion disclaimer.
        "fictional-iid-explicit": RunConfig(
            mode                             = "fl",
            num_silos                        = 2,
            disease_glossary                 = ["velarex", "sornathis"],
            doctor_label_space               = "fictional_disease",
            explicit_real_disease_exclusion  = True,
            end_condition                    = "horizon",
            end_condition_param              = 40,
            min_events_to_train              = 4,
            local_epochs                     = 3,
            sim_days                         = 2,
            max_rounds                       = 20,
            wandb_offline                    = True,
            world_configs                    = [
                WorldConfig(num_agents=75,
                            progressions=["Velarex", "Sornathis"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=75,
                            progressions=["Velarex", "Sornathis"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
            ],
        ),

        # ── ablation-noniid-5s ────────────────────────────────────────────────
        # 3 silos with a monotone disease gradient: pure flu → mixed → pure pneumonia.
        # Silos 0 and 2 never see the other disease locally — max heterogeneity.
        # Used for: centralized oracle, local-only baseline, and FL non-IID run.
        "ablation-noniid-5s": RunConfig(
            mode                = "fl",
            num_silos           = 3,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 40,
            min_events_to_train = 4,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 20,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=75,
                            progressions=["Influenza"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=75,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
                WorldConfig(num_agents=75,
                            progressions=["Bacterial Pneumonia"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.00,
                            initial_seeds=8, contact_rate_sigma=0.5,
                            end_condition="horizon", end_condition_param=40),
            ],
        ),

        # ── long-noniid-5s ────────────────────────────────────────────────────
        # 5-silo non-IID run designed for epidemic-active training across ≥20 FL
        # rounds. Key insight: with β=2.0 and 300 agents, the epidemic arc lasts
        # ~50-60 sim-days. sim_days=2 maps that onto 25-30 FL rounds instead of
        # 12-15 at sim_days=4. Horizon=60 (30 rounds) caps total wall time.
        # Intended for Ollama runs — use_ollama not hard-coded, defaults to True.
        "long-noniid-5s": RunConfig(
            mode                = "fl",
            num_silos           = 5,
            disease_strategy    = "Influenza",
            end_condition       = "horizon",
            end_condition_param = 60,   # 30 rounds × 2 sim_days
            min_events_to_train = 8,
            local_epochs        = 3,
            sim_days            = 2,
            max_rounds          = 30,
            wandb_offline       = True,
            world_configs       = [
                WorldConfig(num_agents=300,
                            progressions=["Influenza"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=300,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.75, 0.25],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=300,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.5, 0.5],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=300,
                            progressions=["Influenza", "Bacterial Pneumonia"],
                            disease_weights=[0.25, 0.75],
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
                WorldConfig(num_agents=300,
                            progressions=["Bacterial Pneumonia"],
                            disease_weights=None,
                            disease_strategy="Influenza", beta_scale=2.0,
                            initial_seeds=5,
                            end_condition="horizon", end_condition_param=60),
            ],
        ),

        # ══ Shared-world presets (Q42) ═══════════════════════════════════════
        # One large SIR world, N hospital districts as FL silos.
        # Eliminates phase-asynchrony noise from independent SIRs.
        # Geographic seeding creates non-IID: hospital 0 contracts flu more,
        # hospital 1 contracts pneumonia more (matching city-hospital intuition).

        # ── shared-iid ────────────────────────────────────────────────────────
        # IID baseline: both hospitals have identical 50/50 disease weights.
        # Establishes that FL works in the easy shared-world case.
        "shared-iid": RunConfig(
            mode                     = "shared_fl",
            n_hospitals              = 2,
            num_agents               = 800,
            progressions             = ["Influenza", "Bacterial Pneumonia"],
            hospital_disease_weights = [[0.5, 0.5], [0.5, 0.5]],
            end_condition            = "horizon",
            end_condition_param      = 200,
            min_events_to_train      = 10,
            local_epochs             = 3,
            sim_days                 = 4,
            max_rounds               = 50,
            wandb_offline            = True,
            use_ollama               = False,
        ),

        # ── shared-noniid ─────────────────────────────────────────────────────
        # Geographic non-IID: hospital 0 district is flu-dominant (80/20),
        # hospital 1 district is pneumonia-dominant (20/80).
        # Same epidemic arc, different local disease distributions — mirrors
        # hospitals in different neighbourhoods of the same city.
        "shared-noniid": RunConfig(
            mode                     = "shared_fl",
            n_hospitals              = 2,
            num_agents               = 800,
            progressions             = ["Influenza", "Bacterial Pneumonia"],
            hospital_disease_weights = [[0.8, 0.2], [0.2, 0.8]],
            end_condition            = "horizon",
            end_condition_param      = 200,
            min_events_to_train      = 10,
            local_epochs             = 3,
            sim_days                 = 4,
            max_rounds               = 50,
            wandb_offline            = True,
            use_ollama               = False,
        ),
    }


PRESET_DESCRIPTIONS = {
    # ── Structured ablation (use with --mode local_only / fl / centralized) ──
    "iid":           "2 silos · 150 agents · flu+pneumonia 50/50 both silos · IID baseline",
    "non-iid":       "2 silos · 150 agents · flu 75/25 vs pneumonia 25/75 · partial heterogeneity",
    "extreme-noniid":"2 silos · 150 agents · flu-only vs pneumonia-only · maximum heterogeneity",
    # ── Other presets ─────────────────────────────────────────────────────────
    "smoke":         "2 silos · 15 agents · Influenza · fast smoke test (offline W&B, no Ollama)",
    "standard":      "5 silos · 300 agents · non-IID gradient · sim_days=2 · epidemic active ≥20 rounds · Ollama-ready",
    "multi-disease": "5 silos · 100 agents · Influenza + Bacterial Pneumonia · Dirichlet non-IID",
    "non-iid-10":    "10 silos · deterministic single-disease (5×Influenza / 5×Bacterial Pneumonia)",
    "static-noniid": "5 silos · no SIR · fixed cases/day · infectious_fraction slider",
    "hard-triage":   "3 silos · Bacterial Pneumonia only · maximum triage difficulty",
    "long-burn":        "3 silos · 300 agents · Influenza + Bacterial Pneumonia · sim_days=2 · embedding studies",
    # ── Fictional disease experiment ──────────────────────────────────────────
    "fictional-noniid":          "2 silos · 75 agents · Velarex-only vs Sornathis-only · fictional non-IID",
    "fictional-iid":             "2 silos · 75 agents · Velarex+Sornathis 50/50 both silos · fictional IID",
    "fictional-noniid-explicit": "2 silos · 75 agents · fictional non-IID + explicit flu/pneumonia exclusion disclaimer",
    "fictional-iid-explicit":    "2 silos · 75 agents · fictional IID + explicit flu/pneumonia exclusion disclaimer",
    # ── 5-silo overnight ablation ─────────────────────────────────────────────
    "ablation-iid-5s":    "5 silos · 300 agents · flu+pneumonia 50/50 all silos · IID ceiling",
    # ── SIR volume-scaling ────────────────────────────────────────────────────
    "sir-cal-2x":  "3 silos · 150 agents/silo · 2× volume-scaling baseline vs ablation-iid-5s",
    "sir-cal-3x":  "3 silos · 225 agents/silo · 3× volume-scaling; targets ~160 events/silo to match controlled",
    "ablation-noniid-5s": "5 silos · 300 agents · pure-flu → mixed → pure-pneumo gradient · max heterogeneity",
    "long-noniid-3s":     "3 silos · 300 agents · flu-only / mixed / pneumo-only · sim_days=2 · Ollama-ready",
    "long-noniid-5s":     "5 silos · 300 agents · non-IID gradient · sim_days=2 → epidemic active ≥20 rounds",
    # ── Shared-world (Q42) ────────────────────────────────────────────────────
    "shared-iid":         "1 world · 2 hospital districts · 800 agents · IID 50/50 · phase-sync baseline",
    "shared-noniid":      "1 world · 2 hospital districts · 800 agents · geo non-IID (80/20 vs 20/80)",
}


# ── Interactive wizard ─────────────────────────────────────────────────────────

def _wizard() -> RunConfig:
    try:
        import questionary
        from questionary import Style
    except ImportError:
        print("Install questionary for the interactive wizard:  pip install questionary")
        sys.exit(1)

    from simulation.progression import PROGRESSION_STRATEGIES
    from simulation.strategies import STRATEGIES

    style = Style([
        ("qmark",     "fg:#22c55e bold"),
        ("question",  "fg:#f8fafc bold"),
        ("answer",    "fg:#60a5fa bold"),
        ("pointer",   "fg:#22c55e bold"),
        ("selected",  "fg:#60a5fa"),
        ("separator", "fg:#6b7280"),
        ("instruction", "fg:#6b7280 italic"),
    ])

    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   FedWorld Simulator — Launcher      ║")
    print("  ╚══════════════════════════════════════╝\n")

    # ── Preset or manual ──────────────────────────────────────────────────────
    presets = _make_presets()
    preset_choices = [
        questionary.Choice("Configure manually", value=None),
    ] + [
        questionary.Choice(f"{name:<15} — {desc}", value=name)
        for name, desc in PRESET_DESCRIPTIONS.items()
    ]
    preset_name = questionary.select(
        "Start from a preset or configure manually:",
        choices=preset_choices,
        style=style,
    ).ask()
    if preset_name is None and preset_name != None:   # Ctrl-C
        sys.exit(0)

    if preset_name:
        cfg = presets[preset_name]
        print(f"\n  Preset '{preset_name}' loaded.\n")
        customize = questionary.confirm(
            "Customize further?", default=False, style=style
        ).ask()
        if customize is None:
            sys.exit(0)
        if not customize:
            return cfg
    else:
        cfg = RunConfig()

    print()

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode_choice = questionary.select(
        "Select mode:",
        choices=[
            questionary.Choice("Town View      — 2D animated agents walking between buildings", value="pygame"),
            questionary.Choice("Analytics TUI  — SIR chart · clinic log · agent panels",       value="tui"),
            questionary.Choice("Roguelike TUI  — ncurses map view of the world",               value="rogue"),
            questionary.Choice("FL Training    — federated LoRA training, logs to W&B",        value="fl"),
        ],
        default=cfg.mode,
        style=style,
    ).ask()
    if mode_choice is None:
        sys.exit(0)
    cfg.mode = mode_choice

    print()

    # ── Disease progressions (multi-select for multi-disease worlds) ──────────
    if cfg.world_configs:
        print(f"  Disease mix defined per-silo by preset world_configs — skipping global progression.\n")
    else:
        prog_choices = [
            questionary.Choice(
                f"{name:<20} — {strat.description}",
                value=name,
                checked=(name in cfg.progressions),
            )
            for name, strat in PROGRESSION_STRATEGIES.items()
        ]
        progs = questionary.checkbox(
            "Disease progressions circulating (space to select, enter to confirm):",
            choices=prog_choices,
            style=style,
        ).ask()
        if not progs:
            sys.exit(0)
        cfg.progressions = progs

        # ── Spread strategy ──────────────────────────────────────────────────
        if "MIMIC" not in cfg.progressions:
            spread_choices = [
                questionary.Choice(f"{name:<20} — {strat.description}", value=name)
                for name, strat in STRATEGIES.items()
            ]
            spread = questionary.select(
                "Transmission spread strategy:",
                choices=spread_choices,
                default=cfg.disease_strategy,
                style=style,
            ).ask()
            if spread is None:
                sys.exit(0)
            cfg.disease_strategy = spread

    print()

    # ── End condition ──────────────────────────────────────────────────────────
    ec_choice = questionary.select(
        "Simulation end condition:",
        choices=[
            questionary.Choice("Extinction        — stop when I = 0 for N consecutive days",  value="extinction"),
            questionary.Choice("No susceptibles   — stop when S = 0 (everyone exposed)",      value="no_susceptibles"),
            questionary.Choice("Horizon           — stop after a fixed number of days",       value="horizon"),
            questionary.Choice("Training budget   — stop after N clinic events collected",    value="budget"),
        ],
        default=cfg.end_condition,
        style=style,
    ).ask()
    if ec_choice is None:
        sys.exit(0)
    cfg.end_condition = ec_choice

    param_prompts = {
        "horizon":    ("Max simulated days:",        90),
        "extinction": ("Consecutive zero-I days:",   3),
        "budget":     ("Target clinic events:",      500),
    }
    if ec_choice in param_prompts:
        label, default = param_prompts[ec_choice]
        param_raw = questionary.text(
            label,
            default=str(cfg.end_condition_param or default),
            style=style,
        ).ask()
        if param_raw is None:
            sys.exit(0)
        cfg.end_condition_param = int(param_raw)
    else:
        cfg.end_condition_param = None

    print()

    # ── Basic params ───────────────────────────────────────────────────────────
    if not cfg.world_configs:
        agents_raw = questionary.text(
            "Agents per world:", default=str(cfg.num_agents), style=style
        ).ask()
        if agents_raw is None:
            sys.exit(0)
        cfg.num_agents = int(agents_raw)

    seed_raw = questionary.text(
        "Random seed:", default=str(cfg.seed), style=style
    ).ask()
    if seed_raw is None:
        sys.exit(0)
    cfg.seed = int(seed_raw)

    # ── FL-specific params ─────────────────────────────────────────────────────
    if cfg.mode == "fl":
        print()
        if not cfg.world_configs:
            silos_raw = questionary.text(
                "Number of FL silos:", default=str(cfg.num_silos), style=style
            ).ask()
            if silos_raw is None:
                sys.exit(0)
            cfg.num_silos = int(silos_raw)

            # ── Non-IID per-silo configuration ────────────────────────────────
            noniid = questionary.confirm(
                "Configure each silo separately (non-IID)?",
                default=False, style=style,
            ).ask()
            if noniid is None:
                sys.exit(0)

            if noniid:
                from fl.train import WorldConfig
                from simulation.progression import PROGRESSION_STRATEGIES
                world_cfgs = []
                print()
                for silo_i in range(cfg.num_silos):
                    print(f"  — Silo {silo_i} configuration —")
                    s_progs = questionary.checkbox(
                        f"  Silo {silo_i} disease progressions:",
                        choices=[
                            questionary.Choice(name, checked=(name in cfg.progressions))
                            for name in PROGRESSION_STRATEGIES
                        ],
                        style=style,
                    ).ask()
                    if not s_progs:
                        sys.exit(0)

                    s_agents_raw = questionary.text(
                        f"  Silo {silo_i} agents:", default=str(cfg.num_agents), style=style
                    ).ask()
                    if s_agents_raw is None:
                        sys.exit(0)

                    s_beta_raw = questionary.text(
                        f"  Silo {silo_i} beta_scale (transmission multiplier):",
                        default="1.0", style=style,
                    ).ask()
                    if s_beta_raw is None:
                        sys.exit(0)

                    world_cfgs.append(WorldConfig(
                        num_agents       = int(s_agents_raw),
                        progressions     = s_progs,
                        disease_strategy = s_progs[0],
                        beta_scale       = float(s_beta_raw),
                        end_condition    = cfg.end_condition,
                        end_condition_param = cfg.end_condition_param,
                    ))
                    print()

                cfg.world_configs = world_cfgs
        else:
            print(f"  Silos: {cfg.num_silos} (from preset world_configs)")

        max_rounds_raw = questionary.text(
            "Max FL rounds (safety cap; run ends when all epidemics extinct):",
            default=str(cfg.max_rounds), style=style
        ).ask()
        if max_rounds_raw is None:
            sys.exit(0)
        cfg.max_rounds = int(max_rounds_raw)

        sim_days_raw = questionary.text(
            "Simulated days per FL round:", default=str(cfg.sim_days), style=style
        ).ask()
        if sim_days_raw is None:
            sys.exit(0)
        cfg.sim_days = int(sim_days_raw)

        min_ev_raw = questionary.text(
            "Min events to train per round:", default=str(cfg.min_events_to_train), style=style
        ).ask()
        if min_ev_raw is None:
            sys.exit(0)
        cfg.min_events_to_train = int(min_ev_raw)

        epochs_raw = questionary.text(
            "Local training epochs:", default=str(cfg.local_epochs), style=style
        ).ask()
        if epochs_raw is None:
            sys.exit(0)
        cfg.local_epochs = int(epochs_raw)

        print()
        project_raw = questionary.text(
            "W&B project name:", default=cfg.wandb_project, style=style
        ).ask()
        if project_raw is None:
            sys.exit(0)
        cfg.wandb_project = project_raw

        run_name_raw = questionary.text(
            "W&B run name (blank = auto):", default="", style=style
        ).ask()
        if run_name_raw is None:
            sys.exit(0)
        cfg.wandb_run_name = run_name_raw.strip() or None

        offline = questionary.confirm(
            "Run W&B offline (no upload)?", default=cfg.wandb_offline, style=style
        ).ask()
        if offline is None:
            sys.exit(0)
        cfg.wandb_offline = offline

    print()
    return cfg


# ── Non-interactive CLI ────────────────────────────────────────────────────────

def _parse_cli() -> RunConfig:
    from simulation.progression import PROGRESSION_STRATEGIES
    from simulation.strategies import STRATEGIES

    p = argparse.ArgumentParser(
        description="FedWorld Simulator — run any mode from the command line",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", choices=list(PRESET_DESCRIPTIONS), default=None,
                   help="Load a named preset; other flags override individual fields")
    p.add_argument("--mode", choices=["tui", "rogue", "pygame", "fl", "local_only", "centralized", "compare",
                                      "static_centralized", "static_local", "shared_fl"], default=None)
    p.add_argument("--agents",   type=int, default=None)
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--progressions", nargs="+", default=None,
                   help="One or more disease progressions (space-separated)")
    p.add_argument("--disease-strategy", default=None,
                   choices=list(STRATEGIES), dest="disease_strategy")
    p.add_argument("--end-condition",    default=None,
                   choices=["horizon", "extinction", "no_susceptibles", "budget"],
                   dest="end_condition")
    p.add_argument("--end-condition-param", type=int, default=None,
                   dest="end_condition_param")
    # FL
    p.add_argument("--silos",      type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None, dest="max_rounds",
                   help="Safety cap on FL rounds; run ends when all epidemics extinct")
    p.add_argument("--sim-days",   type=int, default=None, dest="sim_days")
    p.add_argument("--min-events", type=int, default=None, dest="min_events_to_train",
                   help="Skip LoRA training if fewer events this round")
    p.add_argument("--fedavg-min-examples", type=int, default=None, dest="fedavg_min_examples",
                   help="Exclude silos with fewer training examples than this from FedAvg aggregate (0 = include all)")
    p.add_argument("--epochs",     type=int, default=None, dest="local_epochs")
    p.add_argument("--wandb-project",  default=None, dest="wandb_project")
    p.add_argument("--wandb-run-name", default=None, dest="wandb_run_name")
    p.add_argument("--offline",        action="store_true", default=False, dest="wandb_offline")
    p.add_argument("--no-ollama",      action="store_true", default=False)
    p.add_argument("--ollama",         action="store_true", default=False,
                   help="Force Ollama on even when the preset disables it")
    p.add_argument("--beta-scale",     type=float, default=None, dest="beta_scale",
                   help="Transmission rate multiplier (default 1.0; <1 slows spread)")
    p.add_argument("--initial-seeds",  type=int,   default=None, dest="initial_seeds",
                   help="Number of agents infected at simulation start (default 3)")
    p.add_argument("--contact-rate-sigma", type=float, default=None, dest="contact_rate_sigma",
                   help="Std-dev of per-agent contact-rate heterogeneity (default 0.0 = uniform)")
    p.add_argument("--training-device", type=str, default=None, dest="training_device",
                   choices=["cpu", "cuda"],
                   help="Device for LoRA training (default: cpu — safe when Ollama is on GPU)")
    p.add_argument("--fl-backend", type=str, default=None, dest="fl_backend",
                   choices=["inprocess", "flower"],
                   help="FL aggregation backend: inprocess (default) or flower (Flower protocol)")

    a = p.parse_args()

    # Start from preset or defaults
    if a.preset:
        cfg = _make_presets()[a.preset]
    else:
        cfg = RunConfig()

    # Apply any explicit CLI overrides
    if a.mode              is not None: cfg.mode                = a.mode
    if a.agents            is not None: cfg.num_agents          = a.agents
    if a.seed              is not None: cfg.seed                = a.seed
    if a.progressions      is not None: cfg.progressions        = a.progressions
    if a.disease_strategy  is not None: cfg.disease_strategy    = a.disease_strategy
    if a.end_condition     is not None: cfg.end_condition       = a.end_condition
    if a.end_condition_param is not None: cfg.end_condition_param = a.end_condition_param
    if a.silos             is not None: cfg.num_silos           = a.silos
    if a.max_rounds        is not None: cfg.max_rounds          = a.max_rounds
    if a.sim_days          is not None: cfg.sim_days            = a.sim_days
    if a.min_events_to_train    is not None: cfg.min_events_to_train    = a.min_events_to_train
    if a.fedavg_min_examples    is not None: cfg.fedavg_min_examples    = a.fedavg_min_examples
    if a.local_epochs      is not None: cfg.local_epochs        = a.local_epochs
    if a.wandb_project     is not None: cfg.wandb_project       = a.wandb_project
    if a.wandb_run_name    is not None: cfg.wandb_run_name      = a.wandb_run_name
    if a.wandb_offline:                 cfg.wandb_offline       = True
    if a.no_ollama:                     cfg.use_ollama          = False
    if a.ollama:                        cfg.use_ollama          = True
    if a.beta_scale           is not None: cfg.beta_scale          = a.beta_scale
    if a.initial_seeds        is not None: cfg.initial_seeds       = a.initial_seeds
    if a.contact_rate_sigma   is not None: cfg.contact_rate_sigma  = a.contact_rate_sigma
    if a.training_device is not None:   cfg.training_device     = a.training_device
    if a.fl_backend      is not None:   cfg.fl_backend          = a.fl_backend

    # Ensure mode defaults to fl when a preset is used
    if a.preset and a.mode is None:
        pass  # preset already sets mode

    return cfg


# ── Mode launchers ─────────────────────────────────────────────────────────────

def _launch(cfg: RunConfig) -> None:
    from simulation.world import WorldEngine
    from simulation.progression import PROGRESSION_STRATEGIES
    from simulation.strategies import STRATEGIES
    from simulation.end_conditions import from_config as make_end_condition

    if cfg.mode == "fl":
        _launch_fl(cfg)
        return

    if cfg.mode == "shared_fl":
        _launch_shared(cfg)
        return

    if cfg.mode == "local_only":
        _launch_local_only(cfg)
        return

    if cfg.mode == "centralized":
        _launch_centralized(cfg)
        return

    if cfg.mode == "compare":
        _launch_compare(cfg)
        return

    if cfg.mode in ("static_centralized", "static_local"):
        _launch_static(cfg)
        return

    end_cond    = make_end_condition(cfg.end_condition, cfg.end_condition_param)
    progressions = [PROGRESSION_STRATEGIES[p] for p in cfg.progressions]
    strategy    = STRATEGIES.get(cfg.disease_strategy)

    world = WorldEngine(
        num_agents             = cfg.num_agents,
        seed                   = cfg.seed,
        progression_strategies = progressions,
        disease_strategy       = strategy,
        end_condition          = end_cond,
    )

    from simulation.interaction_log import InteractionLogger
    from simulation.ollama_client   import OllamaDiagnosticClient
    from simulation.patient_llm     import PatientLLMClient

    logger = InteractionLogger()
    world.attach_interaction_logger(logger)
    print(f"Interaction log → {logger.path}")

    patient_llm = PatientLLMClient()
    if patient_llm.health_check():
        world.set_patient_llm(patient_llm)
        print(f"Patient LLM: {patient_llm.model} active")
    else:
        patient_llm = None
        print(f"Patient LLM: unreachable — using templates (ollama pull phi3:mini)")

    client      = OllamaDiagnosticClient(patient_llm=patient_llm)
    ollama_error = None
    if client.health_check():
        q = world.clinic_queue
        world.register_diagnostic_fn(lambda ev, _q=q: client.diagnose(ev, _queue=_q))
        world.event_log.append("Ollama connected — LLM doctor active.")
    else:
        ollama_error = (
            "Ollama unreachable at localhost:11434  |  "
            "fix: ollama serve  &&  ollama pull phi3:mini"
        )

    if cfg.mode == "tui":
        import curses
        from ui.tui import TUI

        def _run(stdscr):
            curses.curs_set(0)
            curses.start_color()
            curses.use_default_colors()
            for i, (fg, bg) in enumerate([
                (curses.COLOR_GREEN, -1), (curses.COLOR_RED, -1),
                (curses.COLOR_YELLOW, -1), (curses.COLOR_CYAN, -1),
                (curses.COLOR_MAGENTA, -1), (curses.COLOR_WHITE, -1),
                (curses.COLOR_BLUE, -1), (curses.COLOR_RED, curses.COLOR_RED),
                (curses.COLOR_BLACK, curses.COLOR_GREEN),
                (curses.COLOR_BLACK, curses.COLOR_YELLOW),
            ], start=1):
                curses.init_pair(i, fg, bg)
            TUI(stdscr, world, ollama_error=ollama_error, ollama_client=client).run()

        curses.wrapper(_run)

    elif cfg.mode == "rogue":
        import curses
        from ui.rogue_tui import RogueTUI
        curses.wrapper(lambda s: RogueTUI(s, world).run())

    elif cfg.mode == "pygame":
        from ui.pygame_world import PygameWorld
        PygameWorld(world).run()



def _make_fl_cfg(cfg: RunConfig) -> "FLTrainConfig":
    from fl.train import FLTrainConfig
    return FLTrainConfig(
        num_silos           = cfg.num_silos,
        max_rounds          = cfg.max_rounds,
        num_agents          = cfg.num_agents,
        sim_days            = cfg.sim_days,
        min_events_to_train = cfg.min_events_to_train,
        fedavg_min_examples = cfg.fedavg_min_examples,
        local_epochs        = cfg.local_epochs,
        progressions        = cfg.progressions,
        disease_strategy    = cfg.disease_strategy,
        seed                = cfg.seed,
        end_condition       = cfg.end_condition,
        end_condition_param = cfg.end_condition_param,
        wandb_project       = cfg.wandb_project,
        wandb_run_name      = cfg.wandb_run_name,
        wandb_offline       = cfg.wandb_offline,
        use_ollama          = cfg.use_ollama,
        world_configs       = cfg.world_configs,
        beta_scale          = cfg.beta_scale,
        initial_seeds       = cfg.initial_seeds,
        contact_rate_sigma  = cfg.contact_rate_sigma,
        training_device     = cfg.training_device,
        fl_backend          = cfg.fl_backend,
        disease_glossary                 = cfg.disease_glossary,
        doctor_label_space               = cfg.doctor_label_space,
        explicit_real_disease_exclusion  = cfg.explicit_real_disease_exclusion,
    )


def _make_shared_cfg(cfg: RunConfig) -> "SharedWorldConfig":
    from fl.train import SharedWorldConfig
    return SharedWorldConfig(
        n_hospitals              = cfg.n_hospitals,
        n_agents                 = cfg.num_agents,
        progressions             = cfg.progressions,
        hospital_disease_weights = cfg.hospital_disease_weights,
        end_condition            = cfg.end_condition,
        end_condition_param      = cfg.end_condition_param or 200,
        min_events_to_train      = cfg.min_events_to_train,
        local_epochs             = cfg.local_epochs,
        sim_days                 = cfg.sim_days,
        max_rounds               = cfg.max_rounds,
        seed                     = cfg.seed,
        training_device          = cfg.training_device,
        wandb_project            = cfg.wandb_project,
        wandb_run_name           = cfg.wandb_run_name,
        wandb_offline            = cfg.wandb_offline,
        use_ollama               = cfg.use_ollama,
    )


def _launch_fl(cfg: RunConfig) -> None:
    from fl.train import run_federated_training
    run_federated_training(_make_fl_cfg(cfg))


def _launch_shared(cfg: RunConfig) -> None:
    from fl.train import run_shared_world_training
    run_shared_world_training(_make_shared_cfg(cfg))


def _launch_local_only(cfg: RunConfig) -> None:
    from fl.train import run_local_only_training
    run_local_only_training(_make_fl_cfg(cfg))


def _launch_centralized(cfg: RunConfig) -> None:
    from fl.train import run_centralized_training
    run_centralized_training(_make_fl_cfg(cfg))


def _launch_compare(cfg: RunConfig) -> None:
    from fl.train import run_comparison
    run_comparison(_make_fl_cfg(cfg))


def _launch_static(cfg: RunConfig) -> None:
    from fl.train import run_static_training
    mode = "centralized" if cfg.mode == "static_centralized" else "local"
    run_static_training(_make_fl_cfg(cfg), mode=mode)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            cfg = _wizard()
        else:
            cfg = _parse_cli()
        _launch(cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.")
