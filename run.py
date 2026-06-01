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
  smoke        2 silos · 15 agents · Standard Flu · fast smoke test (offline W&B)
  standard     3 silos · 60 agents · Flu + Corona · typical research run
  multi-disease 5 silos · 100 agents · 4 diseases circulating simultaneously
  non-iid      5 silos · asymmetric disease mix, beta scale, and population per silo
  hard-triage  3 silos · Slow Burn + Deadly · maximum triage difficulty
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
    disease_strategy:   str        = "Standard Flu"
    end_condition:      str        = "extinction"
    end_condition_param: Optional[int] = None
    # FL-only
    num_silos:          int        = 3
    max_rounds:         int        = 100
    sim_days:           int        = 7
    min_events_to_train: int       = 10
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

    def __post_init__(self):
        if self.progressions is None:
            self.progressions = ["Standard Flu"]


# ── Presets ───────────────────────────────────────────────────────────────────

def _make_presets() -> dict[str, RunConfig]:
    """
    Named experiment presets. Each returns a fully configured RunConfig.
    CLI flags applied after --preset override individual fields.
    """
    from fl.train import WorldConfig

    return {
        # ── smoke ─────────────────────────────────────────────────────────────
        # Minimal run for quick pipeline testing — no Ollama, offline W&B, tiny world.
        "smoke": RunConfig(
            mode                = "fl",
            num_agents          = 15,
            num_silos           = 2,
            progressions        = ["Standard Flu"],
            disease_strategy    = "Standard Flu",
            end_condition       = "extinction",
            min_events_to_train = 3,
            local_epochs        = 1,
            wandb_offline       = True,
            use_ollama          = False,
        ),

        # ── standard ──────────────────────────────────────────────────────────
        # Typical single-session research run: two circulating diseases, Ollama enabled.
        "standard": RunConfig(
            mode                = "fl",
            num_agents          = 60,
            num_silos           = 3,
            progressions        = ["Standard Flu", "Mild Corona"],
            disease_strategy    = "Standard Flu",
            end_condition       = "extinction",
            min_events_to_train = 10,
            local_epochs        = 10,
        ),

        # ── multi-disease ─────────────────────────────────────────────────────
        # Three archetypes in circulation (Dirichlet α=0.3 assigns weights per silo).
        # Tests non-IID disease identity under mixed spread without explicit WorldConfig.
        "multi-disease": RunConfig(
            mode                = "fl",
            num_agents          = 100,
            num_silos           = 5,
            progressions        = ["Standard Flu", "Mild Corona", "Slow Burn"],
            disease_strategy    = "Standard Flu",
            end_condition       = "extinction",
            min_events_to_train = 10,
            local_epochs        = 10,
        ),

        # ── non-iid ───────────────────────────────────────────────────────────
        # Three silos with explicitly different disease profiles and population sizes.
        # Silo 0: urban hospital — flu + corona, large population, high transmission.
        # Silo 1: general ward   — flu + sepsis, moderate size.
        # Silo 2: rural clinic   — slow-burn sepsis only, small population, low beta.
        "non-iid": RunConfig(
            mode                = "fl",
            num_agents          = 300,
            num_silos           = 5,
            disease_strategy    = "Standard Flu",
            end_condition       = "extinction",
            min_events_to_train = 8,
            local_epochs        = 3,
            sim_days            = 2,
            world_configs       = [
                # All silos see all 3 diseases — non-IID is in the prevalence weights.
                # S0: Flu-dominant
                WorldConfig(num_agents=300, progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.55, 0.25, 0.20],
                            disease_strategy="Standard Flu", beta_scale=1.2),
                # S1: Corona-dominant
                WorldConfig(num_agents=300, progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.15, 0.65, 0.20],
                            disease_strategy="Standard Flu", beta_scale=1.0),
                # S2: SlowBurn-dominant
                WorldConfig(num_agents=300, progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.20, 0.20, 0.60],
                            disease_strategy="Standard Flu", beta_scale=0.9),
                # S3: Flu+Corona mixed
                WorldConfig(num_agents=300, progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.40, 0.45, 0.15],
                            disease_strategy="Standard Flu", beta_scale=1.1),
                # S4: Flu+SlowBurn mixed
                WorldConfig(num_agents=300, progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.35, 0.15, 0.50],
                            disease_strategy="Standard Flu", beta_scale=0.95),
            ],
        ),

        # ── static-noniid ─────────────────────────────────────────────────────
        # Non-IID without SIR dynamics. Each silo generates a fixed number of
        # clinic visits per round; `infectious_fraction` is the "slider" that
        # controls the proportion that are infectious disease cases.
        # Removes epidemic temporal distribution shift — clean FL baseline.
        "static-noniid": RunConfig(
            mode                = "fl",
            num_agents          = 1,    # unused in static mode
            num_silos           = 5,
            disease_strategy    = "Standard Flu",
            end_condition       = "horizon",
            end_condition_param = 30,   # fixed 30 rounds
            min_events_to_train = 8,
            local_epochs        = 3,
            sim_days            = 2,
            world_configs       = [
                WorldConfig(num_agents=1,
                            progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.55, 0.25, 0.20],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=20,
                            end_condition="horizon", end_condition_param=30),
                WorldConfig(num_agents=1,
                            progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.15, 0.65, 0.20],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=20,
                            end_condition="horizon", end_condition_param=30),
                WorldConfig(num_agents=1,
                            progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.20, 0.20, 0.60],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=20,
                            end_condition="horizon", end_condition_param=30),
                WorldConfig(num_agents=1,
                            progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.40, 0.45, 0.15],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=20,
                            end_condition="horizon", end_condition_param=30),
                WorldConfig(num_agents=1,
                            progressions=["Standard Flu", "Mild Corona", "Slow Burn"],
                            disease_weights=[0.35, 0.15, 0.50],
                            static_mode=True, infectious_fraction=0.60, cases_per_day=20,
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
            disease_strategy    = "Standard Flu",
            end_condition       = "extinction",
            min_events_to_train = 8,
            local_epochs        = 3,
            world_configs       = [
                # Flu-dominant silos (0-3)
                WorldConfig(num_agents=100, progressions=["Standard Flu"],
                            disease_strategy="Standard Flu", beta_scale=1.2),
                WorldConfig(num_agents=80,  progressions=["Standard Flu"],
                            disease_strategy="Standard Flu", beta_scale=1.0),
                WorldConfig(num_agents=120, progressions=["Standard Flu"],
                            disease_strategy="Standard Flu", beta_scale=1.3),
                WorldConfig(num_agents=60,  progressions=["Standard Flu"],
                            disease_strategy="Standard Flu", beta_scale=0.9),
                # Corona-dominant silos (4-6)
                WorldConfig(num_agents=100, progressions=["Mild Corona"],
                            disease_strategy="Standard Flu", beta_scale=1.1),
                WorldConfig(num_agents=80,  progressions=["Mild Corona"],
                            disease_strategy="Standard Flu", beta_scale=1.0),
                WorldConfig(num_agents=60,  progressions=["Mild Corona"],
                            disease_strategy="Standard Flu", beta_scale=0.8),
                # Sepsis-dominant silos (7-9)
                WorldConfig(num_agents=80,  progressions=["Slow Burn"],
                            disease_strategy="Standard Flu", beta_scale=0.9),
                WorldConfig(num_agents=50,  progressions=["Slow Burn"],
                            disease_strategy="Standard Flu", beta_scale=0.7),
                WorldConfig(num_agents=40,  progressions=["Slow Burn"],
                            disease_strategy="Standard Flu", beta_scale=0.6),
            ],
        ),

        # ── hard-triage ───────────────────────────────────────────────────────
        # Slow Burn (sepsis-like plateau, muted symptoms) + Mild Corona (silent
        # hypoxia). Both are designed to fool symptom-based triage.
        "hard-triage": RunConfig(
            mode                = "fl",
            num_agents          = 80,
            num_silos           = 3,
            progressions        = ["Slow Burn", "Mild Corona"],
            disease_strategy    = "Standard Flu",
            end_condition       = "extinction",
            min_events_to_train = 8,
            local_epochs        = 10,
        ),

        # ── long-burn ─────────────────────────────────────────────────────────
        # Designed for embedding evolution studies: large populations + slower
        # spread keep all silos alive for 30+ rounds at 2 sim-days/round.
        "long-burn": RunConfig(
            mode                = "fl",
            num_agents          = 300,
            num_silos           = 3,
            progressions        = ["Slow Burn", "Mild Corona"],
            disease_strategy    = "Standard Flu",
            end_condition       = "horizon",
            end_condition_param = 80,
            min_events_to_train = 5,
            local_epochs        = 10,
            sim_days            = 2,
            beta_scale          = 0.7,
            initial_seeds       = 12,
        ),
    }


PRESET_DESCRIPTIONS = {
    "smoke":        "2 silos · 15 agents · Standard Flu · fast smoke test (offline W&B, no Ollama)",
    "standard":     "3 silos · 60 agents · Flu + Corona · typical research run",
    "multi-disease":"5 silos · 100 agents · 3 archetypes · Dirichlet α=0.3 non-IID",
    "non-iid":      "3 silos · explicit disease profiles (Flu+Corona / Flu+Sepsis / Sepsis)",
    "non-iid-10":   "10 silos · deterministic single-disease (4×Flu / 3×Corona / 3×Sepsis)",
    "static-noniid":"5 silos · no SIR · fixed cases/day · infectious_fraction slider",
    "hard-triage":  "3 silos · Slow Burn + Mild Corona · maximum triage difficulty",
    "long-burn":    "3 silos · 300 agents · Slow Burn + Corona · sim_days=2 · embedding studies",
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
            questionary.Choice("Analytics TUI  — SIR chart · clinic log · agent panels", value="tui"),
            questionary.Choice("Roguelike TUI  — ncurses map view of the world",          value="rogue"),
            questionary.Choice("FL Training    — federated LoRA training, logs to W&B",   value="fl"),
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
    p.add_argument("--mode", choices=["tui", "rogue", "fl", "centralized", "compare"], default=None)
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
    p.add_argument("--epochs",     type=int, default=None, dest="local_epochs")
    p.add_argument("--wandb-project",  default=None, dest="wandb_project")
    p.add_argument("--wandb-run-name", default=None, dest="wandb_run_name")
    p.add_argument("--offline",        action="store_true", default=False, dest="wandb_offline")
    p.add_argument("--no-ollama",      action="store_true", default=False)
    p.add_argument("--beta-scale",     type=float, default=None, dest="beta_scale",
                   help="Transmission rate multiplier (default 1.0; <1 slows spread)")
    p.add_argument("--initial-seeds",  type=int,   default=None, dest="initial_seeds",
                   help="Number of agents infected at simulation start (default 3)")

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
    if a.min_events_to_train is not None: cfg.min_events_to_train = a.min_events_to_train
    if a.local_epochs      is not None: cfg.local_epochs        = a.local_epochs
    if a.wandb_project     is not None: cfg.wandb_project       = a.wandb_project
    if a.wandb_run_name    is not None: cfg.wandb_run_name      = a.wandb_run_name
    if a.wandb_offline:                 cfg.wandb_offline       = True
    if a.no_ollama:                     cfg.use_ollama          = False
    if a.beta_scale    is not None:     cfg.beta_scale          = a.beta_scale
    if a.initial_seeds is not None:     cfg.initial_seeds       = a.initial_seeds

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

    if cfg.mode == "centralized":
        _launch_centralized(cfg)
        return

    if cfg.mode == "compare":
        _launch_compare(cfg)
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
        print(f"Patient LLM: unreachable — using templates (ollama pull tinyllama)")

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



def _make_fl_cfg(cfg: RunConfig) -> "FLTrainConfig":
    from fl.train import FLTrainConfig
    return FLTrainConfig(
        num_silos           = cfg.num_silos,
        max_rounds          = cfg.max_rounds,
        num_agents          = cfg.num_agents,
        sim_days            = cfg.sim_days,
        min_events_to_train = cfg.min_events_to_train,
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
    )


def _launch_fl(cfg: RunConfig) -> None:
    from fl.train import run_federated_training
    run_federated_training(_make_fl_cfg(cfg))


def _launch_centralized(cfg: RunConfig) -> None:
    from fl.train import run_centralized_training
    run_centralized_training(_make_fl_cfg(cfg))


def _launch_compare(cfg: RunConfig) -> None:
    from fl.train import run_comparison
    run_comparison(_make_fl_cfg(cfg))


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
