#!/usr/bin/env python3
"""
FedWorld interactive launcher.

Run with no arguments for the guided wizard:
    python run.py

Or pass --help for the full non-interactive CLI:
    python run.py --help

Modes
-----
  tui        Analytics dashboard (SIR chart, clinic log, agent panels)
  rogue      Roguelike ncurses map view
  pygame     Pygame town view with moving agents
  fl         Federated LoRA training across N silos (logs to W&B)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional


# ── Config object shared by all modes ─────────────────────────────────────────

@dataclass
class RunConfig:
    mode:               str   = "tui"
    num_agents:         int   = 30
    seed:               int   = 42
    progression:        str   = "Standard Flu"
    disease_strategy:   str   = "Standard Flu"
    end_condition:      str   = "horizon"
    end_condition_param: Optional[int] = 90
    # FL-only
    num_silos:          int   = 3
    num_rounds:         int   = 10
    sim_days:           int   = 7
    min_events_to_train: int  = 10
    local_epochs:       int   = 3
    wandb_project:      str   = "fedworld"
    wandb_run_name:     Optional[str] = None
    wandb_offline:      bool  = False


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

    cfg = RunConfig()

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode_choice = questionary.select(
        "Select mode:",
        choices=[
            questionary.Choice("Analytics TUI  — SIR chart · clinic log · agent panels", value="tui"),
            questionary.Choice("Roguelike TUI  — ncurses map view of the world",          value="rogue"),
            questionary.Choice("Pygame         — visual town with moving agents",          value="pygame"),
            questionary.Choice("FL Training    — federated LoRA training, logs to W&B",   value="fl"),
        ],
        style=style,
    ).ask()
    if mode_choice is None:
        sys.exit(0)
    cfg.mode = mode_choice

    print()

    # ── Disease progression ────────────────────────────────────────────────────
    prog_choices = [
        questionary.Choice(
            f"{name:<20} — {strat.description}",
            value=name,
        )
        for name, strat in PROGRESSION_STRATEGIES.items()
    ]
    prog = questionary.select(
        "Disease progression strategy:",
        choices=prog_choices,
        style=style,
    ).ask()
    if prog is None:
        sys.exit(0)
    cfg.progression = prog

    # ── Spread strategy (only if not MIMIC — MIMIC bundles its own dynamics) ──
    if prog != "MIMIC":
        spread_choices = [
            questionary.Choice(f"{name:<20} — {strat.description}", value=name)
            for name, strat in STRATEGIES.items()
        ]
        spread = questionary.select(
            "Transmission spread strategy:",
            choices=spread_choices,
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
            questionary.Choice("Horizon           — stop after a fixed number of days",       value="horizon"),
            questionary.Choice("No susceptibles   — stop when S = 0 (everyone exposed)",      value="no_susceptibles"),
            questionary.Choice("Extinction        — stop when I = 0 for N consecutive days",  value="extinction"),
            questionary.Choice("Training budget   — stop after N clinic events collected",    value="budget"),
        ],
        style=style,
    ).ask()
    if ec_choice is None:
        sys.exit(0)
    cfg.end_condition = ec_choice

    # Parameter for conditions that need one
    param_prompts = {
        "horizon":    ("Max simulated days:",        90),
        "extinction": ("Consecutive zero-I days:",   3),
        "budget":     ("Target clinic events:",      500),
    }
    if ec_choice in param_prompts:
        label, default = param_prompts[ec_choice]
        param_raw = questionary.text(
            label,
            default=str(default),
            style=style,
        ).ask()
        if param_raw is None:
            sys.exit(0)
        cfg.end_condition_param = int(param_raw)
    else:
        cfg.end_condition_param = None

    print()

    # ── Basic params ───────────────────────────────────────────────────────────
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
        silos_raw = questionary.text(
            "Number of FL silos:", default=str(cfg.num_silos), style=style
        ).ask()
        if silos_raw is None:
            sys.exit(0)
        cfg.num_silos = int(silos_raw)

        rounds_raw = questionary.text(
            "FL rounds:", default=str(cfg.num_rounds), style=style
        ).ask()
        if rounds_raw is None:
            sys.exit(0)
        cfg.num_rounds = int(rounds_raw)

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
            "Run W&B offline (no upload)?", default=False, style=style
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
    p.add_argument("--mode", choices=["tui", "rogue", "pygame", "fl"], default="tui")
    p.add_argument("--agents",   type=int, default=30)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--progression",      default="Standard Flu",
                   choices=list(PROGRESSION_STRATEGIES))
    p.add_argument("--disease-strategy", default="Standard Flu",
                   choices=list(STRATEGIES), dest="disease_strategy")
    p.add_argument("--end-condition",    default="horizon",
                   choices=["horizon", "extinction", "no_susceptibles", "budget"],
                   dest="end_condition")
    p.add_argument("--end-condition-param", type=int, default=90,
                   dest="end_condition_param")
    # FL
    p.add_argument("--silos",      type=int, default=3)
    p.add_argument("--rounds",     type=int, default=10)
    p.add_argument("--sim-days",   type=int, default=7,   dest="sim_days")
    p.add_argument("--min-events", type=int, default=10,  dest="min_events_to_train",
                   help="Skip LoRA training if fewer events this round")
    p.add_argument("--epochs",     type=int, default=3,   dest="local_epochs")
    p.add_argument("--wandb-project",  default="fedworld",  dest="wandb_project")
    p.add_argument("--wandb-run-name", default=None,         dest="wandb_run_name")
    p.add_argument("--offline",   action="store_true",  dest="wandb_offline")

    a = p.parse_args()
    return RunConfig(
        mode               = a.mode,
        num_agents         = a.agents,
        seed               = a.seed,
        progression        = a.progression,
        disease_strategy   = a.disease_strategy,
        end_condition      = a.end_condition,
        end_condition_param= a.end_condition_param,
        num_silos           = a.silos,
        num_rounds          = a.rounds,
        sim_days            = a.sim_days,
        min_events_to_train = a.min_events_to_train,
        local_epochs        = a.local_epochs,
        wandb_project      = a.wandb_project,
        wandb_run_name     = a.wandb_run_name,
        wandb_offline      = a.wandb_offline,
    )


# ── Mode launchers ─────────────────────────────────────────────────────────────

def _launch(cfg: RunConfig) -> None:
    from simulation.world import WorldEngine
    from simulation.progression import PROGRESSION_STRATEGIES
    from simulation.strategies import STRATEGIES
    from simulation.end_conditions import from_config as make_end_condition

    end_cond = make_end_condition(cfg.end_condition, cfg.end_condition_param)
    progression = PROGRESSION_STRATEGIES[cfg.progression]
    strategy    = STRATEGIES.get(cfg.disease_strategy)

    if cfg.mode == "fl":
        _launch_fl(cfg)
        return

    world = WorldEngine(
        num_agents           = cfg.num_agents,
        seed                 = cfg.seed,
        progression_strategy = progression,
        disease_strategy     = strategy,
        end_condition        = end_cond,
    )

    from simulation.interaction_log import InteractionLogger
    from simulation.ollama_client import OllamaDiagnosticClient

    logger = InteractionLogger()
    world.attach_interaction_logger(logger)
    print(f"Interaction log → {logger.path}")

    client = OllamaDiagnosticClient()
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
        from ui.pygame_ui import PygameUI
        PygameUI(world).run()


def _launch_fl(cfg: RunConfig) -> None:
    from fl.train import FLTrainConfig, run_federated_training

    fl_cfg = FLTrainConfig(
        num_silos            = cfg.num_silos,
        num_rounds           = cfg.num_rounds,
        num_agents           = cfg.num_agents,
        sim_days             = cfg.sim_days,
        min_events_to_train  = cfg.min_events_to_train,
        local_epochs         = cfg.local_epochs,
        progression          = cfg.progression,
        seed                 = cfg.seed,
        end_condition        = cfg.end_condition,
        end_condition_param  = cfg.end_condition_param,
        wandb_project        = cfg.wandb_project,
        wandb_run_name       = cfg.wandb_run_name,
        wandb_offline        = cfg.wandb_offline,
    )
    run_federated_training(fl_cfg)


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
