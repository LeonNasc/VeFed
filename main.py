#!/usr/bin/env python3
"""
Federated Simulated World — launcher

  python main.py          →  analytics TUI  (panels, SIR chart, clinic log)
  python main.py --rogue  →  roguelike TUI  (agents as chars on a dungeon map)
"""
import argparse
import curses
from simulation.world import WorldEngine
from simulation.ollama_client import OllamaDiagnosticClient, OllamaUnavailableError
from ui.tui import TUI


def _setup_colors() -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1,  curses.COLOR_GREEN,   -1)
    curses.init_pair(2,  curses.COLOR_RED,     -1)
    curses.init_pair(3,  curses.COLOR_YELLOW,  -1)
    curses.init_pair(4,  curses.COLOR_CYAN,    -1)
    curses.init_pair(5,  curses.COLOR_MAGENTA, -1)
    curses.init_pair(6,  curses.COLOR_WHITE,   -1)
    curses.init_pair(7,  curses.COLOR_BLUE,    -1)
    curses.init_pair(8,  curses.COLOR_RED,     curses.COLOR_RED)
    curses.init_pair(9,  curses.COLOR_BLACK,   curses.COLOR_GREEN)
    curses.init_pair(10, curses.COLOR_BLACK,   curses.COLOR_YELLOW)


def _build_world_and_ollama():
    world  = WorldEngine(num_agents=30, seed=42)
    client = OllamaDiagnosticClient()
    ollama_error = None
    if client.health_check():
        q = world.clinic_queue
        world.register_diagnostic_fn(
            lambda ev, _q=q: client.diagnose(ev, _queue=_q)
        )
        world.event_log.append("Ollama (mistral) connected — LLM doctor active.")
    else:
        ollama_error = (
            "Ollama unreachable at localhost:11434  |  "
            "fix: ollama serve  &&  ollama pull mistral"
        )
    return world, client, ollama_error


def _run_analytics(stdscr) -> None:
    _setup_colors()
    world, client, ollama_error = _build_world_and_ollama()
    TUI(stdscr, world, ollama_error=ollama_error, ollama_client=client).run()


def _run_rogue(stdscr) -> None:
    from ui.rogue_tui import RogueTUI
    _setup_colors()
    world, client, ollama_error = _build_world_and_ollama()
    RogueTUI(stdscr, world, ollama_error=ollama_error).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Simulated World simulator"
    )
    parser.add_argument(
        "--rogue", action="store_true",
        help="Launch roguelike dungeon view instead of the analytics TUI"
    )
    args = parser.parse_args()

    try:
        if args.rogue:
            curses.wrapper(_run_rogue)
        else:
            curses.wrapper(_run_analytics)
    except KeyboardInterrupt:
        pass
