#!/usr/bin/env python3
"""
Federated Simulated World — launcher

  python main.py          →  analytics TUI  (panels, SIR chart, clinic log)
  python main.py --pygame →  pygame town view (agents as squares in a town)
"""
import argparse
import curses
from simulation.world import WorldEngine
from simulation.ollama_client import OllamaDiagnosticClient, OllamaUnavailableError
from simulation.interaction_log import InteractionLogger
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


def _build_world(seed: int = 42, num_agents: int = 30):
    world  = WorldEngine(num_agents=num_agents, seed=seed)
    logger = InteractionLogger()
    world.attach_interaction_logger(logger)
    print(f"Interaction log → {logger.path}")

    client = OllamaDiagnosticClient()
    ollama_error = None
    if client.health_check():
        q = world.clinic_queue
        world.register_diagnostic_fn(
            lambda ev, _q=q: client.diagnose(ev, _queue=_q)
        )
        world.event_log.append("Ollama connected — LLM doctor active.")
    else:
        ollama_error = (
            "Ollama unreachable at localhost:11434  |  "
            "fix: ollama serve  &&  ollama pull phi3:mini"
        )
    return world, client, ollama_error


def _run_analytics(stdscr, num_agents: int = 30, seed: int = 42,
                   _out: list | None = None) -> None:
    _setup_colors()
    world, client, ollama_error = _build_world(seed=seed, num_agents=num_agents)
    tui = TUI(stdscr, world, ollama_error=ollama_error, ollama_client=client)
    tui.run()
    if _out is not None:
        _out.append(tui._export_summary)


def _run_pygame(num_agents: int, seed: int) -> None:
    from ui.pygame_ui import PygameUI
    world, _, _ = _build_world(seed=seed, num_agents=num_agents)
    PygameUI(world).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Simulated World simulator"
    )
    parser.add_argument(
        "--pygame", action="store_true",
        help="Launch pygame town view instead of the analytics TUI"
    )
    parser.add_argument("--agents", type=int, default=30,
                        help="Number of agents (default: 30)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    try:
        if args.pygame:
            _run_pygame(num_agents=args.agents, seed=args.seed)
        else:
            _out: list = []
            curses.wrapper(_run_analytics, args.agents, args.seed, _out)
            if _out and _out[0]:
                print(_out[0])
    except KeyboardInterrupt:
        pass
