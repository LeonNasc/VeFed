#!/usr/bin/env python3
"""
Federated Simulated World — ncurses TUI
Based on: Nascimento et al., "Federated Simulated World Formalization"
"""
import curses
from simulation.world import WorldEngine
from simulation.ollama_client import OllamaDiagnosticClient, OllamaUnavailableError
from ui.tui import TUI


def main(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()

    # Color pairs
    curses.init_pair(1,  curses.COLOR_GREEN,   -1)  # healthy / OK
    curses.init_pair(2,  curses.COLOR_RED,     -1)  # infected / danger
    curses.init_pair(3,  curses.COLOR_YELLOW,  -1)  # recovering / warning
    curses.init_pair(4,  curses.COLOR_CYAN,    -1)  # headers / info
    curses.init_pair(5,  curses.COLOR_MAGENTA, -1)  # clinic / LLM
    curses.init_pair(6,  curses.COLOR_WHITE,   -1)  # normal text
    curses.init_pair(7,  curses.COLOR_BLUE,    -1)  # locations
    curses.init_pair(8,  curses.COLOR_RED,     curses.COLOR_RED)
    curses.init_pair(9,  curses.COLOR_BLACK,   curses.COLOR_GREEN)
    curses.init_pair(10, curses.COLOR_BLACK,   curses.COLOR_YELLOW)

    world  = WorldEngine(num_agents=30, seed=42)
    client = OllamaDiagnosticClient()

    # ── Ollama health check ───────────────────────────────────────────────────
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

    tui = TUI(stdscr, world, ollama_error=ollama_error, ollama_client=client)
    tui.run()


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
