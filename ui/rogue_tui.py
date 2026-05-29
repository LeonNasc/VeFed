"""
RogueTUI — roguelike dungeon view for the Federated Simulated World.

Each location is rendered as a walled room; agents appear as stable
single-character glyphs whose color encodes health status:

  green  = susceptible
  red    = infected (bold)
  yellow = recovering
  magenta = hospitalised

Room wall characters vary by location type:
  # home   = work   ~ third place   + hospital

Run with:  python main.py --rogue
"""
from __future__ import annotations
import curses
import random
import string
import time

from simulation.models import HealthStatus, LocationType
from simulation.world import WorldEngine
from simulation.strategies import STRATEGIES

C_GREEN   = 1
C_RED     = 2
C_YELLOW  = 3
C_CYAN    = 4
C_MAGENTA = 5
C_WHITE   = 6
C_BLUE    = 7

WALL_CHAR: dict[LocationType, str] = {
    LocationType.HOME:      '#',
    LocationType.WORK:      '=',
    LocationType.THIRD:     '~',
    LocationType.HOSPITAL:  '+',
    LocationType.COMMUTING: '.',
}

GLYPH_POOL = string.ascii_lowercase + string.digits + string.ascii_uppercase

SIDEBAR_W = 36


def _phase(tick: int) -> tuple[str, int]:
    """Return (label, colour_pair) for the current time-of-day phase."""
    if tick < 72:   return "SLEEPING ", C_BLUE
    if tick < 84:   return "COMMUTING", C_YELLOW
    if tick < 192:  return "WORKING  ", C_CYAN
    if tick < 204:  return "COMMUTING", C_YELLOW
    if tick < 240:  return "SOCIAL   ", C_GREEN
    return               "EVENING  ", C_MAGENTA


class RogueTUI:
    """Roguelike dungeon-style view of the Federated Simulated World."""

    TICK_INTERVAL = 0.08   # seconds between auto-ticks

    def __init__(self, stdscr, world: WorldEngine,
                 ollama_error: str | None = None):
        self.stdscr       = stdscr
        self.world        = world
        self.running      = False
        self.ollama_error = ollama_error
        self._last_tick   = 0.0
        self.scroll_log   = 0
        self._strategy_names = list(STRATEGIES.keys())
        self._strategy_idx   = 0
        self._H = self._W = 0  # set on first _draw

        # Assign a stable single-char glyph to every agent (persists across ticks)
        self._glyphs: dict[str, str] = {
            ag.id: GLYPH_POOL[i % len(GLYPH_POOL)]
            for i, ag in enumerate(world.agents)
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.stdscr.nodelay(True)
        self.stdscr.timeout(50)

        while True:
            key = self.stdscr.getch()
            if key in (ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                self.world.step_tick()
            elif key in (ord('r'), ord('R')):
                self.running = not self.running
            elif key in (ord('c'), ord('C')):
                loc = list(self.world.locations.keys())[0]
                self.world.inject_disease_cloud(loc, 0.5)
            elif key in (ord('d'), ord('D')):
                loc = random.choice(list(self.world.locations.values()))
                self.world.inject_disease_cloud(loc.id, 0.6)
                self.world.event_log.append(f"Manual cloud → {loc.id}")
            elif key in (ord('s'), ord('S')):
                self._strategy_idx = (self._strategy_idx + 1) % len(self._strategy_names)
                name = self._strategy_names[self._strategy_idx]
                self.world.set_disease_strategy(STRATEGIES[name])
            elif key == curses.KEY_DOWN:
                self.scroll_log = min(
                    self.scroll_log + 1,
                    max(0, len(self.world.event_log) - 5))
            elif key == curses.KEY_UP:
                self.scroll_log = max(0, self.scroll_log - 1)

            if self.running:
                now = time.monotonic()
                if now - self._last_tick >= self.TICK_INTERVAL:
                    self.world.step_tick()
                    self._last_tick = now

            self._draw()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._H, self._W = self.stdscr.getmaxyx()
        self._map_w = self._W - SIDEBAR_W
        self._map_h = self._H - 1       # last row reserved for status bar
        self.stdscr.erase()
        self._draw_map()
        self._draw_sidebar()
        self._draw_status()
        curses.doupdate()

    # ── Map (dungeon grid) ────────────────────────────────────────────────────

    def _draw_map(self) -> None:
        w    = self.world
        locs = [l for l in w.locations.values()
                if l.type != LocationType.COMMUTING]
        if not locs:
            return

        max_row = max(l.row for l in locs)
        max_col = max(l.col for l in locs)

        room_h = max(4, self._map_h // (max_row + 1))
        room_w = max(8, self._map_w // (max_col + 1))

        for loc in locs:
            self._draw_room(loc,
                            loc.row * room_h,
                            loc.col * room_w,
                            room_h, room_w, w)

    def _draw_room(self, loc, top: int, left: int,
                   room_h: int, room_w: int, world) -> None:
        wc     = WALL_CHAR.get(loc.type, '#')
        is_hot = loc.ambient_exposure() > 0.1

        if is_hot:
            wall_attr = curses.color_pair(C_RED) | curses.A_BOLD
        elif loc.type == LocationType.HOSPITAL:
            wall_attr = curses.color_pair(C_MAGENTA)
        elif loc.type == LocationType.WORK:
            wall_attr = curses.color_pair(C_BLUE)
        elif loc.type == LocationType.THIRD:
            wall_attr = curses.color_pair(C_YELLOW)
        else:
            wall_attr = curses.color_pair(C_WHITE)

        # Top and bottom walls
        for c in range(room_w):
            self._ch(top,              left + c, wc, wall_attr)
            self._ch(top + room_h - 1, left + c, wc, wall_attr)
        # Side walls
        for r in range(1, room_h - 1):
            self._ch(top + r, left,              wc, wall_attr)
            self._ch(top + r, left + room_w - 1, wc, wall_attr)

        # Location label on the top wall
        label = f"{loc.short_name}{loc.id[-2:]}"
        for i, ch in enumerate(label[:room_w - 2]):
            self._ch(top, left + 1 + i, ch, wall_attr | curses.A_BOLD)

        # Floor tiles
        for r in range(1, room_h - 1):
            for c in range(1, room_w - 1):
                self._ch(top + r, left + c, '.', curses.color_pair(C_WHITE))

        # Agents inside the room
        inner_h = room_h - 2
        inner_w = room_w - 2
        agents_here = [world.agents_by_id[aid] for aid in loc.agents_present
                       if aid in world.agents_by_id]
        for idx, agent in enumerate(agents_here):
            r = idx // inner_w
            c = idx %  inner_w
            if r >= inner_h:
                break
            glyph = self._glyphs.get(agent.id, '?')
            if agent.hospitalised:
                attr = curses.color_pair(C_MAGENTA) | curses.A_BOLD
            elif agent.status == HealthStatus.INFECTED:
                attr = curses.color_pair(C_RED) | curses.A_BOLD
            elif agent.status == HealthStatus.RECOVERING:
                attr = curses.color_pair(C_YELLOW)
            elif agent.status == HealthStatus.RECOVERED:
                attr = curses.color_pair(C_GREEN)
            else:
                attr = curses.color_pair(C_GREEN)
            self._ch(top + 1 + r, left + 1 + c, glyph, attr)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _draw_sidebar(self) -> None:
        w   = self.world
        sir = w.sir_model
        sx  = self._map_w
        col = sx + 1

        # Vertical divider
        for r in range(self._map_h):
            self._ch(r, sx, '│', curses.color_pair(C_WHITE))

        # ── Clock ─────────────────────────────────────────────────────────
        t       = w.current_tick
        day_pct = t / w.TICKS_PER_DAY
        hour    = int(day_pct * 24)
        minute  = int((day_pct * 24 - hour) * 60)
        phase, ph_color = _phase(t)

        row = 0
        self._str(row,     col, "── CLOCK ──────────────────────",
                  curses.color_pair(C_CYAN) | curses.A_BOLD)
        self._str(row + 1, col, f"DAY {w.current_day:04d}   {hour:02d}:{minute:02d}",
                  curses.color_pair(C_CYAN) | curses.A_BOLD)
        self._str(row + 2, col, phase,
                  curses.color_pair(ph_color) | curses.A_BOLD)
        row += 4

        # ── SIR stats ─────────────────────────────────────────────────────
        total = max(1, sir.total)
        self._str(row,     col, "── SIR ────────────────────────",
                  curses.color_pair(C_CYAN) | curses.A_BOLD)
        self._str(row + 1, col, f"S: {sir.S:3d}  ({sir.S/total*100:4.1f}%)",
                  curses.color_pair(C_GREEN))
        self._str(row + 2, col, f"I: {sir.I:3d}  ({sir.I/total*100:4.1f}%)",
                  curses.color_pair(C_RED) | curses.A_BOLD)
        self._str(row + 3, col, f"R: {sir.R:3d}  ({sir.R/total*100:4.1f}%)",
                  curses.color_pair(C_YELLOW))
        self._str(row + 4, col, f"FL rounds: {w.fl_round}",
                  curses.color_pair(C_MAGENTA))
        row += 6

        # ── Clinic / last consultation ────────────────────────────────────
        row = self._draw_clinic_section(row, col)

        # ── Legend ────────────────────────────────────────────────────────
        self._str(row,     col, "── LEGEND ─────────────────────",
                  curses.color_pair(C_CYAN) | curses.A_BOLD)
        self._str(row + 1, col, "green  = susceptible",  curses.color_pair(C_GREEN))
        self._str(row + 2, col, "red    = infected",     curses.color_pair(C_RED) | curses.A_BOLD)
        self._str(row + 3, col, "yellow = recovering",   curses.color_pair(C_YELLOW))
        self._str(row + 4, col, "magenta= hospitalised", curses.color_pair(C_MAGENTA))
        self._str(row + 5, col, "red ## = disease cloud",curses.color_pair(C_RED))
        row += 7

        # ── Event log ─────────────────────────────────────────────────────
        self._str(row, col, "── EVENT LOG ──────────────────",
                  curses.color_pair(C_CYAN) | curses.A_BOLD)
        log_rows = self._map_h - row - 2
        if log_rows > 0:
            log   = w.event_log
            start = max(0, len(log) - log_rows - self.scroll_log)
            lines = log[start: start + log_rows]
            for i, line in enumerate(lines):
                attr = curses.color_pair(C_WHITE)
                if "Infected" in line or "cloud" in line.lower():
                    attr = curses.color_pair(C_RED)
                elif "discharged" in line or "recovery" in line:
                    attr = curses.color_pair(C_GREEN)
                elif "FL" in line or "Round" in line:
                    attr = curses.color_pair(C_MAGENTA)
                self._str(row + 1 + i, col, line[:SIDEBAR_W - 2], attr)

        if self.ollama_error:
            self._str(self._map_h - 1, col, "Ollama offline",
                      curses.color_pair(C_RED))

    # ── Clinic section (last consultation) ───────────────────────────────────

    def _draw_clinic_section(self, row: int, col: int) -> int:
        """Draw last patient/doctor conversation in sidebar; returns next row."""
        w         = self.world
        processed = w.clinic_queue.processed
        text_w    = SIDEBAR_W - 2

        self._str(row, col, "── CLINIC ──────────────────────",
                  curses.color_pair(C_MAGENTA) | curses.A_BOLD)
        row += 1

        if not processed:
            self._str(row, col, "no cases yet", curses.color_pair(C_WHITE))
            return row + 2

        ev         = processed[-1]
        ag         = w.agents_by_id.get(ev.agent_id)
        name       = (ag.name if ag else ev.agent_id)[:16]
        action_str = ev.action.value if ev.action else "pending"
        label_str  = ev.oracle_label or "?"

        if action_str == "hospitalise":
            out_attr = curses.color_pair(C_RED) | curses.A_BOLD
        elif action_str == "resolve":
            out_attr = curses.color_pair(C_YELLOW)
        else:
            out_attr = curses.color_pair(C_GREEN)

        self._str(row, col, f"{name} | {label_str} → {action_str}"[:text_w], out_attr)
        row += 1

        for turn in ev.conversation:
            role   = turn.get("role", "?")
            text   = turn.get("text", "")
            prefix = "[P] " if role == "patient" else "[D] "
            attr   = (curses.color_pair(C_GREEN) if role == "patient"
                      else curses.color_pair(C_CYAN))
            line   = (prefix + text)[:text_w]
            self._str(row, col, line, attr)
            row += 1
            if row >= self._map_h - 14:   # leave room for legend + event log
                break

        return row + 1   # spacer

    # ── Status bar ───────────────────────────────────────────────────────────

    def _draw_status(self) -> None:
        H, W  = self.stdscr.getmaxyx()
        mode  = "AUTO  " if self.running else "PAUSED"
        mattr = (curses.color_pair(C_GREEN) | curses.A_BOLD
                 if self.running else curses.color_pair(C_YELLOW) | curses.A_BOLD)
        bar = (" [SPC] step  [R] run/pause  [D] disease cloud  "
               "[S] strategy  [↑↓] log  [Q] quit  ")
        try:
            self.stdscr.addstr(H - 1, 0,
                               bar[:W - 9].ljust(W - 9),
                               curses.color_pair(C_WHITE))
            self.stdscr.addstr(H - 1, W - 8, f"  {mode}", mattr)
        except curses.error:
            pass
        self.stdscr.noutrefresh()

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _ch(self, y: int, x: int, ch: str, attr: int = 0) -> None:
        try:
            if 0 <= y < self._H - 1 and 0 <= x < self._W - 1:
                self.stdscr.addch(y, x, ch, attr)
        except curses.error:
            pass

    def _str(self, y: int, x: int, text: str, attr: int = 0) -> None:
        try:
            max_w = self._W - x - 1
            if max_w > 0 and 0 <= y < self._H - 1:
                self.stdscr.addstr(y, x, text[:max_w], attr)
        except curses.error:
            pass
