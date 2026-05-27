"""
TUI — ncurses interface for the Federated Simulated World.

Layout (four panes):
┌─────────────────────────┬──────────────────┐
│  WORLD MAP              │  SIR CURVE       │
│  (location grid +       │  + stats         │
│   agent glyphs)         ├──────────────────┤
│                         │  AGENTS          │
├─────────────────────────┤  (scrollable)    │
│  CLINIC / LLM DOCTOR    ├──────────────────┤
│  (queue + outcomes)     │  EVENT LOG       │
├─────────────────────────┴──────────────────┤
│  STATUS BAR  [SPACE]=tick [r]=run [q]=quit │
└────────────────────────────────────────────┘
"""
from __future__ import annotations
import curses
import time
from simulation.models import HealthStatus, LocationType
from simulation.world import WorldEngine
from simulation.strategies import STRATEGIES

# Colour pair aliases
C_GREEN   = 1
C_RED     = 2
C_YELLOW  = 3
C_CYAN    = 4
C_MAGENTA = 5
C_WHITE   = 6
C_BLUE    = 7


class Panel:
    """Thin wrapper around a curses window with border + title."""
    def __init__(self, h: int, w: int, y: int, x: int, title: str = ""):
        self.win   = curses.newwin(h, w, y, x)
        self.h     = h
        self.w     = w
        self.title = title

    def clear_draw(self) -> None:
        self.win.erase()
        self.win.border()
        if self.title:
            label = f" {self.title} "
            self.win.addstr(0, 2, label, curses.color_pair(C_CYAN) | curses.A_BOLD)

    def safe_addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        try:
            max_w = self.w - x - 1
            if max_w > 0 and y < self.h - 1 and y > 0:
                self.win.addstr(y, x, text[:max_w], attr)
        except curses.error:
            pass

    def refresh(self) -> None:
        self.win.noutrefresh()


class TUI:
    """Main TUI controller."""

    TICK_INTERVAL = 0.06   # seconds between auto-ticks when running

    def __init__(self, stdscr, world: WorldEngine):
        self.stdscr  = stdscr
        self.world   = world
        self.running = False   # auto-step mode
        self.scroll_agent = 0
        self.scroll_log   = 0
        self.agent_filter = "ALL"   # ALL | S | I | R
        self._last_tick_time = 0.0
        self._clinic_msgs: list[str] = []
        self._strategy_names = list(STRATEGIES.keys())
        self._strategy_idx   = 0
        self._build_layout()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        H, W = self.stdscr.getmaxyx()
        mid_x  = W // 2
        top_h  = H * 3 // 5
        bot_h  = H - top_h - 1   # 1 for status bar

        sir_h    = top_h // 2
        agent_h  = top_h - sir_h
        log_h    = bot_h

        self.p_map    = Panel(top_h,     mid_x,       0,         0,     "WORLD MAP")
        self.p_sir    = Panel(sir_h,     W - mid_x,   0,         mid_x, "SIR DYNAMICS")
        self.p_agents = Panel(agent_h,   W - mid_x,   sir_h,     mid_x, "AGENTS")
        self.p_clinic = Panel(bot_h,     mid_x,       top_h,     0,     "CLINIC / LLM DOCTOR")
        self.p_log    = Panel(log_h,     W - mid_x,   top_h,     mid_x, "EVENT LOG")
        self.status_y = H - 1

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
                self.world.inject_disease_cloud(
                    list(self.world.locations.keys())[0], intensity=0.5
                )
            elif key == curses.KEY_DOWN:
                self.scroll_log   = min(self.scroll_log + 1,
                                        max(0, len(self.world.event_log) - 5))
            elif key == curses.KEY_UP:
                self.scroll_log   = max(0, self.scroll_log - 1)
            elif key == curses.KEY_NPAGE:
                self.scroll_agent = min(self.scroll_agent + 1,
                                        max(0, len(self.world.agents) - 3))
            elif key == curses.KEY_PPAGE:
                self.scroll_agent = max(0, self.scroll_agent - 1)
            elif key in (ord('f'), ord('F')):
                modes = ["ALL", "S", "I", "R"]
                idx   = modes.index(self.agent_filter)
                self.agent_filter = modes[(idx + 1) % len(modes)]
            elif key in (ord('s'), ord('S')):
                self._strategy_idx = (self._strategy_idx + 1) % len(self._strategy_names)
                name = self._strategy_names[self._strategy_idx]
                self.world.set_disease_strategy(STRATEGIES[name])

            elif key in (ord('d'), ord('D')):
                # Inject cloud at random location
                import random
                loc = random.choice(list(self.world.locations.values()))
                self.world.inject_disease_cloud(loc.id, 0.6)
                self.world.event_log.append(f"Manual cloud → {loc.id}")

            # Auto-tick
            if self.running:
                now = time.monotonic()
                if now - self._last_tick_time >= self.TICK_INTERVAL:
                    self.world.step_tick()
                    self._last_tick_time = now

            self._draw()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._draw_map()
        self._draw_sir()
        self._draw_agents()
        self._draw_clinic()
        self._draw_log()
        self._draw_status()
        curses.doupdate()

    # ── Map panel ────────────────────────────────────────────────────────────

    def _draw_map(self) -> None:
        p = self.p_map
        w = self.world
        p.clear_draw()

        day_pct = w.current_tick / w.TICKS_PER_DAY
        hour    = int(day_pct * 24)
        minute  = int((day_pct * 24 - hour) * 60)
        # Day counter — shown prominently in top-centre of map panel
        day_str = f" DAY {w.current_day:04d}  {hour:02d}:{minute:02d} "
        p.safe_addstr(0, max(2, (p.w - len(day_str)) // 2),
                      day_str,
                      curses.color_pair(C_CYAN) | curses.A_BOLD)
        strategy_str = f" {w.strategy.name} "
        p.safe_addstr(0, p.w - len(strategy_str) - 1,
                      strategy_str,
                      curses.color_pair(C_YELLOW))

        # Gather all locations with known grid positions
        locs = [l for l in w.locations.values()
                if l.type != LocationType.COMMUTING]
        if not locs:
            p.refresh()
            return

        max_row = max(l.row for l in locs)
        max_col = max(l.col for l in locs)

        cell_h = max(2, (p.h - 3) // (max_row + 1))
        cell_w = max(6, (p.w - 2) // (max_col + 1))

        for loc in locs:
            draw_r = 1 + loc.row * cell_h
            draw_c = 1 + loc.col * cell_w
            if draw_r >= p.h - 1 or draw_c >= p.w - 1:
                continue

            # Colour by type
            lt = loc.type
            if lt == LocationType.HOSPITAL:
                attr = curses.color_pair(C_MAGENTA)
            elif lt == LocationType.WORK:
                attr = curses.color_pair(C_BLUE)
            elif lt == LocationType.THIRD:
                attr = curses.color_pair(C_YELLOW)
            else:
                attr = curses.color_pair(C_WHITE)

            # Disease cloud overlay
            if loc.ambient_exposure() > 0.1:
                attr = curses.color_pair(C_RED) | curses.A_BOLD

            label = f"{loc.short_name}{loc.id[-2:]}"
            p.safe_addstr(draw_r, draw_c, label, attr | curses.A_BOLD)

            # Agents at this location
            agents_here = [w.agents_by_id[aid] for aid in loc.agents_present
                           if aid in w.agents_by_id]
            glyph_str = ""
            for ag in agents_here[:cell_w - 2]:
                glyph_str += ag.display_char
            p.safe_addstr(draw_r + 1, draw_c, glyph_str[:cell_w - 1],
                          curses.color_pair(C_GREEN))

        # Legend
        legend_y = p.h - 2
        p.safe_addstr(legend_y, 1,
                      "· susc  ✶ infect  ○ recov  | H home  W work  T third  + hosp",
                      curses.color_pair(C_WHITE))
        p.refresh()

    # ── SIR panel ────────────────────────────────────────────────────────────

    def _draw_sir(self) -> None:
        p   = self.p_sir
        sir = self.world.sir_model
        p.clear_draw()
        inner_w = p.w - 2
        inner_h = p.h - 4

        # Numeric summary
        total = max(1, sir.total)
        p.safe_addstr(1, 1,
                      f"S:{sir.S:3d}  I:{sir.I:3d}  R:{sir.R:3d}  N:{total:3d}",
                      curses.color_pair(C_WHITE) | curses.A_BOLD)

        # Sparkline history
        history = sir.history[-(inner_w):]
        if len(history) < 2:
            p.refresh()
            return

        max_n  = max(s + i + r for s, i, r in history) or 1
        blocks = " ▁▂▃▄▅▆▇█"

        for col_i, (s, inf, r) in enumerate(history):
            cx = 1 + col_i
            if cx >= p.w - 1:
                break
            # Draw I (infected) bar
            bar_h = int((inf / max_n) * inner_h)
            for row_off in range(inner_h):
                draw_r = inner_h - row_off + 1
                if draw_r < 1 or draw_r >= p.h - 1:
                    continue
                if row_off < bar_h:
                    p.safe_addstr(draw_r, cx, "│", curses.color_pair(C_RED))
                else:
                    r_bar = int((r / max_n) * inner_h)
                    if row_off < r_bar:
                        p.safe_addstr(draw_r, cx, "│", curses.color_pair(C_GREEN))

        # FL rounds
        fl_y = p.h - 2
        fl_msg = f"FL rounds: {self.world.fl_round}"
        p.safe_addstr(fl_y, 1, fl_msg, curses.color_pair(C_MAGENTA))
        p.refresh()

    # ── Agents panel ─────────────────────────────────────────────────────────

    def _draw_agents(self) -> None:
        p       = self.p_agents
        agents  = self.world.agents
        p.clear_draw()

        # Filter
        if self.agent_filter != "ALL":
            fmap = {"S": HealthStatus.SUSCEPTIBLE,
                    "I": HealthStatus.INFECTED,
                    "R": HealthStatus.RECOVERING}
            target = fmap.get(self.agent_filter)
            agents = [a for a in agents if a.status == target]

        p.safe_addstr(0, p.w - 14,
                      f"[f] filter:{self.agent_filter} ",
                      curses.color_pair(C_CYAN))

        visible_rows = p.h - 2
        start = min(self.scroll_agent, max(0, len(agents) - visible_rows))
        visible = agents[start: start + visible_rows]

        for i, agent in enumerate(visible):
            row = i + 1
            hs  = agent.health_state

            if hs.status == HealthStatus.SUSCEPTIBLE:
                sc    = curses.color_pair(C_GREEN)
                badge = "S"
            elif hs.status == HealthStatus.INFECTED:
                sc    = curses.color_pair(C_RED) | curses.A_BOLD
                badge = "I"
            elif hs.status == HealthStatus.RECOVERING:
                sc    = curses.color_pair(C_YELLOW)
                badge = "R"
            else:
                sc    = curses.color_pair(C_WHITE)
                badge = "V"

            hosp_flag = "+" if agent.hospitalised else " "
            loc_label = agent.current_location[:5]
            sev_bar   = self._mini_bar(hs.severity, 5)
            sym_bar   = self._mini_bar(hs.symptoms, 5)

            line = (f"{badge}{hosp_flag} {agent.name[:14]:<14} "
                    f"@{loc_label:<5} sev[{sev_bar}] sym[{sym_bar}]")
            p.safe_addstr(row, 1, line, sc)

        p.refresh()

    # ── Clinic panel ─────────────────────────────────────────────────────────

    def _draw_clinic(self) -> None:
        p     = self.p_clinic
        queue = self.world.clinic_queue
        p.clear_draw()

        row = 1
        p.safe_addstr(row, 1,
                      f"Queue: {len(queue.patients):2d} waiting   "
                      f"Processed today: {len(queue.processed):3d}",
                      curses.color_pair(C_WHITE) | curses.A_BOLD)
        row += 1

        # Extension point notice
        p.safe_addstr(row, 1,
                      "[ LLM diagnostic fn: world.register_diagnostic_fn(fn) ]",
                      curses.color_pair(C_MAGENTA))
        row += 1
        p.safe_addstr(row, 1, "─" * (p.w - 4), curses.color_pair(C_WHITE))
        row += 1

        # Recent processed cases
        recent = list(queue.processed)[-(p.h - row - 2):]
        for ev in recent:
            action_str = ev.action.value if ev.action else "pending"
            label_str  = ev.oracle_label or ""
            ag         = self.world.agents_by_id.get(ev.agent_id)
            name       = ag.name if ag else ev.agent_id
            sev_str    = f"{ev.severity:.2f}"
            sym_str    = f"{ev.symptoms:.2f}"
            line       = (f"  {name[:14]:<14} sev={sev_str} sym={sym_str}"
                          f"  [{action_str:<12}] {label_str}")

            if action_str == "hospitalise":
                attr = curses.color_pair(C_RED) | curses.A_BOLD
            elif action_str == "resolve":
                attr = curses.color_pair(C_YELLOW)
            else:
                attr = curses.color_pair(C_GREEN)
            p.safe_addstr(row, 1, line, attr)
            row += 1
            if row >= p.h - 1:
                break

        p.refresh()

    # ── Event log panel ──────────────────────────────────────────────────────

    def _draw_log(self) -> None:
        p   = self.p_log
        log = self.world.event_log + self.world.fl_log
        p.clear_draw()

        visible_rows = p.h - 2
        start = max(0, len(log) - visible_rows - self.scroll_log)
        lines = log[start: start + visible_rows]

        for i, line in enumerate(lines):
            row  = i + 1
            attr = curses.color_pair(C_WHITE)
            if "Infected" in line or "cloud" in line.lower():
                attr = curses.color_pair(C_RED)
            elif "discharged" in line or "recovery" in line:
                attr = curses.color_pair(C_GREEN)
            elif "FL" in line or "Round" in line:
                attr = curses.color_pair(C_MAGENTA)
            elif "hospital" in line.lower():
                attr = curses.color_pair(C_YELLOW)
            p.safe_addstr(row, 1, line, attr)

        p.refresh()

    # ── Status bar ───────────────────────────────────────────────────────────

    def _draw_status(self) -> None:
        H, W = self.stdscr.getmaxyx()
        mode = "AUTO" if self.running else "PAUSED"
        mode_attr = (curses.color_pair(C_GREEN) | curses.A_BOLD
                     if self.running else curses.color_pair(C_YELLOW) | curses.A_BOLD)

        bar = (f" [SPACE] step  [R] run/pause  [F] filter  [D] cloud  [S] strategy  "
               f"[↑↓] log  [PgUp/Dn] agents  [Q] quit  ")
        try:
            self.stdscr.addstr(H - 1, 0, bar[:W - 12].ljust(W - 12),
                               curses.color_pair(C_WHITE))
            self.stdscr.addstr(H - 1, W - 11, f"  {mode:<8}",  mode_attr)
        except curses.error:
            pass
        self.stdscr.noutrefresh()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _mini_bar(value: float, width: int) -> str:
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)
