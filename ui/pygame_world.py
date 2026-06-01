"""
PygameWorld — 2D animated town view of the FedWorld simulator.

Agents appear as colored dots that walk between buildings, turn red when
infected, and head to the hospital when they seek care.

Controls
────────
  SPACE     step one tick
  R         toggle auto-run
  ↑ / ↓     speed up / slow down
  Q / ESC   quit
"""
from __future__ import annotations
import hashlib
import math
import time

import pygame

from simulation.models import HealthStatus, LocationType
from simulation.world import WorldEngine

# ── Palette ───────────────────────────────────────────────────────────────────

BG       = ( 28,  28,  32)
ROAD     = ( 50,  50,  56)
ROAD_LN  = ( 65,  65,  72)

BCOL: dict[LocationType, tuple] = {
    LocationType.HOME:      (160, 115,  65),
    LocationType.WORK:      ( 55,  95, 160),
    LocationType.THIRD:     (185, 110,  30),
    LocationType.HOSPITAL:  (225, 225, 230),
    LocationType.COMMUTING: ( 70,  70,  76),
}
BRIM: dict[LocationType, tuple] = {   # darker border per type
    LocationType.HOME:      (120,  80,  40),
    LocationType.WORK:      ( 35,  65, 120),
    LocationType.THIRD:     (140,  75,  15),
    LocationType.HOSPITAL:  (170, 170, 185),
    LocationType.COMMUTING: ( 50,  50,  55),
}

ACOL: dict[HealthStatus, tuple] = {
    HealthStatus.SUSCEPTIBLE: ( 80, 200,  80),
    HealthStatus.INFECTED:    (220,  45,  45),
    HealthStatus.RECOVERING:  (220, 180,  50),
    HealthStatus.RECOVERED:   ( 80, 160, 220),
}
HOSP_RING = (160, 75, 210)   # purple ring drawn around hospitalised agents

WHITE      = (240, 240, 245)
GREY_LT    = (160, 160, 170)
GREY_MD    = (100, 100, 110)
TEXT_DIM   = ( 90,  90, 100)
HOSP_RED   = (200,  40,  40)
SIDEBAR_BG = ( 22,  22,  26)

# ── Layout ────────────────────────────────────────────────────────────────────

CELL      = 150   # px per grid cell
BSIZE     = 90    # building square side
AGENT_R   = 7     # agent dot radius
SIDEBAR_W = 300
MIN_H     = 520

TICK_SPEEDS = [0.40, 0.18, 0.07, 0.025]   # s/tick: slow → fast
LERP_K      = 0.16                          # fraction of gap closed per frame

# time-of-day buckets (tick ranges)
_PHASES = [
    (  0,  72, "SLEEPING",   ( 60,  80, 140)),
    ( 72,  84, "COMMUTING",  (180, 160,  60)),
    ( 84, 192, "WORKING",    ( 60, 130, 190)),
    (192, 204, "COMMUTING",  (180, 160,  60)),
    (204, 240, "SOCIAL",     (100, 180,  90)),
    (240, 288, "EVENING",    (130,  80, 160)),
]


def _phase(tick: int) -> tuple[str, tuple]:
    for lo, hi, label, col in _PHASES:
        if lo <= tick < hi:
            return label, col
    return "EVENING", (130, 80, 160)


def _sir_color(status: HealthStatus) -> tuple:
    return ACOL.get(status, GREY_MD)


# ── Main class ────────────────────────────────────────────────────────────────

class PygameWorld:
    """Pygame 2-D animated view of a WorldEngine."""

    def __init__(self, world: WorldEngine):
        self.world = world
        pygame.init()
        pygame.display.set_caption("FedWorld — Town View")

        # ── pixel centers for each location ──────────────────────────────────
        non_commute = [l for l in world.locations.values()
                       if l.type != LocationType.COMMUTING]
        max_row = max((l.row for l in non_commute), default=1)
        max_col = max((l.col for l in non_commute), default=1)

        map_w  = (max_col + 1) * CELL
        map_h  = (max_row + 1) * CELL
        scr_w  = map_w + SIDEBAR_W
        scr_h  = max(map_h, MIN_H)

        self.screen = pygame.display.set_mode((scr_w, scr_h), pygame.RESIZABLE)
        self._map_w  = map_w
        self._map_h  = map_h
        self.clock   = pygame.time.Clock()

        # Fonts
        self._fsm = pygame.font.SysFont("monospace", 11)
        self._fmd = pygame.font.SysFont("monospace", 13)
        self._flg = pygame.font.SysFont("monospace", 16, bold=True)
        self._fxl = pygame.font.SysFont("monospace", 19, bold=True)

        self._loc_px: dict[str, tuple[float, float]] = {}
        for loc in world.locations.values():
            if loc.type == LocationType.COMMUTING:
                # Place commute hub at map center so commuting agents
                # visually flow through the middle of town.
                self._loc_px[loc.id] = (map_w / 2, map_h / 2)
            else:
                cx = loc.col * CELL + CELL // 2
                cy = loc.row * CELL + CELL // 2
                self._loc_px[loc.id] = (float(cx), float(cy))

        # Per-agent deterministic scatter offset within a building
        self._off: dict[str, tuple[float, float]] = {}
        for ag in world.agents:
            h  = int(hashlib.md5(ag.id.encode()).hexdigest()[:8], 16)
            ox = ((h & 0xFF) - 128) * 0.24          # ±30 px
            oy = (((h >> 8) & 0xFF) - 128) * 0.24
            self._off[ag.id] = (ox, oy)

        # Smooth render positions (lerped each frame)
        self._px: dict[str, list[float]] = {}
        for ag in world.agents:
            tx, ty = self._agent_target(ag)
            self._px[ag.id] = [tx, ty]

        # UI state
        self._running    = False
        self._speed_idx  = 1
        self._last_tick  = 0.0
        self._frame      = 0

    # ── Target pixel for an agent ─────────────────────────────────────────────

    def _agent_target(self, ag) -> tuple[float, float]:
        bx, by = self._loc_px.get(ag.current_location,
                                   (self._map_w / 2, self._map_h / 2))
        ox, oy = self._off[ag.id]
        return bx + ox, by + oy

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        while True:
            self.clock.tick(30)
            self._frame += 1

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); return
                elif ev.type == pygame.KEYDOWN:
                    self._key(ev.key)

            if self._running:
                now = time.monotonic()
                if now - self._last_tick >= TICK_SPEEDS[self._speed_idx]:
                    self.world.step_tick()
                    self._last_tick = now

            # Lerp agent positions
            for ag in self.world.agents:
                tx, ty = self._agent_target(ag)
                pos    = self._px[ag.id]
                pos[0] += (tx - pos[0]) * LERP_K
                pos[1] += (ty - pos[1]) * LERP_K

            self._draw()
            pygame.display.flip()

    def _key(self, key: int) -> None:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            pygame.quit(); raise SystemExit
        elif key == pygame.K_SPACE:
            self.world.step_tick()
            self._last_tick = time.monotonic()
        elif key == pygame.K_r:
            self._running = not self._running
            self._last_tick = time.monotonic()
        elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
            self._speed_idx = min(self._speed_idx + 1, len(TICK_SPEEDS) - 1)
        elif key in (pygame.K_DOWN, pygame.K_MINUS):
            self._speed_idx = max(self._speed_idx - 1, 0)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        w = self.world
        scr_w, scr_h = self.screen.get_size()

        self.screen.fill(BG)
        self._draw_roads(scr_h)
        self._draw_buildings()
        self._draw_agents()
        self._draw_sidebar(scr_w, scr_h)

    def _draw_roads(self, scr_h: int) -> None:
        """Draw a simple grid road network behind the buildings."""
        s = self.screen
        for c in range((self._map_w // CELL) + 1):
            x = c * CELL + CELL // 2
            pygame.draw.line(s, ROAD, (x, 0), (x, self._map_h), CELL - BSIZE)
        for r in range((self._map_h // CELL) + 1):
            y = r * CELL + CELL // 2
            pygame.draw.line(s, ROAD, (0, y), (self._map_w, y), CELL - BSIZE)
        # Lighter road-centre lines
        for c in range((self._map_w // CELL) + 1):
            x = c * CELL + CELL // 2
            pygame.draw.line(s, ROAD_LN, (x, 0), (x, self._map_h), 1)
        for r in range((self._map_h // CELL) + 1):
            y = r * CELL + CELL // 2
            pygame.draw.line(s, ROAD_LN, (0, y), (self._map_w, y), 1)

    def _draw_buildings(self) -> None:
        s = self.screen
        half = BSIZE // 2

        for loc in self.world.locations.values():
            if loc.type == LocationType.COMMUTING:
                continue

            cx, cy = int(self._loc_px[loc.id][0]), int(self._loc_px[loc.id][1])
            rect   = pygame.Rect(cx - half, cy - half, BSIZE, BSIZE)
            col    = BCOL.get(loc.type, GREY_MD)
            rim    = BRIM.get(loc.type, GREY_MD)

            # Shadow
            shadow = rect.move(3, 3)
            pygame.draw.rect(s, (15, 15, 18), shadow, border_radius=6)

            # Body
            pygame.draw.rect(s, col, rect, border_radius=6)
            pygame.draw.rect(s, rim, rect, width=2, border_radius=6)

            # Hospital cross
            if loc.type == LocationType.HOSPITAL:
                cw, ch = 8, 26
                pygame.draw.rect(s, HOSP_RED,
                                 (cx - cw // 2, cy - ch // 2, cw, ch))
                pygame.draw.rect(s, HOSP_RED,
                                 (cx - ch // 2, cy - cw // 2, ch, cw))

            # Type label (small, bottom of building)
            lbl_map = {
                LocationType.HOME:     "home",
                LocationType.WORK:     "work",
                LocationType.THIRD:    "social",
                LocationType.HOSPITAL: "HOSPITAL",
            }
            lbl = lbl_map.get(loc.type, "")
            if lbl:
                surf = self._fsm.render(lbl, True, WHITE if loc.type != LocationType.HOSPITAL else GREY_MD)
                s.blit(surf, (cx - surf.get_width() // 2, cy + half // 2 - 4))

    def _draw_agents(self) -> None:
        s     = self.screen
        frame = self._frame

        for ag in self.world.agents:
            status     = ag.health_state.status
            hospitalised = getattr(ag, "hospitalised", False)

            px, py = int(self._px[ag.id][0]), int(self._px[ag.id][1])
            col    = ACOL.get(status, GREY_MD)

            # Pulse radius for infected agents
            r = AGENT_R
            if status == HealthStatus.INFECTED:
                r = AGENT_R + int(2 * abs(math.sin(frame * 0.12)))

            # Hospitalised: draw a purple ring around them
            if hospitalised:
                pygame.draw.circle(s, HOSP_RING, (px, py), r + 4, width=2)

            # Drop shadow
            pygame.draw.circle(s, (15, 15, 18), (px + 2, py + 2), r)
            # Agent dot
            pygame.draw.circle(s, col, (px, py), r)
            # Bright specular highlight
            pygame.draw.circle(s, _lighten(col, 80), (px - 2, py - 2), max(2, r // 3))

    def _draw_sidebar(self, scr_w: int, scr_h: int) -> None:
        s = self.screen
        w = self.world

        sx = self._map_w
        pygame.draw.rect(s, SIDEBAR_BG, (sx, 0, scr_w - sx, scr_h))
        pygame.draw.line(s, GREY_MD, (sx, 0), (sx, scr_h), 1)

        x = sx + 14
        y = 18

        def text(msg, font, color=WHITE, indent=0):
            nonlocal y
            surf = font.render(msg, True, color)
            s.blit(surf, (x + indent, y))
            y += surf.get_height() + 3

        def sep(gap=8):
            nonlocal y
            y += gap
            pygame.draw.line(s, GREY_MD, (x, y), (scr_w - 14, y), 1)
            y += gap

        # ── Header ───────────────────────────────────────────────────────────
        text("FedWorld", self._fxl, WHITE)
        phase_lbl, phase_col = _phase(w.current_tick)
        text(f"Day {w.current_day:>3}  T{w.current_tick:>3}/287", self._fmd, GREY_LT)
        text(phase_lbl, self._flg, phase_col)
        sep()

        # ── SIR counts ───────────────────────────────────────────────────────
        text("SIR", self._flg, GREY_LT)
        total = max(len(w.agents), 1)
        sir_data = [
            ("S", w.sir_model.S, ACOL[HealthStatus.SUSCEPTIBLE]),
            ("I", w.sir_model.I, ACOL[HealthStatus.INFECTED]),
            ("R", w.sir_model.R, ACOL[HealthStatus.RECOVERING]),
        ]
        bar_w = scr_w - sx - 28
        for label, count, col in sir_data:
            frac = count / total
            blen = int(frac * bar_w)
            bar_rect = pygame.Rect(x, y, blen, 14)
            pygame.draw.rect(s, col, bar_rect, border_radius=3)
            pygame.draw.rect(s, GREY_MD, pygame.Rect(x, y, bar_w, 14),
                             width=1, border_radius=3)
            lbl_surf = self._fsm.render(f"{label} {count:>3} ({frac*100:.0f}%)",
                                        True, WHITE)
            s.blit(lbl_surf, (x + 4, y))
            y += 20

        # Hospitalised (tracked via agent.hospitalised flag, not HealthStatus)
        hosp = sum(1 for ag in w.agents if getattr(ag, "hospitalised", False))
        text(f"  hosp {hosp:>3}", self._fsm, HOSP_RING)
        sep()

        # ── Legend ───────────────────────────────────────────────────────────
        text("AGENTS", self._flg, GREY_LT)
        for status, col in ACOL.items():
            pygame.draw.circle(s, col, (x + 7, y + 7), 6)
            lbl_surf = self._fsm.render(status.value, True, GREY_LT)
            s.blit(lbl_surf, (x + 18, y + 1))
            y += 17
        # Hospitalised indicator
        pygame.draw.circle(s, GREY_LT, (x + 7, y + 7), 6)
        pygame.draw.circle(s, HOSP_RING, (x + 7, y + 7), 10, width=2)
        lbl_surf = self._fsm.render("hospitalised", True, GREY_LT)
        s.blit(lbl_surf, (x + 18, y + 1))
        y += 17
        sep()

        # ── Recent events ────────────────────────────────────────────────────
        text("EVENTS", self._flg, GREY_LT)
        recent = list(w.event_log)[-10:]
        for ev in recent:
            short = ev[:36] + "…" if len(ev) > 36 else ev
            text(short, self._fsm, TEXT_DIM)

        sep()

        # ── Controls ─────────────────────────────────────────────────────────
        speed_labels = ["slow", "normal", "fast", "turbo"]
        run_lbl = f"{'[RUNNING]' if self._running else '[PAUSED] '}"
        text(f"{run_lbl}  {speed_labels[self._speed_idx]}", self._fmd, GREY_LT)
        text("SPACE step  R run  ↑↓ speed", self._fsm, TEXT_DIM)
        text("Q quit", self._fsm, TEXT_DIM)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lighten(col: tuple, amt: int) -> tuple:
    return tuple(min(255, c + amt) for c in col)
