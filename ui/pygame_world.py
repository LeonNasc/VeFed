"""
PygameWorld — 2D animated town view of the FedWorld simulator.

A fixed road network (8 junctions, 10 edges) defines the town.  Buildings
are placed near their gateway junction by type.  Agents walk at constant
speed along the road graph — building → junction → road path → junction →
building — so movement always follows visible roads.  Commuting is a silent
logical state; agents simply stay put until a real location is assigned.

Controls
────────
  SPACE     step one tick
  R         toggle auto-run
  ↑ / ↓     speed up / slow down
  Q / ESC   quit
"""
from __future__ import annotations

import collections
import hashlib
import math
import time

import pygame

from simulation.models import HealthStatus, LocationType
from simulation.world import WorldEngine

# ── Palette ───────────────────────────────────────────────────────────────────

BG_COL     = ( 28,  36,  28)
ROAD_COL   = ( 58,  54,  46)
ROAD_MRK   = ( 78,  74,  64)   # centre-line / junction dots

BCOL: dict[LocationType, tuple] = {
    LocationType.HOME:     (160, 115,  65),
    LocationType.WORK:     ( 55,  95, 160),
    LocationType.THIRD:    (185, 110,  30),
    LocationType.HOSPITAL: (225, 225, 230),
}
BRIM: dict[LocationType, tuple] = {
    LocationType.HOME:     (120,  80,  40),
    LocationType.WORK:     ( 35,  65, 120),
    LocationType.THIRD:    (140,  75,  15),
    LocationType.HOSPITAL: (170, 170, 185),
}
ACOL: dict[HealthStatus, tuple] = {
    HealthStatus.SUSCEPTIBLE: ( 80, 200,  80),
    HealthStatus.INFECTED:    (220,  45,  45),
    HealthStatus.RECOVERING:  (220, 180,  50),
    HealthStatus.RECOVERED:   ( 80, 160, 220),
}
HOSP_RING = (160,  75, 210)
HOSP_RED  = (200,  40,  40)

WHITE      = (240, 240, 245)
GREY_LT    = (160, 160, 170)
GREY_MD    = (100, 100, 110)
TEXT_DIM   = ( 90,  90, 100)
SIDEBAR_BG = ( 22,  22,  26)

# ── Sizes / speeds ────────────────────────────────────────────────────────────

BSIZE      = 78    # building square side (px)
AGENT_R    = 7
SIDEBAR_W  = 300
MAP_W      = 730
MAP_H      = 520
ROAD_W     = 22    # road stroke width
TICK_SPEEDS = [0.40, 0.18, 0.07, 0.025]

_FPS           = 30
_JOURNEY_PX    = 280   # typical home→work pixel distance
_JOURNEY_TICKS = 10    # target: complete a journey in this many sim ticks


def _agent_speed(speed_idx: int) -> float:
    """px/frame so a typical journey finishes in _JOURNEY_TICKS sim ticks."""
    frames_per_tick = max(1.0, TICK_SPEEDS[speed_idx] * _FPS)
    return _JOURNEY_PX / (_JOURNEY_TICKS * frames_per_tick)

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


# ── Road network ──────────────────────────────────────────────────────────────
#
#  Node layout (pixel coords on 730×520 map):
#
#   0(185,115) ──── 3(435,115) ──── 6(615,115)
#      │                │                │
#   1(185,270) ──── 4(435,270) ──── 7(615,270)
#      │                │
#   2(185,415) ──── 5(435,415)
#
#  Homes cluster left of 0/1/2.
#  Works cluster around 3/4/5/7.
#  Social clusters below 5.
#  Hospital sits above/right of 6.

_NODES: list[tuple[float, float]] = [
    (185, 115),  # 0  top-left
    (185, 270),  # 1  mid-left
    (185, 415),  # 2  bot-left
    (435, 115),  # 3  top-center
    (435, 270),  # 4  center-hub
    (435, 415),  # 5  bot-center
    (615, 115),  # 6  top-right
    (615, 270),  # 7  mid-right
]

_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2),        # left spine
    (3, 4), (4, 5),        # center spine
    (6, 7),                # right spine
    (0, 3), (3, 6),        # top cross
    (1, 4), (4, 7),        # mid cross
    (2, 5),                # bot cross
]

# Building slots: (dx, dy, gateway_node_index)
# dx/dy are pixel offsets from the gateway junction.
_SLOTS: dict[LocationType, list[tuple[int, int, int]]] = {
    LocationType.HOME: [
        (-105, -42, 0), (-105,  42, 0),
        (-105, -42, 1), (-105,  42, 1),
        (-105, -42, 2), (-105,  42, 2),
    ],
    LocationType.WORK: [
        (-105, -42, 3), (  10, -42, 3),
        ( -30, -30, 4), (  30,  30, 4),
        (-105,  42, 5), (  10,  42, 5),
        (  50,   0, 7),
    ],
    LocationType.THIRD: [
        ( -90,  65, 5), (  10,  65, 5),
    ],
    LocationType.HOSPITAL: [
        (  75, -20, 6),
    ],
}


# ── Layout ────────────────────────────────────────────────────────────────────

def _layout(world: WorldEngine) -> tuple[dict[str, tuple[float, float]],
                                         dict[str, int]]:
    """
    Assign pixel centres and gateway node indices to every non-commuting
    location.  Returns (loc_px, gateways).
    """
    by_type: dict[LocationType, list] = {}
    for loc in world.locations.values():
        if loc.type != LocationType.COMMUTING:
            by_type.setdefault(loc.type, []).append(loc)

    loc_px:   dict[str, tuple[float, float]] = {}
    gateways: dict[str, int]                 = {}

    for ltype, slots in _SLOTS.items():
        for i, loc in enumerate(by_type.get(ltype, [])):
            dx, dy, node = slots[i % len(slots)]
            nx, ny = _NODES[node]
            loc_px[loc.id]   = (float(nx + dx), float(ny + dy))
            gateways[loc.id] = node

    return loc_px, gateways


# ── Road-graph BFS ────────────────────────────────────────────────────────────

def _build_adj() -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(len(_NODES))}
    for a, b in _EDGES:
        adj[a].append(b)
        adj[b].append(a)
    return adj


_ADJ = _build_adj()


def _bfs(src: int, dst: int) -> list[int]:
    """Return list of node indices from src to dst (inclusive)."""
    if src == dst:
        return [src]
    q: collections.deque = collections.deque([[src]])
    visited = {src}
    while q:
        path = q.popleft()
        for nb in _ADJ[path[-1]]:
            if nb == dst:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                q.append(path + [nb])
    return [src, dst]


# ── Main class ────────────────────────────────────────────────────────────────

class PygameWorld:

    def __init__(self, world: WorldEngine):
        self.world = world
        pygame.init()
        pygame.display.set_caption("FedWorld — Town View")

        scr_w = MAP_W + SIDEBAR_W
        scr_h = max(MAP_H, 520)
        self.screen = pygame.display.set_mode((scr_w, scr_h), pygame.RESIZABLE)
        self.clock  = pygame.time.Clock()
        self._frame = 0

        self._fsm = pygame.font.SysFont("monospace", 11)
        self._fmd = pygame.font.SysFont("monospace", 13)
        self._flg = pygame.font.SysFont("monospace", 16, bold=True)
        self._fxl = pygame.font.SysFont("monospace", 19, bold=True)

        self._loc_px, self._gateway = _layout(world)

        # Per-agent deterministic scatter offset within a building
        self._off: dict[str, tuple[float, float]] = {}
        for ag in world.agents:
            h  = int(hashlib.md5(ag.id.encode()).hexdigest()[:8], 16)
            ox = ((h & 0xFF) - 128) * 0.22
            oy = (((h >> 8) & 0xFF) - 128) * 0.22
            self._off[ag.id] = (ox, oy)

        # Per-agent visual state
        self._pos:           dict[str, list[float]]          = {}
        self._wps:           dict[str, list[tuple[float, float]]] = {}
        self._last_real_loc: dict[str, str]                  = {}

        for ag in world.agents:
            loc = world.locations.get(ag.current_location)
            if loc and loc.type != LocationType.COMMUTING:
                self._last_real_loc[ag.id] = ag.current_location
            t = self._building_pos(ag.current_location, ag.id)
            self._pos[ag.id] = list(t)
            self._wps[ag.id] = []

        self._bg = self._render_bg()

        self._running   = False
        self._speed_idx = 1
        self._last_tick = 0.0

    # ── Position helpers ──────────────────────────────────────────────────────

    def _building_pos(self, loc_id: str, agent_id: str) -> tuple[float, float]:
        bx, by = self._loc_px.get(loc_id, (MAP_W / 2, MAP_H / 2))
        ox, oy = self._off[agent_id]
        return bx + ox, by + oy

    def _set_destination(self, ag, new_loc_id: str) -> None:
        """Compute road-following waypoints to new_loc_id."""
        src_loc  = self._last_real_loc.get(ag.id)
        src_node = self._gateway.get(src_loc, 0) if src_loc else self._nearest_node(self._pos[ag.id])
        dst_node = self._gateway.get(new_loc_id, src_node)

        path = _bfs(src_node, dst_node)

        wps: list[tuple[float, float]] = []
        # Step 1: walk from current building out to its road junction
        wps.append(_NODES[path[0]])
        # Step 2: follow road junctions to destination
        for n in path[1:]:
            wps.append(_NODES[n])
        # Step 3: walk from destination junction into the building
        bx, by = self._loc_px[new_loc_id]
        ox, oy = self._off[ag.id]
        wps.append((bx + ox, by + oy))

        self._wps[ag.id] = wps

    def _nearest_node(self, pos: list[float]) -> int:
        px, py = pos
        return min(range(len(_NODES)),
                   key=lambda i: math.hypot(_NODES[i][0] - px, _NODES[i][1] - py))

    # ── Background surface (drawn once) ───────────────────────────────────────

    def _render_bg(self) -> pygame.Surface:
        surf = pygame.Surface((MAP_W, MAP_H))
        surf.fill(BG_COL)

        # Spur roads: each building connects to its gateway junction.
        # Drawn first so main roads paint over the junction end cleanly.
        for loc_id, (bx, by) in self._loc_px.items():
            node_idx = self._gateway.get(loc_id)
            if node_idx is not None:
                nx, ny = _NODES[node_idx]
                pygame.draw.line(surf, ROAD_COL,
                                 (int(bx), int(by)), (int(nx), int(ny)), 12)

        # Main road surfaces
        for a, b in _EDGES:
            ax, ay = int(_NODES[a][0]), int(_NODES[a][1])
            bx, by = int(_NODES[b][0]), int(_NODES[b][1])
            pygame.draw.line(surf, ROAD_COL, (ax, ay), (bx, by), ROAD_W)

        # Junction dots on top
        for nx, ny in _NODES:
            pygame.draw.circle(surf, ROAD_MRK, (int(nx), int(ny)), ROAD_W // 2)

        return surf

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

            self._update_agents()
            self._draw()
            pygame.display.flip()

    def _update_agents(self) -> None:
        speed = _agent_speed(self._speed_idx)
        for ag in self.world.agents:
            loc = self.world.locations.get(ag.current_location)
            if loc and loc.type != LocationType.COMMUTING:
                if ag.current_location != self._last_real_loc.get(ag.id):
                    self._set_destination(ag, ag.current_location)   # reads old _last_real_loc as source
                    self._last_real_loc[ag.id] = ag.current_location # update after

            wps = self._wps[ag.id]
            pos = self._pos[ag.id]

            if wps:
                tx, ty = wps[0]
                dx, dy = tx - pos[0], ty - pos[1]
                dist   = math.hypot(dx, dy)
                if dist <= speed:
                    pos[0], pos[1] = tx, ty
                    wps.pop(0)
                else:
                    pos[0] += dx / dist * speed
                    pos[1] += dy / dist * speed

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
        scr_w, scr_h = self.screen.get_size()
        self.screen.blit(self._bg, (0, 0))
        self._draw_buildings()
        self._draw_agents()
        self._draw_sidebar(scr_w, scr_h)

    def _draw_buildings(self) -> None:
        s    = self.screen
        half = BSIZE // 2

        for loc in self.world.locations.values():
            if loc.type == LocationType.COMMUTING or loc.id not in self._loc_px:
                continue

            cx, cy = int(self._loc_px[loc.id][0]), int(self._loc_px[loc.id][1])
            rect   = pygame.Rect(cx - half, cy - half, BSIZE, BSIZE)
            col    = BCOL.get(loc.type, GREY_MD)
            rim    = BRIM.get(loc.type, GREY_MD)

            pygame.draw.rect(s, (15, 15, 18), rect.move(3, 3), border_radius=6)
            pygame.draw.rect(s, col,           rect,            border_radius=6)
            pygame.draw.rect(s, rim,           rect, width=2,   border_radius=6)

            if loc.type == LocationType.HOSPITAL:
                cw, ch = 8, 26
                pygame.draw.rect(s, HOSP_RED, (cx - cw // 2, cy - ch // 2, cw, ch))
                pygame.draw.rect(s, HOSP_RED, (cx - ch // 2, cy - cw // 2, ch, cw))

            lbl = {LocationType.HOME: "home", LocationType.WORK: "work",
                   LocationType.THIRD: "social", LocationType.HOSPITAL: "HOSPITAL"}.get(loc.type, "")
            if lbl:
                tc   = WHITE if loc.type != LocationType.HOSPITAL else GREY_MD
                surf = self._fsm.render(lbl, True, tc)
                s.blit(surf, (cx - surf.get_width() // 2, cy + half // 2 - 4))

    def _draw_agents(self) -> None:
        s     = self.screen
        frame = self._frame

        for ag in self.world.agents:
            status       = ag.health_state.status
            hospitalised = getattr(ag, "hospitalised", False)
            px, py       = int(self._pos[ag.id][0]), int(self._pos[ag.id][1])
            col          = ACOL.get(status, GREY_MD)

            r = AGENT_R
            if status == HealthStatus.INFECTED:
                r = AGENT_R + int(2 * abs(math.sin(frame * 0.12)))

            if hospitalised:
                pygame.draw.circle(s, HOSP_RING, (px, py), r + 4, width=2)

            pygame.draw.circle(s, (15, 15, 18), (px + 2, py + 2), r)
            pygame.draw.circle(s, col,           (px, py),         r)
            pygame.draw.circle(s, _lighten(col, 80), (px - 2, py - 2), max(2, r // 3))

    def _draw_sidebar(self, scr_w: int, scr_h: int) -> None:
        s = self.screen
        w = self.world

        sx = MAP_W
        pygame.draw.rect(s, SIDEBAR_BG, (sx, 0, scr_w - sx, scr_h))
        pygame.draw.line(s, GREY_MD,    (sx, 0), (sx, scr_h), 1)

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

        text("FedWorld", self._fxl, WHITE)
        phase_lbl, phase_col = _phase(w.current_tick)
        text(f"Day {w.current_day:>3}  T{w.current_tick:>3}/287", self._fmd, GREY_LT)
        text(phase_lbl, self._flg, phase_col)
        sep()

        text("SIR", self._flg, GREY_LT)
        total    = max(len(w.agents), 1)
        sir_data = [
            ("S", w.sir_model.S, ACOL[HealthStatus.SUSCEPTIBLE]),
            ("I", w.sir_model.I, ACOL[HealthStatus.INFECTED]),
            ("R", w.sir_model.R, ACOL[HealthStatus.RECOVERING]),
        ]
        bar_w = scr_w - sx - 28
        for label, count, col in sir_data:
            frac  = count / total
            blen  = int(frac * bar_w)
            pygame.draw.rect(s, col, pygame.Rect(x, y, blen, 14), border_radius=3)
            pygame.draw.rect(s, GREY_MD, pygame.Rect(x, y, bar_w, 14), width=1, border_radius=3)
            lsurf = self._fsm.render(f"{label} {count:>3} ({frac*100:.0f}%)", True, WHITE)
            s.blit(lsurf, (x + 4, y))
            y += 20

        hosp = sum(1 for ag in w.agents if getattr(ag, "hospitalised", False))
        text(f"  hosp {hosp:>3}", self._fsm, HOSP_RING)
        sep()

        text("AGENTS", self._flg, GREY_LT)
        for status, col in ACOL.items():
            pygame.draw.circle(s, col, (x + 7, y + 7), 6)
            lsurf = self._fsm.render(status.value, True, GREY_LT)
            s.blit(lsurf, (x + 18, y + 1))
            y += 17
        pygame.draw.circle(s, GREY_LT,   (x + 7, y + 7),  6)
        pygame.draw.circle(s, HOSP_RING, (x + 7, y + 7), 10, width=2)
        lsurf = self._fsm.render("hospitalised", True, GREY_LT)
        s.blit(lsurf, (x + 18, y + 1))
        y += 17
        sep()

        text("EVENTS", self._flg, GREY_LT)
        for ev in list(w.event_log)[-10:]:
            short = ev[:36] + "…" if len(ev) > 36 else ev
            text(short, self._fsm, TEXT_DIM)
        sep()

        speed_labels = ["slow", "normal", "fast", "turbo"]
        run_lbl = "[RUNNING]" if self._running else "[PAUSED] "
        text(f"{run_lbl}  {speed_labels[self._speed_idx]}", self._fmd, GREY_LT)
        text("SPACE step  R run  ↑↓ speed", self._fsm, TEXT_DIM)
        text("Q quit", self._fsm, TEXT_DIM)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lighten(col: tuple, amt: int) -> tuple:
    return tuple(min(255, c + amt) for c in col)
