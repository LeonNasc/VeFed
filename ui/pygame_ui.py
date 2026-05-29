"""
Pygame city visualisation for the Federated Simulated World.

City is divided into four districts:
  Residential (NW)  – HOME locations
  Commercial  (NE)  – WORK locations
  Leisure     (SW)  – THIRD-place venues
  Hospital    (SE)  – medical facilities

Streets are the visible road-colour background between building blocks.
Agents commuting between districts appear as coloured dots on the road network.

Controls:
  Space       – pause / resume
  + / -       – increase / decrease simulation speed
  Q / Escape  – quit
"""
from __future__ import annotations
import math
import random
import threading
import time

import pygame

from simulation.models import HealthStatus, LocationType
from simulation.world import WorldEngine

# ── Palette ──────────────────────────────────────────────────────────────────
BG         = ( 18,  20,  28)
HEADER_BG  = ( 28,  30,  42)
SIDEBAR_BG = ( 24,  26,  36)
GRID_LINE  = ( 40,  42,  55)

ROAD_COL   = ( 30,  31,  36)   # asphalt — fills everything not covered by a building
KERB_COL   = ( 48,  50,  58)   # district boundary line colour

LOC_FILL = {
    LocationType.HOME:     (118,  82,  52),
    LocationType.WORK:     ( 52,  78, 118),
    LocationType.THIRD:    ( 44, 108,  56),
    LocationType.HOSPITAL: (172, 172, 195),
}
LOC_BORDER = {
    LocationType.HOME:     (185, 138,  88),
    LocationType.WORK:     ( 92, 132, 178),
    LocationType.THIRD:    ( 78, 158,  86),
    LocationType.HOSPITAL: (215, 215, 238),
}

DISTRICT_LABEL_COL = ( 60,  64,  78)

WINDOW_COL  = (118, 158, 210)   # office window tint
ROOF_COL    = ( 88,  55,  32)   # home rooftop strip
PARK_COL    = ( 38,  92,  46)   # leisure inner green

LOC_LABEL_COL = (225, 225, 235)

AGENT_COLOR = {
    HealthStatus.SUSCEPTIBLE: ( 78, 196,  78),
    HealthStatus.INFECTED:    (220,  55,  55),
    HealthStatus.RECOVERING:  (238, 158,  38),
    HealthStatus.RECOVERED:   ( 78, 155, 218),
}

TEXT_FG  = (200, 205, 215)
TEXT_DIM = ( 98, 108, 122)
SIR_S    = ( 78, 196,  78)
SIR_I    = (220,  55,  55)
SIR_R    = ( 78, 155, 218)

# ── Dimensions ────────────────────────────────────────────────────────────────
WIN_W     = 1280
WIN_H     = 820
HEADER_H  = 54
STATUS_H  = 28
SIDEBAR_W = 300
MAP_PAD   = 10

ROAD_W    = 14   # street width — gap left around each building block
BLOCK_PAD =  5   # extra inset from cell edge to building rect
AGENT_SZ  =  7
AGENT_GAP =  2
LABEL_H   = 15

# District extents as fractions of map area: (x0, y0, x1, y1)
DISTRICT_FRACS = {
    LocationType.HOME:     (0.00, 0.00, 0.52, 0.60),
    LocationType.WORK:     (0.52, 0.00, 1.00, 0.52),
    LocationType.THIRD:    (0.00, 0.60, 0.62, 1.00),
    LocationType.HOSPITAL: (0.62, 0.52, 1.00, 1.00),
}

DISTRICT_NAMES = {
    LocationType.HOME:     "RESIDENTIAL",
    LocationType.WORK:     "COMMERCIAL",
    LocationType.THIRD:    "LEISURE",
    LocationType.HOSPITAL: "HOSPITAL DISTRICT",
}

_ROAD_SAMPLE_N = 500   # pre-sampled road positions for commuter dots


class PygameUI:
    SPEEDS = [1, 2, 4, 8, 16, 32]

    def __init__(self, world: WorldEngine):
        self.world      = world
        self._speed_idx = 2
        self._paused    = False
        self._lock      = threading.Lock()
        self._running   = True
        self._last_tick = time.monotonic()

        pygame.init()
        pygame.display.set_caption("Federated Simulated World")
        self.screen  = pygame.display.set_mode((WIN_W, WIN_H))
        self.font_sm = pygame.font.SysFont("monospace", 11)
        self.font_md = pygame.font.SysFont("monospace", 13)
        self.font_lg = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_xl = pygame.font.SysFont("monospace", 18, bold=True)
        self.clock   = pygame.time.Clock()

        self._build_layout()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        map_x = MAP_PAD
        map_y = HEADER_H + MAP_PAD
        map_w = WIN_W - SIDEBAR_W - 2 * MAP_PAD
        map_h = WIN_H - HEADER_H - STATUS_H - 2 * MAP_PAD
        self._map_rect = pygame.Rect(map_x, map_y, map_w, map_h)

        by_type: dict[LocationType, list] = {lt: [] for lt in DISTRICT_FRACS}
        for loc in self.world.locations.values():
            if loc.type in DISTRICT_FRACS:
                by_type[loc.type].append(loc)

        self._loc_rects:      dict[str, pygame.Rect]         = {}
        self._district_rects: dict[LocationType, pygame.Rect] = {}

        for loc_type, (fx0, fy0, fx1, fy1) in DISTRICT_FRACS.items():
            dx = int(map_x + fx0 * map_w)
            dy = int(map_y + fy0 * map_h)
            dw = int((fx1 - fx0) * map_w)
            dh = int((fy1 - fy0) * map_h)
            self._district_rects[loc_type] = pygame.Rect(dx, dy, dw, dh)

            locs = by_type[loc_type]
            if not locs:
                continue

            n    = len(locs)
            cols = max(1, round(n ** 0.5))
            rows = (n + cols - 1) // cols

            # Half-road margin on each district edge; full road gap between cells
            ux = dx + ROAD_W // 2
            uy = dy + ROAD_W // 2
            uw = dw - ROAD_W
            uh = dh - ROAD_W

            cell_w = uw / cols if cols == 1 else (uw - (cols - 1) * ROAD_W) / cols
            cell_h = uh / rows if rows == 1 else (uh - (rows - 1) * ROAD_W) / rows

            for i, loc in enumerate(locs):
                c  = i % cols
                r  = i // cols
                bx = int(ux + c * (cell_w + ROAD_W)) + BLOCK_PAD
                by = int(uy + r * (cell_h + ROAD_W)) + BLOCK_PAD
                bw = max(14, int(cell_w) - 2 * BLOCK_PAD)
                bh = max(14, int(cell_h) - 2 * BLOCK_PAD)
                self._loc_rects[loc.id] = pygame.Rect(bx, by, bw, bh)

        self._road_pts = self._sample_road_points(_ROAD_SAMPLE_N)

    def _sample_road_points(self, n: int) -> list[tuple[int, int]]:
        """Sample random pixel positions that fall in the road network (not on buildings)."""
        rng   = random.Random(42)
        rects = list(self._loc_rects.values())
        mx, my = self._map_rect.x, self._map_rect.y
        mw, mh = self._map_rect.w, self._map_rect.h
        pts, tries = [], 0
        while len(pts) < n and tries < n * 25:
            px = rng.randint(mx, mx + mw - 1)
            py = rng.randint(my, my + mh - 1)
            if not any(r.inflate(4, 4).collidepoint(px, py) for r in rects):
                pts.append((px, py))
            tries += 1
        return pts

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        while self._running:
            self._handle_events()
            self._step_sim()
            self._draw()
            self.clock.tick(60)
        pygame.quit()

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    self._running = False
                elif ev.key == pygame.K_SPACE:
                    self._paused = not self._paused
                    self._last_tick = time.monotonic()
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._speed_idx = min(self._speed_idx + 1, len(self.SPEEDS) - 1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._speed_idx = max(self._speed_idx - 1, 0)

    def _step_sim(self):
        if self._paused:
            self._last_tick = time.monotonic()
            return
        tps   = self.SPEEDS[self._speed_idx]
        now   = time.monotonic()
        steps = min(int((now - self._last_tick) * tps), tps)
        if steps < 1:
            return
        self._last_tick = now
        for _ in range(steps):
            with self._lock:
                self.world.step_tick()

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(ROAD_COL)
        self._draw_district_zones()
        self._draw_buildings()
        self._draw_commuters()
        self._draw_header()
        self._draw_sidebar()
        self._draw_status_bar()
        pygame.display.flip()

    # ── District zones ────────────────────────────────────────────────────────

    def _draw_district_zones(self):
        for loc_type, rect in self._district_rects.items():
            # Subtle boundary outline
            pygame.draw.rect(self.screen, KERB_COL, rect, width=1)
            # District name label
            label = DISTRICT_NAMES.get(loc_type, "")
            surf  = self.font_sm.render(label, True, DISTRICT_LABEL_COL)
            self.screen.blit(surf, (rect.x + 4, rect.y + 3))

    # ── Buildings ─────────────────────────────────────────────────────────────

    def _draw_buildings(self):
        w = self.world
        for loc_id, rect in self._loc_rects.items():
            loc         = w.locations[loc_id]
            agents_here = [a for a in w.agents if a.current_location == loc_id]
            self._draw_building(loc, rect)
            self._draw_agents_in(agents_here, rect)

    def _draw_building(self, loc, rect: pygame.Rect):
        fill   = LOC_FILL.get(loc.type,   (80, 80, 80))
        border = LOC_BORDER.get(loc.type, (120, 120, 120))

        pygame.draw.rect(self.screen, fill, rect, border_radius=3)
        self._draw_building_details(loc, rect)

        cloud = loc.ambient_exposure()
        if cloud > 0.005:
            alpha   = min(190, int(cloud * 480))
            overlay = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            overlay.fill((220, 40, 40, alpha))
            self.screen.blit(overlay, (rect.x, rect.y))

        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=3)

        label = self.font_sm.render(f"{loc.short_name} {loc.id}", True, LOC_LABEL_COL)
        self.screen.blit(label, (rect.x + 3, rect.y + 2))

    def _draw_building_details(self, loc, rect: pygame.Rect):
        if loc.type == LocationType.WORK and rect.w > 22 and rect.h > 28:
            # Grid of office windows
            win_w, win_h = 4, 4
            gap_x, gap_y = 4, 6
            wx = rect.x + 5
            while wx + win_w < rect.right - 4:
                wy = rect.y + LABEL_H + 5
                while wy + win_h < rect.bottom - 4:
                    pygame.draw.rect(self.screen, WINDOW_COL, (wx, wy, win_w, win_h))
                    wy += win_h + gap_y
                wx += win_w + gap_x

        elif loc.type == LocationType.HOME and rect.h > 18:
            # Darker rooftop strip
            roof = pygame.Rect(rect.x + 2, rect.y + LABEL_H, rect.w - 4, 4)
            pygame.draw.rect(self.screen, ROOF_COL, roof)

        elif loc.type == LocationType.THIRD and rect.w > 20 and rect.h > 24:
            # Green interior
            inner = pygame.Rect(rect.x + 4, rect.y + LABEL_H + 4,
                                rect.w - 8, rect.h - LABEL_H - 8)
            if inner.w > 4 and inner.h > 4:
                pygame.draw.rect(self.screen, PARK_COL, inner, border_radius=2)

        elif loc.type == LocationType.HOSPITAL:
            # Red cross centred in the lower part of the building
            cx  = rect.centerx
            cy  = rect.y + LABEL_H + (rect.h - LABEL_H) // 2
            arm = max(5, min(rect.w, rect.h - LABEL_H) // 4)
            pygame.draw.line(self.screen, (220, 40, 40), (cx, cy - arm), (cx, cy + arm), 3)
            pygame.draw.line(self.screen, (220, 40, 40), (cx - arm, cy), (cx + arm, cy), 3)

    def _draw_agents_in(self, agents, rect: pygame.Rect):
        if not agents:
            return
        inner_x = rect.x + 4
        inner_y = rect.y + LABEL_H
        inner_w = rect.w - 8
        cols    = max(1, inner_w // (AGENT_SZ + AGENT_GAP))
        for i, agent in enumerate(agents):
            ax = inner_x + (i % cols) * (AGENT_SZ + AGENT_GAP)
            ay = inner_y + (i // cols) * (AGENT_SZ + AGENT_GAP)
            if ay + AGENT_SZ > rect.bottom - 2:
                break
            pygame.draw.rect(
                self.screen,
                AGENT_COLOR.get(agent.status, (128, 128, 128)),
                (ax, ay, AGENT_SZ, AGENT_SZ),
            )

    # ── Commuters on roads ────────────────────────────────────────────────────

    def _draw_commuters(self):
        commuting = [a for a in self.world.agents
                     if a.current_type == LocationType.COMMUTING]
        for i, agent in enumerate(commuting):
            if not self._road_pts:
                break
            px, py = self._road_pts[i % len(self._road_pts)]
            pygame.draw.rect(
                self.screen,
                AGENT_COLOR.get(agent.status, (128, 128, 128)),
                (px, py, AGENT_SZ - 1, AGENT_SZ - 1),
            )

    # ── Header ────────────────────────────────────────────────────────────────

    def _draw_header(self):
        pygame.draw.rect(self.screen, HEADER_BG, (0, 0, WIN_W, HEADER_H))
        w = self.world

        surf = self.font_xl.render("Federated Simulated World", True, TEXT_FG)
        self.screen.blit(surf, (12, 8))

        tick   = w.current_tick
        hour   = (tick * 5) // 60
        minute = (tick * 5) % 60
        surf   = self.font_md.render(f"Day {w.current_day:04d}   {hour:02d}:{minute:02d}",
                                     True, TEXT_DIM)
        self.screen.blit(surf, (12, 32))

        sir_x = 360
        for lbl, val, col in [("S", w.sir_model.S, SIR_S),
                               ("I", w.sir_model.I, SIR_I),
                               ("R", w.sir_model.R, SIR_R)]:
            surf = self.font_lg.render(f"{lbl}:{val:3d}", True, col)
            self.screen.blit(surf, (sir_x, 16))
            sir_x += 90

        speed_s = "PAUSED" if self._paused else f"x{self.SPEEDS[self._speed_idx]}"
        color   = (255, 200, 60) if self._paused else TEXT_DIM
        surf    = self.font_md.render(speed_s, True, color)
        self.screen.blit(surf, (638, 20))

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _draw_sidebar(self):
        sx = WIN_W - SIDEBAR_W
        sy = HEADER_H
        sh = WIN_H - HEADER_H - STATUS_H
        pygame.draw.rect(self.screen, SIDEBAR_BG, (sx, sy, SIDEBAR_W, sh))
        pygame.draw.line(self.screen, GRID_LINE, (sx, sy), (sx, sy + sh), 1)

        y = sy + 10
        y = self._draw_sir_chart(sx + 8, y, SIDEBAR_W - 16, 110)
        y += 14

        for status, color in AGENT_COLOR.items():
            pygame.draw.rect(self.screen, color, (sx + 8, y + 1, 10, 10))
            surf = self.font_sm.render(f" {status.value}", True, TEXT_DIM)
            self.screen.blit(surf, (sx + 20, y))
            y += 14
        y += 6

        surf = self.font_sm.render("── events ──────────────", True, TEXT_DIM)
        self.screen.blit(surf, (sx + 8, y)); y += 16

        for entry in reversed(self.world.event_log[-22:]):
            surf = self.font_sm.render(entry[:36], True, TEXT_FG)
            self.screen.blit(surf, (sx + 8, y)); y += 13
            if y > sy + sh - 16:
                break

    def _draw_sir_chart(self, x: int, y: int, w: int, h: int) -> int:
        hist = self.world.sir_model.history
        pygame.draw.rect(self.screen, (28, 30, 44), (x, y, w, h))
        if len(hist) >= 2:
            N    = max(1, self.world.sir_model.total)
            data = hist[-(w // 2):]

            def _sparkline(idx, color):
                pts = [(x + int(i * (w - 2) / max(1, len(data) - 1)),
                        y + h - 2 - int(row[idx] / N * (h - 6)))
                       for i, row in enumerate(data)]
                if len(pts) >= 2:
                    pygame.draw.lines(self.screen, color, False, pts, 2)

            _sparkline(0, SIR_S)
            _sparkline(1, SIR_I)
            _sparkline(2, SIR_R)

        for i, (lbl, col) in enumerate([("S", SIR_S), ("I", SIR_I), ("R", SIR_R)]):
            surf = self.font_sm.render(lbl, True, col)
            self.screen.blit(surf, (x + i * 28, y + h + 2))
        return y + h + 16

    # ── Status bar ────────────────────────────────────────────────────────────

    def _draw_status_bar(self):
        y = WIN_H - STATUS_H
        pygame.draw.rect(self.screen, HEADER_BG, (0, y, WIN_W, STATUS_H))
        commuting = sum(1 for a in self.world.agents
                        if a.current_type == LocationType.COMMUTING)
        hint = f"  Space=pause   +/-=speed   Q=quit      commuting: {commuting}"
        surf = self.font_sm.render(hint, True, TEXT_DIM)
        self.screen.blit(surf, (8, y + 8))
