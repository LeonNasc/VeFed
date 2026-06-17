"""
Controlled data-arrival schedules for the scheduled FL ablation.

Each function returns a list of ints of length n_rounds where
schedule[r] = number of training events revealed to all silos in round r.
The sum equals n_total (rounding remainder is added to the last round).

Shapes
------
flat      — uniform; no temporal structure
ramp      — linearly increasing; data-rich late rounds
gaussian  — bell curve centred at mid-run; SIR-like but phase-synced
burst     — exponential decay; heavy early, trails off (epidemic fast-peak)
parabola  — U-shape; high at both ends, trough in middle (two-wave)
"""
from __future__ import annotations
import math


def _normalize(raw: list[float], n_total: int) -> list[int]:
    s = sum(raw)
    if s == 0:
        raise ValueError("schedule weights sum to zero")
    # Cumulative allocation: distribute rounding error across all rounds
    # rather than dumping the remainder onto the last round.
    cumf = cumi = 0
    counts = []
    for v in raw:
        cumf += v / s * n_total
        this = max(0, round(cumf) - cumi)
        counts.append(this)
        cumi += this
    return counts


def flat_schedule(n_total: int, n_rounds: int) -> list[int]:
    return _normalize([1.0] * n_rounds, n_total)


def ramp_schedule(n_total: int, n_rounds: int) -> list[int]:
    return _normalize([float(r + 1) for r in range(n_rounds)], n_total)


def gaussian_schedule(
    n_total: int,
    n_rounds: int,
    mu: float | None = None,
    sigma: float | None = None,
) -> list[int]:
    if mu is None:
        mu = (n_rounds - 1) / 2.0
    if sigma is None:
        sigma = n_rounds / 5.0
    raw = [math.exp(-0.5 * ((r - mu) / sigma) ** 2) for r in range(n_rounds)]
    return _normalize(raw, n_total)


def burst_schedule(
    n_total: int,
    n_rounds: int,
    tau: float | None = None,
) -> list[int]:
    """Exponential decay — data-heavy early, tapers off."""
    if tau is None:
        tau = n_rounds / 4.0
    raw = [math.exp(-r / tau) for r in range(n_rounds)]
    return _normalize(raw, n_total)


def parabola_schedule(n_total: int, n_rounds: int) -> list[int]:
    """U-shape — high at both ends, trough in the middle."""
    mid = (n_rounds - 1) / 2.0
    raw = [(r - mid) ** 2 + 0.5 for r in range(n_rounds)]
    return _normalize(raw, n_total)


SCHEDULES: dict[str, callable] = {
    "flat":     flat_schedule,
    "ramp":     ramp_schedule,
    "gaussian": gaussian_schedule,
    "burst":    burst_schedule,
    "parabola": parabola_schedule,
}


def make_schedule(name: str, n_total: int, n_rounds: int, **kwargs) -> list[int]:
    if name not in SCHEDULES:
        raise ValueError(f"unknown schedule {name!r}; choose from {list(SCHEDULES)}")
    return SCHEDULES[name](n_total, n_rounds, **kwargs)
