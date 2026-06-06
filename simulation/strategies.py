"""
Disease strategies — Strategy Pattern for infection and progression (§3.1–3.2).

Swap strategies at runtime:
    world.set_disease_strategy(AggressiveFluStrategy())

To implement a custom strategy, subclass DiseaseStrategy and override:
    - transmission_rate(location_type) -> float   [β modifier]
    - initial_state()                  -> (severity, symptoms)
    - progression_matrix()             -> (A: 2×2, B: 2-vec, sigma_eta)
    - recovery_threshold()             -> (min_days, max_severity)
    - cloud_decay_rate()               -> float
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from simulation.models import LocationType


# ─── Base interface ───────────────────────────────────────────────────────────

class DiseaseStrategy(ABC):
    """
    Abstract strategy — defines all tuneable epidemiological parameters.
    Concrete strategies override only what differs.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    def transmission_rate(self, location_type: LocationType) -> float:
        """
        β multiplier for a given location type (eq. 2).
        Base rate in WorldEngine is multiplied by this value.
        """
        return 1.0

    def cloud_decay_rate(self) -> float:
        """Disease cloud intensity loss per tick (§3.1 ω decay)."""
        return 0.05


# ─── Concrete strategies ──────────────────────────────────────────────────────

class StandardFluStrategy(DiseaseStrategy):
    """
    Baseline seasonal influenza — moderate spread, predictable progression.
    Default strategy used when none is set explicitly.
    """

    name        = "Standard Flu"
    description = "Seasonal influenza — moderate β, 7–14 day recovery."

    def transmission_rate(self, location_type: LocationType) -> float:
        return {
            LocationType.HOME:      0.6,
            LocationType.WORK:      1.0,
            LocationType.THIRD:     1.4,
            LocationType.COMMUTING: 0.8,
            LocationType.HOSPITAL:  0.3,
        }.get(location_type, 1.0)





class AggressiveFluStrategy(DiseaseStrategy):
    """
    High-virulence variant — faster spread, higher initial severity,
    slower recovery. Models a novel strain or immunologically naive population.
    """

    name        = "Aggressive Flu"
    description = "High-virulence variant — fast spread, slow recovery."

    def transmission_rate(self, location_type: LocationType) -> float:
        base = StandardFluStrategy().transmission_rate(location_type)
        return base * 1.6   # 60 % more transmissible everywhere



    def recovery_threshold(self) -> tuple[int, float]:
        return 12, 0.10   # needs more days and lower severity to recover

    def cloud_decay_rate(self) -> float:
        return 0.03        # lingers longer in the environment


class MildCoronaStrategy(DiseaseStrategy):
    """
    Mild respiratory coronavirus — very fast spread but low severity.
    Most agents recover quickly; a small fraction escalate.
    """

    name        = "Mild Corona"
    description = "Mild coronavirus — high β, low severity, quick recovery."

    def transmission_rate(self, location_type: LocationType) -> float:
        base = StandardFluStrategy().transmission_rate(location_type)
        return base * 2.0   # highly contagious



    def recovery_threshold(self) -> tuple[int, float]:
        return 4, 0.20   # recovers quickly

    def cloud_decay_rate(self) -> float:
        return 0.08       # clears from air faster


class SlowBurnStrategy(DiseaseStrategy):
    """
    Slow-progressing chronic pathogen — low daily β, but symptoms
    accumulate over weeks. Models diseases with long incubation periods.
    """

    name        = "Slow Burn"
    description = "Slow-progressing pathogen — low daily β, long course."

    def transmission_rate(self, location_type: LocationType) -> float:
        base = StandardFluStrategy().transmission_rate(location_type)
        return base * 0.4   # hard to catch...



    def recovery_threshold(self) -> tuple[int, float]:
        return 20, 0.08   # ... but takes a long time to shake off

    def cloud_decay_rate(self) -> float:
        return 0.02


# ─── Registry (for TUI display / selection) ───────────────────────────────────

STRATEGIES: dict[str, DiseaseStrategy] = {
    s.name: s for s in [
        StandardFluStrategy(),
        AggressiveFluStrategy(),
        MildCoronaStrategy(),
        SlowBurnStrategy(),
    ]
}

# Alias so presets using the new disease names resolve correctly.
# Influenza → StandardFluStrategy (moderate-spread baseline)
# Bacterial Pneumonia → SlowBurnStrategy (lower transmissibility, slower progression)
STRATEGIES["Influenza"]           = StandardFluStrategy()
STRATEGIES["Bacterial Pneumonia"] = SlowBurnStrategy()
