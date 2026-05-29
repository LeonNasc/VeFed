"""
Disease progression strategies — separate from transmission (DiseaseStrategy).

Each strategy defines the shape of an individual's illness curve:
  - Severity s(t): a skewed bell — logistic rise, exponential decay
  - Symptoms σ(t): proportional to ds/dt (derivative of severity)

                  peak
          ╭───────────╮
         /              ╲
        /                 ╲___________  ← slow decay (deadly)
       /                              ╲
──────/                                ╲──────
    rise                              cleared

σ is high on the upslope (patient feels worse),
near zero at the plateau (feels stable — dangerous for deadly diseases),
negative on the downslope (patient feels better).

Parameters per strategy:
  peak_severity      — ceiling of the bell          [0..1]
  peak_severity_sd   — individual variation (η)     [0..1]
  rise_days          — days from infection to peak
  decay_rate         — daily fractional drop after peak
                       small (0.03) = long plateau (deadly)
                       large (0.25) = clears fast (mild)
  symptom_sensitivity — how strongly σ tracks ds/dt

Extension point for MIMIC:
  Subclass DiseaseProgressionStrategy and override sample_trajectory().
  Return a MimicTrajectory that provides s(t) from real patient data.
  SymptomNarrator.followup_answer() will accept per-variable measurements
  in a future refactor — for now the σ band drives language generation.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.mimic_db import MimicDatabase


# ─── Trajectory ───────────────────────────────────────────────────────────────

@dataclass
class DiseaseTrajectory:
    """
    Per-agent illness curve, sampled once at infection time.

    s(t) is evaluated day-by-day via step():
      Rising phase  (day < peak_day):  logistic approach to peak_severity
      Declining phase (day >= peak_day): exponential decay from peak

    σ(t) = max(slope_component, floor_component)
      slope_component = symptom_sensitivity × Δs  (high on upslope, 0 at plateau)
      floor_component = severity_floor_weight × s(t)  (baseline from absolute severity)

    The floor ensures σ > 0 at the plateau so agents at high severity still
    report symptoms and seek care. Deadly/SlowBurn use a low floor (research
    challenge: plateau is still hard to triage, just not completely silent).
    """
    peak_severity:         float   # individual peak height
    rise_days:             float   # days from day-0 to peak
    decay_rate:            float   # daily fractional drop after peak
    symptom_sensitivity:   float   # slope_component = sensitivity × Δs
    severity_floor_weight: float = 0.25  # floor_component = weight × severity

    # Runtime state — updated by step()
    _day:          int   = 0
    _severity:     float = 0.0
    _prev_severity: float = 0.0
    _at_peak:      bool  = False

    def step(self) -> tuple[float, float]:
        """
        Advance one day. Returns (severity, symptoms).
        Called by HealthState.step() — do not call directly.
        """
        self._prev_severity = self._severity
        self._day += 1

        if not self._at_peak:
            # Logistic rise: s approaches peak_severity with rate k
            # s(t) = peak / (1 + exp(-k*(t - t_half)))
            # where t_half = rise_days / 2
            k = 6.0 / max(self.rise_days, 1.0)   # steepness
            t_half = self.rise_days / 2.0
            self._severity = self.peak_severity / (
                1.0 + math.exp(-k * (self._day - t_half))
            )
            if self._day >= self.rise_days:
                self._at_peak  = True
                self._severity = self.peak_severity
        else:
            # Exponential decay after peak
            self._severity = self._severity * (1.0 - self.decay_rate)

        self._severity = max(0.0, min(1.0, self._severity))

        delta = self._severity - self._prev_severity
        slope_sig = max(0.0, self.symptom_sensitivity * delta)
        floor_sig = self.severity_floor_weight * self._severity
        symptoms  = min(1.0, max(slope_sig, floor_sig))

        return self._severity, symptoms

    def apply_treatment(self, decay_boost: float = 0.15) -> None:
        """
        Treatment accelerates recovery — increases decay_rate.
        Called by ClinicQueue.apply_outcomes() on RESOLVE action.
        """
        self.decay_rate = min(0.99, self.decay_rate + decay_boost)

    @property
    def day(self) -> int:
        return self._day

    @property
    def cleared(self) -> bool:
        return self._at_peak and self._severity < 0.02

    @property
    def trend(self) -> str:
        """Day-over-day severity direction — used for temporal symptom language."""
        delta = self._severity - self._prev_severity
        if delta > 0.015:
            return "worsening"
        elif delta < -0.015:
            return "improving"
        return "stable"


# ─── Abstract base ────────────────────────────────────────────────────────────

class DiseaseProgressionStrategy(ABC):
    """
    Abstract strategy — defines how individual trajectories are sampled.
    Override sample_trajectory() to implement custom or data-driven curves.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        """
        Sample one individual's illness trajectory at infection time.
        rng provides reproducible individual variation (η in §3.2).
        """
        ...

    def _sample(self, rng: random.Random, mean: float, sd: float,
                lo: float = 0.01, hi: float = 1.0) -> float:
        """Gaussian sample clamped to [lo, hi]."""
        return max(lo, min(hi, rng.gauss(mean, sd)))


# ─── Concrete strategies ──────────────────────────────────────────────────────

class StandardFluProgression(DiseaseProgressionStrategy):
    """
    Seasonal influenza — moderate peak, clears in ~10 days.
    Symmetric-ish bell; patient clearly feels worse then better.
    """
    name        = "Standard Flu"
    description = "Moderate peak (~0.45), clears in ~10 days."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.45, 0.10, 0.15, 0.75),
            rise_days             = self._sample(rng, 3.0,  0.5,  1.5,  5.0),
            decay_rate            = self._sample(rng, 0.18, 0.03, 0.08, 0.35),
            symptom_sensitivity   = self._sample(rng, 3.5,  0.5,  1.5,  6.0),
            severity_floor_weight = 0.25,
        )


class AggressiveFluProgression(DiseaseProgressionStrategy):
    """
    High-virulence strain — high peak, slow decay, long illness.
    Patient feels very ill for 2+ weeks.
    """
    name        = "Aggressive Flu"
    description = "High peak (~0.70), slow decay — ~18 day illness."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.70, 0.08, 0.45, 0.92),
            rise_days             = self._sample(rng, 2.5,  0.5,  1.0,  4.0),
            decay_rate            = self._sample(rng, 0.07, 0.02, 0.03, 0.14),
            symptom_sensitivity   = self._sample(rng, 4.0,  0.5,  2.0,  7.0),
            severity_floor_weight = 0.22,
        )


class PersistentFluProgression(DiseaseProgressionStrategy):
    """
    Persistent respiratory infection — flu-like onset but very slow recovery.

    Designed to produce training events across many FL rounds:
      - Quick rise (3 d): agents hit clinic in round 1
      - Moderate-high peak (~0.65): ~35 % of agents cross the 0.70 severity
        override and seek care every day regardless of σ
      - Very slow decay (rate ~0.025): illness lasts 5–7 weeks, so later
        rounds still have data

    Intended use: pair with AggressiveFluStrategy (transmission) so the
    epidemic spreads across the population in the first 1–2 rounds, then
    agents linger and generate training examples for the remaining rounds.
    """
    name        = "Persistent Flu"
    description = "Fast onset (~3 d), moderate-high peak (~0.65), very slow decay — 5–7 week illness."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.65, 0.10, 0.40, 0.90),
            rise_days             = self._sample(rng, 3.0,  0.5,  1.5,  5.0),
            decay_rate            = self._sample(rng, 0.015, 0.003, 0.008, 0.025),
            symptom_sensitivity   = self._sample(rng, 4.0,  0.5,  2.0,  6.5),
            severity_floor_weight = 0.25,
        )


class MildCoronaProgression(DiseaseProgressionStrategy):
    """
    Mild coronavirus — low peak, broad curve (slow rise AND slow decay),
    patient feels slightly off for a long time but never very ill.
    """
    name        = "Mild Corona"
    description = "Low peak (~0.25), broad shallow curve — ~14 days."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.25, 0.07, 0.08, 0.45),
            rise_days             = self._sample(rng, 5.0,  1.0,  2.0,  9.0),
            decay_rate            = self._sample(rng, 0.10, 0.02, 0.05, 0.20),
            symptom_sensitivity   = self._sample(rng, 2.5,  0.5,  1.0,  5.0),
            severity_floor_weight = 0.20,
        )


class SlowBurnProgression(DiseaseProgressionStrategy):
    """
    Slow-progressing pathogen — moderate peak, very slow decay.
    Severity stays high for weeks. Low floor weight preserves the research
    challenge: plateau symptoms are muted but not zero, so the LLM doctor
    still receives a weak signal rather than complete silence.
    """
    name        = "Slow Burn"
    description = "Moderate peak, very slow decay — weeks-long illness."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.60, 0.10, 0.35, 0.85),
            rise_days             = self._sample(rng, 6.0,  1.0,  3.0, 10.0),
            decay_rate            = self._sample(rng, 0.03, 0.01, 0.01, 0.07),
            symptom_sensitivity   = self._sample(rng, 3.0,  0.5,  1.5,  5.5),
            severity_floor_weight = 0.15,
        )


class DeadlyProgression(DiseaseProgressionStrategy):
    """
    Life-threatening disease — very high peak, near-zero decay rate.
    Severity plateaus near top for an extended period. The very low floor
    weight (0.12) keeps the plateau signal muted — the patient reports
    "not getting better" language rather than screaming severity — making
    this the hardest triage case for the LLM doctor.
    Patient will NOT recover without treatment (hospitalisation or resolve).
    """
    name        = "Deadly"
    description = "Very high peak (~0.88), near-zero decay — fatal without treatment."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.88, 0.05, 0.75, 0.99),
            rise_days             = self._sample(rng, 2.0,  0.5,  1.0,  3.5),
            decay_rate            = self._sample(rng, 0.01, 0.005, 0.002, 0.025),
            symptom_sensitivity   = self._sample(rng, 4.5,  0.5,  2.5,  7.0),
            severity_floor_weight = 0.12,
        )


# ─── MIMIC progression ────────────────────────────────────────────────────────

class MimicProgression(DiseaseProgressionStrategy):
    """
    MIMIC-III/IV dataset-driven progression.

    Draws patient trajectories from a MimicDatabase (mock or real).
    Returned trajectories are MimicDiseaseTrajectory objects backed by the
    patient's severity_series from the database record.

    Usage with mock data (no MIMIC access needed):
        from simulation.mimic_db import MockMimicDatabase
        db   = MockMimicDatabase()
        prog = MimicProgression(db, diagnosis_group="sepsis")
        world = WorldEngine(progression_strategy=prog)

    Usage with real MIMIC (future):
        db   = RealMimicDatabase("/path/to/mimic_csv/")
        prog = MimicProgression(db, diagnosis_group="sepsis")

    diagnosis_group: "sepsis" | "pneumonia" | "heart_failure" | None (any group).
    """
    name        = "MIMIC"
    description = "MIMIC dataset-driven progression."

    def __init__(
        self,
        database: Optional["MimicDatabase"] = None,
        diagnosis_group: Optional[str] = None,
        severity_floor_weight: float = 0.20,
    ):
        self._db            = database
        self._group         = diagnosis_group
        self._floor_weight  = severity_floor_weight

    def load(self, database: "MimicDatabase") -> None:
        """Wire in a database after construction (mirrors the old API contract)."""
        self._db = database

    def sample_trajectory(self, rng: random.Random) -> "MimicDiseaseTrajectory":  # type: ignore[override]
        if self._db is None:
            raise RuntimeError(
                "MimicProgression has no database. "
                "Pass one at construction: MimicProgression(MockMimicDatabase()) "
                "or call .load(db) before simulating."
            )
        from simulation.mimic_db import MimicDiseaseTrajectory
        record = self._db.random_subject(self._group, rng)
        return MimicDiseaseTrajectory(record, self._floor_weight)


# ─── Registry ─────────────────────────────────────────────────────────────────

def _make_mimic_mock() -> MimicProgression:
    from simulation.mimic_db import MockMimicDatabase
    return MimicProgression(MockMimicDatabase(), diagnosis_group=None)


PROGRESSION_STRATEGIES: dict[str, DiseaseProgressionStrategy] = {
    s.name: s for s in [
        StandardFluProgression(),
        AggressiveFluProgression(),
        PersistentFluProgression(),
        MildCoronaProgression(),
        SlowBurnProgression(),
        DeadlyProgression(),
        _make_mimic_mock(),
    ]
}
