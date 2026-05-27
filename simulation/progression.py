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


# ─── Trajectory ───────────────────────────────────────────────────────────────

@dataclass
class DiseaseTrajectory:
    """
    Per-agent illness curve, sampled once at infection time.

    s(t) is evaluated day-by-day via step():
      Rising phase  (day < peak_day):  logistic approach to peak_severity
      Declining phase (day >= peak_day): exponential decay from peak

    σ(t) = symptom_sensitivity × Δs   (finite difference, clipped to [0,1])
    Negative Δs (improving) maps to σ = 0 — patient reports feeling better,
    not a negative symptom value.
    """
    peak_severity:      float   # individual peak height
    rise_days:          float   # days from day-0 to peak
    decay_rate:         float   # daily fractional drop after peak
    symptom_sensitivity: float  # σ = sensitivity × |Δs| on upslope

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

        # σ = sensitivity × Δs — positive only on upslope
        delta = self._severity - self._prev_severity
        symptoms = max(0.0, min(1.0, self.symptom_sensitivity * delta))

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
            peak_severity       = self._sample(rng, 0.45, 0.10, 0.15, 0.75),
            rise_days           = self._sample(rng, 3.0,  0.5,  1.5,  5.0),
            decay_rate          = self._sample(rng, 0.18, 0.03, 0.08, 0.35),
            symptom_sensitivity = self._sample(rng, 3.5,  0.5,  1.5,  6.0),
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
            peak_severity       = self._sample(rng, 0.70, 0.08, 0.45, 0.92),
            rise_days           = self._sample(rng, 2.5,  0.5,  1.0,  4.0),
            decay_rate          = self._sample(rng, 0.07, 0.02, 0.03, 0.14),
            symptom_sensitivity = self._sample(rng, 4.0,  0.5,  2.0,  7.0),
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
            peak_severity       = self._sample(rng, 0.25, 0.07, 0.08, 0.45),
            rise_days           = self._sample(rng, 5.0,  1.0,  2.0,  9.0),
            decay_rate          = self._sample(rng, 0.10, 0.02, 0.05, 0.20),
            symptom_sensitivity = self._sample(rng, 2.5,  0.5,  1.0,  5.0),
        )


class SlowBurnProgression(DiseaseProgressionStrategy):
    """
    Slow-progressing pathogen — moderate peak, very slow decay.
    Severity stays high for weeks; plateau causes σ ≈ 0 despite
    dangerous internal state. Hard to triage correctly.
    """
    name        = "Slow Burn"
    description = "Moderate peak, very slow decay — weeks-long illness."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity       = self._sample(rng, 0.60, 0.10, 0.35, 0.85),
            rise_days           = self._sample(rng, 6.0,  1.0,  3.0, 10.0),
            decay_rate          = self._sample(rng, 0.03, 0.01, 0.01, 0.07),
            symptom_sensitivity = self._sample(rng, 3.0,  0.5,  1.5,  5.5),
        )


class DeadlyProgression(DiseaseProgressionStrategy):
    """
    Life-threatening disease — very high peak, near-zero decay rate.
    Severity plateaus near top for an extended period; σ collapses to 0
    at the plateau even though the patient is critically ill.
    Patient will NOT recover without treatment (hospitalisation or resolve).
    """
    name        = "Deadly"
    description = "Very high peak (~0.88), near-zero decay — fatal without treatment."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity       = self._sample(rng, 0.88, 0.05, 0.75, 0.99),
            rise_days           = self._sample(rng, 2.0,  0.5,  1.0,  3.5),
            decay_rate          = self._sample(rng, 0.01, 0.005, 0.002, 0.025),
            symptom_sensitivity = self._sample(rng, 4.5,  0.5,  2.5,  7.0),
        )


# ─── MIMIC stub — extension point ────────────────────────────────────────────

class MimicProgression(DiseaseProgressionStrategy):
    """
    MIMIC-III/IV dataset-driven progression — future extension point.

    Contract:
      Load a MIMIC cohort CSV with columns:
        subject_id, day, severity_proxy (e.g. SOFA score normalised to [0,1])

      sample_trajectory() should:
        1. Draw a random subject_id from the cohort
        2. Return a MimicTrajectory that wraps that patient's time series
           and provides s(t) as a lookup rather than a computed curve

      SymptomNarrator.followup_answer() will be extended to accept
      per-variable measurements (HR, temp, SpO2, ...) mapped to language
      bands independently, giving richer LLM doctor interactions.

    Raises NotImplementedError until dataset is loaded.
    """
    name        = "MIMIC"
    description = "MIMIC dataset-driven progression (not yet loaded)."

    def __init__(self, dataset_path: str | None = None):
        self._dataset_path = dataset_path
        self._cohort       = None   # populated by load()

    def load(self, path: str) -> None:
        """Load and validate the MIMIC cohort CSV."""
        raise NotImplementedError(
            "MimicProgression.load() not yet implemented.\n"
            "Expected CSV columns: subject_id, day, severity_proxy [0..1]"
        )

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        raise NotImplementedError(
            "MimicProgression requires a loaded dataset. Call load() first."
        )


# ─── Registry ─────────────────────────────────────────────────────────────────

PROGRESSION_STRATEGIES: dict[str, DiseaseProgressionStrategy] = {
    s.name: s for s in [
        StandardFluProgression(),
        AggressiveFluProgression(),
        MildCoronaProgression(),
        SlowBurnProgression(),
        DeadlyProgression(),
    ]
}
