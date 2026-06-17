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
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.mimic_db import MimicDatabase


# ─── Trajectory ───────────────────────────────────────────────────────────────

MANAGEMENT_TIERS: tuple[str, ...] = ("home rest", "treat", "hospitalise")
"""Severity-to-management mapping thresholds (0.0–0.40 / 0.40–0.70 / 0.70+)."""


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
    disease_name:          str   = "unknown"  # human-readable; set by strategy
    icd_code:              str   = "B99.9"    # ICD-10 training target; set by strategy
    incubation_days:       int   = 0          # days post-infection with no symptoms or transmission
    # Per-variable vital multipliers — scales SEVERITY_SLOPES for this disease.
    # Keys match NORMAL_RANGES in case_table.py.  Missing keys default to 1.0.
    # These make each disease clinically distinguishable in CaseTable vitals and
    # in the NEWS2-based ground-truth management decision.
    vital_slopes: dict = field(default_factory=dict)

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

        if self._day <= self.incubation_days:
            # Latent/incubation phase — no symptoms, no infectiousness
            self._severity = 0.0
            return 0.0, 0.0

        # Logistic rise / exponential decay measured from end of incubation
        effective_day = self._day - self.incubation_days

        if not self._at_peak:
            # Logistic rise: s approaches peak_severity with rate k
            # s(t) = peak / (1 + exp(-k*(t - t_half)))
            # where t_half = rise_days / 2
            k = 6.0 / max(self.rise_days, 1.0)   # steepness
            t_half = self.rise_days / 2.0
            self._severity = self.peak_severity / (
                1.0 + math.exp(-k * (effective_day - t_half))
            )
            if effective_day >= self.rise_days:
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
    """Seasonal influenza — moderate peak, clears in ~10 days."""
    name        = "Standard Flu"
    icd_code    = "J11.1"   # Influenza w/ other respiratory manifestations, virus not identified
    description = "Moderate peak (~0.45), clears in ~10 days."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.45, 0.10, 0.15, 0.75),
            rise_days             = self._sample(rng, 3.0,  0.5,  1.5,  5.0),
            decay_rate            = self._sample(rng, 0.18, 0.03, 0.08, 0.35),
            symptom_sensitivity   = self._sample(rng, 3.5,  0.5,  1.5,  6.0),
            severity_floor_weight = 0.25,
            disease_name          = self.name,
            icd_code              = self.icd_code,
            incubation_days       = 2,   # 1-4 d; 2 d typical for influenza
            # Fever-dominant: high temp + HR, SpO2 largely preserved
            vital_slopes = {"temp": 1.8, "HR": 1.3, "CRP": 1.5,
                            "SpO2": 0.4, "RR":  0.7, "BP_sys": 0.9},
        )


class AggressiveFluProgression(DiseaseProgressionStrategy):
    """High-virulence novel influenza — high peak, slow decay, long illness."""
    name        = "Aggressive Flu"
    icd_code    = "J09.X1"  # Novel influenza A virus with pneumonia
    description = "High peak (~0.70), slow decay — ~18 day illness."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.70, 0.08, 0.45, 0.92),
            rise_days             = self._sample(rng, 2.5,  0.5,  1.0,  4.0),
            decay_rate            = self._sample(rng, 0.07, 0.02, 0.03, 0.14),
            symptom_sensitivity   = self._sample(rng, 4.0,  0.5,  2.0,  7.0),
            severity_floor_weight = 0.22,
            disease_name          = self.name,
            icd_code              = self.icd_code,
            incubation_days       = 1,   # aggressive strain — short incubation
            # Viral pneumonia: high fever + significant respiratory compromise
            vital_slopes = {"temp": 2.0, "HR": 1.5, "CRP": 2.5, "WBC": 2.0,
                            "SpO2": 2.0, "RR":  1.8, "BP_sys": 1.2},
        )


class PersistentFluProgression(DiseaseProgressionStrategy):
    """
    Persistent respiratory infection — flu-like onset but very slow recovery.

    Designed to produce training events across many FL rounds:
      - Quick rise (3 d): agents hit clinic in round 1
      - Moderate-high peak (~0.65): ~35 % of agents cross the 0.70 severity
        override and seek care regardless of σ
      - Very slow decay (rate ~0.015): illness lasts 8+ weeks

    Intended use: pair with AggressiveFluStrategy (transmission) so the
    epidemic spreads fast then agents linger across rounds.
    """
    name        = "Persistent Flu"
    icd_code    = "J10.89"  # Influenza due to other identified virus, other manifestations
    description = "Fast onset (~3 d), moderate-high peak (~0.65), very slow decay — 8-week illness."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.65, 0.10, 0.40, 0.90),
            rise_days             = self._sample(rng, 3.0,  0.5,  1.5,  5.0),
            decay_rate            = self._sample(rng, 0.015, 0.003, 0.008, 0.025),
            symptom_sensitivity   = self._sample(rng, 4.0,  0.5,  2.0,  6.5),
            severity_floor_weight = 0.25,
            disease_name          = self.name,
            icd_code              = self.icd_code,
            incubation_days       = 2,
            # Sustained flu: moderate fever + mild SpO2 compromise, prolonged inflammation
            vital_slopes = {"temp": 1.5, "HR": 1.2, "CRP": 1.8, "WBC": 1.2,
                            "SpO2": 0.8, "RR":  1.0, "BP_sys": 0.9},
        )


class MildCoronaProgression(DiseaseProgressionStrategy):
    """Mild coronavirus — low peak, broad shallow curve."""
    name        = "Mild Corona"
    icd_code    = "U07.2"   # COVID-19, virus not identified
    description = "Low peak (~0.25), broad shallow curve — ~14 days."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.25, 0.07, 0.08, 0.45),
            rise_days             = self._sample(rng, 5.0,  1.0,  2.0,  9.0),
            decay_rate            = self._sample(rng, 0.10, 0.02, 0.05, 0.20),
            symptom_sensitivity   = self._sample(rng, 2.5,  0.5,  1.0,  5.0),
            severity_floor_weight = 0.20,
            disease_name          = self.name,
            icd_code              = self.icd_code,
            incubation_days       = 4,   # SARS-CoV-2-like: 4-6 d median
            # Silent hypoxia: notable SpO2 drop with low fever (the clinical puzzle)
            vital_slopes = {"SpO2": 1.8, "temp": 0.5, "HR": 0.8, "RR": 1.0,
                            "WBC": 0.6, "CRP": 1.2, "fatigue": 2.0},
        )


class SlowBurnProgression(DiseaseProgressionStrategy):
    """
    Slow-progressing sepsis-like pathogen — moderate peak, very slow decay.
    Low floor weight preserves the research challenge: plateau σ is muted.
    """
    name        = "Slow Burn"
    icd_code    = "A41.9"   # Sepsis, unspecified organism
    description = "Moderate peak, very slow decay — weeks-long illness."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.60, 0.10, 0.35, 0.85),
            rise_days             = self._sample(rng, 6.0,  1.0,  3.0, 10.0),
            decay_rate            = self._sample(rng, 0.03, 0.01, 0.01, 0.07),
            symptom_sensitivity   = self._sample(rng, 3.0,  0.5,  1.5,  5.5),
            severity_floor_weight = 0.15,
            disease_name          = self.name,
            icd_code              = self.icd_code,
            incubation_days       = 4,   # slow-onset sepsis: subtle prodrome
            # Sepsis pattern: circulatory failure — high lactate, BP drops,
            # compensatory tachycardia; fever unreliable (can be absent or low)
            vital_slopes = {"lactate": 4.0, "BP_sys": 2.5, "HR": 2.0, "RR": 1.5,
                            "WBC": 2.5, "CRP": 3.0, "SpO2": 1.2, "temp": 0.4},
        )


class DeadlyProgression(DiseaseProgressionStrategy):
    """
    Life-threatening pneumonia — very high peak, near-zero decay.
    Patient will NOT recover without treatment.
    """
    name        = "Deadly"
    icd_code    = "J18.9"   # Pneumonia, unspecified organism
    description = "Very high peak (~0.88), near-zero decay — fatal without treatment."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.88, 0.05, 0.75, 0.99),
            rise_days             = self._sample(rng, 2.0,  0.5,  1.0,  3.5),
            decay_rate            = self._sample(rng, 0.01, 0.005, 0.002, 0.025),
            symptom_sensitivity   = self._sample(rng, 4.5,  0.5,  2.5,  7.0),
            severity_floor_weight = 0.12,
            disease_name          = self.name,
            icd_code              = self.icd_code,
            incubation_days       = 1,   # acute pneumonia: rapid onset
            # Respiratory failure: severe SpO2 drop + high RR are the defining signs;
            # septic physiology with BP drop and tachycardia at severe stages
            vital_slopes = {"SpO2": 3.0, "RR": 2.5, "BP_sys": 1.5, "HR": 1.5,
                            "WBC": 2.0, "CRP": 2.0, "temp": 1.0},
        )


class InfluenzaProgression(DiseaseProgressionStrategy):
    name        = "Influenza"
    icd_code    = "J10.1"
    description = "Sudden onset, fever+myalgia dominant. Fast spread, clears ~8 days."

    def sample_trajectory(self, rng):
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.55, 0.12, 0.20, 0.82),
            rise_days             = self._sample(rng, 2.5,  0.5,  1.5,  4.0),
            decay_rate            = self._sample(rng, 0.17, 0.03, 0.08, 0.30),
            symptom_sensitivity   = self._sample(rng, 3.5,  0.5,  1.5,  6.0),
            severity_floor_weight = 0.25,
            disease_name          = "influenza",
            icd_code              = self.icd_code,
            incubation_days       = 2,
            vital_slopes = {"temp": 2.0, "HR": 1.3, "CRP": 1.5,
                            "SpO2": 0.3, "RR": 0.6, "BP_sys": 0.8},
        )


class BacterialPneumoniaProgression(DiseaseProgressionStrategy):
    name        = "Bacterial Pneumonia"
    icd_code    = "J18.9"
    description = "Slower onset, respiratory dominant. Less contagious, higher severity, longer recovery."

    def sample_trajectory(self, rng):
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.75, 0.12, 0.35, 0.97),
            rise_days             = self._sample(rng, 4.5,  0.8,  2.5,  7.0),
            decay_rate            = self._sample(rng, 0.07, 0.02, 0.03, 0.13),
            symptom_sensitivity   = self._sample(rng, 3.0,  0.5,  1.5,  5.5),
            severity_floor_weight = 0.28,
            disease_name          = "pneumonia",
            icd_code              = self.icd_code,
            incubation_days       = 3,
            vital_slopes = {"SpO2": 2.5, "RR": 2.2, "temp": 1.4, "HR": 1.4,
                            "WBC": 2.0, "CRP": 2.5, "BP_sys": 1.2},
        )


# ─── Fictional diseases (unknown-disease experiment) ─────────────────────────
#
# SIR dynamics are calibrated to match influenza (Velarex) and pneumonia
# (Sornathis) so epidemic arc properties are preserved.
# The disease_name is intentionally fictional — phi3:mini has no prior.

class VelarexProgression(DiseaseProgressionStrategy):
    """
    Fictional fast-spreading inflammatory disease (musculoskeletal / peripheral).
    Epidemic dynamics similar to influenza; symptom profile completely invented.
    See simulation/fictional_diseases.py for LLM prompt definitions.
    """
    name        = "Velarex"
    icd_code    = "X01.V"   # synthetic code — not a real ICD entry
    description = "Sudden-onset joint warmth, extremity mottling, photophobia, metallic taste."

    def sample_trajectory(self, rng):
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.50, 0.12, 0.18, 0.78),
            rise_days             = self._sample(rng, 2.5,  0.5,  1.5,  4.0),
            decay_rate            = self._sample(rng, 0.17, 0.03, 0.08, 0.30),
            symptom_sensitivity   = self._sample(rng, 3.5,  0.5,  1.5,  6.0),
            severity_floor_weight = 0.20,
            disease_name          = "velarex",
            icd_code              = self.icd_code,
            incubation_days       = 2,
            # Inflammatory tachycardia; no respiratory signature (SpO2/RR minimal)
            vital_slopes = {"HR": 1.8, "temp": 1.2, "CRP": 1.0,
                            "SpO2": 0.1, "RR": 0.2, "BP_sys": 0.5},
        )


class SornathisProgression(DiseaseProgressionStrategy):
    """
    Fictional slow-spreading neuro-respiratory disease (paraesthesia + dry cough).
    Epidemic dynamics similar to bacterial pneumonia; symptom profile completely invented.
    See simulation/fictional_diseases.py for LLM prompt definitions.
    """
    name        = "Sornathis"
    icd_code    = "X01.S"   # synthetic code — not a real ICD entry
    description = "Gradual-onset peripheral tingling, dry cough, blurred vision, night sweats."

    def sample_trajectory(self, rng):
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.72, 0.13, 0.32, 0.96),
            rise_days             = self._sample(rng, 4.5,  0.8,  2.5,  7.0),
            decay_rate            = self._sample(rng, 0.07, 0.02, 0.03, 0.13),
            symptom_sensitivity   = self._sample(rng, 3.0,  0.5,  1.5,  5.5),
            severity_floor_weight = 0.28,
            disease_name          = "sornathis",
            icd_code              = self.icd_code,
            incubation_days       = 3,
            # Respiratory compromise + CRP; no inflammatory tachycardia
            vital_slopes = {"SpO2": 2.0, "RR": 1.8, "CRP": 2.2,
                            "temp": 0.8, "HR": 0.7, "BP_sys": 1.0},
        )


class MorvenProgression(DiseaseProgressionStrategy):
    """
    Novel/emerging fictional disease for the unknown-disease experiment.
    GI-neurological profile: abdominal cramping, confusion episodes, cold sensitivity.
    Deliberately ambiguous — shares mild fever with Velarex and neurological element
    with Sornathis, but the GI + confusion combination is unique.
    NOT added to PROGRESSION_STRATEGIES by default; injected mid-run in one silo only.
    See simulation/fictional_diseases.py for LLM prompt definitions.
    """
    name        = "Morven"
    icd_code    = "X01.M"   # synthetic code
    description = "Recurring abdominal cramps, confusion episodes, cold sensitivity, mild fever."

    def sample_trajectory(self, rng):
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.45, 0.12, 0.18, 0.72),
            rise_days             = self._sample(rng, 3.5,  0.6,  2.0,  5.5),
            decay_rate            = self._sample(rng, 0.11, 0.02, 0.05, 0.18),
            symptom_sensitivity   = self._sample(rng, 3.2,  0.5,  1.5,  5.5),
            severity_floor_weight = 0.22,
            disease_name          = "morven",
            icd_code              = self.icd_code,
            incubation_days       = 3,
            # GI-inflammatory + mild fever; no respiratory, no musculoskeletal peak
            vital_slopes = {"HR": 1.1, "temp": 1.0, "WBC": 1.8, "CRP": 1.3,
                            "SpO2": 0.05, "RR": 0.1, "BP_sys": 0.4},
        )


# ─── Benchmark probe disease ──────────────────────────────────────────────────

class BenchmarkFeverProgression(DiseaseProgressionStrategy):
    """
    Synthetic viral haemorrhagic-like fever — used only for embedding probes.
    NOT added to PROGRESSION_STRATEGIES (not circulated in simulations).

    Design goals:
    - Wide severity variance (SD 0.25) → all three management tiers naturally represented
    - Distinctive vital signature (massively elevated CRP, coagulopathy proxy, SpO2 drop)
      so the model *can* separate it from existing diseases if it learns correctly
    - ICD A99.0: synthetic code, clearly marks probe events in scatter plots
    """
    name        = "Benchmark Fever"
    icd_code    = "A99.0"
    description = "Synthetic benchmark — wide severity range, distinctive haemorrhagic vital profile."

    def sample_trajectory(self, rng: random.Random) -> DiseaseTrajectory:
        return DiseaseTrajectory(
            peak_severity         = self._sample(rng, 0.60, 0.25, 0.12, 0.96),
            rise_days             = self._sample(rng, 4.0,  0.8,  2.0,  7.0),
            decay_rate            = self._sample(rng, 0.06, 0.02, 0.02, 0.12),
            symptom_sensitivity   = self._sample(rng, 5.0,  0.8,  2.5,  8.0),
            severity_floor_weight = 0.30,
            disease_name          = self.name,
            icd_code              = self.icd_code,
            incubation_days       = 3,
            vital_slopes = {
                "CRP":    4.5,   # massively elevated — hallmark of this disease
                "WBC":    3.0,   # leukocytosis
                "temp":   2.5,   # high sustained fever
                "HR":     1.8,   # tachycardia
                "SpO2":   1.5,   # moderate hypoxia
                "BP_sys": 1.8,   # hypotension at severe stages
                "RR":     1.4,
            },
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
        InfluenzaProgression(),
        BacterialPneumoniaProgression(),
        VelarexProgression(),
        SornathisProgression(),
        MorvenProgression(),
    ]
}
# Legacy classes (StandardFluProgression, AggressiveFluProgression,
# PersistentFluProgression, MildCoronaProgression, SlowBurnProgression,
# DeadlyProgression) are retained but excluded from the registry.
# BenchmarkFeverProgression and MimicProgression are also retained for
# probes/stubs but not in the default registry.
