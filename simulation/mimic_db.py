"""
MIMIC database abstraction — interface + mock implementation.

MimicDatabase defines the three-method interface that both MockMimicDatabase
and any future real CSV loader must satisfy. Code built on this interface
(MimicCaseTable, MimicProgression, MimicDiseaseTrajectory) works identically
against both; swapping mock→real is a one-line change at construction time.

Real MIMIC integration notes (for future reference):
  Tables needed:
    ADMISSIONS  — hadm_id, admittime, dischtime, diagnosis
    CHARTEVENTS — hadm_id, charttime, itemid, value
    LABEVENTS   — hadm_id, charttime, itemid, value

  ItemID mapping (MIMIC-III subset):
    211   → HR           (bpm)
    678   → temp         (°F → subtract 32 × 5/9 for °C)
    618   → RR           (breaths/min)
    646   → SpO2         (%)
    51    → BP_sys       (arterial)
    8368  → BP_dia
    1127  → WBC          (×10⁹/L, Lab)
    8440  → CRP          (mg/L, not always present in MIMIC-III)
    818   → lactate      (mmol/L, Lab)

  Severity proxy: normalised SOFA score.
    SOFA range 0–24; normalise to [0,1] by dividing by 24.
    Can be approximated from component values (resp, coag, liver, CVS, CNS, renal).

  See MimicDatabase docstring for the expected CSV schema.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class MimicRecord:
    """
    One patient's ICU stay — mirrors what we'd load from MIMIC CSV exports.

    chartevents: {day: {variable: value}}  (day 0 = admission day)
    severity_series: normalised SOFA/APACHE proxy per day, shape [0,1]

    MIMIC equivalents:
      subject_id  → ADMISSIONS.subject_id
      hadm_id     → ADMISSIONS.hadm_id
      diagnosis   → ADMISSIONS.diagnosis (free-text, grouped to 3 categories here)
    """
    subject_id:      int
    hadm_id:         int
    diagnosis_group: str   # "sepsis" | "pneumonia" | "heart_failure"
    chartevents:     dict[int, dict[str, float]] = field(default_factory=dict)
    severity_series: list[float]                 = field(default_factory=list)

    @property
    def los_days(self) -> int:
        """Length of stay in days."""
        return len(self.severity_series)

    def vitals_at_day(self, day: int) -> dict[str, float]:
        return self.chartevents.get(day, {})

    def severity_at_day(self, day: int) -> float:
        if day < 0 or day >= len(self.severity_series):
            return 0.0
        return self.severity_series[day]


# ─── Abstract interface ────────────────────────────────────────────────────────

class MimicDatabase(ABC):
    """
    Interface contract for MIMIC data access.

    Implement this with:
      - MockMimicDatabase()          — synthetic data, no files needed
      - RealMimicDatabase(csv_dir)   — loads from MIMIC CSV exports (future)

    Expected CSV schema for RealMimicDatabase (one row per patient-day):
      subject_id, hadm_id, day, diagnosis_group,
      HR, temp, RR, SpO2, BP_sys, BP_dia, WBC, CRP, lactate,
      pain, fatigue, nausea, severity_proxy
    """

    @abstractmethod
    def cohort_ids(self, diagnosis_group: Optional[str] = None) -> list[int]:
        """Return subject_ids, optionally filtered by diagnosis group."""
        ...

    @abstractmethod
    def get_record(self, subject_id: int) -> MimicRecord:
        """Retrieve a patient record by subject_id. Raises KeyError if missing."""
        ...

    def random_subject(
        self,
        diagnosis_group: Optional[str] = None,
        rng: Optional[random.Random]   = None,
    ) -> MimicRecord:
        """Pick a random patient, optionally from a diagnosis group."""
        rng = rng or random.Random()
        ids = self.cohort_ids(diagnosis_group)
        if not ids:
            raise ValueError(f"No subjects for diagnosis_group={diagnosis_group!r}")
        return self.get_record(rng.choice(ids))


# ─── Per-group generation parameters ──────────────────────────────────────────

_GROUP_PARAMS: dict[str, dict] = {
    "sepsis": {
        "peak_severity_mean":  0.78, "peak_severity_sd": 0.08,
        "onset_days_mean":     3.0,  "onset_days_sd":    1.0,
        "recovery_days_mean": 15.0,  "recovery_days_sd": 4.0,
        "los_mean":           18.0,  "los_sd":           5.0,
        # Vital amplifiers relative to the base CaseTable slopes
        "amplifiers": {
            "HR": 1.4, "temp": 1.3, "WBC": 1.8, "CRP": 2.0,
            "lactate": 1.6, "SpO2": 1.2, "BP_sys": 1.1,
        },
    },
    "pneumonia": {
        "peak_severity_mean":  0.58, "peak_severity_sd": 0.10,
        "onset_days_mean":     3.0,  "onset_days_sd":    1.0,
        "recovery_days_mean": 10.0,  "recovery_days_sd": 2.5,
        "los_mean":           11.0,  "los_sd":           3.0,
        "amplifiers": {
            "RR": 1.8, "SpO2": 1.6, "temp": 1.2, "WBC": 1.3, "CRP": 1.4,
        },
    },
    "heart_failure": {
        "peak_severity_mean":  0.62, "peak_severity_sd": 0.10,
        "onset_days_mean":     2.0,  "onset_days_sd":    0.8,
        "recovery_days_mean": 12.0,  "recovery_days_sd": 3.5,
        "los_mean":           13.0,  "los_sd":           4.0,
        "amplifiers": {
            "HR": 1.3, "BP_sys": 0.8, "BP_dia": 0.8,
            "SpO2": 1.3, "fatigue": 1.5,
        },
    },
}

_VARIABLES = [
    "HR", "temp", "RR", "SpO2", "BP_sys", "BP_dia",
    "WBC", "CRP", "lactate", "pain", "fatigue", "nausea",
]

# Midpoints of NORMAL_RANGES from case_table.py — duplicated here to avoid import
_MIDPOINTS = {
    "HR": 80.0, "temp": 37.0, "RR": 16.0, "SpO2": 97.5,
    "BP_sys": 115.0, "BP_dia": 75.0,
    "WBC": 7.5, "CRP": 5.0, "lactate": 1.25,
    "pain": 1.0, "fatigue": 1.5, "nausea": 1.0,
}

_SLOPES = {
    "HR": +8.0, "temp": +0.35, "RR": +3.0, "SpO2": -1.5,
    "BP_sys": -4.0, "BP_dia": -2.0,
    "WBC": +2.0, "CRP": +15.0, "lactate": +0.4,
    "pain": +1.2, "fatigue": +1.5, "nausea": +0.8,
}


# ─── Mock implementation ───────────────────────────────────────────────────────

class MockMimicDatabase(MimicDatabase):
    """
    Synthetic ICU patient database — no MIMIC access required.

    Generates n_patients_per_group records per diagnosis group at construction.
    Vitals are correlated with diagnosis-group-specific severity trajectories,
    matching the variable schema of CaseTable for a direct drop-in swap.

    Usage:
        db = MockMimicDatabase()
        rec = db.random_subject("sepsis")
        ct  = MimicCaseTable(rec, rng)
        prog = MimicProgression(db, diagnosis_group="sepsis")

    Replace with RealMimicDatabase(csv_dir) when MIMIC data is available.
    """

    def __init__(self, n_patients_per_group: int = 50, seed: int = 0):
        rng = random.Random(seed)
        self._records: dict[int, MimicRecord] = {}
        subject_id = 10001   # start above plausible real MIMIC range for clarity

        for group, params in _GROUP_PARAMS.items():
            for _ in range(n_patients_per_group):
                rec = _generate_record(
                    subject_id, subject_id + 80000, group, params, rng
                )
                self._records[rec.subject_id] = rec
                subject_id += 1

    def cohort_ids(self, diagnosis_group: Optional[str] = None) -> list[int]:
        if diagnosis_group is None:
            return list(self._records.keys())
        return [
            sid for sid, rec in self._records.items()
            if rec.diagnosis_group == diagnosis_group
        ]

    def get_record(self, subject_id: int) -> MimicRecord:
        if subject_id not in self._records:
            raise KeyError(f"subject_id {subject_id} not in MockMimicDatabase")
        return self._records[subject_id]


# ─── MimicDiseaseTrajectory ────────────────────────────────────────────────────

class MimicDiseaseTrajectory:
    """
    Drop-in replacement for DiseaseTrajectory backed by a MimicRecord.

    Provides the same interface consumed by HealthState.step() and
    CaseTable._severity_at_day(). Severity values come directly from the
    record's normalised severity_series instead of a parametric curve.

    Treatment accelerates recovery by applying a cumulative severity reduction.
    """

    def __init__(self, record: MimicRecord, severity_floor_weight: float = 0.20):
        self._record               = record
        self.severity_floor_weight = severity_floor_weight
        self._day          = 0
        self._severity     = 0.0
        self._prev_severity = 0.0
        self._treatment_reduction = 0.0   # accumulated from apply_treatment()

    def step(self) -> tuple[float, float]:
        self._prev_severity = self._severity
        raw_sev = self._record.severity_at_day(self._day)
        self._severity = max(0.0, raw_sev - self._treatment_reduction)
        self._day += 1

        delta     = self._severity - self._prev_severity
        slope_sig = max(0.0, 3.0 * delta)   # fixed sensitivity for MIMIC trajectories
        floor_sig = self.severity_floor_weight * self._severity
        symptoms  = min(1.0, max(slope_sig, floor_sig))

        return self._severity, symptoms

    def apply_treatment(self, decay_boost: float = 0.15) -> None:
        self._treatment_reduction = min(1.0, self._treatment_reduction + decay_boost * 2.5)

    @property
    def _at_peak(self) -> bool:
        return self._day > self._record.los_days // 2

    @property
    def decay_rate(self) -> float:
        return 0.01   # nominal; triggers INFECTED → RECOVERING transition in HealthState

    @property
    def cleared(self) -> bool:
        return self._severity < 0.02

    @property
    def day(self) -> int:
        return self._day

    @property
    def trend(self) -> str:
        delta = self._severity - self._prev_severity
        if delta > 0.015:
            return "worsening"
        elif delta < -0.015:
            return "improving"
        return "stable"

    # Compatibility shims for CaseTable._severity_at_day (not called by MimicCaseTable)
    @property
    def rise_days(self) -> float:
        return max(1.0, self._record.los_days / 3.0)

    @property
    def peak_severity(self) -> float:
        return max(self._record.severity_series, default=0.5)


# ─── Generation helpers ────────────────────────────────────────────────────────

def _generate_record(
    subject_id: int,
    hadm_id: int,
    group: str,
    params: dict,
    rng: random.Random,
) -> MimicRecord:
    peak = max(0.10, min(0.99, rng.gauss(params["peak_severity_mean"],
                                          params["peak_severity_sd"])))
    onset    = max(1.0, rng.gauss(params["onset_days_mean"],    params["onset_days_sd"]))
    recovery = max(3.0, rng.gauss(params["recovery_days_mean"], params["recovery_days_sd"]))
    los      = int(max(onset + recovery * 0.5,
                       rng.gauss(params["los_mean"], params["los_sd"])))
    los = max(3, min(60, los))

    severity_series: list[float] = []
    for day in range(los):
        s = _severity_at_day(day, peak, onset, recovery)
        severity_series.append(round(s, 3))

    amplifiers = params.get("amplifiers", {})
    chartevents: dict[int, dict[str, float]] = {}
    for day in range(los):
        sev       = severity_series[day]
        sev_lagged = severity_series[max(0, day - 1)]   # temp lags 1 day
        row: dict[str, float] = {}
        for var in _VARIABLES:
            mid   = _MIDPOINTS[var]
            slope = _SLOPES[var]
            amp   = amplifiers.get(var, 1.0)
            s_eff = sev_lagged if var == "temp" else sev
            val   = mid + slope * s_eff * 10 * amp
            val  += rng.gauss(0, abs(slope) * amp * 0.25)
            # Clamp physiological limits
            if var == "SpO2":     val = min(100.0, max(60.0, val))
            elif var == "BP_sys": val = min(220.0, max(60.0, val))
            elif var == "BP_dia": val = min(120.0, max(40.0, val))
            else:                 val = max(0.0, val)
            row[var] = round(val, 1)
        chartevents[day] = row

    return MimicRecord(
        subject_id      = subject_id,
        hadm_id         = hadm_id,
        diagnosis_group = group,
        chartevents     = chartevents,
        severity_series = severity_series,
    )


def _severity_at_day(day: float, peak: float, onset: float, recovery: float) -> float:
    """Parametric severity curve matching DiseaseTrajectory's shape."""
    if day < onset:
        k = 6.0 / max(onset, 1.0)
        s = peak / (1.0 + math.exp(-k * (day - onset / 2.0)))
    else:
        decay_rate = -math.log(0.02) / max(recovery, 1.0)
        s = peak * math.exp(-decay_rate * (day - onset))
    return max(0.0, min(1.0, s))
