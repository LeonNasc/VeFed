"""
Synthetic case table generation — bridge to MIMIC.

Each infected agent gets a CaseTable: a time series of vitals, labs, and
clinical measurements correlated with their DiseaseTrajectory severity curve.

Structure mirrors MIMIC-III/IV CHARTEVENTS schema:
  day → {variable_name: value}

Variables tracked:
  Vitals:  HR (heart rate), temp (°C), RR (respiratory rate), SpO2 (%), BP_sys, BP_dia
  Labs:    WBC (×10⁹/L), CRP (mg/L), lactate (mmol/L)
  Subjective: pain (0-10), fatigue (0-10), nausea (0-10)

Values are sampled with realistic correlations:
  - HR ↑ as severity ↑ and temp ↑
  - RR ↑ as severity ↑
  - SpO2 ↓ as severity ↑
  - temp peaks ~1 day after severity peak (lag)
  - WBC, CRP rise with infection, decline on recovery
  - Subjective scores track symptom σ derivative

Extension point for MIMIC:
  Replace `_generate_synthetic()` with `_load_from_mimic(subject_id)`.
  The rest of the code (banding, phrase selection) stays identical.
"""
from __future__ import annotations
import random
import math


# ─── Normal ranges (for banding) ──────────────────────────────────────────────

NORMAL_RANGES = {
    # Vitals
    "HR":      (60,  100),   # bpm
    "temp":    (36.5, 37.5), # °C
    "RR":      (12,  20),    # breaths/min
    "SpO2":    (95,  100),   # %
    "BP_sys":  (90,  140),   # mmHg
    "BP_dia":  (60,  90),    # mmHg
    # Labs
    "WBC":     (4.0, 11.0),  # ×10⁹/L
    "CRP":     (0,   10),    # mg/L
    "lactate": (0.5, 2.0),   # mmol/L
    # Subjective
    "pain":    (0, 2),       # 0-10 scale
    "fatigue": (0, 3),
    "nausea":  (0, 2),
}

# Severity-to-value slopes — how much each variable changes per 0.1 severity
SEVERITY_SLOPES = {
    "HR":      +8,     # +8 bpm per 0.1 severity
    "temp":    +0.35,  # +0.35°C per 0.1 severity
    "RR":      +3,
    "SpO2":    -1.5,   # oxygen drops
    "BP_sys":  -4,     # slight drop (vasodilation)
    "BP_dia":  -2,
    "WBC":     +2.0,   # immune response
    "CRP":     +15,
    "lactate": +0.4,
    "pain":    +1.2,
    "fatigue": +1.5,
    "nausea":  +0.8,
}


# ─── Case table ───────────────────────────────────────────────────────────────

class CaseTable:
    """
    Time series of clinical measurements for one patient.
    Generated synthetically from a DiseaseTrajectory, or later loaded from MIMIC.
    """

    def __init__(self, trajectory, rng: random.Random, max_days: int = 30):
        self._trajectory = trajectory
        self._rng        = rng
        self._data       = self._generate_synthetic(max_days)

    def get(self, variable: str, day: int) -> float | None:
        """Retrieve value for a given variable on a given day."""
        if variable not in self._data:
            return None
        series = self._data[variable]
        if day < 0 or day >= len(series):
            return None
        return series[day]

    def band(self, variable: str, day: int) -> str:
        """
        Classify value as 'normal' | 'borderline' | 'abnormal'.
        Used for phrase selection.
        """
        val = self.get(variable, day)
        if val is None:
            return "normal"
        lo, hi = NORMAL_RANGES.get(variable, (0, 100))
        margin = (hi - lo) * 0.15   # ±15% is borderline

        if val < lo - margin or val > hi + margin:
            return "abnormal"
        elif val < lo or val > hi:
            return "borderline"
        else:
            return "normal"

    def _generate_synthetic(self, max_days: int) -> dict[str, list[float]]:
        """
        Generate correlated time series from the trajectory.
        Each variable = baseline + slope × severity(t) + noise + lag.
        """
        data = {var: [] for var in NORMAL_RANGES.keys()}

        for day in range(max_days):
            # Evaluate trajectory at this day
            # (We don't actually step it — just peek at internal state)
            # For generation, we'll simulate the curve ourselves
            sev = self._severity_at_day(day)

            for var in NORMAL_RANGES.keys():
                baseline_lo, baseline_hi = NORMAL_RANGES[var]
                baseline = (baseline_lo + baseline_hi) / 2.0
                slope    = SEVERITY_SLOPES.get(var, 0)

                # Temp lags severity by ~1 day
                lag = 1 if var == "temp" else 0
                sev_lagged = self._severity_at_day(max(0, day - lag))

                value = baseline + slope * sev_lagged * 10  # *10 because slope is per 0.1 sev
                noise = self._rng.gauss(0, abs(slope) * 0.3)  # proportional noise
                value += noise
                value = max(0, value)  # clamp negatives

                # SpO2 and BP can't exceed physiological limits
                if var == "SpO2":
                    value = min(100, value)
                elif var == "BP_sys":
                    value = max(60, min(220, value))
                elif var == "BP_dia":
                    value = max(40, min(120, value))

                data[var].append(round(value, 1))

        return data

    def _severity_at_day(self, day: int) -> float:
        """
        Reconstruct severity at a given day from the trajectory params.
        This mirrors DiseaseTrajectory.step() logic without actually stepping.
        """
        t = self._trajectory
        if day < t.rise_days:
            # Logistic rise
            k = 6.0 / max(t.rise_days, 1.0)
            t_half = t.rise_days / 2.0
            s = t.peak_severity / (1.0 + math.exp(-k * (day - t_half)))
        else:
            # Exponential decay
            days_past_peak = day - t.rise_days
            s = t.peak_severity * ((1.0 - t.decay_rate) ** days_past_peak)
        return max(0.0, min(1.0, s))

    @property
    def variables(self) -> list[str]:
        return list(self._data.keys())


# ─── MIMIC loader stub ────────────────────────────────────────────────────────

class MimicCaseTable(CaseTable):
    """
    Case table backed by a MimicRecord (mock or real MIMIC data).

    Populates self._data directly from chartevents — same get/band interface
    as the synthetic CaseTable. Swap in MockMimicDatabase for development or
    RealMimicDatabase for production; this class is unaffected.

    Usage:
        from simulation.mimic_db import MockMimicDatabase
        db  = MockMimicDatabase()
        rec = db.random_subject("sepsis")
        ct  = MimicCaseTable(rec, rng)

    When a variable is missing for a given day, the midpoint of its normal
    range is used as a safe fallback.
    """

    def __init__(self, record, rng: random.Random):
        # Bypass CaseTable.__init__ — populate _data from the record directly
        self._trajectory = None   # unused for MIMIC records
        self._rng        = rng
        self._data       = self._load_from_record(record)

    def _load_from_record(self, record) -> dict[str, list[float]]:
        data: dict[str, list[float]] = {var: [] for var in NORMAL_RANGES}
        for day in range(record.los_days):
            vitals = record.vitals_at_day(day)
            for var in NORMAL_RANGES:
                lo, hi  = NORMAL_RANGES[var]
                fallback = (lo + hi) / 2.0
                data[var].append(vitals.get(var, fallback))
        return data

    # _severity_at_day is never called on MimicCaseTable — overridden for safety
    def _severity_at_day(self, day: int) -> float:
        return 0.0
