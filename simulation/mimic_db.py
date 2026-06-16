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
    # ── MIMIC cohort groups used by MIMICDataSource ───────────────────────────
    # Vital amplifier values calibrated to published literature patterns.
    "influenza": {
        "peak_severity_mean":  0.48, "peak_severity_sd": 0.10,
        "onset_days_mean":     2.0,  "onset_days_sd":    0.8,
        "recovery_days_mean":  7.0,  "recovery_days_sd": 2.0,
        "los_mean":            7.0,  "los_sd":           2.0,
        # Influenza: fever-dominant, achiness, mild tachycardia; SpO2 stays normal
        "amplifiers": {
            "HR": 1.4, "temp": 1.8, "WBC": 1.2, "CRP": 1.3,
            "fatigue": 1.3, "pain": 1.4,
            "SpO2": 0.3,  # SpO2 drop minimal for uncomplicated influenza
            "RR":   0.5,
        },
    },
    "bacterial_pneumonia": {
        "peak_severity_mean":  0.62, "peak_severity_sd": 0.10,
        "onset_days_mean":     3.0,  "onset_days_sd":    1.0,
        "recovery_days_mean": 10.0,  "recovery_days_sd": 2.5,
        "los_mean":           11.0,  "los_sd":           3.0,
        # Pneumonia: SpO2↓↓, RR↑↑, moderate fever; responds well to antibiotics
        "amplifiers": {
            "SpO2": 2.0, "RR": 2.2, "temp": 1.3, "WBC": 1.8,
            "CRP": 2.0, "HR": 1.2,
        },
    },
    "covid": {
        # COVID-19 (novel disease / Morven analog):
        # "Silent hypoxia" — SpO2 drops sharply, often before severe dyspnoea is felt.
        # Extreme fatigue disproportionate to other symptoms (long-COVID precursor).
        # Variable fever (often low-grade or absent early).
        # GI symptoms in a subset (nausea). Mild leukopenia (WBC normal or low).
        "peak_severity_mean":  0.65, "peak_severity_sd": 0.12,
        "onset_days_mean":     4.0,  "onset_days_sd":    1.5,
        "recovery_days_mean": 14.0,  "recovery_days_sd": 5.0,
        "los_mean":           14.0,  "los_sd":           5.0,
        "amplifiers": {
            "SpO2":   2.5,   # dominant marker — silent desaturation
            "fatigue": 3.0,  # extreme fatigue, disproportionate
            "HR":     1.3,   # secondary tachycardia
            "temp":   0.7,   # variable/low-grade fever
            "RR":     1.4,   # delayed respiratory compromise
            "WBC":    0.6,   # mild leukopenia (viral — unlike bacterial)
            "CRP":    1.8,   # elevated inflammatory marker
            "nausea": 1.6,   # GI involvement in subset
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

    _ICD_BY_GROUP: dict[str, str] = {
        "sepsis":        "A41.9",   # Sepsis, unspecified organism
        "pneumonia":     "J18.9",   # Pneumonia, unspecified organism
        "heart_failure": "I50.9",   # Heart failure, unspecified
    }

    def __init__(self, record: MimicRecord, severity_floor_weight: float = 0.20):
        self._record               = record
        self.severity_floor_weight = severity_floor_weight
        self._day          = 0
        self._severity     = 0.0
        self._prev_severity = 0.0
        self._treatment_reduction = 0.0   # accumulated from apply_treatment()
        self.disease_name = f"MIMIC-{record.diagnosis_group}"
        self.icd_code     = self._ICD_BY_GROUP.get(record.diagnosis_group, "B99.9")

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


# ─── Real MIMIC-IV CSV loader ─────────────────────────────────────────────────

class RealMimicDatabase(MimicDatabase):
    """
    Loads patient records from preprocessed MIMIC-IV CSV exports.

    Expects a single wide-format CSV (one row per patient-day) at csv_path.
    Generate it from raw MIMIC-IV with scripts/preprocess_mimic.py (see that
    file for the SQL extraction queries and itemid list).

    Expected CSV columns:
        subject_id, hadm_id, day, diagnosis_group,
        HR, temp, RR, SpO2, BP_sys, BP_dia, WBC, CRP, lactate,
        pain, fatigue, nausea, severity_proxy

    diagnosis_group must be one of: influenza, bacterial_pneumonia, covid,
    sepsis, pneumonia, heart_failure.

    MIMIC-IV raw tables needed (if building the preprocessed CSV from scratch):
        admissions.csv.gz         — subject_id, hadm_id
        diagnoses_icd.csv.gz      — hadm_id, icd_code, icd_version
        chartevents.csv.gz        — hadm_id, itemid, charttime, valuenum

    ICD-10 cohort filters applied by the preprocessing script:
        influenza:           J09.x, J10.x, J11.x
        bacterial_pneumonia: J13.x, J14.x, J15.x, J18.x
        covid:               U07.1

    MIMIC-IV chartevents itemids used:
        220045 → HR      (bpm)
        223761 → temp    (°F; converted to °C by (F-32)×5/9)
        220210 → RR      (breaths/min)
        220277 → SpO2    (%)
        220179 → BP_sys  (mmHg)
        220180 → BP_dia  (mmHg)

    MIMIC-III compatibility note: itemids differ (211→HR, 678→temp, etc.).
    The preprocessing script handles both versions.

    Note on COVID data: standard MIMIC-IV covers admissions through 2019.
    COVID patients (ICD U07.1) appear in MIMIC-IV v2.0+ (2020–2022 data).
    If COVID data is unavailable, MockMimicDatabase generates synthetic COVID
    vitals calibrated to published literature.
    """

    def __init__(self, csv_path: str | "Path", max_patients: Optional[int] = None):
        from pathlib import Path
        import csv as _csv

        self._records: dict[int, MimicRecord] = {}
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"RealMimicDatabase: preprocessed CSV not found at {path}\n"
                "Run scripts/preprocess_mimic.py against your MIMIC-IV download first."
            )

        # Group rows by (subject_id, hadm_id)
        _rows: dict[tuple, list[dict]] = {}
        opener = _csv.DictReader
        if str(path).endswith(".gz"):
            import gzip, io
            raw = gzip.open(path, "rt", encoding="utf-8")
        else:
            raw = open(path, newline="", encoding="utf-8")

        with raw as f:
            for row in opener(f):
                key = (int(row["subject_id"]), int(row["hadm_id"]))
                _rows.setdefault(key, []).append(row)
                if max_patients and len(_rows) >= max_patients:
                    break

        for (sid, hid), rows in _rows.items():
            rows.sort(key=lambda r: int(r["day"]))
            group = rows[0]["diagnosis_group"]
            los   = len(rows)

            severity_series: list[float] = []
            chartevents: dict[int, dict[str, float]] = {}
            for row in rows:
                day = int(row["day"])
                sev = float(row.get("severity_proxy", 0.0))
                severity_series.append(round(sev, 3))
                vitals: dict[str, float] = {}
                for var in _VARIABLES:
                    val_str = row.get(var, "")
                    if val_str and val_str not in ("", "NA", "nan"):
                        raw_val = float(val_str)
                        if var == "temp":
                            raw_val = (raw_val - 32.0) * 5.0 / 9.0  # °F → °C
                        vitals[var] = round(raw_val, 1)
                chartevents[day] = vitals

            self._records[sid] = MimicRecord(
                subject_id      = sid,
                hadm_id         = hid,
                diagnosis_group = group,
                chartevents     = chartevents,
                severity_series = severity_series,
            )

    def cohort_ids(self, diagnosis_group: Optional[str] = None) -> list[int]:
        if diagnosis_group is None:
            return list(self._records.keys())
        return [sid for sid, r in self._records.items()
                if r.diagnosis_group == diagnosis_group]

    def get_record(self, subject_id: int) -> MimicRecord:
        if subject_id not in self._records:
            raise KeyError(f"subject_id {subject_id} not in RealMimicDatabase")
        return self._records[subject_id]
