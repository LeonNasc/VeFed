#!/usr/bin/env python3
"""
Preprocess raw MIMIC-IV CSV exports → wide-format vitals table for RealMimicDatabase.

Usage:
    python scripts/preprocess_mimic.py \\
        --mimic-dir /path/to/mimic-iv/hosp \\
        --chartevents /path/to/mimic-iv/icu/chartevents.csv.gz \\
        --out data/mimic_vitals.csv

Download MIMIC-IV from: https://physionet.org/content/mimiciv/
  Required files:
    hosp/admissions.csv.gz
    hosp/diagnoses_icd.csv.gz
    icu/chartevents.csv.gz          (large — ~32 GB uncompressed)

  Alternatively use MIMIC-IV-Extract (preprocessed hourly vitals in parquet)
  or BigQuery MIMIC-IV if you have access.

ICD cohort filters applied:
    influenza           J09.x, J10.x, J11.x    (ICD-10)  or  480-488 (ICD-9)
    bacterial_pneumonia J13.x, J14.x, J15.x, J18.x       or  481-483, 485-486
    covid               U07.1                              (ICD-10 only)

MIMIC-IV chartevents itemids used:
    220045 → HR     (bpm)
    223761 → temp   (°F → converted to °C here)
    220210 → RR     (breaths/min)
    220277 → SpO2   (%)
    220179 → BP_sys (mmHg)
    220180 → BP_dia (mmHg)

Output CSV schema (one row per subject-day):
    subject_id, hadm_id, day, diagnosis_group,
    HR, temp, RR, SpO2, BP_sys, BP_dia, WBC, CRP, lactate,
    pain, fatigue, nausea, severity_proxy

severity_proxy: normalised NEWS2 score (0–1, computed from vitals).
WBC / CRP / lactate: not in chartevents — set to NaN unless you also process
    labevents (itemids: 51301→WBC, 50889→CRP, 50813→lactate).

Note on COVID data:
    Standard MIMIC-IV v1.0 covers ICU admissions through 2019 and does NOT
    include COVID patients.  MIMIC-IV v2.0+ (released 2022) extends to 2022
    and includes COVID (U07.1) admissions.  Check your version with:
        head -2 hosp/admissions.csv.gz | zcat | grep -o '202[0-9]'
    If no COVID data is available, MockMimicDatabase generates synthetic
    COVID vitals calibrated to the published "silent hypoxia" pattern.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path


# ── ICD cohort filters ────────────────────────────────────────────────────────

ICD10_PREFIXES: dict[str, tuple[str, ...]] = {
    "influenza":           ("J09", "J10", "J11"),
    "bacterial_pneumonia": ("J13", "J14", "J15", "J18"),
    "covid":               ("U07",),
}

ICD9_PREFIXES: dict[str, tuple[str, ...]] = {
    "influenza":           ("480", "481", "487", "488"),
    "bacterial_pneumonia": ("481", "482", "485", "486"),
}


def _icd_group(code: str, version: str) -> str | None:
    code = code.strip().upper().replace(".", "")
    prefixes = ICD10_PREFIXES if version == "10" else ICD9_PREFIXES
    for group, pfxs in prefixes.items():
        if any(code.startswith(p.replace(".", "")) for p in pfxs):
            return group
    return None


# ── Chartevents itemids ───────────────────────────────────────────────────────

ITEMID_TO_VAR: dict[int, str] = {
    220045: "HR",
    223761: "temp",    # °F
    220210: "RR",
    220277: "SpO2",
    220179: "BP_sys",
    220180: "BP_dia",
}

NEEDED_ITEMIDS = set(ITEMID_TO_VAR.keys())


def _news2_severity(row: dict) -> float:
    """Normalised NEWS2 score (0–1) from vital dict."""
    score = 0
    rr = row.get("RR") or 16
    if rr <= 8:       score += 3
    elif rr <= 11:    score += 1
    elif rr <= 20:    score += 0
    elif rr <= 24:    score += 2
    else:             score += 3

    spo2 = row.get("SpO2") or 98
    if spo2 <= 91:    score += 3
    elif spo2 <= 93:  score += 2
    elif spo2 <= 95:  score += 1

    bp = row.get("BP_sys") or 120
    if bp <= 90:      score += 3
    elif bp <= 100:   score += 2
    elif bp <= 110:   score += 1
    elif bp >= 220:   score += 3

    hr = row.get("HR") or 75
    if hr <= 40:      score += 3
    elif hr <= 50:    score += 1
    elif hr <= 90:    score += 0
    elif hr <= 110:   score += 1
    elif hr <= 130:   score += 2
    else:             score += 3

    temp = row.get("temp") or 37.0
    if temp <= 35.0:   score += 3
    elif temp <= 36.0: score += 1
    elif temp <= 38.0: score += 0
    elif temp <= 39.0: score += 1
    else:              score += 2

    return round(score / 20.0, 3)   # max theoretical NEWS2 = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir",    required=True,
                    help="Path to MIMIC-IV hosp/ directory (admissions + diagnoses)")
    ap.add_argument("--chartevents",  required=True,
                    help="Path to icu/chartevents.csv.gz")
    ap.add_argument("--out",          default="data/mimic_vitals.csv")
    ap.add_argument("--max-patients", type=int, default=None,
                    help="Cap on total patients (for testing)")
    args = ap.parse_args()

    hosp_dir = Path(args.mimic_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: build hadm_id → (subject_id, diagnosis_group) ────────────────
    print("Loading admissions...")
    hadm_subject: dict[int, int] = {}
    adm_path = hosp_dir / "admissions.csv.gz"
    with gzip.open(adm_path, "rt") as f:
        for row in csv.DictReader(f):
            hadm_subject[int(row["hadm_id"])] = int(row["subject_id"])

    print("Loading diagnoses...")
    hadm_group: dict[int, str] = {}
    diag_path = hosp_dir / "diagnoses_icd.csv.gz"
    with gzip.open(diag_path, "rt") as f:
        for row in csv.DictReader(f):
            hid  = int(row["hadm_id"])
            if hid in hadm_group:
                continue   # already assigned
            grp = _icd_group(row["icd_code"], row.get("icd_version", "10"))
            if grp:
                hadm_group[hid] = grp

    target_hadms = set(hadm_group.keys())
    print(f"Cohort sizes: { {g: sum(1 for v in hadm_group.values() if v==g) for g in ICD10_PREFIXES} }")

    # ── Step 2: aggregate chartevents by hadm_id × day ───────────────────────
    print("Processing chartevents (this may take several minutes)...")
    # day_vitals[hadm_id][day][var] → list of values (averaged later)
    day_vitals: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    from datetime import datetime
    hadm_admit: dict[int, datetime] = {}

    # Re-read admissions for admit times
    with gzip.open(adm_path, "rt") as f:
        for row in csv.DictReader(f):
            hid = int(row["hadm_id"])
            if hid in target_hadms:
                try:
                    hadm_admit[hid] = datetime.fromisoformat(
                        row["admittime"].replace(" ", "T"))
                except (ValueError, KeyError):
                    pass

    ce_path = Path(args.chartevents)
    opener  = gzip.open if str(ce_path).endswith(".gz") else open
    n_processed = 0
    with opener(ce_path, "rt") as f:
        for row in csv.DictReader(f):
            item = int(row.get("itemid", 0))
            if item not in NEEDED_ITEMIDS:
                continue
            hid = int(row.get("hadm_id", 0))
            if hid not in target_hadms:
                continue
            try:
                val = float(row["valuenum"])
            except (ValueError, TypeError):
                continue

            var = ITEMID_TO_VAR[item]
            if var == "temp":
                val = (val - 32.0) * 5.0 / 9.0   # °F → °C

            admit = hadm_admit.get(hid)
            if admit is None:
                continue
            try:
                ct = datetime.fromisoformat(
                    row["charttime"].replace(" ", "T"))
                day = (ct - admit).days
            except (ValueError, KeyError):
                continue
            if day < 0:
                continue

            day_vitals[hid][day][var].append(val)
        n_processed += 1

    print(f"Processed {n_processed} chartevent rows across {len(day_vitals)} admissions")

    # ── Step 3: write output CSV ──────────────────────────────────────────────
    VARS = ["HR", "temp", "RR", "SpO2", "BP_sys", "BP_dia",
            "WBC", "CRP", "lactate", "pain", "fatigue", "nausea"]
    n_written = 0
    with open(out_path, "w", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["subject_id", "hadm_id", "day", "diagnosis_group"]
            + VARS + ["severity_proxy"],
        )
        writer.writeheader()

        for hid, days in day_vitals.items():
            if hid not in hadm_group:
                continue
            sid   = hadm_subject.get(hid)
            if sid is None:
                continue
            group = hadm_group[hid]

            for day in sorted(days.keys()):
                vitals = days[day]
                row: dict = {
                    "subject_id":       sid,
                    "hadm_id":          hid,
                    "day":              day,
                    "diagnosis_group":  group,
                }
                agg: dict = {}
                for var in VARS:
                    vals = vitals.get(var, [])
                    row[var] = round(sum(vals) / len(vals), 2) if vals else ""
                    if vals:
                        agg[var] = sum(vals) / len(vals)
                row["severity_proxy"] = _news2_severity(agg)
                writer.writerow(row)
                n_written += 1

            if args.max_patients and n_written >= args.max_patients * 14:
                break

    print(f"Wrote {n_written} patient-day rows to {out_path}")


if __name__ == "__main__":
    main()
