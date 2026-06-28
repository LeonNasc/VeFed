#!/usr/bin/env python3
"""
Preprocess MIMIC-IV-ED event log (MIMICEL, physionet mimicel-ed/2.1.0) into a
per-stay snapshot table for the unknown-disease detection experiment.

Unlike scripts/preprocess_mimic.py (full MIMIC-IV chartevents + admissions,
multi-day ICU trajectories), this dataset is a single flat event log per ED
visit: vitals come from "Vital sign check" rows, chief complaint from
"Triage in the ED" rows, and ICD diagnosis from "Discharge from the ED" rows,
all keyed by stay_id. Each stay becomes ONE snapshot row (day=0) — there is
no multi-day progression in ED data, so this feeds the static-mode (one
encounter per visit) experiment path, not the SIR multi-day path.

Cohorts are classified by keyword match on icd_title (robust across ICD-9/
ICD-10 vintages, verified against a full-dataset frequency breakdown).
Counts in this export (distinct ED stays, "Discharge from the ED" rows):
    pneumonia (any)      9,726       sepsis/septicemia   3,724
    uti                  7,146       influenza           2,187
    meningitis             181       malaria                21
No COVID (U07.x) or dengue cases exist in this MIMIC-IV-ED export.

Select which three become the experiment's diagnosis_group via --cohorts
(comma-separated, must be keys of KEYWORD_BUCKETS below).

Usage:
    python scripts/preprocess_mimicel.py \\
        --csv MIMIC/physionet.org/files/mimicel-ed/2.1.0/data/mimicel.csv \\
        --out MIMIC/mimic_unknown_disease_pool.csv \\
        --cohorts pneumonia,sepsis,uti

Output columns:
    stay_id, subject_id, hadm_id, diagnosis_group, icd_code,
    HR, temp, RR, SpO2, BP_sys, BP_dia, pain, acuity, chiefcomplaint
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

# Keyword buckets matched against icd_title (uppercased). First match wins.
KEYWORD_BUCKETS: dict[str, tuple[str, ...]] = {
    "pneumonia":    (r"\bPNEUMONIA\b", r"\bBRONCHOPNEUMONIA\b"),
    "sepsis":       (r"\bSEPSIS\b", r"SEPTICEMIA", r"\bSEPTIC\b"),
    "uti":          (r"URINARY TRACT INFECTION", r"\bUTI\b", r"CYSTITIS"),
    "influenza":    (r"\bINFLUENZA\b", r"\bFLU\b"),
    "meningitis":   (r"MENINGITIS",),
    "malaria":      (r"MALARIA",),
    "pharyngitis":  (r"PHARYNGITIS", r"TONSILLITIS", r"STREP THROAT"),
    "tuberculosis": (r"TUBERCUL",),
}

# MIMICEL column name -> our canonical vital name
VITAL_COLS = {
    "temperature": "temp",
    "heartrate":   "HR",
    "resprate":    "RR",
    "o2sat":       "SpO2",
    "sbp":         "BP_sys",
    "dbp":         "BP_dia",
    "pain":        "pain",
}


def _icd_group(title: str, active_buckets: list[str]) -> str | None:
    title = title.strip().upper()
    for bucket in active_buckets:
        if any(re.search(p, title) for p in KEYWORD_BUCKETS[bucket]):
            return bucket
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="MIMIC/mimic_unknown_disease_pool.csv")
    ap.add_argument("--cohorts", default="pneumonia,sepsis,uti",
                    help=f"Comma-separated, from: {list(KEYWORD_BUCKETS)}")
    args = ap.parse_args()

    active_buckets = [c.strip() for c in args.cohorts.split(",")]
    unknown = [c for c in active_buckets if c not in KEYWORD_BUCKETS]
    if unknown:
        raise ValueError(f"Unknown cohort(s) {unknown}; choose from {list(KEYWORD_BUCKETS)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diagnosis_group: dict[int, tuple[str, str]] = {}   # stay_id -> (group, icd_code)
    chiefcomplaint:  dict[int, str]              = {}
    acuity:          dict[int, str]              = {}
    subject_hadm:    dict[int, tuple[int, int]]  = {}
    vitals_sum:      dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    vitals_cnt:      dict[int, dict[str, int]]   = defaultdict(lambda: defaultdict(int))

    print(f"Streaming {args.csv} ...")
    n_rows = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}

        def g(row, col):
            i = idx[col]
            return row[i] if i < len(row) else ""

        for row in reader:
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                print(f"  {n_rows:,} rows...")

            sid_str = g(row, "stay_id")
            if not sid_str:
                continue
            sid = int(float(sid_str))

            activity = g(row, "activity")

            if sid not in subject_hadm:
                subj = g(row, "subject_id")
                hadm = g(row, "hadm_id")
                subject_hadm[sid] = (
                    int(float(subj)) if subj else 0,
                    int(float(hadm)) if hadm else 0,
                )

            if activity == "Discharge from the ED":
                code  = g(row, "icd_code")
                title = g(row, "icd_title")
                if code and title and sid not in diagnosis_group:
                    grp = _icd_group(title, active_buckets)
                    if grp:
                        diagnosis_group[sid] = (grp, code)

            elif activity == "Triage in the ED":
                cc = g(row, "chiefcomplaint")
                if cc and sid not in chiefcomplaint:
                    chiefcomplaint[sid] = cc
                ac = g(row, "acuity")
                if ac and sid not in acuity:
                    acuity[sid] = ac

            elif activity == "Vital sign check":
                for src_col, dst_col in VITAL_COLS.items():
                    val_str = g(row, src_col)
                    if val_str:
                        try:
                            vitals_sum[sid][dst_col] += float(val_str)
                            vitals_cnt[sid][dst_col] += 1
                        except ValueError:
                            pass

    print(f"Processed {n_rows:,} rows; {len(diagnosis_group)} stays matched a cohort")

    fieldnames = ["stay_id", "subject_id", "hadm_id", "diagnosis_group", "icd_code",
                  "HR", "temp", "RR", "SpO2", "BP_sys", "BP_dia", "pain", "acuity",
                  "chiefcomplaint"]
    n_written = 0
    group_counts: dict[str, int] = defaultdict(int)
    with open(out_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for sid, (grp, code) in diagnosis_group.items():
            cc = chiefcomplaint.get(sid, "")
            if not cc:
                continue   # no text -> useless for the text-comparison experiment
            subj, hadm = subject_hadm.get(sid, (0, 0))
            row = {
                "stay_id": sid, "subject_id": subj, "hadm_id": hadm,
                "diagnosis_group": grp, "icd_code": code,
                "acuity": acuity.get(sid, ""),
                "chiefcomplaint": cc,
            }
            for var in ["HR", "temp", "RR", "SpO2", "BP_sys", "BP_dia", "pain"]:
                cnt = vitals_cnt[sid].get(var, 0)
                row[var] = round(vitals_sum[sid][var] / cnt, 1) if cnt else ""
            writer.writerow(row)
            n_written += 1
            group_counts[grp] += 1

    print(f"Wrote {n_written} stays to {out_path}")
    print(f"Cohort sizes: {dict(group_counts)}")


if __name__ == "__main__":
    main()
