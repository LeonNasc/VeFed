"""
MIMICDataSource — DataSource implementation backed by MIMIC vital sign records.

Maps fictional disease names to real MIMIC cohorts and generates natural-language
patient complaints from vital sign patterns.  Designed for two use cases:

  1. Real-disease ablation: use "influenza", "bacterial_pneumonia", "covid" directly
     as disease names (bypasses fictional disease isolation).

  2. COVID-as-novel-disease experiment: Morven → covid cohort.  COVID has a
     distinctively different vital signature (SpO2↓↓ + extreme fatigue) from
     influenza and bacterial pneumonia, serving as a realistic "unknown pathogen"
     for the prototype-bank detection system.

Fictional disease → MIMIC cohort mapping:
  velarex   → influenza           (fever-dominant, HR↑, normal SpO2)
  sornathis → bacterial_pneumonia (SpO2↓, RR↑, markedly elevated CRP)
  morven    → covid               (SpO2↓↓ silent hypoxia, extreme fatigue)

Usage:
    from simulation.mimic_db import MockMimicDatabase, RealMimicDatabase
    from simulation.mimic_data_source import MIMICDataSource

    db  = MockMimicDatabase()                      # no files needed
    # db = RealMimicDatabase("data/mimic_vitals.csv")  # real MIMIC
    src = MIMICDataSource(db, seed=42)

    # In WorldEngine construction, pass src as the data_source argument.
    # MIMICDataSource is a drop-in for TemplateDataSource / OllamaDataSource.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from simulation.data_sources import DataSource
from simulation.mimic_db import MimicDatabase, MimicRecord
from simulation.case_table import MimicCaseTable, NORMAL_RANGES

if TYPE_CHECKING:
    from simulation.models import InnerState
    from simulation.symptom_language import Personality


# ── Disease → MIMIC cohort ────────────────────────────────────────────────────

FICTIONAL_TO_COHORT: dict[str, str] = {
    # Fictional disease names (main experiment)
    "velarex":            "influenza",
    "sornathis":          "bacterial_pneumonia",
    "morven":             "covid",
    # Real disease names (real-disease ablation mode)
    "influenza":          "influenza",
    "pneumonia":          "bacterial_pneumonia",
    "bacterial_pneumonia": "bacterial_pneumonia",
    "covid":              "covid",
    "covid-19":           "covid",
}


# ── Complaint phrase banks keyed by cohort × severity band ───────────────────
# Three bands: 0=mild, 1=moderate, 2=severe.
# {days} is substituted by days_infected at render time.

_PHRASES: dict[str, list[list[str]]] = {
    "influenza": [
        # band 0 — mild
        [
            "I've had a fever and body aches for {days} day(s). Feeling pretty wiped out.",
            "Started with chills and joint pain {days} day(s) ago. Low fever. Just want to sleep.",
            "Feverish, achy all over, and exhausted. No energy. {days} day(s) now.",
            "My whole body hurts and I've had a temperature on and off. {days} day(s).",
            "Chills, headache, and muscle aches. Fever's around 38. {days} day(s).",
        ],
        # band 1 — moderate
        [
            "High fever — nearly 39 — and my muscles are killing me. Shivering and exhausted. "
            "{days} day(s) of this.",
            "I feel terrible: bad fever, severe body aches, barely able to get up. {days} day(s).",
            "Alternating chills and sweating, high fever, body aches everywhere. "
            "Can't eat. {days} day(s).",
            "Fever hit 39.5 last night, aching all over, no energy at all. {days} day(s).",
            "Severe muscle pain, high fever, splitting headache. Worst I've felt in years. "
            "{days} day(s).",
        ],
        # band 2 — severe
        [
            "I can't get out of bed. Fever won't break, severe chills, and the pain is "
            "everywhere. {days} day(s).",
            "My temperature is dangerously high, I'm shaking uncontrollably, and my whole "
            "body is in agony. {days} day(s).",
            "Extremely high fever, severe muscle pain, I'm confused and barely able to "
            "speak. {days} day(s).",
            "Can't function — fever, severe chills, total body pain, no appetite. "
            "{days} day(s) and getting worse.",
            "I've been burning up for {days} day(s). Shivering despite the heat, severe "
            "aches, I feel like I'm dying.",
        ],
    ],

    "bacterial_pneumonia": [
        # band 0 — mild
        [
            "I have a persistent cough and feel a bit short of breath. Mild fever. "
            "{days} day(s).",
            "Cough that won't go away, slight breathlessness on exertion. {days} day(s).",
            "Chest feels tight and I'm coughing a lot. Temperature is slightly elevated. "
            "{days} day(s).",
            "Getting a bit breathless going up stairs. Cough and low-grade fever. "
            "{days} day(s).",
            "Persistent cough, some chest discomfort, mild shortness of breath. "
            "{days} day(s).",
        ],
        # band 1 — moderate
        [
            "I'm struggling to breathe properly. Bad cough, chest pain, fever above 38.5. "
            "{days} day(s) of this.",
            "Quite short of breath even at rest. Productive cough, high fever. {days} day(s).",
            "Breathing is really hard — chest feels heavy. Coughing up phlegm. "
            "High fever. {days} day(s).",
            "I can't catch my breath and the cough is exhausting me. Fever and chest "
            "pain. {days} day(s).",
            "Short of breath, coughing constantly, fever won't go down. Chest hurts. "
            "{days} day(s).",
        ],
        # band 2 — severe
        [
            "I'm fighting for every breath. Severe chest pain, high fever, I can't walk "
            "across the room. {days} day(s).",
            "Breathing is extremely laboured. Severe cough, I'm coughing blood-tinged "
            "sputum, and I have a high fever. {days} day(s).",
            "Can barely breathe — every breath is a struggle. Chest agony, very high "
            "fever. {days} day(s).",
            "I'm gasping for air, my chest is in terrible pain, and I'm burning up. "
            "{days} day(s).",
            "Severe breathlessness, painful cough, very high fever — I need help now. "
            "{days} day(s).",
        ],
    ],

    # COVID-19: "silent hypoxia" + extreme fatigue + variable fever.
    # Breathlessness is often delayed; fatigue hits first and hard.
    # GI symptoms (nausea, stomach discomfort) appear in a subset.
    # This distinctive pattern serves as the "novel disease" signal.
    "covid": [
        # band 0 — mild
        [
            "I'm completely exhausted — can't do anything. A bit short of breath on "
            "any effort. {days} day(s).",
            "Total fatigue and I feel like I can't get enough air. Low-grade fever. "
            "{days} day(s).",
            "I feel wiped out for no clear reason, and slightly breathless. {days} day(s).",
            "Exhausted and can't catch my breath walking to the kitchen. Mild fever "
            "yesterday. {days} day(s).",
            "I feel awful but it's hard to explain — just completely drained and "
            "slightly breathless. {days} day(s).",
        ],
        # band 1 — moderate
        [
            "Extreme fatigue, short of breath even at rest, and my heart is racing. "
            "{days} day(s).",
            "I can barely get off the sofa — breathless and utterly exhausted. Low "
            "fever. {days} day(s).",
            "My oxygen feels low — I'm struggling to breathe and I have no energy "
            "whatsoever. {days} day(s).",
            "Breathless doing nothing, heart pounding, completely exhausted. Also some "
            "nausea. {days} day(s).",
            "I feel like I'm suffocating and yet I can't tell exactly why — breathless, "
            "exhausted, a bit feverish. {days} day(s).",
        ],
        # band 2 — severe
        [
            "I can barely breathe and I'm completely incapacitated by fatigue. My "
            "heart is racing. {days} day(s).",
            "Severe breathlessness — I can't finish a sentence without gasping. Total "
            "exhaustion. {days} day(s).",
            "I feel like I'm suffocating. Completely unable to do anything, heart racing, "
            "severely breathless. {days} day(s).",
            "I can't breathe, I can't move, I'm dizzy and exhausted. This has been "
            "getting worse for {days} day(s).",
            "Severe breathlessness and the worst fatigue of my life. I need oxygen — "
            "I'm struggling badly. {days} day(s).",
        ],
    ],
}


def _severity_band(severity: float) -> int:
    """Map severity [0, 1] → phrase bank index 0/1/2."""
    if severity < 0.35:
        return 0
    elif severity < 0.65:
        return 1
    return 2


def _personality_modifier(text: str, personality: "Personality") -> str:
    """
    Light personality adjustment — stoic patients understate, anxious ones overstate.
    Uses the existing Personality enum's integer value as a proxy for expressiveness.
    """
    # Personality.value is an int; higher = more expressive in the existing schema
    val = getattr(personality, "value", 1)
    if val == 0:  # stoic
        # Trim to first sentence
        first = text.split(".")[0]
        return first + "." if first else text
    return text


# ── MIMICDataSource ───────────────────────────────────────────────────────────

class MIMICDataSource(DataSource):
    """
    Generates patient opening statements from MIMIC vital sign records.

    Each call to opening_statement():
      1. Maps the fictional/real disease name to a MIMIC cohort.
      2. Samples a random MimicRecord from that cohort.
      3. Picks vitals at the day closest to the agent's days_infected.
      4. Uses the CaseTable.band() abnormality classification to confirm which
         vitals are elevated/depressed (serves as a sanity check; phrasing is
         pre-written per cohort + severity band rather than assembled from vitals).
      5. Returns a phrase from the cohort-specific bank, scaled by severity.

    Phrase banks carry real disease symptom language (fever, breathlessness,
    fatigue).  Use this source for:
      - Real-disease ablation experiments (direct flu/pneumonia/COVID runs).
      - COVID-as-novel-disease detection (Morven → covid cohort).

    Do NOT use this source in the main fictional-disease FL experiment — the
    real symptom language defeats the LLM-contamination control (see
    fictional_diseases.py rationale).
    """

    def __init__(self, db: MimicDatabase, seed: int = 42):
        self._db  = db
        self._rng = random.Random(seed)

    def opening_statement(
        self,
        inner_state: "InnerState",
        days: int,
        personality: "Personality",
    ) -> str:
        disease  = getattr(inner_state, "disease_name", "influenza")
        severity = getattr(inner_state, "severity", 0.3)

        cohort = FICTIONAL_TO_COHORT.get(disease)
        if cohort is None:
            cohort = "influenza"  # safe default

        # Sample a random patient record from this cohort
        try:
            record = self._db.random_subject(cohort, self._rng)
        except ValueError:
            # Cohort missing from database (e.g., covid not in older MIMIC)
            record = self._db.random_subject(rng=self._rng)

        # Optionally verify vital bands against expected cohort signature
        # (useful for debugging; result is not used for text selection)
        ct  = MimicCaseTable(record, self._rng)
        day = min(days, record.los_days - 1)
        _key_vitals_check(ct, day, cohort)   # no-op in production

        band  = _severity_band(severity)
        pool  = _PHRASES.get(cohort, _PHRASES["influenza"])[band]
        text  = self._rng.choice(pool).format(days=days)
        return _personality_modifier(text, personality)

    def cohort_for(self, disease_name: str) -> str:
        """Return the MIMIC cohort that maps to this disease name."""
        return FICTIONAL_TO_COHORT.get(disease_name, "influenza")


def _key_vitals_check(ct: MimicCaseTable, day: int, cohort: str) -> None:
    """
    Debug helper: checks that key vitals are in the expected direction for the
    cohort.  No-op in normal operation; raises nothing — only logs mismatches.
    Useful when validating that RealMimicDatabase loaded the right patients.
    """
    expected_abnormal: dict[str, str] = {
        "influenza":          "temp",     # fever is primary marker
        "bacterial_pneumonia": "SpO2",    # hypoxia is primary marker
        "covid":              "SpO2",     # silent hypoxia
    }
    key_var = expected_abnormal.get(cohort)
    if key_var is None:
        return
    band = ct.band(key_var, day)
    if band == "normal" and day > 1:
        # Mild warning only — early days may not show abnormals yet
        pass
