"""
Real-MIMIC-grounded text generation for the unknown-disease detection experiment.

Three mechanisms over the same underlying real MIMIC-IV-ED row (chiefcomplaint +
vitals), parallel to the baseline/template/ollama axis used for the fictional
disease sweep:
  mimic_raw()            -- the bare chiefcomplaint field, nothing else (terse,
                             often 1-3 words: "FEVER/TRAVEL", "Cough")
  mimic_phrase_library()  -- deterministic: chiefcomplaint + real vitals -> fuller
                             templated phrase, no LLM
  mimic_guided_ollama()    -- phi3:mini writes a naturalistic patient statement,
                             grounded in the same real chiefcomplaint + vitals
                             (not a canned per-disease phrase bank)
"""
from __future__ import annotations

import random

_VITAL_PHRASES = {
    "temp":  lambda v: f"temperature {v}°F",
    "HR":    lambda v: f"heart rate {v} bpm",
    "RR":    lambda v: f"breathing rate {v}/min",
    "SpO2":  lambda v: f"oxygen saturation {v}%",
    "pain":  lambda v: f"pain {v}/10",
}

_VITAL_COLS = ["temp", "HR", "RR", "SpO2", "pain"]


def _vitals_clauses(row: dict) -> list[str]:
    clauses = []
    for col in _VITAL_COLS:
        val = row.get(col)
        if val:
            clauses.append(_VITAL_PHRASES[col](val))
    return clauses


def mimic_raw(row: dict) -> str:
    """Bare chiefcomplaint field — the original (terse) mechanism."""
    return (row.get("chiefcomplaint") or "").strip()


def mimic_phrase_library(row: dict) -> str:
    """Deterministic enrichment: chiefcomplaint + real vitals -> fuller phrase."""
    cc = mimic_raw(row)
    clauses = _vitals_clauses(row)
    if not cc:
        return ""
    if not clauses:
        return cc
    return f"{cc}. Vitals on arrival: {', '.join(clauses)}."


def mimic_guided_ollama(row: dict, client, personality, rng: random.Random) -> str:
    """
    phi3:mini writes a naturalistic patient statement grounded in the real
    MIMIC chiefcomplaint + vitals. Falls back to mimic_phrase_library() if
    Ollama is unavailable or raises.
    """
    cc = mimic_raw(row)
    if not cc:
        return ""
    clauses = _vitals_clauses(row)
    context = f"You came to the emergency department for: {cc}."
    if clauses:
        context += f" Your vitals are: {'; '.join(clauses)}."
    try:
        text = client.complaint_opening(context, personality)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    return mimic_phrase_library(row)
