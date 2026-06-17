"""
Case summary compilation: structured clinical note from nurse-patient conversation.

After the multi-turn conversation the attending doctor compiles a brief structured
note. DistilBERT then classifies this summary instead of the raw patient turns.

Two implementations:
  TemplateCaseSummarizer  — deterministic extraction from InnerState + conversation
  OllamaCaseSummarizer    — phi3:mini writes the note from the dialogue transcript

Both expose the same interface:
    summarizer.compile(event: DiagnosticEvent) -> str

Usage in WorldEngine (set AgentConfig.case_summarizer to a CaseSummarizer instance):
    WorldEngine compiles the summary after conversation generation and stores it in
    DiagnosticEvent.case_summary; _normalize_item() in learner.py uses it as text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.models import DiagnosticEvent


# ── Severity helpers ──────────────────────────────────────────────────────────

def _sev_band(severity: float) -> str:
    if severity < 0.30:
        return "mild"
    if severity < 0.60:
        return "moderate"
    return "severe"


_VITAL_NAMES = {
    "temp":  "temperature",
    "HR":    "heart rate",
    "SpO2":  "oxygen saturation",
    "RR":    "respiratory rate",
    "BP_sys":"blood pressure",
}

_VITAL_UNITS = {
    "temp":   "°C",
    "HR":     " bpm",
    "SpO2":   "%",
    "RR":     " breaths/min",
    "BP_sys": " mmHg",
}

_FATIGUE_BAND = [(3, "none"), (5, "mild"), (7, "moderate"), (10, "severe")]
_PAIN_BAND    = [(3, "none"), (5, "mild"), (7, "moderate"), (10, "severe")]

def _band(value: float, bands: list) -> str:
    for threshold, label in bands:
        if value <= threshold:
            return label
    return bands[-1][1]


# ── Abstract base ─────────────────────────────────────────────────────────────

class CaseSummarizer(ABC):
    @abstractmethod
    def compile(self, event: "DiagnosticEvent") -> str: ...


# ── Template summarizer ───────────────────────────────────────────────────────

class TemplateCaseSummarizer(CaseSummarizer):
    """
    Deterministic structured note.  No LLM required.

    Format:
        Duration: N days | Severity: mild/moderate/severe | Trend: stable/worsening
        Chief complaint: <first patient turn, ≤100 chars>
        History: <remaining patient turns joined>
        Key finding: <most abnormal vital if present>
        Fatigue: X/10 | Pain: X/10
    """

    def compile(self, event: "DiagnosticEvent") -> str:
        patient_turns = [
            t["text"] for t in event.conversation if t.get("role") == "patient"
        ]
        chief = patient_turns[0][:100].rstrip() if patient_turns else "No complaint recorded."
        history = " ".join(patient_turns[1:]).strip() if len(patient_turns) > 1 else ""

        sev    = _sev_band(event.severity)
        days   = event.days_infected or 0
        trend  = getattr(event, "trend", None) or "stable"

        lines = [
            f"Duration: {days} day{'s' if days != 1 else ''} | Severity: {sev} | Trend: {trend}",
            f"Chief complaint: {chief}",
        ]
        if history:
            lines.append(f"History: {history[:200]}")

        ct  = getattr(event, "case_table", None)
        day = int(event.days_infected or 0)

        top_vital = getattr(event, "top_vital", None)
        fatigue   = getattr(event, "fatigue",   None)
        pain      = getattr(event, "pain",       None)

        if ct is not None:
            if top_vital is None:
                _PRIORITY = ["temp", "RR", "SpO2", "HR"]
                for band_target in ("abnormal", "borderline"):
                    for var in _PRIORITY:
                        b = ct.band(var, day)
                        if b == band_target:
                            top_vital = (var, ct.get(var, day), b)
                            break
                    if top_vital:
                        break
            if fatigue is None:
                fatigue = ct.get("fatigue", day)
            if pain is None:
                pain = ct.get("pain", day)

        if top_vital:
            var, val, band = top_vital
            name  = _VITAL_NAMES.get(var, var)
            unit  = _VITAL_UNITS.get(var, "")
            val_s = f"{val:.1f}" if isinstance(val, float) else str(val)
            lines.append(f"Key finding: {name} {val_s}{unit} ({band})")

        if fatigue is not None and pain is not None:
            lines.append(
                f"Fatigue: {_band(fatigue, _FATIGUE_BAND)} | Pain: {_band(pain, _PAIN_BAND)}"
            )

        return "\n".join(lines)


# ── Ollama summarizer ─────────────────────────────────────────────────────────

_SYSTEM_SUMMARIZE = (
    "You are an attending physician reviewing a nurse-patient triage conversation. "
    "Write a concise structured case summary in 3-4 sentences covering: "
    "chief complaint, key symptoms and duration, notable clinical features, "
    "and preliminary severity assessment. Be specific and clinical."
)


class OllamaCaseSummarizer(CaseSummarizer):
    """
    Uses phi3:mini to write the structured case summary from the conversation.

    Falls back to TemplateCaseSummarizer if Ollama is unavailable.
    """

    def __init__(self, client=None):
        from simulation.patient_llm import PatientLLMClient
        self._client   = client or PatientLLMClient()
        self._fallback = TemplateCaseSummarizer()

    def compile(self, event: "DiagnosticEvent") -> str:
        dialogue = self._format_dialogue(event.conversation)
        if not dialogue:
            return self._fallback.compile(event)
        try:
            return self._client._call(_SYSTEM_SUMMARIZE, dialogue)
        except Exception:
            return self._fallback.compile(event)

    @staticmethod
    def _format_dialogue(turns: list) -> str:
        lines = []
        for t in turns:
            role = t.get("role", "?").capitalize()
            text = t.get("text", "").strip()
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)
