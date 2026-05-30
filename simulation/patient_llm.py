"""
Patient LLM — small frozen Ollama model for natural-language patient statements.

Generates opening complaints and follow-up symptom answers.  Never fine-tuned;
shared across all WorldEngine silos as a single instance created at startup.

Falls back to SymptomNarrator templates (via caller) if Ollama is unreachable.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error

OLLAMA_URL    = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "tinyllama"
TIMEOUT_SEC   = 20

_PERSONALITY_TONE = {
    "stoic":   "You tend to downplay how bad things feel and keep answers brief.",
    "neutral": "You describe things matter-of-factly.",
    "anxious": "You are visibly worried and tend to emphasise how bad things feel.",
}

# Severity → qualitative condition description passed to the LLM.
# Thresholds match _severity_perceived_band() in symptom_language.py.
_SEVERITY_DESC = [
    (0.35, "You feel slightly off — mild fatigue and a vague sense of not being right."),
    (0.65, "You feel quite unwell — noticeable fever or body aches and low energy."),
    (1.01, "You feel very ill — high fever, exhaustion, and significant discomfort."),
]

_SYSTEM_OPENING = """\
You are a patient visiting a doctor. Describe your symptoms in 1-2 sentences.
Rules:
- First person only. Plain, everyday language — no medical jargon or diagnoses.
- Do not mention specific numbers or test results.
- 1-2 sentences maximum.
- Tone: {tone}"""

_SYSTEM_FOLLOWUP = """\
You are a patient at a doctor's clinic. Answer the doctor's question briefly.
Rules:
- First person. 1-2 sentences maximum.
- Plain language — describe how things feel, not clinical values.
- Tone: {tone}"""


class PatientLLMUnavailableError(RuntimeError):
    pass


class PatientLLMClient:
    """
    Generates natural-language patient statements via a small frozen local LLM.

    Shared across all silos — instantiate once at startup, pass to every
    WorldEngine and OllamaDiagnosticClient.  The model is never updated.
    """

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
                 timeout: int = TIMEOUT_SEC):
        self.model   = model
        self.url     = url
        self.timeout = timeout

    # ── Public interface ──────────────────────────────────────────────────────

    def opening_statement(self, severity: float, days: int, personality) -> str:
        """
        Generate an opening patient complaint from health-state scalars.
        severity : float 0-1 (absolute illness level, not rate of change)
        days     : days since infection
        personality : Personality enum
        """
        tone     = _PERSONALITY_TONE.get(personality.value, _PERSONALITY_TONE["neutral"])
        cond     = next(d for thr, d in _SEVERITY_DESC if severity < thr)
        user_msg = (
            f"You have been unwell for {days} day(s). {cond}\n"
            f"Tell the doctor why you came today."
        )
        return self._call(_SYSTEM_OPENING.format(tone=tone), user_msg)

    def complaint_opening(self, prompt_context: str, personality) -> str:
        """
        Generate an opening statement for a non-infectious complaint visit.
        prompt_context describes the specific complaint (e.g. "You have lower back
        pain…"); the model generates a natural first-person clinic opening.
        """
        tone     = _PERSONALITY_TONE.get(personality.value, _PERSONALITY_TONE["neutral"])
        system   = _SYSTEM_OPENING.format(tone=tone)
        user_msg = f"{prompt_context}\nTell the doctor why you came today."
        return self._call(system, user_msg)

    def followup_answer(self, question: str, severity: float, personality,
                        case_table=None, day: int = 0) -> str:
        """
        Answer a doctor follow-up question in character.
        Uses case_table to give the patient qualitative awareness of relevant
        vitals (e.g. "breathing is notably worse") without leaking numbers —
        the doctor gets precise vitals through the vitals_request step.
        """
        tone     = _PERSONALITY_TONE.get(personality.value, _PERSONALITY_TONE["neutral"])
        cond     = next(d for thr, d in _SEVERITY_DESC if severity < thr)
        context  = self._relevant_context(question, case_table, day)
        user_msg = (
            f'The doctor asked: "{question}"\n'
            f"How you currently feel: {cond}"
        )
        if context:
            user_msg += f"\nAdditional context you're aware of: {context}"
        return self._call(_SYSTEM_FOLLOWUP.format(tone=tone), user_msg)

    def health_check(self) -> bool:
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model":    self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":   False,
            "options":  {"temperature": 0.75, "num_predict": 60},
        }).encode()
        req = urllib.request.Request(
            self.url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
                return body.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as exc:
            raise PatientLLMUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise PatientLLMUnavailableError(str(exc)) from exc

    def _relevant_context(self, question: str, case_table, day: int) -> str | None:
        """
        Map the doctor's question to a qualitative band description for the
        relevant vital.  Patients are aware of how things feel, not precise
        readings — precise numbers go to the doctor via vitals_request.
        """
        if case_table is None:
            return None
        q = question.lower()

        LOOKUP = [
            (("heart", "pulse", "palpitation"),  "HR",      "heart rate"),
            (("fever", "temperature", "hot"),     "temp",    "temperature"),
            (("breath", "breathing", "respir"),   "RR",      "breathing"),
            (("oxygen", "spo2", "saturation"),    "SpO2",    "oxygen level"),
            (("pain", "hurt", "ache"),            "pain",    "pain"),
            (("tired", "fatigue", "energy"),      "fatigue", "fatigue"),
            (("nausea", "sick", "vomit"),         "nausea",  "nausea"),
        ]
        for keywords, var, label in LOOKUP:
            if any(w in q for w in keywords):
                band = case_table.band(var, day)
                return {
                    "normal":     f"your {label} feels normal",
                    "borderline": f"your {label} seems slightly off",
                    "abnormal":   f"your {label} is noticeably abnormal",
                }.get(band)
        return None
