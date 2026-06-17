"""
DataSource — abstract interface for patient complaint generation.

Each concrete class encapsulates one observation modality:
  TemplateDataSource      — SymptomNarrator template phrases (fast, no LLM)
  PhraseLibraryDataSource — curated phrase banks from phrase_sampler.py
  OllamaDataSource        — live local LLM via PatientLLMClient
  MIMICDataSource         — MIMIC vital-sign patterns → complaint text

Two methods:
  opening_statement() — single-turn monologue (backward-compatible)
  full_conversation() — multi-turn patient–nurse exchange via state machine

WorldEngine calls full_conversation() when AgentConfig.multi_turn=True
(default True after the state-machine refactor). Training data then uses
the concatenated patient turns instead of the opening statement alone.
"""
from __future__ import annotations

import dataclasses
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.models import InnerState
    from simulation.symptom_language import Personality
    from simulation.conversation import ConversationRecord


# ── Abstract base ─────────────────────────────────────────────────────────────

class DataSource(ABC):
    """Generates patient complaint text for an infectious clinic visit."""

    @abstractmethod
    def opening_statement(
        self,
        inner_state: "InnerState",
        days: int,
        personality: "Personality",
    ) -> str: ...

    def full_conversation(
        self,
        inner_state: "InnerState",
        days: int,
        personality: "Personality",
    ) -> "ConversationRecord":
        """
        Multi-turn patient–nurse exchange via the conversation state machine.

        Default implementation: wraps opening_statement() in a ConversationRecord
        with no nurse probes.  Concrete subclasses override to add probes.
        """
        from simulation.conversation import ConversationRecord, ConversationTurn
        opener = self.opening_statement(inner_state, days, personality)
        rec = ConversationRecord(personality_key=personality.value.lower())
        rec.turns.append(ConversationTurn("patient", opener))
        return rec


# ── Concrete implementations ──────────────────────────────────────────────────

class TemplateDataSource(DataSource):
    """
    Uses SymptomNarrator.full_opening_statement — template-driven, no LLM.

    confusion_rate: fraction of visits where text is generated for a different
    disease while ground-truth label stays correct (irreducible noise floor).
    """

    _INFECTIOUS = ("influenza", "pneumonia")

    def __init__(self, seed: int = 42, confusion_rate: float = 0.0):
        from simulation.symptom_language import SymptomNarrator
        self._rng            = random.Random(seed)
        self._narrator       = SymptomNarrator(rng=self._rng)
        self._confusion_rate = confusion_rate

    def opening_statement(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> str:
        state = inner_state
        if self._confusion_rate > 0 and self._rng.random() < self._confusion_rate:
            candidates = [d for d in self._INFECTIOUS
                          if d != getattr(inner_state, "disease_name", "unknown")]
            if candidates:
                state = dataclasses.replace(
                    inner_state, disease_name=self._rng.choice(candidates))
        return self._narrator.full_opening_statement(state, days, personality)

    def full_conversation(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> "ConversationRecord":
        from simulation.conversation import simulate_conversation
        opener        = self.opening_statement(inner_state, days, personality)
        probe_banks   = _probe_banks_for(getattr(inner_state, "disease_name", ""))
        return simulate_conversation(
            opener, inner_state, days, personality,
            self._rng, probe_banks, self._narrator,
        )


class PhraseLibraryDataSource(DataSource):
    """
    Uses curated phrase banks (PHRASES dict in phrase_sampler.py).

    Higher lexical diversity than TemplateDataSource; same confusion_rate
    semantics — swaps disease key before phrase lookup.
    """

    _INFECTIOUS = ("influenza", "pneumonia")

    # Severity float → label used as key in PHRASES
    _BAND_LABELS = {0: "mild", 1: "moderate", 2: "severe"}

    def __init__(self, seed: int = 42, confusion_rate: float = 0.0):
        from simulation.phrase_sampler import PhraseLibrary
        self._lib            = PhraseLibrary(seed=seed, confusion_rate=0.0)
        self._rng            = self._lib._rng
        self._confusion_rate = confusion_rate

    def opening_statement(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> str:
        from simulation.symptom_language import _severity_perceived_band
        disease = getattr(inner_state, "disease_name", "influenza")
        if self._confusion_rate > 0 and self._rng.random() < self._confusion_rate:
            candidates = [d for d in self._INFECTIOUS if d != disease]
            if candidates:
                disease = self._rng.choice(candidates)
        band     = _severity_perceived_band(inner_state.severity, personality)
        severity = self._BAND_LABELS.get(band, "moderate")
        result   = self._lib.sample(disease, severity, personality.value, days=days)
        return result["text"]

    def full_conversation(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> "ConversationRecord":
        from simulation.conversation import simulate_conversation
        opener      = self.opening_statement(inner_state, days, personality)
        probe_banks = _probe_banks_for(getattr(inner_state, "disease_name", ""))
        return simulate_conversation(
            opener, inner_state, days, personality,
            self._rng, probe_banks,
        )


class OllamaDataSource(DataSource):
    """
    Generates opening statements via a local Ollama LLM (PatientLLMClient).

    The LLM receives severity + days only — no explicit disease name — so
    confusion_rate is a no-op here; the model's output is already ambiguous.
    Falls back to TemplateDataSource if the client raises.
    """

    def __init__(self, client=None, seed: int = 42):
        self._client   = client   # PatientLLMClient; injected at runtime
        self._fallback = TemplateDataSource(seed=seed)

    def opening_statement(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> str:
        if self._client is not None:
            try:
                return self._client.opening_statement(
                    inner_state.severity, days, personality)
            except Exception:
                pass
        return self._fallback.opening_statement(inner_state, days, personality)


class FictionalDataSource(DataSource):
    """
    DataSource backed by the fictional disease phrase banks (Velarex, Sornathis,
    Morven).  Used when WorldEngine runs with fictional progressions so that
    clinic text matches the fictional disease identity.

    Supports the full conversation state machine via simulate_conversation().
    """

    _BAND_LABELS = {0: "mild", 1: "moderate", 2: "severe"}

    def __init__(self, seed: int = 42):
        from simulation.phrase_sampler import PhraseLibrary  # noqa: F401 — just for import check
        from simulation.fictional_diseases import FICTIONAL_DISEASES  # noqa: F401
        import random
        self._rng   = random.Random(seed)

    def opening_statement(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> str:
        from simulation.fictional_diseases import FICTIONAL_DISEASES
        from simulation.symptom_language import _severity_perceived_band
        disease = getattr(inner_state, "disease_name", "velarex")
        info    = FICTIONAL_DISEASES.get(disease)
        if info is None:
            return f"I've been feeling unwell for {days} days."
        band    = _severity_perceived_band(inner_state.severity, personality)
        phrases = info["phrase_banks"][band]
        return self._rng.choice(phrases).format(days=days)

    def full_conversation(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> "ConversationRecord":
        from simulation.conversation import simulate_conversation
        opener      = self.opening_statement(inner_state, days, personality)
        probe_banks = _probe_banks_for(getattr(inner_state, "disease_name", ""))
        return simulate_conversation(
            opener, inner_state, days, personality,
            self._rng, probe_banks,
        )


class OllamaFictionalDataSource(DataSource):
    """
    Generates fictional-disease patient text via phi3:mini (PatientLLMClient).

    Builds a disease-specific symptom context from the fictional disease profile
    and passes it to PatientLLMClient.complaint_opening(), so the LLM produces
    naturally-worded clinic complaints grounded in each disease's unique signature:
      velarex   — high fever, reddish/mottled fingers+toes, light sensitivity
      sornathis — chest tightness, SpO2 drop, blurred vision, night sweats
      morven    — abdominal cramping, confusion episodes, cold sensitivity

    Falls back to FictionalDataSource if Ollama is unavailable.
    """

    _SYMPTOM_CONTEXT: dict[str, list[str]] = {
        "velarex": [
            "You have a high fever, aching muscles, and you have noticed your fingers "
            "and toes look reddish and mottled. Bright light bothers your eyes.",
            "You have a fever with severe muscle aches. Your fingers and toes are "
            "blotchy and red, and light sensitivity is making things uncomfortable.",
            "You are running a high fever, your whole body aches, your fingers and toes "
            "are alarmingly red and mottled, and even dim light feels painful.",
        ],
        "sornathis": [
            "You have mild chest tightness and some shortness of breath. "
            "Your vision has been slightly blurry and you've had some earache.",
            "You are struggling to breathe properly and your chest feels very tight. "
            "Your vision keeps going blurry and you have had drenching night sweats.",
            "You are severely short of breath — it is hard to complete sentences. "
            "Chest pressure is constant, vision is blurry, and you're drenched in sweat at night.",
        ],
        "morven": [
            "You have had recurring cramping pains in your abdomen, feel a bit confused "
            "at times, and are very sensitive to cold.",
            "Your abdomen keeps cramping painfully and you've had some worrying episodes "
            "of confusion. You feel cold all the time despite no low temperature.",
            "Severe abdominal cramps keep coming in waves and you've had multiple "
            "frightening episodes where you felt confused and disoriented. Cold sensitivity is extreme.",
        ],
    }

    def __init__(self, client=None, seed: int = 42):
        self._client   = client   # PatientLLMClient, injected at call time if None
        self._fallback = FictionalDataSource(seed=seed)

    def _get_client(self):
        if self._client is not None:
            return self._client
        from simulation.patient_llm import PatientLLMClient
        return PatientLLMClient()

    def opening_statement(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> str:
        from simulation.symptom_language import _severity_perceived_band
        disease  = getattr(inner_state, "disease_name", "velarex")
        contexts = self._SYMPTOM_CONTEXT.get(disease)
        if not contexts:
            return self._fallback.opening_statement(inner_state, days, personality)
        band    = _severity_perceived_band(inner_state.severity, personality)
        band    = min(band, len(contexts) - 1)
        context = f"You have been unwell for {days} day(s). {contexts[band]}"
        try:
            return self._get_client().complaint_opening(context, personality)
        except Exception:
            return self._fallback.opening_statement(inner_state, days, personality)

    def full_conversation(self, inner_state: "InnerState", days: int,
                          personality: "Personality") -> "ConversationRecord":
        from simulation.conversation import simulate_conversation
        opener      = self.opening_statement(inner_state, days, personality)
        probe_banks = _probe_banks_for(getattr(inner_state, "disease_name", ""))
        return simulate_conversation(
            opener, inner_state, days, personality,
            self._fallback._rng, probe_banks,
        )


def _probe_banks_for(disease_name: str) -> dict | None:
    """
    Return the probe-response banks for a disease, or None if unavailable.

    Checks fictional_diseases.py first (velarex / sornathis / morven), then
    returns None for real-disease names (template fallback handles those).
    """
    try:
        from simulation.fictional_diseases import FICTIONAL_DISEASES
        d = FICTIONAL_DISEASES.get(disease_name)
        if d:
            return d.get("probe_responses")
    except ImportError:
        pass
    return None


def make_mimic_data_source(
    csv_path: str | None = None,
    seed: int = 42,
    n_mock_patients: int = 100,
) -> "DataSource":
    """
    Factory for MIMICDataSource.

    csv_path=None  → MockMimicDatabase (no files needed; synthetic vitals)
    csv_path=<path> → RealMimicDatabase loaded from preprocessed CSV

    The preprocessed CSV is generated from raw MIMIC-IV by
    scripts/preprocess_mimic.py.  See RealMimicDatabase docstring for the
    expected schema and ICD cohort filters.
    """
    from simulation.mimic_db import MockMimicDatabase, RealMimicDatabase
    from simulation.mimic_data_source import MIMICDataSource

    if csv_path is None:
        db = MockMimicDatabase(n_patients_per_group=n_mock_patients // 6, seed=seed)
    else:
        db = RealMimicDatabase(csv_path)
    return MIMICDataSource(db, seed=seed)
