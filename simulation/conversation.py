"""
Conversation state machine for patient–nurse interactions.

Replaces the single `opening_statement()` call with a multi-turn exchange
that varies in length and style based on personality, severity, and disease.

Key behavioural differences from the single-turn baseline:
  STOIC    → terse opener + 1 probe answered briefly; nurse moves on
  NEUTRAL  → opener + 2 probes; patient answers fully
  ANXIOUS  → verbose opener + 1-2 probes; patient over-answers and may
              volunteer extra information or ask the nurse a question back

This introduces realistic noise:
  - Variable embedding length (stoic = short, anxious = long)
  - Disease-specific symptom mentions spread across turns (not all in opener)
  - Personality distortion of severity perception persists across turns
  - Some signal only appears in the "other symptoms" probe answer

State machine flow
──────────────────
  [P: opener]
  [N: probe_1 — duration]
  [P: answer_1]
  (if neutral or anxious and severity ≤ 0.7):
    [N: probe_2 — other_symptoms]
    [P: answer_2]
  (if anxious and random < 0.40):
    [P: elaboration — volunteer extra detail unprompted]

Conversation.turns returns the full exchange as a list of (role, text) pairs.
The concatenated patient text is what gets embedded by DistilBERT.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.symptom_language import Personality


# ── Nurse question templates per probe type ───────────────────────────────────

NURSE_QUESTIONS: dict[str, list[str]] = {
    "duration": [
        "How long have you been feeling this way?",
        "When did your symptoms first start?",
        "How many days has this been going on for?",
        "Could you tell me how long this has been happening?",
    ],
    "onset": [
        "Did this come on suddenly or more gradually?",
        "Was there anything that seemed to trigger it?",
        "Can you describe how it started?",
    ],
    "other_symptoms": [
        "Are there any other symptoms I should know about?",
        "Is there anything else you've been experiencing alongside these symptoms?",
        "Have you noticed anything else unusual?",
        "Anything else at all that you've observed?",
    ],
    "severity_scale": [
        "On a scale of one to ten, how would you rate how you're feeling overall?",
        "How much has this been affecting your daily activities?",
        "Would you say your symptoms are getting worse, staying the same, or improving?",
    ],
    "treatment": [
        "Have you taken anything for this, such as paracetamol or ibuprofen?",
        "Have you tried any treatment or remedies so far?",
        "Are you on any regular medication that I should know about?",
    ],
    "location": [
        "Where exactly do you feel this most — can you point to it?",
        "Is the discomfort in one place or more general?",
        "Can you describe where you're feeling it?",
    ],
}

# Standard probe sequence: probe_1 is always duration; probe_2 is always
# other_symptoms.  Additional probes draw from the remainder.
_PROBE_SEQUENCE = ["duration", "other_symptoms", "onset", "treatment"]


# ── Elaboration bridges (anxious patients volunteer extra info) ───────────────

_ELABORATION_BRIDGES: dict[str, list[str]] = {
    "stoic":   [],   # stoic never volunteers unprompted
    "neutral": [
        "Actually, there's one more thing I should mention.",
        "Oh — I almost forgot.",
        "I should add,",
    ],
    "anxious": [
        "I'm also really worried because",
        "Another thing that's been frightening me is",
        "I didn't mention it before but it's been on my mind —",
        "Oh, and I nearly forgot — this is also concerning me:",
        "One more thing — I'm not sure if it's relevant but",
    ],
}

# Stoic patient closure phrases (used after probe_1 to signal they are done)
_STOIC_CLOSURES = [
    "That's really all I can tell you.",
    "I think that covers it.",
    "I've told you what I know.",
    "That's about the size of it.",
]

# Anxious patient question-back phrases (interjected at end of probe answer)
_ANXIOUS_QUESTION_BACKS = [
    " Is that a bad sign?",
    " Should I be worried about that?",
    " Do you think it could be something serious?",
    " That is normal, isn't it?",
    " I keep reading things online and worrying.",
]


# ── ConversationTurn and ConversationRecord ───────────────────────────────────

@dataclass
class ConversationTurn:
    role: str    # "patient" | "nurse"
    text: str


@dataclass
class ConversationRecord:
    """Result of simulate_conversation(). Consumed by DataSource.full_conversation()."""
    turns:           list[ConversationTurn] = field(default_factory=list)
    probe_count:     int = 0
    personality_key: str = "neutral"

    def patient_turns(self) -> list[str]:
        return [t.text for t in self.turns if t.role == "patient"]

    def patient_text(self, separator: str = " ") -> str:
        """Concatenated patient speech — what gets embedded by DistilBERT."""
        return separator.join(self.patient_turns())

    def as_dialogue(self) -> str:
        """Full formatted exchange for logging / research notes inspection."""
        lines = []
        for t in self.turns:
            role = "Patient" if t.role == "patient" else "Nurse"
            lines.append(f"{role}: {t.text}")
        return "\n".join(lines)


# ── Core simulation function ──────────────────────────────────────────────────

def simulate_conversation(
    opener: str,
    inner_state,
    days: int,
    personality: "Personality",
    rng: random.Random,
    probe_responses: dict | None = None,
    narrator=None,
) -> ConversationRecord:
    """
    Run the patient–nurse state machine for one clinic visit.

    Parameters
    ----------
    opener          : The patient's opening complaint (already generated).
    inner_state     : InnerState snapshot (severity, disease_name, trend, …).
    days            : Days infected.
    personality     : Personality enum value.
    rng             : Shared RNG for reproducibility.
    probe_responses : Per-probe-type phrase banks for the current disease.
                      Keys: "duration" | "onset" | "other_symptoms" | …
                      Values: {"stoic": [str, …], "neutral": [str, …], "anxious": [str, …]}
    narrator        : Optional SymptomNarrator instance for fallback followup_answer().

    Returns
    -------
    ConversationRecord with all turns.
    """
    from simulation.symptom_language import Personality

    pkey = personality.value.lower()  # "stoic" | "neutral" | "anxious"
    severity = getattr(inner_state, "severity", 0.4)
    record = ConversationRecord(personality_key=pkey)

    # ── Opener ────────────────────────────────────────────────────────────────
    record.turns.append(ConversationTurn("patient", opener))

    # ── Determine probe count ─────────────────────────────────────────────────
    # Base count by personality
    base_probes = {"stoic": 1, "neutral": 2, "anxious": 1}.get(pkey, 2)
    # Severe presentations → nurse skips extra probing, moves to vitals
    if severity >= 0.75:
        base_probes = max(1, base_probes - 1)
    probe_count = base_probes
    record.probe_count = probe_count

    # ── Run probes ────────────────────────────────────────────────────────────
    for i, probe_type in enumerate(_PROBE_SEQUENCE[:probe_count]):
        # Nurse question
        q_pool = NURSE_QUESTIONS.get(probe_type, NURSE_QUESTIONS["other_symptoms"])
        nurse_q = rng.choice(q_pool)
        record.turns.append(ConversationTurn("nurse", nurse_q))

        # Patient answer
        answer = _patient_answer(
            probe_type, pkey, inner_state, days, severity, rng,
            probe_responses, narrator,
        )

        # Anxious: small chance to ask a question back (only on first probe)
        if pkey == "anxious" and i == 0 and rng.random() < 0.30:
            answer = answer.rstrip(".,") + rng.choice(_ANXIOUS_QUESTION_BACKS)

        record.turns.append(ConversationTurn("patient", answer))

        # Stoic: after probe_1, they may close the conversation
        if pkey == "stoic" and i == 0 and rng.random() < 0.50:
            closure = rng.choice(_STOIC_CLOSURES)
            record.turns[-1] = ConversationTurn(
                "patient", record.turns[-1].text + " " + closure
            )

    # ── Anxious elaboration (unprompted, after all probes) ────────────────────
    if pkey in ("neutral", "anxious"):
        p_elaborate = {"neutral": 0.20, "anxious": 0.55}.get(pkey, 0.0)
        if rng.random() < p_elaborate and probe_responses:
            extra = _pick_phrase(
                probe_responses.get("elaboration", probe_responses.get("other_symptoms", {})),
                pkey, rng,
            )
            if extra:
                bridge = rng.choice(_ELABORATION_BRIDGES.get(pkey, [""]) or [""])
                full = (bridge + " " + extra).strip() if bridge else extra
                record.turns.append(ConversationTurn("patient", full))

    return record


# ── Answer generation ─────────────────────────────────────────────────────────

def _patient_answer(
    probe_type: str,
    pkey: str,
    inner_state,
    days: int,
    severity: float,
    rng: random.Random,
    probe_responses: dict | None,
    narrator,
) -> str:
    """Generate a patient answer for a given probe type."""

    # 1. Try disease-specific probe-response bank
    if probe_responses:
        phrase = _pick_phrase(probe_responses.get(probe_type, {}), pkey, rng)
        if phrase:
            return phrase.format(days=days)

    # 2. Fallback: SymptomNarrator.followup_answer() (variable-specific)
    if narrator is not None:
        from simulation.symptom_language import Personality
        p_enum = next(
            (p for p in Personality if p.value == pkey),
            Personality.NEUTRAL,
        )
        # Build a synthetic question text so _infer_variable can map it
        synthetic_q = {
            "duration":       "how long have you had this?",
            "onset":          "did this come on suddenly?",
            "other_symptoms": "any other symptoms?",
            "severity_scale": "how severe is the pain?",
            "treatment":      "any medication?",
            "location":       "where does it hurt?",
        }.get(probe_type, "how are you feeling?")
        return narrator.followup_answer(
            inner_state.symptoms if hasattr(inner_state, "symptoms") else severity,
            p_enum,
            synthetic_q,
        )

    # 3. Generic fallback
    band = 0 if severity < 0.35 else (1 if severity < 0.65 else 2)
    generic = [
        {"stoic": "Not sure. Just as I described.", "neutral": "A few days, I think.", "anxious": "Exactly as I said — I've been very worried about it."},
        {"stoic": "I'd say moderate.", "neutral": "Quite unwell, I'd say.", "anxious": "Pretty bad — I'd say seven or eight out of ten, honestly."},
        {"stoic": "Quite bad.", "neutral": "It's been really affecting me.", "anxious": "The worst I've ever felt — I'm very scared."},
    ]
    return generic[band].get(pkey, "I'm not sure.")


def _pick_phrase(bank: dict, pkey: str, rng: random.Random) -> str | None:
    """Pick a random phrase from bank[pkey], falling back to 'neutral'."""
    pool = bank.get(pkey) or bank.get("neutral") or []
    if not pool:
        return None
    return rng.choice(pool)
