"""
Symptom language generation — expanded for case-table-driven reporting.

Two layers:
  1. Simple symptom-band phrases (original) — for general complaints
  2. Variable-specific phrases (new) — for vitals, labs, subjective measures

Patients report either "fuzzy" (subjective) or "measured" (precise) randomly,
with personality biasing the style and perception.
"""
from __future__ import annotations
import random
from enum import Enum


class Personality(Enum):
    STOIC   = "stoic"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"


# ─── Symptom banding (layer 1 — original, preserved) ──────────────────────────

def _band(sigma: float) -> int:
    """0 = low, 1 = medium, 2 = high"""
    if sigma < 0.30:
        return 0
    elif sigma < 0.65:
        return 1
    else:
        return 2


def _perceived_band(sigma: float, personality: Personality) -> int:
    raw = _band(sigma)
    if personality == Personality.STOIC:
        return max(0, raw - 1)
    elif personality == Personality.ANXIOUS:
        return min(2, raw + 1)
    return raw


# Layer 1 phrases — EXPANDED significantly
OPENING_PHRASES = [
    # band 0 — low
    [
        "I've been a bit under the weather for {days} day(s), nothing serious really.",
        "Honestly I'm probably fine, but I've had a mild headache and some tiredness for {days} day(s).",
        "I don't want to waste your time, I just feel slightly off. It's been {days} day(s).",
        "I've had a slight scratchy throat for {days} day(s) and maybe a touch of fatigue.",
        "Not sure it's worth mentioning but I've felt a little drained the past {days} day(s).",
        "Just a minor cold I think, started {days} day(s) ago.",
        "I'm mostly okay, just a bit of a sniffle for {days} day(s) now.",
        "Feeling a tiny bit run down, it's been going on {days} day(s).",
        "I have a light cough and feel mildly tired, {days} day(s) so far.",
        "It's probably nothing but I've been slightly achy for {days} day(s).",
    ],
    # band 1 — medium
    [
        "I've been feeling quite unwell for {days} day(s) — fever, body aches, and I'm exhausted.",
        "For the past {days} day(s) I've had chills, a persistent headache and I can barely get out of bed.",
        "I think I have a proper flu. Fever on and off, sore throat, {days} day(s) now.",
        "I feel rotten. Hot and cold sweats, aching all over, going on {days} day(s).",
        "It started as a sniffle but now after {days} day(s) I have a real fever and muscle pain.",
        "I've been sick for {days} day(s) with fever, chills, and a bad cough.",
        "This has been going on {days} day(s) — high temperature, body aches, no energy.",
        "I'm really unwell. Fever, headache, can't sleep properly. {days} day(s) like this.",
        "Feeling quite ill for {days} day(s) — sweating, shivering, very weak.",
        "It's been {days} day(s) of fever and fatigue, I'm struggling to function.",
    ],
    # band 2 — high
    [
        "I feel terrible. I've had a high fever, I can barely breathe properly, and it's been {days} day(s).",
        "I'm really worried — I've been bedridden for {days} day(s), fever won't break, chest feels tight.",
        "This is serious, I think. High fever, no appetite, very short of breath. {days} day(s) like this.",
        "I can't function. Severe headache, fever, I've been shaking. This has gone on {days} day(s).",
        "I feel like I'm getting worse, not better. Breathing is difficult, high fever, {days} day(s).",
        "I'm genuinely frightened. {days} day(s) of high fever, chest pain, extreme weakness.",
        "This is the worst I've ever felt. {days} day(s) now — can't breathe well, burning fever.",
        "I need help. {days} day(s) of severe symptoms, fever spiking, barely able to move.",
        "I'm very ill. High fever for {days} day(s), difficulty breathing, drenched in sweat.",
        "Something is very wrong. {days} day(s) — chest tightness, fever over 39, can't catch my breath.",
    ],
]


# ─── Multi-variable phrase bank (layer 2 — new) ──────────────────────────────

VARIABLE_PHRASES = {
    "HR": {
        "normal": {
            "fuzzy": [
                "My heart feels fine, beating normally.",
                "No issues with my heart, it seems steady.",
                "Pulse feels regular to me.",
            ],
            "measured": [
                "They checked my heart rate and it was {value}.",
                "The nurse said my pulse was {value}.",
                "It measured {value} when they took it.",
            ],
        },
        "borderline": {
            "fuzzy": [
                "My heart feels a bit fast sometimes.",
                "I think my pulse is slightly elevated.",
                "It feels like my heart is working harder than usual.",
                "I can feel my heartbeat more than normal.",
            ],
            "measured": [
                "My heart rate was {value}, they said it's a little high.",
                "It came back as {value}, which seemed elevated.",
                "They got {value} when they checked.",
            ],
        },
        "abnormal": {
            "fuzzy": [
                "My heart is racing, I can feel it pounding.",
                "My pulse is really fast, I'm very aware of it.",
                "It feels like my heart is going to burst out of my chest.",
                "I can hear my heartbeat in my ears, it's so fast.",
                "My heart won't slow down, it's been racing for hours.",
            ],
            "measured": [
                "The nurse measured it at {value}, she looked concerned.",
                "It was {value} — they said that's quite high.",
                "My heart rate hit {value}, I saw it on the monitor.",
                "They recorded {value}, much higher than normal.",
            ],
        },
    },
    "temp": {
        "normal": {
            "fuzzy": [
                "No fever, I feel a normal temperature.",
                "I don't think I have a fever.",
                "Temperature seems fine to me.",
            ],
            "measured": [
                "My temperature was {value} degrees.",
                "The thermometer read {value}.",
                "It came out as {value}°C.",
            ],
        },
        "borderline": {
            "fuzzy": [
                "I feel a bit warm, maybe slightly feverish.",
                "I might have a low-grade fever, hard to tell.",
                "I feel warmer than usual but not burning up.",
                "Slightly flushed, possibly a mild fever.",
            ],
            "measured": [
                "It was {value}, just above normal.",
                "The thermometer showed {value}°C.",
                "Temperature came back {value}, a bit elevated.",
            ],
        },
        "abnormal": {
            "fuzzy": [
                "I'm burning up, definitely have a high fever.",
                "I feel extremely hot, sweating heavily.",
                "My forehead is on fire, I'm drenched in sweat.",
                "I can't cool down, fever is really high.",
                "Shaking with chills despite feeling scorching hot.",
            ],
            "measured": [
                "My temperature was {value}, that's really high.",
                "The thermometer hit {value}°C.",
                "It peaked at {value}, I was shocked.",
                "Measured {value}°C — they said I need to bring it down urgently.",
            ],
        },
    },
    "RR": {
        "normal": {
            "fuzzy": [
                "Breathing normally, no issues there.",
                "My breathing is fine.",
                "No trouble with my breath.",
            ],
            "measured": [
                "Respiratory rate was {value}.",
                "They counted {value} breaths per minute.",
            ],
        },
        "borderline": {
            "fuzzy": [
                "I'm breathing a bit faster than usual.",
                "Slightly short of breath when I move around.",
                "I feel like I'm breathing harder.",
            ],
            "measured": [
                "They said my breathing rate was {value}, a bit quick.",
                "Respiratory rate came back {value}.",
            ],
        },
        "abnormal": {
            "fuzzy": [
                "I'm really struggling to breathe, it's labored.",
                "Can't catch my breath, feels very tight.",
                "Breathing is rapid and shallow, I can't get enough air.",
                "Every breath is effort, I feel like I'm suffocating.",
            ],
            "measured": [
                "My respiratory rate was {value}, way too fast.",
                "They recorded {value} breaths per minute — very high.",
            ],
        },
    },
    "SpO2": {
        "normal": {
            "fuzzy": [
                "Breathing feels fine, no issues with oxygen.",
                "My oxygen seems okay.",
            ],
            "measured": [
                "Oxygen saturation was {value}%.",
                "The pulse ox read {value}%.",
                "SpO2 measured {value}%.",
            ],
        },
        "borderline": {
            "fuzzy": [
                "I feel a bit lightheaded, maybe not getting enough oxygen.",
                "Slightly dizzy, could be oxygen.",
            ],
            "measured": [
                "Oxygen level was {value}%, they said it's borderline.",
                "SpO2 came back {value}%.",
            ],
        },
        "abnormal": {
            "fuzzy": [
                "I feel very lightheaded and faint.",
                "My lips are tingling, I think my oxygen is low.",
                "Dizzy and struggling, I don't think I'm getting enough air.",
            ],
            "measured": [
                "Oxygen saturation dropped to {value}%, they were worried.",
                "SpO2 was only {value}% — they put me on oxygen.",
            ],
        },
    },
    "pain": {
        "normal": {
            "fuzzy": [
                "No pain to speak of.",
                "I'm not in pain.",
                "Pain-wise I'm fine.",
            ],
            "measured": [
                "Pain level is about {value} out of 10.",
                "I'd say {value}/10 for pain.",
            ],
        },
        "borderline": {
            "fuzzy": [
                "A bit of discomfort, mild aching.",
                "Some pain but it's manageable.",
                "Low-level ache, nothing severe.",
            ],
            "measured": [
                "Pain is around {value} out of 10.",
                "I'd rate it {value}/10.",
            ],
        },
        "abnormal": {
            "fuzzy": [
                "I'm in a lot of pain, it's constant.",
                "Pain is severe, I can barely move.",
                "Hurts a great deal, really struggling.",
            ],
            "measured": [
                "Pain level is {value} out of 10, it's bad.",
                "I told them {value}/10, it's really severe.",
            ],
        },
    },
    "fatigue": {
        "normal": {
            "fuzzy": [
                "Energy levels are okay.",
                "I'm not particularly tired.",
                "Feeling alright, not fatigued.",
            ],
            "measured": [
                "Fatigue is maybe {value} out of 10.",
            ],
        },
        "borderline": {
            "fuzzy": [
                "I'm more tired than usual.",
                "Feeling a bit drained.",
                "Energy is low, I'm fatigued.",
            ],
            "measured": [
                "Fatigue level around {value}/10.",
            ],
        },
        "abnormal": {
            "fuzzy": [
                "I'm utterly exhausted, can barely stay awake.",
                "So tired I can't function, completely drained.",
                "Extreme fatigue, I've never felt this wiped out.",
            ],
            "measured": [
                "Fatigue is {value} out of 10, I'm running on empty.",
            ],
        },
    },
}

# ─── Temporal trend phrases (appended to opening statement) ──────────────────

TREND_ADDONS = {
    "worsening": {
        "STOIC":   ["though I notice it's not improving"],
        "NEUTRAL": [
            "and I'm getting worse, not better",
            "and it seems to be escalating",
            "and I feel more unwell than yesterday",
        ],
        "ANXIOUS": [
            "and I'm really worried because it's getting worse",
            "and it's escalating — I'm genuinely scared",
            "and I'm deteriorating, I can feel it",
        ],
    },
    "stable": {
        "STOIC":   ["not getting better, not getting worse"],
        "NEUTRAL": [
            "and there's no sign of improvement",
            "and it's been the same for a while now",
        ],
        "ANXIOUS": [
            "and nothing is improving — I'm very worried",
            "and I can't seem to shake it, which frightens me",
        ],
    },
    "improving": {
        "STOIC":   ["though it seems to be slowly resolving"],
        "NEUTRAL": [
            "though I think I'm slowly getting better",
            "and it feels like it might be turning a corner",
        ],
        "ANXIOUS": [
            "and I think I might be very slowly getting better, though I'm still worried",
        ],
    },
}

# ─── Specific vital mention phrases (appended after trend) ────────────────────

VITAL_ADDONS = {
    "temp": {
        "abnormal":   [
            "I have a very high fever.",
            "My temperature is really elevated.",
            "I'm burning up.",
        ],
        "borderline": [
            "I have a slight fever.",
            "I feel a bit warm, possibly mildly feverish.",
        ],
    },
    "RR": {
        "abnormal":   [
            "I'm struggling to breathe.",
            "Breathing is very laboured.",
            "I can't catch my breath.",
        ],
        "borderline": [
            "I'm a bit short of breath.",
            "Breathing feels harder than normal.",
        ],
    },
    "SpO2": {
        "abnormal":   [
            "I feel lightheaded and can't get enough air.",
            "I think my oxygen might be low.",
        ],
        "borderline": ["I feel slightly faint when I move around."],
    },
    "HR": {
        "abnormal":   [
            "My heart is racing.",
            "My pulse is pounding — I can feel it.",
            "My heart won't slow down.",
        ],
        "borderline": ["My heart feels a little fast."],
    },
}


BACKGROUND_PHRASES = {
    Personality.STOIC: [
        "I thought I'd stop in — not quite myself lately, probably nothing.",
        "Nothing dramatic, just wanted a routine check.",
        "Felt a bit run-down this week, might be stress.",
        "Mild fatigue, thought it worth a quick check.",
        "I'm mostly fine, just wanted to rule anything out.",
    ],
    Personality.NEUTRAL: [
        "I've had a mild headache and some tiredness, nothing serious.",
        "Just wanted to check in — I've felt slightly off for a few days.",
        "I haven't been sleeping great and feel a bit run-down.",
        "Nothing specific, just not feeling 100% lately.",
        "I've had a vague sense of fatigue, thought I'd get it checked out.",
        "I feel mostly fine, just a little less energetic than usual.",
    ],
    Personality.ANXIOUS: [
        "I'm probably fine, but I've been feeling off and wanted to be sure.",
        "I know it's likely nothing but I haven't felt right and it's worrying me.",
        "I've been a bit tired and anxious about it, wanted to get checked.",
        "I feel like something might be wrong — I can't put my finger on it.",
        "I couldn't sleep worrying about how I've been feeling lately.",
    ],
}


def background_opening(personality: Personality, rng: random.Random) -> str:
    """Opening complaint for a healthy/worried-well visitor with no active disease."""
    return rng.choice(BACKGROUND_PHRASES[personality])


PERSONALITY_PREFIX = {
    Personality.STOIC: [
        "I suppose ", "If I'm honest, ", "Well, ", "To be fair, ", "",
    ],
    Personality.NEUTRAL: [
        "", "", "Yes, ", "Actually, ", "",
    ],
    Personality.ANXIOUS: [
        "I'm really worried but ", "This is what concerns me — ",
        "I don't want to overreact but ", "Honestly it frightens me — ", "",
    ],
}


# ─── SymptomNarrator ──────────────────────────────────────────────────────────

def _severity_perceived_band(severity: float, personality: Personality) -> int:
    """
    Map absolute severity → language band, adjusted for personality.

    Used by full_opening_statement so plateau patients still sound ill
    (σ is near-zero at plateau but severity is high — language should reflect
    actual sickness, not just rate of change).
    """
    if severity < 0.35:
        base = 0
    elif severity < 0.65:
        base = 1
    else:
        base = 2

    if personality == Personality.STOIC:
        return max(0, base - 1)   # stoic downplays: high→medium, medium→low
    elif personality == Personality.ANXIOUS:
        return min(2, base + 1)   # anxious exaggerates: low→medium, medium→high
    return base


class SymptomNarrator:
    """Generates natural-language patient reports from case tables."""

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()

    def opening_statement(self, symptoms: float, days: int,
                          personality: Personality) -> str:
        band    = _perceived_band(symptoms, personality)
        phrases = OPENING_PHRASES[band]
        phrase  = self._rng.choice(phrases)
        return phrase.format(days=days)

    def full_opening_statement(self, inner_state, days: int,
                               personality: Personality) -> str:
        """
        Richer opening statement using InnerState.

        Composed of three optional parts:
          1. Base severity phrase — uses absolute severity (not σ) for banding
             so plateau patients sound appropriately ill.
          2. Temporal trend — appended ~65 % of the time (more for anxious,
             less for stoic).
          3. Most abnormal vital — appended ~50 % of the time if one exists.

        The LLM doctor still must work from this language; severity is not
        exposed numerically. The challenge comes from personality distortion,
        the plateau (severity high but trends "stable"), and vague vital phrasing.
        """
        band   = _severity_perceived_band(inner_state.severity, personality)
        base   = self._rng.choice(OPENING_PHRASES[band]).format(days=days)
        parts  = [base]

        # Trend addon
        p_trend = {
            Personality.STOIC:   0.35,
            Personality.NEUTRAL: 0.65,
            Personality.ANXIOUS: 0.85,
        }.get(personality, 0.65)
        if self._rng.random() < p_trend:
            pkey = personality.value.upper()
            addon_pool = TREND_ADDONS.get(inner_state.trend, {}).get(pkey, [])
            if addon_pool:
                addon = self._rng.choice(addon_pool)
                # Join naturally: first letter lowercase, prepend comma
                if addon and addon[0].islower():
                    parts[-1] = parts[-1].rstrip(".")
                    parts.append(", " + addon + ".")
                else:
                    parts.append(addon)

        # Vital addon
        if inner_state.top_vital and self._rng.random() < 0.50:
            var, _val, band_str = inner_state.top_vital
            phrases = VITAL_ADDONS.get(var, {}).get(band_str, [])
            if phrases:
                parts.append(" " + self._rng.choice(phrases))

        return "".join(parts)

    def report_variable(self, variable: str, value: float, band: str,
                        personality: Personality, force_measured: bool = False) -> str:
        if variable not in VARIABLE_PHRASES:
            return f"I'm not sure about {variable}."

        if force_measured:
            mode = "measured"
        else:
            p_measured = {
                Personality.STOIC:   0.30,
                Personality.NEUTRAL: 0.50,
                Personality.ANXIOUS: 0.60,
            }.get(personality, 0.50)
            mode = "measured" if self._rng.random() < p_measured else "fuzzy"

        band_shift = {
            Personality.STOIC:   -1,
            Personality.NEUTRAL:  0,
            Personality.ANXIOUS: +1,
        }.get(personality, 0)

        band_order = ["normal", "borderline", "abnormal"]
        band_idx   = band_order.index(band) + band_shift
        band_idx   = max(0, min(2, band_idx))
        perceived_band = band_order[band_idx]

        phrases = VARIABLE_PHRASES[variable][perceived_band][mode]
        phrase  = self._rng.choice(phrases)

        if mode == "measured" and "{value}" in phrase:
            if variable in ("HR", "RR"):
                val_str = f"{int(value)}"
            elif variable in ("temp",):
                val_str = f"{value:.1f}"
            elif variable in ("SpO2",):
                val_str = f"{int(value)}"
            elif variable in ("pain", "fatigue", "nausea"):
                val_str = f"{int(value)}"
            else:
                val_str = f"{value:.1f}"
            phrase = phrase.format(value=val_str)

        prefix_pool = PERSONALITY_PREFIX[personality]
        prefix = self._rng.choice(prefix_pool)
        if prefix and phrase and phrase[0].islower():
            return prefix + phrase
        return phrase

    def followup_answer(self, symptoms: float, personality: Personality,
                        question_text: str, case_table=None, day: int = 0) -> str:
        if case_table is not None:
            var, force_measured = self._infer_variable(question_text)
            if var and var in case_table.variables:
                value = case_table.get(var, day)
                band  = case_table.band(var, day)
                if value is not None:
                    return self.report_variable(var, value, band, personality,
                                                force_measured)

        return self._simple_followup_answer(symptoms, personality, question_text)

    def _infer_variable(self, question: str) -> tuple[str | None, bool]:
        q = question.lower()
        force = any(w in q for w in ("vital", "number", "measure", "read", "rate", "level"))

        if any(w in q for w in ("heart", "pulse", "bpm")):
            return "HR", force
        if any(w in q for w in ("fever", "temperature", "hot", "warm")):
            return "temp", force
        if any(w in q for w in ("breath", "breathing", "respiratory")):
            return "RR", force
        if any(w in q for w in ("oxygen", "spo2", "saturation")):
            return "SpO2", force
        if any(w in q for w in ("pain", "hurt", "ache")):
            return "pain", force
        if any(w in q for w in ("tired", "fatigue", "energy", "exhaust")):
            return "fatigue", force
        if any(w in q for w in ("nausea", "nauseous", "sick", "vomit")):
            return "nausea", force

        return None, False

    def _simple_followup_answer(self, symptoms: float, personality: Personality,
                                 question_text: str) -> str:
        band = _perceived_band(symptoms, personality)
        generic = [
            ["I feel mostly okay.", "Not too bad really.", "Just a bit off."],
            ["I'm quite unwell.", "Feeling pretty rough.", "Not great at all."],
            ["I feel terrible.", "Very ill indeed.", "This is really serious."],
        ]
        phrase = self._rng.choice(generic[band])
        prefix_pool = PERSONALITY_PREFIX[personality]
        prefix = self._rng.choice(prefix_pool)
        if prefix and phrase[0].islower():
            return prefix + phrase
        return phrase
