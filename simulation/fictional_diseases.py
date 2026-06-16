"""
Fictional disease definitions for the clean-paper FL experiment.

Real-world analog mapping
─────────────────────────
  Velarex   ↔  Influenza          (fast-spreading; similar SIR dynamics)
  Sornathis ↔  Bacterial Pneumonia (slow-spreading; higher severity ceiling)

  This mapping is ONLY at the epidemic-dynamics level.  The symptom profiles
  are completely invented so the LLM has no pretraining prior for either name.
  See REAL_DISEASE_ANALOG below for the programmatic anchor used by reporting
  and result-interpretation code.  Full rationale: docs/fictional_diseases.md.

Why fictional diseases?
  phi3:mini has memorised influenza and pneumonia from pretraining. Even when
  silos see only one disease, the LLM can generate plausible cross-disease
  symptom text from memory — not from the SIR signal. This inflates local
  diag_acc and undermines the non-IID FL claim.

  By using invented names + novel symptom profiles defined entirely in the
  system prompt, the model has no prior to fall back on:
    - Patient text is generated from the provided definition only.
    - The doctor must reason from conversation evidence, not memorised patterns.
    - The non-IID isolation is real: a pure-Velarex silo genuinely cannot
      describe or diagnose Sornathis without FL weight sharing.

Three prompt variants (selected via OllamaDiagnosticClient):
  standard            disease_glossary set, explicit_exclusion=False
                      Novel names only; no mention of real diseases in prompts.
  explicit-exclusion  disease_glossary set, explicit_exclusion=True
                      Novel names + LLM told that "influenza"/"pneumonia" are
                      unknown terminology in this world, though the clinical
                      syndromes themselves still exist and can be recognised.
                      Ablation: tests naming novelty vs. explicit suppression.
  real-disease        disease_glossary=None
                      Standard influenza/pneumonia labels; LLM uses pretraining
                      knowledge freely.  Serves as the contaminated baseline.

Each disease dict contains:
  name                : label used in DIAGNOSTIC_LABELS and ground_truth
  display_name        : human-readable name shown in prompts
  real_disease_analog : the real-world disease this stands in for (reporting only)
  description         : one-sentence system-prompt definition
  symptoms            : bullet-point symptom profile injected into nurse/doctor prompts
  vitals              : which vitals are characteristically abnormal
  severity_profile    : typical severity distribution (mild/moderate/severe)
  phrase_banks        : SymptomNarrator phrase banks, 3 bands (mild/moderate/severe)

Integration points:
  1. progression.py   — VelarexProgression / SornathisProgression
  2. learner.py       — 'velarex', 'sornathis' in DISEASE_LABELS and DIAGNOSTIC_LABELS
  3. ollama_client.py — OllamaDiagnosticClient(disease_glossary=GLOSSARY)
  4. symptom_language.py — SymptomNarrator._phrase_banks update
"""

from __future__ import annotations

# ── Disease A: Velarex ───────────────────────────────────────────────────────
#
# A fast-spreading inflammatory condition affecting joints and peripheral
# circulation. Resembles influenza in epidemic dynamics but has a completely
# distinct symptom profile that phi3:mini has no prior knowledge of.

VELAREX: dict = {
    "name":                "velarex",
    "display_name":        "Velarex",
    "real_disease_analog": "influenza",   # SIR dynamics match; symptom profile invented
    # ── Probe-response banks for conversation state machine ──────────────────
    # Keys match simulation/conversation.py _PROBE_SEQUENCE.
    # Reveal disease-specific details that aren't always in the opener.
    "probe_responses": {
        "duration": {
            "stoic":   ["{days} days.", "About {days} days.", "Started {days} days ago."],
            "neutral": ["It's been {days} days now.", "About {days} days — it came on fairly suddenly.", "I'd say {days} days, give or take."],
            "anxious": ["Exactly {days} days — I've been counting. It started so abruptly it frightened me."],
        },
        "onset": {
            "stoic":   ["Came on suddenly, within a day.", "Quite abrupt onset."],
            "neutral": ["It came on quite suddenly — within hours, really. One morning I just woke up with it.", "Rapid onset, a day or two at most."],
            "anxious": ["Very suddenly — I went to bed fine and woke up with my joints on fire. The speed of it really scared me."],
        },
        "other_symptoms": {
            "stoic":   ["My fingers and toes look reddish. Light bothers me somewhat."],
            "neutral": ["Yes — my fingers and toes have gone quite red and mottled, almost blotchy. And I have a sensitivity to light, bright light is uncomfortable. Oh, and there's a persistent metallic or bitter taste in my mouth.", "My extremities look mottled — fingers and toes, reddish and blotchy — and I have photophobia. And this strange metallic taste."],
            "anxious": ["Yes, quite a few things. My fingers and toes look really alarming — red and blotchy, almost mottled-looking. Any bright light is very painful for me right now. And I have this horrible persistent metallic taste that won't go away no matter what I eat. I keep looking at my hands wondering what's happening to me."],
        },
        "severity_scale": {
            "stoic":   ["About a five, I'd say. I'm managing.", "Maybe a four or five."],
            "neutral": ["I'd say a six out of ten. It's affecting my ability to do things.", "Probably a six — it's quite disruptive."],
            "anxious": ["Eight, maybe nine. It's really bad and I'm very scared about the mottling on my extremities."],
        },
        "treatment": {
            "stoic":   ["Just paracetamol.", "Paracetamol, nothing else."],
            "neutral": ["I've taken paracetamol, which took the edge off the fever but hasn't done much for the joint pain.", "Paracetamol and ibuprofen. Slight improvement in joint stiffness but the other symptoms remain."],
            "anxious": ["I've tried paracetamol, ibuprofen, ice on my joints — nothing really touches the metallic taste or the light sensitivity. I've been trying everything."],
        },
        "elaboration": {
            "neutral": ["I should mention — the joint stiffness is worst in the mornings and seems to ease slightly by afternoon.", "One thing I noticed: the mottling gets worse when I'm cold."],
            "anxious": ["I've also been checking online and everything I read makes me more worried. The combination of symptoms is very unusual. I've never had anything like this before.", "I'm also worried because a neighbour seemed to have similar symptoms last week — could this be spreading?"],
        },
    },
    "description":  (
        "Velarex is an infectious disease characterised by sudden-onset joint "
        "warmth and stiffness, redness or mottling of the extremities (fingers, "
        "toes), photophobia (sensitivity to light), a persistent metallic taste, "
        "and a low-grade fever. It is NOT a respiratory illness — there is no "
        "cough, no chest pain, and breathing is typically unaffected."
    ),
    "symptoms": [
        "Sudden-onset warmth and aching in multiple joints (especially hands and knees)",
        "Visible redness or blotchy mottling of fingers and toes",
        "Photophobia — discomfort or pain when exposed to bright light",
        "Persistent metallic or bitter taste in the mouth",
        "Low-grade fever (typically 37.5–38.5 °C)",
        "General fatigue and malaise, but NOT shortness of breath",
    ],
    "vitals": {
        "HR":   "mildly to moderately elevated (inflammatory tachycardia)",
        "temp": "mildly elevated (37.5–38.5 °C)",
        "SpO2": "normal (≥ 96 %)",
        "RR":   "normal (12–18 breaths/min)",
        "CRP":  "moderately elevated (inflammatory marker)",
    },
    "severity_profile": "Predominantly mild-to-moderate. Severe cases rare (<10%). "
                        "Does NOT cause respiratory distress.",
    # SymptomNarrator phrase banks — 3 bands (0=mild, 1=moderate, 2=severe)
    "phrase_banks": [
        # band 0 — mild
        [
            "My joints feel warm and a bit stiff — fingers and knees mostly. "
            "Also a weird metallic taste in my mouth. {days} day(s) now.",
            "I've noticed my hands look a bit red and blotchy, and light feels "
            "uncomfortable to look at. Started {days} day(s) ago.",
            "Low fever, aching joints in my hands and feet, and a strange taste "
            "like metal. {days} day(s) of this.",
            "Fingers look mottled and feel warm. Some joint stiffness and mild "
            "light sensitivity. {days} day(s).",
            "Slight fever, stiff achy knees, and my fingers look oddly red. "
            "Metallic taste too. {days} day(s).",
        ],
        # band 1 — moderate
        [
            "My joints are really painful — hands, knees, shoulders all warm and "
            "swollen. Strong metallic taste and I can't stand bright light. "
            "{days} day(s) like this.",
            "Severe joint aching, my fingers and toes are bright red and mottled. "
            "Any light hurts my eyes. Feverish. {days} day(s).",
            "I can barely open my hands — the joints are so inflamed. Red blotchy "
            "extremities, bad metallic taste, light sensitivity. {days} day(s).",
            "Fever, very swollen and warm joints everywhere, mottled red hands. "
            "Looking at any light is painful. {days} day(s) and worsening.",
            "Joint pain all over, especially hands and feet. Bitter metallic taste "
            "won't go away. Eyes hurt in light. {days} day(s).",
        ],
        # band 2 — severe
        [
            "I can't move my joints at all — the swelling and heat are extreme. "
            "My extremities are deep red and mottled. Any light is agony. "
            "High fever. {days} day(s) of this.",
            "Completely incapacitated by joint pain. Fingers and toes look "
            "alarming — dark mottled red. Can't open my eyes in light. "
            "Burning fever. {days} day(s).",
            "Worst I've ever felt. All major joints seized up and hot. "
            "Extremities mottled. Light feels like needles. {days} day(s) and "
            "getting much worse.",
            "I haven't been able to walk for {days} day(s). Joints on fire, "
            "hands and feet deep red and mottled. Can't bear any light. "
            "High fever, barely eating.",
            "My joints are so inflamed I can't dress myself. Hands and feet "
            "are alarmingly discoloured. Severe light sensitivity. {days} day(s).",
        ],
    ],
}


# ── Disease B: Sornathis ─────────────────────────────────────────────────────
#
# A slow-spreading neuro-respiratory condition. Resembles pneumonia in epidemic
# dynamics but has a distinct symptom profile with no real-world analog that
# phi3:mini would recognise.

SORNATHIS: dict = {
    "name":                "sornathis",
    "display_name":        "Sornathis",
    "real_disease_analog": "bacterial_pneumonia",   # SIR dynamics match; symptom profile invented
    "probe_responses": {
        "duration": {
            "stoic":   ["{days} days.", "About {days} days, started gradually."],
            "neutral": ["It's been building for {days} days. It started quite gradually — not all at once.", "About {days} days, came on slowly over a few days."],
            "anxious": ["{days} days and it's been getting worse, not better. It started subtly but I've noticed it escalating day by day."],
        },
        "onset": {
            "stoic":   ["Gradual, over a few days.", "Slow onset."],
            "neutral": ["It came on gradually. First just a tingling in my hands, then the cough developed over a day or two.", "Slow onset — it built up progressively over several days."],
            "anxious": ["It crept up on me — I thought it was nothing at first, just some tingling in my fingers. Then the cough started and then my vision started going blurry. I got more and more worried as each new symptom appeared."],
        },
        "other_symptoms": {
            "stoic":   ["My vision goes blurry occasionally. Some earache. Sweating at night.", "Night sweats. Vision episodes. Earache."],
            "neutral": ["Yes — I get these brief episodes where my vision blurs or doubles, which is unsettling. I also have an earache or a pressure feeling in my ears. And I've been waking up completely drenched in sweat, even when the room isn't hot.", "Blurred vision episodes, earache, and quite significant night sweats."],
            "anxious": ["Several more things actually, and they're all worrying me. I keep getting these episodes where my vision suddenly goes blurry or I see double — it lasts a few seconds and then clears. My ears feel pressurised and aching. And every night I wake up absolutely soaked in sweat. I keep wondering if these things are connected."],
        },
        "severity_scale": {
            "stoic":   ["Five or six.", "About a five."],
            "neutral": ["I'd say a six — the breathing difficulty is the most limiting. The tingling and vision episodes are unsettling on top of that.", "Six out of ten. Multiple things going on at once."],
            "anxious": ["Eight out of ten easily. The vision episodes scare me the most — what if they get longer? And the breathlessness is getting worse."],
        },
        "treatment": {
            "stoic":   ["Nothing yet.", "Paracetamol for the earache. Nothing for the rest."],
            "neutral": ["Paracetamol for the earache and a bit of ibuprofen. I haven't tried anything for the tingling or the vision episodes.", "Painkillers for the earache. Nothing effective for the other symptoms."],
            "anxious": ["I've tried everything — paracetamol, steam inhalation for the cough, even eye drops for the vision. Nothing has helped and I'm getting increasingly worried. I tried to look up what could cause all these things together."],
        },
        "elaboration": {
            "neutral": ["I should mention the breathing gets noticeably worse when I'm walking upstairs — I get quite winded.", "The tingling has spread further up my arms since it started — it's not just my hands anymore."],
            "anxious": ["I've been keeping a symptom diary because I was so worried. The vision episodes are getting slightly longer each day. And yesterday I noticed the tingling is now up to my elbows. I'm very scared about what this could be.", "My partner has been really worried too — they said I looked unwell before I even noticed the symptoms myself."],
        },
    },
    "description":  (
        "Sornathis is an infectious disease characterised by gradual-onset "
        "tingling or numbness in the hands and feet, a persistent dry (non-"
        "productive) cough, intermittent episodes of blurred vision, earache, "
        "night sweats, and progressive fatigue. Breathing becomes laboured in "
        "moderate-to-severe cases. It is NOT associated with joint pain or "
        "photophobia."
    ),
    "symptoms": [
        "Tingling or numbness in hands and feet (peripheral paraesthesia)",
        "Persistent dry cough (non-productive — no mucus or phlegm)",
        "Intermittent episodes of blurred or doubled vision",
        "Earache or a feeling of pressure in the ears",
        "Night sweats without necessarily a daytime fever",
        "Progressive fatigue and difficulty breathing on exertion in moderate/severe cases",
    ],
    "vitals": {
        "SpO2": "below normal in moderate/severe cases (< 94 %)",
        "RR":   "elevated (> 20 breaths/min in moderate/severe cases)",
        "temp": "mildly elevated or normal (37.0–38.0 °C)",
        "HR":   "normal to mildly elevated",
        "CRP":  "markedly elevated",
    },
    "severity_profile": "Predominantly moderate-to-severe. Higher severity ceiling "
                        "than Velarex. Can cause significant respiratory compromise.",
    "phrase_banks": [
        # band 0 — mild
        [
            "My hands and feet have this odd tingling feeling, and I have a dry "
            "cough that won't go away. {days} day(s) now.",
            "I've been getting brief episodes where my vision goes a bit blurry, "
            "and my hands feel numb. Dry cough too. {days} day(s).",
            "Tingling in my fingers and toes, earache on and off, and waking up "
            "sweaty at night. {days} day(s) of this.",
            "Dry persistent cough, slight ear pressure, and strange tingling in "
            "my feet. {days} day(s).",
            "My hands feel numb or tingly, some earache, and a dry cough. "
            "Also sweaty at night. {days} day(s).",
        ],
        # band 1 — moderate
        [
            "The tingling in my hands and feet is constant now, my vision keeps "
            "going blurry, and I'm getting breathless just walking around. "
            "Dry cough. {days} day(s).",
            "I'm waking up drenched every night, my hands are numb, I have bad "
            "earache, and I feel quite short of breath. {days} day(s).",
            "Persistent dry cough, my vision keeps blurring, feet and hands feel "
            "like they're asleep. Struggling with stairs. {days} day(s).",
            "Earache, night sweats, numbness all the way up to my elbows, and "
            "I'm breathless doing very little. {days} day(s).",
            "I feel exhausted, tingling up my arms and legs, vision episodes, "
            "and a cough that won't quit. Getting harder to breathe. {days} day(s).",
        ],
        # band 2 — severe
        [
            "I'm struggling badly to breathe. Hands and feet completely numb, "
            "vision going blurry every few minutes, drenched in sweat. {days} "
            "day(s) and deteriorating fast.",
            "I can barely breathe — every breath takes effort. My whole body "
            "tingles, I can't see properly, and the earache is unbearable. "
            "{days} day(s).",
            "Severe breathlessness, can't feel my hands or feet properly, "
            "vision keeps blacking out at the edges. Night sweats are extreme. "
            "{days} day(s).",
            "I feel like I'm suffocating. Total numbness in extremities, vision "
            "episodes, earache, and I can't get out of bed. {days} day(s).",
            "Can't breathe, can't see straight, hands and feet completely numb. "
            "Soaked through every night. {days} day(s) of this and terrified.",
        ],
    ],
}


# ── Disease C: Morven Syndrome (novel / emerging disease) ────────────────────
#
# A GI-neurological disease deliberately designed to be ambiguous:
#   - Shares mild fever + joint ache with Velarex  → early cases may mislabel
#   - Shares neurological element with Sornathis, but confusion not tingling
#   - Distinct: abdominal cramping + cold sensitivity found in neither known disease
#
# As cases accumulate the GI + confusion signature becomes a separable cluster.
# This creates the intended "hard case" for embedding-space detection.

MORVEN: dict = {
    "name":                "morven",
    "display_name":        "Morven Syndrome",
    "real_disease_analog": "emerging",   # no real-world analog — genuinely novel
    "probe_responses": {
        "duration": {
            "stoic":   ["{days} days.", "About {days} days."],
            "neutral": ["About {days} days. The cramping came first, then the cold feeling, then the confusion episodes.", "{days} days — the symptoms have built up in stages."],
            "anxious": ["{days} days, though it feels much longer. I've been keeping track because the confusion episodes have been frightening me so much."],
        },
        "onset": {
            "stoic":   ["Fairly sudden.", "Came on over a day, maybe two."],
            "neutral": ["It came on fairly suddenly — the cramping and cold feeling appeared within a day of each other. The confusion episodes started a day or two later.", "Rapid onset — most of it appeared within the first day or two."],
            "anxious": ["It started quite suddenly, which was alarming. The stomach cramping hit first — I thought it was just indigestion — but then I felt this strange deep cold that I couldn't shake, and then I had my first confusion episode and I was really frightened."],
        },
        "other_symptoms": {
            "stoic":   ["My knees and hips ache. The confusion episodes are the main other thing — I lose track for a few minutes.", "Knee and hip ache. Confusion that clears after a minute or two."],
            "neutral": ["Yes — I have a mild aching in my knees and hips, which is separate from the cramping. And the confusion or mental fogginess episodes — I lose track of my thoughts for a few minutes then it clears. That's been the most unsettling part.", "Knee and hip ache. And these brief episodes where I feel confused or foggy — they last maybe two to five minutes then clear completely."],
            "anxious": ["Quite a few things. My knees and hips ache on and off. But the thing that terrifies me most is these episodes of confusion or mental fogginess — I literally cannot follow a thought for a few minutes. Once I forgot where I was in my own house. Then it cleared and I was back to normal. I don't know what to make of it. Is that neurological?"],
        },
        "severity_scale": {
            "stoic":   ["About a five.", "Five, maybe six."],
            "neutral": ["I'd say a six. The cramping waves are very uncomfortable and the confusion episodes are disconcerting, even if they're brief.", "Six out of ten. The combination of things is hard to deal with."],
            "anxious": ["Eight or nine. Mainly because of the confusion — the cramping I could perhaps tolerate, but losing mental clarity is terrifying. And the constant cold feeling is exhausting."],
        },
        "treatment": {
            "stoic":   ["Nothing for the confusion. Paracetamol for the aches.", "Nothing that's helped."],
            "neutral": ["Paracetamol for the joint ache. I tried a hot water bottle for the cold feeling but I genuinely cannot get warm. Nothing has touched the cramping or the confusion episodes.", "Painkillers and warmth for the cold, but nothing effective."],
            "anxious": ["I've tried everything — hot baths, extra blankets, paracetamol, antacids for the cramping. Nothing helps me get warm. And I obviously can't do anything about the confusion episodes except wait them out, which is very frightening."],
        },
        "elaboration": {
            "neutral": ["I should mention the confusion episodes seem to be getting slightly longer — they were a minute or two at first, now more like five minutes.", "I also noticed I feel lightheaded just before the confusion episodes hit — almost like a warning."],
            "anxious": ["The worst part is I'm not sure if it's safe for me to be alone. What if I have a long confusion episode while I'm on the stairs? I've been calling my family every few hours just to check in. This has been extremely stressful.", "I've been charting the cramping episodes — they come in waves roughly every four to six hours. I don't know if that pattern means anything medically."],
        },
    },
    "description":  (
        "Morven Syndrome is an infectious disease characterised by recurring "
        "waves of abdominal cramping and nausea, episodes of sudden confusion "
        "or mental fogginess (lasting minutes), unusual sensitivity to cold "
        "(patients feel deeply chilled even in warm environments), mild joint "
        "aching mainly in the knees and hips, and a low-grade fever. It is NOT "
        "associated with respiratory symptoms, visual changes, or extremity "
        "mottling."
    ),
    "symptoms": [
        "Recurring waves of abdominal cramping and nausea (not diarrhoea or vomiting)",
        "Brief episodes of sudden confusion or mental fogginess (minutes at a time)",
        "Unusual sensitivity to cold — feeling deeply chilled regardless of temperature",
        "Mild aching in knees and hips",
        "Low-grade fever (37.5–38.5 °C)",
    ],
    "vitals": {
        "HR":    "mildly elevated (85–100 bpm) due to low-grade fever",
        "temp":  "low-grade fever (37.5–38.5 °C)",
        "SpO2":  "normal (≥ 96 %) — no respiratory compromise",
        "RR":    "normal (12–18 breaths/min)",
        "WBC":   "mildly elevated — moderate inflammatory response",
        "CRP":   "moderately elevated",
        "BP_sys": "normal to low (patient may feel lightheaded during confusion episodes)",
    },
    "severity_profile": "Predominantly mild-to-moderate. Rarely severe. "
                        "Confusion episodes become longer and more frequent in moderate cases.",
    "phrase_banks": [
        # band 0 — mild
        [
            "I keep getting these stomach cramps in waves. Not diarrhoea, just "
            "cramping. And I feel oddly cold all the time. {days} day(s) of this.",
            "Waves of nausea and stomach cramps, and I've been feeling strangely "
            "confused for a few minutes at a time. Very cold too. {days} day(s).",
            "My stomach keeps cramping in waves, my knees ache a bit, and I feel "
            "chilled to the bone even indoors. {days} day(s).",
            "Brief episodes where my head goes foggy — a few minutes then it clears. "
            "Also stomach cramps and a cold feeling I can't shake. {days} day(s).",
            "Mild knee and hip ache, recurring stomach cramps, and I feel freezing "
            "cold despite the heating being on. Low fever too. {days} day(s).",
        ],
        # band 1 — moderate
        [
            "The stomach cramping is coming in strong waves now and I keep getting "
            "confused — I can't follow a simple thought for minutes at a time. "
            "Deeply cold. {days} day(s).",
            "Bad abdominal cramping in waves, longer confusion episodes, and I feel "
            "frozen inside. My knees and hips ache badly. {days} day(s).",
            "I can't warm up no matter what I do. Stomach cramps every few hours, "
            "and these spells where I lose track of where I am. {days} day(s).",
            "Recurring waves of cramping and nausea, longer and longer confusion "
            "episodes, and a deep cold feeling all through my body. {days} day(s).",
            "Abdominal spasms, joint ache, and I keep getting these frightening "
            "spells of confusion. I'm also freezing and can't get warm. {days} day(s).",
        ],
        # band 2 — severe
        [
            "The confusion is almost constant — I struggle to hold a conversation. "
            "Violent stomach cramps, can't feel warm, aching all over. {days} day(s).",
            "I'm confused most of the time, the cramping is severe and relentless, "
            "and I feel like I'm freezing from the inside out. {days} day(s).",
            "Near-constant disorientation, terrible abdominal cramping, and an "
            "overwhelming cold that won't lift. Can barely function. {days} day(s).",
            "Can't think straight at all. Severe cramping waves every hour, my "
            "whole body is cold, and my joints ache badly. {days} day(s).",
            "Confusion, severe cramping, freezing cold inside, severe joint pain — "
            "I can't care for myself. This has gone on {days} day(s).",
        ],
    ],
}


# ── Registry ──────────────────────────────────────────────────────────────────

FICTIONAL_DISEASES: dict[str, dict] = {
    "velarex":   VELAREX,
    "sornathis": SORNATHIS,
    "morven":    MORVEN,   # novel/emerging — used only in unknown disease experiment
}

# ── Real-world analog mapping ─────────────────────────────────────────────────
# Authoritative lookup for reporting and result-interpretation layers.
# Maps fictional name → real-world disease name used in the standard label space.
# The correspondence is at the epidemic-dynamics level only; symptom profiles differ.
REAL_DISEASE_ANALOG: dict[str, str] = {
    d["name"]: d["real_disease_analog"]
    for d in FICTIONAL_DISEASES.values()
}
# e.g. {"velarex": "influenza", "sornathis": "bacterial_pneumonia"}

# Reverse: real name → fictional name (for converting standard runs to fictional)
FICTIONAL_NAME_FOR: dict[str, str] = {v: k for k, v in REAL_DISEASE_ANALOG.items()}


# ── Prompt builders ───────────────────────────────────────────────────────────

def _disease_block(d: dict) -> str:
    """Format a disease definition as a bullet-point block for system prompts."""
    bullet_symptoms = "\n".join(f"  - {s}" for s in d["symptoms"])
    bullet_vitals   = "\n".join(f"  - {k}: {v}" for k, v in d["vitals"].items())
    return (
        f"{d['display_name'].upper()}\n"
        f"  Definition: {d['description']}\n"
        f"  Key symptoms:\n{bullet_symptoms}\n"
        f"  Characteristic vitals:\n{bullet_vitals}\n"
        f"  Severity: {d['severity_profile']}"
    )


_EXPLICIT_EXCLUSION_BLOCK = """\
IMPORTANT — NOMENCLATURE RESTRICTION:
In this world the disease names "influenza", "pneumonia", and "the flu" are NOT
known terminology.  The conditions themselves may still exist under different
names — use the condition definitions above to identify them.  If a symptom
pattern reminds you of influenza or pneumonia from your training, that clinical
reasoning is valid; you must simply label the condition using the names listed
above (e.g. Velarex or Sornathis), never using the words "influenza", "pneumonia",
or "flu" in your response."""


def make_nurse_system_prompt(diseases: list[str] | None = None,
                              explicit_exclusion: bool = False) -> str:
    """
    Nurse system prompt with fictional disease definitions injected.

    explicit_exclusion: when True, adds a hard disclaimer forbidding the model
        from applying any pretraining knowledge of influenza/pneumonia.  Use this
        as an ablation against the fictional-disease approach: same novel names,
        but the LLM is also explicitly told that real diseases do not exist.
    """
    active = [FICTIONAL_DISEASES[n] for n in (diseases or list(FICTIONAL_DISEASES))]
    blocks = "\n\n".join(_disease_block(d) for d in active)
    exclusion = f"\n{_EXPLICIT_EXCLUSION_BLOCK}\n" if explicit_exclusion else ""
    return f"""\
You are a triage nurse. Assess the severity of the patient's condition. Each reply is ONE JSON object — no other text.

You are working in a clinic where patients may have one of the following conditions.
Do NOT use any outside medical knowledge — base your assessment only on the definitions below.
{exclusion}
KNOWN CONDITIONS IN THIS CLINIC
────────────────────────────────
{blocks}
────────────────────────────────

First ask the patient one follow-up question about their symptoms, then request their vitals.

To ask a question:
{{"type": "question", "text": "your question here"}}

To request vitals (after the patient has answered):
{{"type": "vitals_request"}}

To give your triage assessment (only after seeing vitals):
{{"type": "triage", "severity": "discharge or mild or moderate or severe or critical", "notes": "one sentence"}}

Severity guide — discharge: no concerning signs; mild: rest at home; moderate: clinic treatment; severe: urgent care; critical: immediate hospitalisation.
Output only one JSON object per turn. Never assess before seeing vitals.
"""


def make_doctor_system_prompt(diseases: list[str] | None = None,
                               explicit_exclusion: bool = False) -> str:
    """
    Doctor system prompt listing only the fictional disease names as options.

    explicit_exclusion: when True, adds a hard disclaimer forbidding the model
        from applying any pretraining knowledge of influenza/pneumonia.
    """
    active      = [FICTIONAL_DISEASES[n] for n in (diseases or list(FICTIONAL_DISEASES))]
    blocks      = "\n\n".join(_disease_block(d) for d in active)
    name_list   = " or ".join(d["name"] for d in active) + " or non-infectious or unknown"
    exclusion   = f"\n{_EXPLICIT_EXCLUSION_BLOCK}\n" if explicit_exclusion else ""
    return f"""\
You are a diagnostic doctor. The triage nurse has already assessed severity. Your task is to identify the disease.
Each reply is ONE JSON object — no other text.

You are working in a clinic where patients may have one of the following conditions.
Do NOT use any outside medical knowledge — base your diagnosis only on the definitions below.
{exclusion}
KNOWN CONDITIONS IN THIS CLINIC
────────────────────────────────
{blocks}
────────────────────────────────

You may ask the patient one clarifying question if needed, then give your diagnosis.

To ask a question:
{{"type": "question", "text": "your question here"}}

To give your diagnosis:
{{"type": "diagnosis", "disease": "{name_list}", "notes": "one sentence", "triage_confirmed": true or false}}

Choose "unknown" when the symptom pattern does not match any known condition.
Choose "non-infectious" when the patient appears healthy or has a minor non-infectious complaint.
Output only one JSON object per turn.
"""


def make_patient_system_prompt(disease_name: str,
                                explicit_exclusion: bool = False) -> str:
    """
    Patient system prompt for fictional diseases.

    explicit_exclusion: when True, adds an explicit sentence forbidding the
        patient LLM from drawing on any knowledge of influenza/pneumonia.
    """
    d = FICTIONAL_DISEASES[disease_name]
    bullet_symptoms = "\n".join(f"  - {s}" for s in d["symptoms"])
    extra = (
        "\nThe words \"influenza\", \"pneumonia\", and \"the flu\" are not known "
        "in this world. Do not use those names. Simply describe how you feel "
        "based on the symptoms listed below — the doctor will identify the illness.\n"
        if explicit_exclusion else ""
    )
    return f"""\
You are a patient visiting a clinic. You have been diagnosed internally with {d['display_name']}.

{d['display_name']} is defined as follows — this is the ONLY source of truth for your symptoms.
Do NOT describe symptoms from real-world diseases. Only describe the symptoms listed below.
{extra}
YOUR SYMPTOMS ({d['display_name']}):
{bullet_symptoms}

When asked questions by the nurse or doctor, answer naturally in first person using only these symptoms.
Describe how severe they feel based on how unwell you feel overall.
Do not mention the disease name — just describe your experience.
"""


# ── DIAGNOSTIC_LABELS extension ───────────────────────────────────────────────
# These labels replace influenza/* and pneumonia/* in the classifier label space.

FICTIONAL_DIAGNOSTIC_LABELS: list[str] = [
    "velarex/mild",
    "velarex/moderate",
    "velarex/severe",
    "sornathis/mild",
    "sornathis/moderate",
    "sornathis/severe",
    "non-infectious",
]
