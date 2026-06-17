"""
Background-visit complaint types for non-epidemic clinic visitors.

Each complaint carries:
  - An ICD-10 code and default management tier (ground-truth label)
  - A prompt_context injected into the patient LLM's system prompt so it
    generates a naturalistically typed opening statement for that condition
  - A `disease` ground-truth tag — "non-infectious" for genuine worried-well
    visits (back pain, anxiety, routine follow-up — no underlying illness),
    or a real (mild, self-limiting) disease name for OTHER_DISEASE_COMPLAINTS
    below. Background visitors are drawn from the union of both pools, so the
    "background" class isn't one trivially-typed bucket — the classifier has
    to tell apart several real low-acuity illnesses, not just spot a template.

They are shared across all silos (IID baseline) and give the classifier real
negative examples for the escalation task.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NonInfectiousComplaint:
    name:           str
    icd_code:       str
    management:     str   # "home rest" | "treat"
    prompt_context: str   # injected into patient LLM; describes the complaint
    disease:        str = "non-infectious"   # gt_disease tag; overridden for real illnesses


NON_INFECTIOUS_COMPLAINTS: tuple[NonInfectiousComplaint, ...] = (
    NonInfectiousComplaint(
        name       = "Low back pain",
        icd_code   = "M54.5",
        management = "home rest",
        prompt_context = (
            "You have lower back pain that started 3–5 days ago. "
            "The pain is dull and worsens when sitting for long periods or bending. "
            "You have no fever, no leg numbness, and are otherwise well."
        ),
    ),
    NonInfectiousComplaint(
        name       = "Headache",
        icd_code   = "R51",
        management = "home rest",
        prompt_context = (
            "You have had a tension headache for 1–2 days. "
            "It is a dull pressure across your forehead. "
            "You feel otherwise healthy — no fever, no neck stiffness, no vomiting."
        ),
    ),
    NonInfectiousComplaint(
        name       = "Generalised anxiety",
        icd_code   = "F41.1",
        management = "treat",
        prompt_context = (
            "You have been feeling anxious and stressed for the past week. "
            "You are sleeping poorly, feel tense, and cannot switch off your thoughts. "
            "You came to talk to a doctor about how you have been feeling mentally."
        ),
    ),
    NonInfectiousComplaint(
        name       = "Hypertension follow-up",
        icd_code   = "Z87.39",
        management = "treat",
        prompt_context = (
            "You are here for a routine blood pressure check. "
            "You have no new symptoms and feel well. "
            "Your doctor asked you to come back to make sure your blood pressure is controlled."
        ),
    ),
    NonInfectiousComplaint(
        name       = "Fatigue",
        icd_code   = "R53.83",
        management = "home rest",
        prompt_context = (
            "You have been feeling unusually tired for about a week. "
            "You sleep enough but wake up feeling unrefreshed. "
            "No fever, no specific pain — just persistent low energy and difficulty concentrating."
        ),
    ),
)


# Real (mild, self-limiting, non-epidemic) illnesses that show up in the
# background clinic population alongside the worried-well. Unlike the SIR
# diseases above, these do not spread between agents — they are sampled
# independently per visit — but they carry genuine `disease` ground-truth
# tags so the diagnostic classifier must distinguish them from each other,
# from the epidemic diseases, AND from genuine non-infectious complaints.
OTHER_DISEASE_COMPLAINTS: tuple[NonInfectiousComplaint, ...] = (
    NonInfectiousComplaint(
        name       = "Common cold",
        icd_code   = "J00",
        management = "home rest",
        disease    = "common_cold",
        prompt_context = (
            "You have had a runny nose, mild sore throat, and occasional sneezing "
            "for about two days. You feel a bit run down but have no fever, "
            "and you're still able to go about your day."
        ),
    ),
    NonInfectiousComplaint(
        name       = "Gastroenteritis",
        icd_code   = "A09",
        management = "home rest",
        disease    = "gastroenteritis",
        prompt_context = (
            "You've had an upset stomach with loose stools and mild nausea since "
            "yesterday — you think it might be something you ate. No blood, no "
            "severe pain, and you're managing to keep fluids down."
        ),
    ),
    NonInfectiousComplaint(
        name       = "Urinary tract infection",
        icd_code   = "N39.0",
        management = "treat",
        disease    = "uti",
        prompt_context = (
            "You've noticed a burning feeling when you urinate and the urge to "
            "go more often than usual, for the past day or two. No fever, no "
            "back pain — otherwise you feel fine."
        ),
    ),
    NonInfectiousComplaint(
        name       = "Migraine",
        icd_code   = "G43.9",
        management = "treat",
        disease    = "migraine",
        prompt_context = (
            "You've had a throbbing headache on one side of your head for several "
            "hours, with some sensitivity to light. You get these occasionally and "
            "recognise the pattern — no fever, no neck stiffness, no vision changes."
        ),
    ),
)
