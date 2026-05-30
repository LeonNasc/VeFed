"""
Non-infectious complaint types for background clinic visitors.

Each complaint carries:
  - An ICD-10 code and default management tier (ground-truth label)
  - A prompt_context injected into the patient LLM's system prompt so it
    generates a naturalistically typed opening statement for that condition

These replace the old single-bucket Z00.0 "worried-well" background events.
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
