"""
Ollama diagnostic client — multi-turn LLM doctor (§3.4).

Pipeline per patient:
  1. Patient sends opening statement (natural language, σ-derived)
  2. Doctor responds — either asks a follow-up (INQUIRY) or delivers triage
  3. If follow-up: patient answers in character, doctor gives final decision
  Max 2 follow-up turns enforced.

The LLM never sees severity s — only the patient's natural-language report.
Raises OllamaUnavailableError if Ollama is unreachable.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from simulation.models import DiagnosticAction, DiagnosticEvent

OLLAMA_URL    = "http://localhost:11434/api/chat"
MODEL         = "mistral"
TIMEOUT_SEC   = 30
MAX_FOLLOWUPS = 2

SYSTEM_PROMPT = """\
You are a clinical triage doctor in a busy clinic. A patient describes their symptoms \
in their own words. Your job is to triage them using only what they tell you.

You may ask up to 2 follow-up questions to gather more information before deciding. \
Each response must be a single JSON object — nothing else.

While gathering information, respond with:
{"type": "question", "text": "<your follow-up question>"}

When ready to decide, respond with:
{"type": "decision", "action": "<home_recovery|resolve|hospitalise>", \
"label": "<mild|moderate|severe>", "diagnosis": "<your clinical diagnosis>", \
"notes": "<one sentence rationale>"}

Rules:
- home_recovery: mild symptoms, patient can rest at home
- resolve: moderate symptoms, needs treatment/prescription
- hospitalise: severe symptoms, requires immediate admission
- Base your decision solely on what the patient reports. Do not ask for numbers.
- Be concise. One question at a time.
"""


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaDiagnosticClient:
    """
    Multi-turn Ollama doctor. Blocks until the full conversation completes.
    Conversation is recorded on DiagnosticEvent.conversation as [{role, text}].
    """

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL,
                 timeout: int = TIMEOUT_SEC):
        self.model   = model
        self.url     = url
        self.timeout = timeout

    # ── Public interface ──────────────────────────────────────────────────────

    def diagnose(self, event: DiagnosticEvent, _queue=None) -> DiagnosticEvent:
        """
        Run the full patient-doctor pipeline for one DiagnosticEvent.
        event.conversation must already contain the opening patient statement.
        Raises OllamaUnavailableError if Ollama is unreachable.
        """
        messages       = self._build_initial_messages(event)
        followups_used = 0

        def _status(msg: str) -> None:
            if _queue:
                _queue.update_status(event.agent_id, msg)

        while True:
            _status(f"Doctor thinking… (turn {followups_used + 1})")
            raw    = self._call_ollama(messages)
            parsed = self._parse_response(raw)

            if parsed["type"] == "question" and followups_used < MAX_FOLLOWUPS:
                q_text = parsed["text"]
                event.conversation.append({"role": "doctor",  "text": q_text})
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                _status(f"Patient answering follow-up {followups_used + 1}/{MAX_FOLLOWUPS}…")

                from simulation.symptom_language import Personality, SymptomNarrator
                personality = event.personality or Personality.NEUTRAL
                answer      = SymptomNarrator().followup_answer(
                    event.symptoms, personality, q_text
                )
                event.conversation.append({"role": "patient", "text": answer})
                messages.append({"role": "user", "content": answer})
                followups_used += 1

            else:
                if parsed["type"] == "question":
                    parsed = self._force_decision(messages)

                action_map = {
                    "home_recovery": DiagnosticAction.RECOVER,
                    "resolve":       DiagnosticAction.RESOLVE,
                    "hospitalise":   DiagnosticAction.HOSPITALISE,
                    "hospitalize":   DiagnosticAction.HOSPITALISE,
                }
                event.action       = action_map.get(
                    parsed.get("action", "").lower(), DiagnosticAction.RECOVER
                )
                event.oracle_label = parsed.get("label", "unknown")
                event.diagnosis    = parsed.get("diagnosis", "")[:100]
                event.notes        = parsed.get("notes", "")[:160]
                event.conversation.append({
                    "role": "doctor",
                    "text": f"[{event.oracle_label}] {event.notes}",
                })
                break

        return event

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

    def _build_initial_messages(self, event: DiagnosticEvent) -> list[dict]:
        opening = next(
            (t["text"] for t in event.conversation if t["role"] == "patient"),
            "I am not feeling well.",
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": opening},
        ]

    def _call_ollama(self, messages: list[dict]) -> str:
        payload = json.dumps({
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "format":   "json",
            "options":  {"temperature": 0.2, "num_predict": 120},
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
                return body.get("message", {}).get("content", "")
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.url}: {exc.reason}"
            ) from exc
        except Exception as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

    def _parse_response(self, raw: str) -> dict:
        try:
            data = json.loads(raw)
            if "type" not in data:
                data["type"] = "decision" if "action" in data else "question"
            return data
        except (json.JSONDecodeError, KeyError):
            return {"type": "decision", "action": "home_recovery",
                    "label": "parse_error",
                    "notes": f"Unparseable output: {raw[:60]}"}

    def _force_decision(self, messages: list[dict]) -> dict:
        messages = messages + [{
            "role": "user",
            "content": "Please give your final triage decision now."
        }]
        raw    = self._call_ollama(messages)
        parsed = self._parse_response(raw)
        if parsed.get("type") == "decision":
            return parsed
        return {"type": "decision", "action": "home_recovery",
                "label": "unknown", "notes": "Could not obtain decision from model."}
