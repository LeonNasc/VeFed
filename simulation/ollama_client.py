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
MODEL         = "phi3:mini"
TIMEOUT_SEC   = 60
MAX_FOLLOWUPS = 1   # was 2 — each follow-up is a full round-trip; keep to 1

SYSTEM_PROMPT = """\
You are a clinical triage doctor. A patient describes their symptoms in their own words.
Triage them based solely on what they report.

DEFAULT: decide immediately on the first turn.
Only ask ONE follow-up if breathing status or oxygen level is completely absent
and the severity is ambiguous. Never ask more than one question.

Each response must be a single JSON object — no other text.

To ask a follow-up:
{"type": "question", "text": "<one concise question>"}

To give your decision:
{"type": "decision", "action": "<home_recovery|resolve|hospitalise>", \
"label": "<mild|moderate|severe>", "diagnosis": "<brief diagnosis>", \
"notes": "<one sentence rationale>"}

Rules:
- home_recovery : mild — patient rests at home
- resolve       : moderate — needs prescription or treatment plan
- hospitalise   : severe — immediate admission required
- Do not ask for numbers or vitals. One question maximum.
"""


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaDiagnosticClient:
    """
    Multi-turn Ollama doctor. Blocks until the full conversation completes.
    Conversation is recorded on DiagnosticEvent.conversation as [{role, text}].

    Ollama runs as a single detached server; this client is one shared handle
    across all WorldEngine silos — no per-silo instances needed.

    Few-shot learning: call update_examples() after each FL round to inject
    the best past conversations as prior context.  The model then sees proven
    correct exchanges before each new patient, improving speed and accuracy.
    """

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL,
                 timeout: int = TIMEOUT_SEC):
        self.model    = model
        self.url      = url
        self.timeout  = timeout
        self._examples: list[dict] = []   # few-shot examples, updated each FL round

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

    def update_examples(self, events: list, max_per_tier: int = 2) -> None:
        """
        Refresh the few-shot example bank from a batch of processed DiagnosticEvents.

        Selects up to max_per_tier correct conversations per management tier
        (home rest / treat / hospitalise), ordered by fewest turns.  Correct =
        doctor's action matches the oracle management in ground_truth.

        Call this after each FL round so the doctor improves progressively.
        """
        import json
        from simulation.models import DiagnosticAction

        mgmt_to_action = {
            "home rest":   DiagnosticAction.RECOVER,
            "treat":       DiagnosticAction.RESOLVE,
            "hospitalise": DiagnosticAction.HOSPITALISE,
        }

        by_tier: dict[str, list[tuple[int, dict]]] = {t: [] for t in mgmt_to_action}

        for ev in events:
            if not ev.ground_truth or not ev.action or not ev.conversation:
                continue
            parts = ev.ground_truth.rsplit(" / ", 1)
            if len(parts) != 2:
                continue
            expected = mgmt_to_action.get(parts[1])
            if expected != ev.action:
                continue   # incorrect diagnosis — not a useful example

            patient_turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
            if not patient_turns:
                continue

            # Reconstruct the doctor JSON from stored fields
            doctor_json = json.dumps({
                "type":      "decision",
                "action":    ev.action.value,
                "label":     ev.oracle_label or "unknown",
                "diagnosis": ev.diagnosis or "",
                "notes":     ev.notes or "",
            })

            n_turns = len(ev.conversation)
            by_tier[parts[1]].append((n_turns, {
                "patient":     patient_turns[0],
                "doctor_json": doctor_json,
            }))

        # Pick the shortest correct conversation per tier for diversity
        self._examples = []
        for tier_examples in by_tier.values():
            tier_examples.sort(key=lambda x: x[0])
            self._examples.extend(ex for _, ex in tier_examples[:max_per_tier])

    def num_examples(self) -> int:
        return len(self._examples)

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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Inject accumulated few-shot examples as prior conversation turns.
        # The model sees proven correct exchanges before each new patient,
        # learning to decide faster and with better calibration over FL rounds.
        for ex in self._examples:
            messages.append({"role": "user",      "content": ex["patient"]})
            messages.append({"role": "assistant", "content": ex["doctor_json"]})
        messages.append({"role": "user", "content": opening})
        return messages

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
