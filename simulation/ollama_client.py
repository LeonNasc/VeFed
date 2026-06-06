"""
Ollama diagnostic client — multi-turn LLM doctor (§3.4).

Conversation protocol per patient:
  1. Patient sends opening statement (natural language)
  2. Doctor asks ONE symptom follow-up question
  3. Patient answers (PatientLLMClient if available, else SymptomNarrator)
  4. Doctor requests objective vitals: {"type": "vitals_request"}
  5. System injects formatted vitals from CaseTable
  6. Doctor delivers triage decision

The doctor LLM never sees severity s numerically — only natural-language
reports and a vitals panel identical to what a triage nurse would hand over.

Raises OllamaUnavailableError if the doctor model is unreachable.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from simulation.models import DiagnosticAction, DiagnosticEvent

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "phi3:mini"
TIMEOUT_SEC = 60
MAX_TURNS   = 6   # safety cap — prevents loops if model keeps asking questions

NURSE_SYSTEM_PROMPT = """\
You are a triage nurse. Assess the severity of the patient's condition. Each reply is ONE JSON object — no other text.

First ask the patient one follow-up question about their symptoms, then request their vitals.

To ask a question:
{"type": "question", "text": "your question here"}

To request vitals (after the patient has answered):
{"type": "vitals_request"}

To give your triage assessment (only after seeing vitals):
{"type": "triage", "severity": "discharge or mild or moderate or severe or critical", "notes": "one sentence"}

Severity guide — discharge: no concerning signs; mild: rest at home; moderate: clinic treatment; severe: urgent care; critical: immediate hospitalisation.
Output only one JSON object per turn. Never assess before seeing vitals.
"""

DOCTOR_SYSTEM_PROMPT = """\
You are a diagnostic doctor. The triage nurse has already assessed severity. Your task is to identify the disease.
Each reply is ONE JSON object — no other text.

You may ask the patient one clarifying question if needed, then give your diagnosis.

To ask a question:
{"type": "question", "text": "your question here"}

To give your diagnosis:
{"type": "diagnosis", "disease": "influenza or pneumonia or non-infectious or unknown", "notes": "one sentence", "triage_confirmed": true or false}

Choose "unknown" when the symptom pattern does not match any known disease.
Output only one JSON object per turn.
"""

# Legacy alias — kept so external code that imports SYSTEM_PROMPT still works
SYSTEM_PROMPT = DOCTOR_SYSTEM_PROMPT


class OllamaUnavailableError(RuntimeError):
    pass


class TriageNurseClient:
    """
    Single-purpose triage nurse: runs a brief multi-turn conversation to assess
    severity (discharge/mild/moderate/severe/critical) and records it on the event.

    Shares the same Ollama server as the diagnostic doctor but uses NURSE_SYSTEM_PROMPT.
    Called first in the two-agent pipeline before OllamaDiagnosticClient runs.
    """

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL,
                 timeout: int = TIMEOUT_SEC, patient_llm=None):
        self.model        = model
        self.url          = url
        self.timeout      = timeout
        self._patient_llm = patient_llm

    def triage(self, event: DiagnosticEvent, _queue=None,
               _format_vitals=None, _patient_answer=None) -> str:
        """
        Run the nurse conversation. Returns the severity string and stores it
        on event.nurse_severity.  The full nurse conversation is appended to
        event.conversation so the doctor can read it.

        _format_vitals, _patient_answer: callables injected by the pipeline so
        the nurse reuses the same vitals/answer logic without duplicating code.
        """
        def _status(msg):
            if _queue:
                _queue.update_status(event.agent_id, msg)

        messages     = [{"role": "system", "content": NURSE_SYSTEM_PROMPT}]
        opening      = next(
            (t["text"] for t in event.conversation if t["role"] == "patient"),
            "I am not feeling well.",
        )
        messages.append({"role": "user", "content": opening})

        vitals_taken = False
        severity     = "mild"   # fallback

        for turn in range(MAX_TURNS):
            _status(f"Nurse assessing… (turn {turn + 1})")
            raw    = self._call_ollama(messages)
            parsed = self._parse_response(raw)

            if parsed["type"] == "question" and not vitals_taken and parsed.get("text"):
                q_text = parsed["text"]
                event.conversation.append({"role": "nurse",   "text": q_text})
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                _status("Patient answering…")
                answer = _patient_answer(event, q_text) if _patient_answer else "I'm not sure."
                event.conversation.append({"role": "patient", "text": answer})
                messages.append({"role": "user", "content": answer})

            elif parsed["type"] == "vitals_request" and not vitals_taken:
                vitals_str = _format_vitals(event) if _format_vitals else "[VITALS] unavailable"
                event.conversation.append({"role": "nurse",   "text": "[measuring vitals]"})
                event.conversation.append({"role": "vitals",  "text": vitals_str})
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                messages.append({"role": "user",      "content": vitals_str})
                vitals_taken = True

            elif parsed["type"] == "triage":
                severity = parsed.get("severity", "mild").lower()
                if severity not in ("discharge", "mild", "moderate", "severe", "critical"):
                    severity = "mild"
                notes = (parsed.get("notes") or "")[:120]
                event.conversation.append({
                    "role": "nurse",
                    "text": f"[Triage] Severity: {severity}. {notes}",
                })
                break

            else:
                # Force a triage if the nurse got stuck
                messages.append({
                    "role": "user",
                    "content": "Please give your triage severity assessment now.",
                })

        event.nurse_severity = severity
        return severity

    def _call_ollama(self, messages):
        payload = json.dumps({
            "model": self.model, "messages": messages,
            "stream": False, "format": "json",
            "options": {"temperature": 0.2, "num_predict": 100},
        }).encode()
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read()).get("message", {}).get("content", "")
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(f"Cannot reach Ollama at {self.url}: {exc.reason}") from exc
        except Exception as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

    def _parse_response(self, raw: str) -> dict:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"type": "triage", "severity": "mild"}
        t = data.get("type", "")
        if t in ("question", "vitals_request", "triage"):
            return data
        for val in data.values():
            if isinstance(val, dict) and val.get("type") in ("question", "vitals_request", "triage"):
                return val
        if "severity" in data:
            data["type"] = "triage"
            return data
        return {"type": "vitals_request"}


class OllamaDiagnosticClient:
    """
    Multi-turn Ollama doctor. Blocks until the full conversation completes.
    Conversation is recorded on DiagnosticEvent.conversation as [{role, text}].

    patient_llm : PatientLLMClient | None
        If provided, patient follow-up answers are generated by the small
        patient model.  Falls back to SymptomNarrator templates otherwise.

    Few-shot learning: call update_examples() after each FL round to inject
    the best past conversations as prior context.
    """

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL,
                 timeout: int = TIMEOUT_SEC, patient_llm=None):
        self.model             = model
        self.url               = url
        self.timeout           = timeout
        self._patient_llm      = patient_llm
        self._examples:        list[dict] = []
        self._proto_library    = None   # PrototypeLibrary | None
        self._proto_encoder    = None   # (model, tokenizer) for embedding queries
        self._proto_k:         int = 3  # prototypes retrieved per query
        self._global_stereos:  dict    = {}   # {label: centroid} from StereotypeLibrary
        self._n_stereo_silos:  int     = 0
        # Triage nurse — shares Ollama server; runs before doctor in the pipeline
        self._nurse = TriageNurseClient(model=model, url=url,
                                        timeout=timeout, patient_llm=patient_llm)

    # ── Public interface ──────────────────────────────────────────────────────

    def diagnose(self, event: DiagnosticEvent, _queue=None,
                 _proto_lib=None) -> DiagnosticEvent:
        """
        Two-agent clinical pipeline:
          Phase 1 — Triage nurse assesses severity (sets event.nurse_severity)
          Phase 2 — Diagnostic doctor identifies disease (sets event.doctor_disease)

        event.conversation must already contain the opening patient statement.
        Both agents' turns are recorded sequentially on event.conversation.
        Raises OllamaUnavailableError if the Ollama server is unreachable.
        """
        def _status(msg: str) -> None:
            if _queue:
                _queue.update_status(event.agent_id, msg)

        nurse_turns = 0

        # ── Phase 1: Triage nurse ─────────────────────────────────────────────
        _status("Nurse triaging…")
        nurse_start = len(event.conversation)
        severity = self._nurse.triage(
            event,
            _queue         = _queue,
            _format_vitals = self._format_vitals,
            _patient_answer= self._patient_answer,
        )
        nurse_turns = len(event.conversation) - nurse_start

        # Map nurse severity → action
        sev_to_action = {
            "discharge": DiagnosticAction.RECOVER,
            "mild":      DiagnosticAction.RECOVER,
            "moderate":  DiagnosticAction.RESOLVE,
            "severe":    DiagnosticAction.HOSPITALISE,
            "critical":  DiagnosticAction.HOSPITALISE,
        }
        event.action = sev_to_action.get(severity, DiagnosticAction.RECOVER)

        # ── Phase 2: Diagnostic doctor ────────────────────────────────────────
        messages = self._build_doctor_messages(event, proto_lib=_proto_lib)
        doctor_turns = 0

        for turn in range(MAX_TURNS):
            _status(f"Doctor diagnosing… (turn {turn + 1})")
            raw    = self._call_ollama(messages)
            parsed = self._parse_response(raw)
            doctor_turns += 1

            if parsed["type"] == "question" and parsed.get("text"):
                q_text = parsed["text"]
                event.conversation.append({"role": "doctor",  "text": q_text})
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                _status("Patient answering…")
                answer = self._patient_answer(event, q_text)
                event.conversation.append({"role": "patient", "text": answer})
                messages.append({"role": "user", "content": answer})

            else:
                if parsed["type"] != "diagnosis":
                    parsed = self._force_diagnosis(messages)

                disease = (parsed.get("disease") or "unknown").lower()
                if disease not in ("influenza", "pneumonia", "non-infectious", "unknown"):
                    disease = "unknown"
                event.doctor_disease = disease
                event.diagnosis      = (parsed.get("notes") or "")[:100]
                event.oracle_label   = disease
                triage_ok = bool(parsed.get("triage_confirmed", True))
                event.conversation.append({
                    "role": "doctor",
                    "text": (
                        f"Diagnosis: {disease}. "
                        f"Triage {'confirmed' if triage_ok else 'adjusted'}. "
                        f"{event.diagnosis}"
                    ),
                })
                break

        event.num_turns = nurse_turns + doctor_turns
        return event

    def update_global_stereotypes(self, centroids: dict, n_silos: int = 0) -> None:
        """
        Store federated stereotype centroids for prior injection at inference.
        Called by the FL orchestrator after each round's aggregate_stereotypes().
        Pass an empty dict to disable stereotype injection.
        """
        self._global_stereos = centroids
        self._n_stereo_silos = n_silos

    def set_prototype_library(self, library, encoder_model, tokenizer,
                               k: int = 3) -> None:
        """
        Attach a PrototypeLibrary for embedding-based retrieval at inference time.
        When set, retrieved prototypes are injected before the few-shot examples.
        """
        self._proto_library = library
        self._proto_encoder = (encoder_model, tokenizer)
        self._proto_k       = k

    def update_examples(self, events: list, max_per_tier: int = 2) -> None:
        """
        Refresh the few-shot example bank from a batch of processed DiagnosticEvents.

        Selects up to max_per_tier correct conversations per management tier,
        ordered by fewest turns.  Each example stores the full message sequence
        (question → patient answer → vitals_request → vitals → decision) so the
        model sees the correct protocol, not a shortcut opening→decision.
        """
        sev_to_action = {
            "mild":           DiagnosticAction.RECOVER,
            "moderate":       DiagnosticAction.RESOLVE,
            "severe":         DiagnosticAction.HOSPITALISE,
            "non-infectious": DiagnosticAction.RECOVER,
            "none":           DiagnosticAction.RECOVER,
        }

        by_sev: dict[str, list[tuple[int, list]]] = {k: [] for k in ("mild", "moderate", "severe", "non-infectious")}

        for ev in events:
            if not ev.ground_truth or not ev.action or not ev.conversation:
                continue
            gt = ev.ground_truth
            if "/" in gt:
                sev = gt.split("/", 1)[1]
            else:
                sev = "non-infectious"
            expected = sev_to_action.get(sev)
            if expected != ev.action:
                continue

            # Only use conversations that went through the full protocol
            roles = [t["role"] for t in ev.conversation]
            if "doctor" not in roles or len(ev.conversation) < 3:
                continue

            messages = self._conversation_to_messages(ev)
            n_turns  = len(ev.conversation)
            if sev in by_sev:
                by_sev[sev].append((n_turns, messages))

        self._examples = []
        for sev_examples in by_sev.values():
            sev_examples.sort(key=lambda x: x[0])
            self._examples.extend(msgs for _, msgs in sev_examples[:max_per_tier])

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

    def _patient_answer(self, event: DiagnosticEvent, question: str) -> str:
        """Generate a patient follow-up answer, preferring the patient LLM."""
        from simulation.symptom_language import Personality, SymptomNarrator
        personality = event.personality or Personality.NEUTRAL

        if self._patient_llm is not None:
            try:
                return self._patient_llm.followup_answer(
                    question, event.severity, personality,
                    case_table=event.case_table, day=event.days_infected,
                )
            except Exception:
                pass

        return SymptomNarrator().followup_answer(
            event.symptoms, personality, question,
            case_table=event.case_table, day=event.days_infected,
        )

    def _format_vitals(self, event: DiagnosticEvent) -> str:
        """Format CaseTable vitals as a triage-nurse-style panel string."""
        ct  = event.case_table
        day = event.days_infected
        if ct is None:
            return "[VITALS] Unavailable — proceed on clinical history alone."

        hr     = ct.get("HR",      day)
        temp   = ct.get("temp",    day)
        spo2   = ct.get("SpO2",    day)
        rr     = ct.get("RR",      day)
        bp_sys = ct.get("BP_sys",  day)
        bp_dia = ct.get("BP_dia",  day)
        pain   = ct.get("pain",    day)
        fat    = ct.get("fatigue", day)

        parts = []
        if hr     is not None: parts.append(f"HR {hr:.0f} bpm")
        if temp   is not None: parts.append(f"Temp {temp:.1f}°C")
        if spo2   is not None: parts.append(f"SpO2 {spo2:.0f}%")
        if rr     is not None: parts.append(f"RR {rr:.0f}/min")
        if bp_sys is not None and bp_dia is not None:
            parts.append(f"BP {bp_sys:.0f}/{bp_dia:.0f} mmHg")
        if pain   is not None: parts.append(f"Pain {pain:.0f}/10")
        if fat    is not None: parts.append(f"Fatigue {fat:.0f}/10")

        return "[VITALS] " + " | ".join(parts)

    def _build_doctor_messages(self, event: DiagnosticEvent,
                               proto_lib=None) -> list[dict]:
        """
        Build the initial message list for the diagnostic doctor.
        Includes: system prompt, RAG injections, full nurse conversation so far,
        and the nurse's triage summary as context.
        """
        messages = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]

        # Shared query embedding for prototype + stereotype retrieval
        opening = next(
            (t["text"] for t in event.conversation if t["role"] == "patient"),
            "I am not feeling well.",
        )
        query_vec = None
        library   = proto_lib or self._proto_library
        need_embed = (library is not None or bool(self._global_stereos))
        if need_embed and self._proto_encoder is not None:
            try:
                import torch
                enc_model, tokenizer = self._proto_encoder
                device = next(enc_model.parameters()).device
                enc = tokenizer([opening], padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
                enc_model.eval()
                with torch.no_grad():
                    out = enc_model(
                        input_ids      = enc["input_ids"].to(device),
                        attention_mask = enc["attention_mask"].to(device),
                        output_hidden_states = True,
                    )
                query_vec = out.hidden_states[-1][0, 0, :].cpu().numpy()
            except Exception:
                pass

        if library is not None and query_vec is not None:
            try:
                retrieved = library.retrieve(query_vec, k=self._proto_k)
                if retrieved:
                    from fl.prototype_library import prototypes_to_prompt_messages
                    messages.extend(prototypes_to_prompt_messages(retrieved))
            except Exception:
                pass

        if self._global_stereos and query_vec is not None:
            try:
                from fl.stereotype_library import (nearest_centroids,
                                                   stereotypes_to_prompt_messages)
                nearest = nearest_centroids(query_vec, self._global_stereos, k=5)
                if nearest:
                    messages.extend(stereotypes_to_prompt_messages(
                        nearest, n_silos=self._n_stereo_silos,
                    ))
            except Exception:
                pass

        for ex_messages in self._examples:
            messages.extend(ex_messages)

        # Inject nurse triage summary so doctor has context
        if event.nurse_severity:
            messages.append({
                "role": "user",
                "content": (
                    f"[Triage nurse assessment] Severity: {event.nurse_severity}. "
                    f"Patient: {opening}"
                ),
            })
        else:
            messages.append({"role": "user", "content": opening})

        return messages

    def _build_initial_messages(self, event: DiagnosticEvent,
                                proto_lib=None) -> list[dict]:
        opening = next(
            (t["text"] for t in event.conversation if t["role"] == "patient"),
            "I am not feeling well.",
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Compute query embedding once — shared by prototype and stereotype paths.
        # Only encode if at least one retrieval system is active.
        query_vec = None
        library   = proto_lib or self._proto_library
        need_embed = (library is not None or bool(self._global_stereos))
        if need_embed and self._proto_encoder is not None:
            try:
                import torch
                enc_model, tokenizer = self._proto_encoder
                device = next(enc_model.parameters()).device
                enc = tokenizer([opening], padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
                enc_model.eval()
                with torch.no_grad():
                    out = enc_model(
                        input_ids      = enc["input_ids"].to(device),
                        attention_mask = enc["attention_mask"].to(device),
                        output_hidden_states = True,
                    )
                query_vec = out.hidden_states[-1][0, 0, :].cpu().numpy()
            except Exception:
                pass

        # ── Prototype retrieval (exemplar-based few-shot RAG) ─────────────────
        if library is not None and query_vec is not None:
            try:
                retrieved = library.retrieve(query_vec, k=self._proto_k)
                if retrieved:
                    from fl.prototype_library import prototypes_to_prompt_messages
                    messages.extend(prototypes_to_prompt_messages(retrieved))
            except Exception:
                pass

        # ── Stereotype prior (centroid-based classification hint) ─────────────
        if self._global_stereos and query_vec is not None:
            try:
                from fl.stereotype_library import (nearest_centroids,
                                                   stereotypes_to_prompt_messages)
                nearest = nearest_centroids(query_vec, self._global_stereos, k=5)
                if nearest:
                    messages.extend(stereotypes_to_prompt_messages(
                        nearest, n_silos=self._n_stereo_silos,
                    ))
            except Exception:
                pass

        # ── Conversation few-shot examples (appended after retrieval) ─────────
        for ex_messages in self._examples:
            messages.extend(ex_messages)

        messages.append({"role": "user", "content": opening})
        return messages

    def _conversation_to_messages(self, ev: DiagnosticEvent) -> list[dict]:
        """Convert a DiagnosticEvent's conversation into Ollama message format."""
        messages = []
        decision_json = json.dumps({
            "type":      "decision",
            "action":    ev.action.value,
            "label":     ev.oracle_label or "unknown",
            "diagnosis": ev.diagnosis or "",
            "notes":     ev.notes or "",
        })
        for turn in ev.conversation:
            role, text = turn["role"], turn["text"]
            if role == "patient":
                messages.append({"role": "user", "content": text})
            elif role == "vitals":
                messages.append({"role": "user", "content": text})
            elif role == "doctor":
                if "[measuring vitals]" in text:
                    messages.append({"role": "assistant",
                                     "content": json.dumps({"type": "vitals_request"})})
                elif text.startswith("Diagnosis:"):
                    messages.append({"role": "assistant", "content": decision_json})
                else:
                    messages.append({"role": "assistant",
                                     "content": json.dumps({"type": "question", "text": text})})
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
        except (json.JSONDecodeError, ValueError):
            return {"type": "decision", "action": "home_recovery",
                    "label": "parse_error",
                    "notes": f"Unparseable output: {raw[:60]}"}

        known = ("question", "vitals_request", "decision", "triage", "diagnosis")
        top_type = data.get("type", "")
        if top_type in known and (top_type != "question" or data.get("text")):
            return data

        # phi3:mini sometimes nests responses — walk one level
        for val in data.values():
            if not isinstance(val, dict):
                continue
            vtype = val.get("type", "")
            if vtype == "question" and val.get("text"):
                return val
            if vtype in ("vitals_request", "triage", "diagnosis"):
                return val
            if vtype == "decision" and val.get("action"):
                return val

        if "disease" in data:
            data["type"] = "diagnosis"
            return data
        if "severity" in data:
            data["type"] = "triage"
            return data
        if "action" in data:
            data["type"] = "decision"
            return data

        return {"type": "vitals_request"}

    def _force_decision(self, messages: list[dict]) -> dict:
        messages = messages + [{
            "role": "user",
            "content": "Please give your final triage decision now.",
        }]
        raw    = self._call_ollama(messages)
        parsed = self._parse_response(raw)
        if parsed.get("type") == "decision":
            return parsed
        return {"type": "decision", "action": "home_recovery",
                "label": "unknown", "notes": "Could not obtain decision from model."}

    def _force_diagnosis(self, messages: list[dict]) -> dict:
        messages = messages + [{
            "role": "user",
            "content": "Please give your final disease diagnosis now.",
        }]
        raw    = self._call_ollama(messages)
        parsed = self._parse_response(raw)
        if parsed.get("type") == "diagnosis":
            return parsed
        return {"type": "diagnosis", "disease": "unknown",
                "notes": "Could not obtain diagnosis from model.",
                "triage_confirmed": True}
