"""
Federated prototype library for case-based triage.

Each silo maintains a PrototypeLibrary — a collection of CLS embeddings from
processed DiagnosticEvents, indexed by ground-truth label. At each FL round:
  1. Correctly-triaged events are encoded and added to the local library.
  2. A compact representative subset is selected per label category.
  3. The orchestrator merges all silos' subsets into a global library.
  4. The global library is redistributed; the LLM doctor retrieves the k nearest
     prototypes at inference time and uses them as grounded few-shot context.

The prototype IS the embedding — raw patient text never leaves the silo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CasePrototype:
    embedding:  np.ndarray   # CLS vector (768,) from DistilBERT
    outcome:    str          # "{icd} / {management}" ground truth
    silo_id:    int
    round_num:  int


class PrototypeLibrary:
    """
    Per-silo case library. Stores CLS embeddings keyed by outcome label.

    Parameters
    ----------
    max_per_label : int
        Maximum prototypes retained per label after pruning. Oldest entries
        are dropped when the limit is exceeded so the library stays compact.
    silo_id : int
    """

    def __init__(self, silo_id: int = 0, max_per_label: int = 50):
        self.silo_id       = silo_id
        self.max_per_label = max_per_label
        self._cases: dict[str, list[CasePrototype]] = {}

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add(self, embedding: np.ndarray, outcome: str, round_num: int) -> None:
        bucket = self._cases.setdefault(outcome, [])
        bucket.append(CasePrototype(
            embedding = embedding.astype(np.float32),
            outcome   = outcome,
            silo_id   = self.silo_id,
            round_num = round_num,
        ))
        if len(bucket) > self.max_per_label:
            del bucket[0]  # drop oldest

    def add_from_events(self, events: list, model, tokenizer,
                        round_num: int, device: str = "cpu") -> int:
        """
        Encode correctly-triaged events and add their CLS embeddings.

        Only events where the LLM management tier matches ground truth are
        added — we want the library to represent *correct* presentations,
        not confused ones.

        Returns number of prototypes added.
        """
        import torch
        from simulation.models import DiagnosticAction

        tier_map = {
            DiagnosticAction.RECOVER:    "home rest",
            DiagnosticAction.RESOLVE:    "treat",
            DiagnosticAction.HOSPITALISE:"hospitalise",
        }
        added = 0
        texts, outcomes = [], []

        for ev in events:
            if not ev.ground_truth or not ev.action:
                continue
            parts = ev.ground_truth.rsplit(" / ", 1)
            if len(parts) != 2:
                continue
            expected_tier = parts[1]
            actual_tier   = tier_map.get(ev.action, "")
            if expected_tier != actual_tier:
                continue
            patient_turns = [t["text"] for t in ev.conversation
                             if t["role"] == "patient"]
            if not patient_turns:
                continue
            texts.append(patient_turns[-1])
            outcomes.append(ev.ground_truth)

        if not texts:
            return 0

        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        model.eval()
        with torch.no_grad():
            out = model(
                input_ids      = enc["input_ids"].to(device),
                attention_mask = enc["attention_mask"].to(device),
                output_hidden_states = True,
            )
        cls_vecs = out.hidden_states[-1][:, 0, :].cpu().numpy()

        for vec, outcome in zip(cls_vecs, outcomes):
            self.add(vec, outcome, round_num)
            added += 1
        return added

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: np.ndarray, k: int = 3) -> list[CasePrototype]:
        """
        Return the k most similar prototypes to the query CLS vector.
        Similarity: cosine. Searches across all label categories.
        """
        all_protos = [p for bucket in self._cases.values() for p in bucket]
        if not all_protos:
            return []

        mat = np.stack([p.embedding for p in all_protos])  # (N, 768)
        q   = query / (np.linalg.norm(query) + 1e-9)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        sims  = (mat / norms) @ q                          # (N,)
        top_k = min(k, len(all_protos))
        idxs  = np.argpartition(sims, -top_k)[-top_k:]
        idxs  = idxs[np.argsort(sims[idxs])[::-1]]
        return [all_protos[i] for i in idxs]

    # ── Federation helpers ────────────────────────────────────────────────────

    def representative_subset(self, max_per_label: int = 10) -> list[CasePrototype]:
        """
        Select up to max_per_label prototypes per label for sharing.
        Prefers the most recent entries (highest round_num) so the shared
        knowledge reflects the current state of the silo's epidemic.
        """
        result = []
        for bucket in self._cases.values():
            sorted_bucket = sorted(bucket, key=lambda p: p.round_num, reverse=True)
            result.extend(sorted_bucket[:max_per_label])
        return result

    def merge(self, prototypes: list[CasePrototype]) -> None:
        """Add a list of prototypes from other silos into this library."""
        for p in prototypes:
            bucket = self._cases.setdefault(p.outcome, [])
            bucket.append(p)
            if len(bucket) > self.max_per_label:
                del bucket[0]

    def size(self) -> int:
        return sum(len(b) for b in self._cases.values())

    def label_counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._cases.items()}


# ── Federation orchestration ──────────────────────────────────────────────────

def federate_libraries(
    libraries: list[PrototypeLibrary],
    max_per_label: int = 10,
) -> list[CasePrototype]:
    """
    Collect representative subsets from all silos and return a merged global
    prototype set ready to be distributed back to each silo.
    """
    global_set: list[CasePrototype] = []
    for lib in libraries:
        global_set.extend(lib.representative_subset(max_per_label))
    return global_set


# ── LLM prompt injection ──────────────────────────────────────────────────────

ICD3_NAMES = {
    "A41": "Sepsis",
    "A99": "Benchmark fever",
    "F41": "Anxiety",
    "J09": "Severe flu",
    "J10": "Flu",
    "J11": "Flu",
    "J18": "Pneumonia",
    "M54": "Back pain",
    "R51": "Headache",
    "R53": "Fatigue",
    "U07": "Corona",
    "Z87": "Hypertension f/u",
}

MGMT_LABELS = {
    "home rest":   "home rest — self-care",
    "treat":       "treat — medical review needed",
    "hospitalise": "hospitalise — admit now",
}


def prototypes_to_prompt_messages(prototypes: list[CasePrototype]) -> list[dict]:
    """
    Convert retrieved prototypes into Ollama message-format few-shot entries.

    Each prototype becomes a (user, assistant) pair where:
      user      = "A patient presented with [disease], [management tier]."
      assistant = the correct decision JSON for that outcome

    This slots into _build_initial_messages() before the real patient turn.
    """
    import json
    messages = []
    for p in prototypes:
        parts = p.outcome.rsplit(" / ", 1)
        if len(parts) != 2:
            continue
        icd, mgmt = parts
        icd3      = icd[:3]
        disease   = ICD3_NAMES.get(icd3, icd)
        mgmt_desc = MGMT_LABELS.get(mgmt, mgmt)

        action_map = {
            "home rest":   "home_recovery",
            "treat":       "resolve",
            "hospitalise": "hospitalise",
        }
        label_map = {
            "home rest":   "mild",
            "treat":       "moderate",
            "hospitalise": "severe",
        }

        user_text = (
            f"[Retrieved case — {disease}, managed as: {mgmt_desc}. "
            f"Origin: silo {p.silo_id}, round {p.round_num}.]"
        )
        assistant_json = json.dumps({
            "type":      "decision",
            "action":    action_map.get(mgmt, "home_recovery"),
            "label":     label_map.get(mgmt, "unknown"),
            "diagnosis": f"{disease} presentation",
            "notes":     f"Based on similar past case from silo {p.silo_id}.",
        })
        messages.append({"role": "user",      "content": user_text})
        messages.append({"role": "assistant", "content": assistant_json})
    return messages
