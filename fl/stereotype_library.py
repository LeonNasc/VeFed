"""
Federated stereotype library: per-class embedding centroids.

Each silo maintains one centroid per diagnostic class (online mean of CLS
vectors from patient opening statements). After each FL round the orchestrator
aggregates silo centroids via weighted average proportional to cases seen,
producing a global stereotype per class.

The LLM doctor uses these global stereotypes as a classification prior at
inference time — complementing (not replacing) the PrototypeLibrary's
exemplar-based few-shot RAG.

Compared to PrototypeLibrary (individual exemplars, merged by set union):
  Memory    : O(n_classes × dim) ≈ 7 × 768 floats ≈ 21 KB, fixed
  Aggregation: exact weighted average — no exemplar selection heuristics
  Privacy   : only the centroid is shared, not individual patient embeddings
  Coverage  : all events contribute (not just correctly-triaged ones) —
              the centroid represents what the disease actually looks like
"""
from __future__ import annotations

import numpy as np


class StereotypeLibrary:
    """
    Per-silo centroid tracker.

    Usage per round:
        lib.update_from_events(events, model, tokenizer)
        state = lib.get_state()          # send to orchestrator
        lib.set_global(global_centroids) # receive federated result
        nearest = lib.nearest(query_vec) # at inference time
    """

    def __init__(self):
        self._centroids: dict[str, np.ndarray] = {}  # local running mean
        self._counts:    dict[str, int]         = {}  # cases seen per class
        self._global:    dict[str, np.ndarray]  = {}  # set after federation

    # ── Local accumulation ────────────────────────────────────────────────────

    def update(self, label: str, embedding: np.ndarray) -> None:
        """Online mean update for one CLS vector."""
        n = self._counts.get(label, 0)
        emb = embedding.astype(np.float32)
        if n == 0:
            self._centroids[label] = emb
        else:
            self._centroids[label] = (self._centroids[label] * n + emb) / (n + 1)
        self._counts[label] = n + 1

    def update_from_events(self, events: list, model, tokenizer,
                           device: str = "cpu") -> int:
        """
        Embed events and update local centroids.

        All events with a valid ground_truth and at least one patient turn
        are used — including incorrectly-triaged ones. The centroid should
        represent what the disease looks like to patients, not what the model
        already gets right.

        Returns the number of events processed.
        """
        import torch

        texts, labels = [], []
        for ev in events:
            label = ev.ground_truth
            if not label:
                continue
            turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
            if not turns:
                continue
            texts.append(turns[-1])
            labels.append(label)

        if not texts:
            return 0

        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            out = model(
                input_ids      = enc["input_ids"].to(device),
                attention_mask = enc["attention_mask"].to(device),
                output_hidden_states = True,
            )
        cls_vecs = out.hidden_states[-1][:, 0, :].cpu().numpy()

        for vec, label in zip(cls_vecs, labels):
            self.update(label, vec)
        return len(texts)

    # ── Federation ────────────────────────────────────────────────────────────

    def get_state(self) -> dict[str, tuple[np.ndarray, int]]:
        """Return {label: (centroid, count)} for the orchestrator to aggregate."""
        return {lbl: (self._centroids[lbl].copy(), self._counts[lbl])
                for lbl in self._centroids}

    def set_global(self, centroids: dict[str, np.ndarray]) -> None:
        """Store federated (weighted-average) centroids for use at inference."""
        self._global = {k: v.copy() for k, v in centroids.items()}

    # ── Inference ────────────────────────────────────────────────────────────

    def nearest(self, query: np.ndarray, k: int = 5,
                use_global: bool = True) -> list[tuple[str, float]]:
        """
        Return top-k nearest class centroids by cosine similarity.
        Uses global (federated) centroids if available, else local.
        """
        source = self._global if (use_global and self._global) else self._centroids
        if not source:
            return []
        return nearest_centroids(query, source, k=k)

    def coverage(self) -> int:
        """Number of classes with a centroid in the global (or local) store."""
        return len(self._global or self._centroids)


# ── Module-level helpers ──────────────────────────────────────────────────────

def nearest_centroids(
    query: np.ndarray,
    centroids: dict[str, np.ndarray],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Top-k nearest centroids by cosine similarity. Pure numpy, no model needed."""
    if not centroids:
        return []
    q_norm = query / (np.linalg.norm(query) + 1e-9)
    results = [
        (label, float(np.dot(q_norm, c / (np.linalg.norm(c) + 1e-9))))
        for label, c in centroids.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def aggregate_stereotypes(
    states: list[dict[str, tuple[np.ndarray, int]]],
) -> dict[str, np.ndarray]:
    """
    Weighted average of per-silo centroids.

    global_centroid[label] = Σ(centroid_i × count_i) / Σ(count_i)

    Silos that haven't seen a class contribute nothing — no zero-padding or
    imputation. If only one silo has seen a class its centroid passes through
    unchanged, which is correct behaviour for rare diseases.
    """
    accum:  dict[str, np.ndarray] = {}
    counts: dict[str, int]        = {}

    for state in states:
        for label, (centroid, n) in state.items():
            if n == 0:
                continue
            if label not in accum:
                accum[label]  = centroid.astype(np.float64) * n
                counts[label] = n
            else:
                accum[label]  += centroid.astype(np.float64) * n
                counts[label] += n

    return {lbl: (accum[lbl] / counts[lbl]).astype(np.float32)
            for lbl in accum}


def stereotypes_to_prompt_messages(
    nearest: list[tuple[str, float]],
    n_silos: int = 0,
) -> list[dict]:
    """
    Convert nearest-centroid results into an Ollama (user, assistant) pair.

    Injected before the patient's opening statement — same slot as prototype
    few-shot examples — so both systems can run simultaneously and the LLM
    sees the centroid prior before it reads the actual patient.

    n_silos is shown for transparency; pass 0 to omit it.
    """
    import json

    if not nearest:
        return []

    silo_note = f" across {n_silos} silos" if n_silos > 0 else ""
    top_label, top_sim = nearest[0]
    rest = "; ".join(f"{lbl} {sim:.2f}" for lbl, sim in nearest[1:])
    prior_str = f"{top_label} ({top_sim:.2f})"
    if rest:
        prior_str += f"; also: {rest}"

    user_content = (
        f"[Federated stereotype prior{silo_note}] "
        f"Cosine similarity to federated class archetypes: {prior_str}. "
        f"Use as a starting prior — examine the patient before deciding."
    )
    assistant_content = json.dumps({
        "type":  "acknowledged",
        "notes": f"Stereotype prior noted — leading archetype: {top_label} ({top_sim:.2f}).",
    })

    return [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
