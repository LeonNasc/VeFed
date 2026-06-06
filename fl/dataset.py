"""Per-silo on-disk dataset for federated learning."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _event_to_record(ev, round_num: int) -> Optional[dict]:
    """Convert a DiagnosticEvent to a JSON-serialisable dict."""
    patient_turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
    if not patient_turns or not ev.ground_truth:
        return None
    return {
        "text":          patient_turns[-1],
        "label":         ev.ground_truth,
        "gt_disease":    getattr(ev, "gt_disease",  None) or "",
        "gt_severity":   getattr(ev, "gt_severity", None) or "",
        "round":         round_num,
        "severity":      float(ev.severity) if ev.severity is not None else 0.0,
        "is_background": bool(getattr(ev, "is_background", False)),
        "days_infected": int(getattr(ev, "days_infected", 0) or 0),
        "action":        ev.action.name if ev.action is not None else None,
        "num_turns":     int(ev.num_turns) if ev.num_turns else 0,
    }


class SiloDataset:
    """
    On-disk JSONL dataset for one FL silo.

    Files
    -----
    train.jsonl   — training split  ((1 - holdout_fraction) × events)
    holdout.jsonl — stable test split, never used for training

    Sampling
    --------
    sample_train(cap) returns a stratified-by-label sample of up to `cap`
    records.  When len(train) ≤ cap all records are returned, so early rounds
    train on all available data.  Once the dataset exceeds cap the sample stays
    representative (proportional across labels) and compute stays bounded.

    Crash resilience
    ----------------
    Both files are appended to, so partial runs survive a process kill or
    session crash.  Call clear() at the start of a fresh run to start over.
    """

    def __init__(
        self,
        dataset_dir: "Path | str",
        silo_id:    int,
        holdout_fraction: float = 0.2,
        seed:       int   = 42,
    ):
        self._dir = Path(dataset_dir) / f"silo_{silo_id}"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._train_path   = self._dir / "train.jsonl"
        self._holdout_path = self._dir / "holdout.jsonl"
        self._holdout_frac = holdout_fraction
        self._rng          = random.Random(seed + silo_id * 1000)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, events: list, round_num: int = 0) -> tuple[int, int]:
        """
        Serialise events and append to train / holdout.
        Returns (n_train_added, n_holdout_added).
        """
        n_train = n_holdout = 0
        with self._train_path.open("a") as ft, self._holdout_path.open("a") as fh:
            for ev in events:
                rec = _event_to_record(ev, round_num)
                if rec is None:
                    continue
                if self._rng.random() < self._holdout_frac:
                    fh.write(json.dumps(rec) + "\n")
                    n_holdout += 1
                else:
                    ft.write(json.dumps(rec) + "\n")
                    n_train += 1
        return n_train, n_holdout

    # ── Read ──────────────────────────────────────────────────────────────────

    def sample_train(self, cap: Optional[int] = None) -> list[dict]:
        """
        Stratified-by-label sample from the training split.

        cap=None  → return all records.
        cap=N     → proportional allocation across labels (at least 1 per
                    label), then random sample within each stratum.  Total ≤ N.
                    Returns all records when len(train) ≤ N (early rounds).
        """
        records = self._load(self._train_path)
        if not records or cap is None or len(records) <= cap:
            return list(records)

        by_label: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_label[r["label"]].append(r)

        # Proportional allocation with at-least-1 per label
        alloc: dict[str, int] = {
            lbl: max(1, round(cap * len(items) / len(records)))
            for lbl, items in by_label.items()
        }

        # Trim overshoot caused by rounding
        total = sum(alloc.values())
        for lbl in sorted(alloc, key=alloc.__getitem__, reverse=True):
            if total <= cap:
                break
            if alloc[lbl] > 1:
                alloc[lbl] -= 1
                total -= 1

        sampled: list[dict] = []
        for lbl, n in alloc.items():
            items = by_label[lbl]
            sampled.extend(self._rng.sample(items, min(n, len(items))))

        self._rng.shuffle(sampled)
        return sampled

    def load_holdout(self) -> list[dict]:
        return self._load(self._holdout_path)

    def size(self) -> tuple[int, int]:
        """(n_train, n_holdout)"""
        return self._count(self._train_path), self._count(self._holdout_path)

    def clear(self) -> None:
        """Delete both files — call at the start of a fresh run."""
        for p in (self._train_path, self._holdout_path):
            if p.exists():
                p.unlink()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def _count(self, path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for ln in path.read_text().splitlines() if ln.strip())
