"""
InteractionLogger — writes DiagnosticEvents to a JSONL file for dataset use.

Each line is a self-contained JSON record with all clinical fields plus the
full patient-doctor conversation, making it suitable for LLM fine-tuning or
triage-classification benchmarks.

Usage:
    logger = InteractionLogger()          # auto-names file under interactions/
    world.attach_interaction_logger(logger)
    # events are logged automatically as the clinic queue is processed
"""
from __future__ import annotations
import json
import threading
from datetime import datetime
from pathlib import Path

from simulation.models import DiagnosticEvent


def _serialise(event: DiagnosticEvent, day: int, seed: int) -> dict:
    return {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sim_day":   day,
            "sim_seed":  seed,
        },
        "agent_id":      event.agent_id,
        "severity":      round(event.severity, 4),
        "symptoms":      round(event.symptoms, 4),
        "days_infected": event.days_infected,
        "personality":   event.personality.value if event.personality else None,
        "ground_truth":  event.ground_truth,
        "conversation":  event.conversation,      # [{role, text}, …]
        "action":        event.action.value if event.action else None,
        "oracle_label":  event.oracle_label,
        "diagnosis":     event.diagnosis,
        "notes":         event.notes,
    }


class InteractionLogger:
    """
    Append-mode JSONL writer.  Thread-safe.
    One JSON object per line — easy to stream with pandas or HuggingFace datasets.
    """
    def __init__(self, path: Path | str | None = None):
        if path is None:
            Path("interactions").mkdir(exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path("interactions") / f"interactions_{ts}.jsonl"
        self._path  = Path(path)
        self._fh    = self._path.open("a", encoding="utf-8", buffering=1)
        self._lock  = threading.Lock()

    def log(self, event: DiagnosticEvent, day: int, seed: int = 0) -> None:
        record = _serialise(event, day, seed)
        line   = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        self._fh.close()

    @property
    def path(self) -> Path:
        return self._path
