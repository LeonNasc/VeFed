"""
Result loader for FL experiments.

load_results(root)  →  list[dict], one dict per run.

Standard fields on every run dict
──────────────────────────────────
  path              str          absolute path of the run directory
  mode              str          "generate" | "federated" | "local_only" | "centralized"
  n_silos           int
  n_rounds          int
  seed              int | None
  final_diag_acc    float | None
  final_triage_acc  float | None
  wall_seconds      float | None
  rounds            list[dict]   per-round rows (see below)

Extra fields inferred from path for scheduled runs  (mode == "generate")
  generator         str          template | phrase_library
  schedule          str          flat | gaussian | ramp | burst | parabola | …
  distribution      str          iid | noniid

Extra fields from summary for SIR-based runs
  experiment        str          wandb_run_name or run_id timestamp
  progressions      list[str]

Per-round row keys (union across formats — absent key means not recorded)
  round             int
  agg_diag_acc      float | None
  agg_triage_acc    float | None
  mean_loss         float | None
  n_silos_trained   int          (SIR runs)
  silo_diag         list         per-silo diagnostic accuracy
  silo_triage       list         per-silo triage accuracy
  sir_i             list[int]    infected count per silo (SIR runs)
  n_reveal          int          events revealed this round (scheduled runs)
  cursors           list[int]    cumulative event counts (scheduled runs)
  train_sizes       list[int]    (scheduled runs)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(x: Any) -> Any:
    """Convert float NaN/inf to None so rows are JSON-safe."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _safe_dict(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _infer_scheduled_meta(run_dir: Path, summary: dict) -> dict:
    """
    Scheduled / fast-ablation layout:
      {root}/{generator}/{schedule}/{dist}/{tag}

    Seed is encoded in the tag name as …_seed{N}.
    """
    parts = run_dir.parts
    meta: dict = {
        "generator":    parts[-4] if len(parts) >= 4 else "",
        "schedule":     parts[-3] if len(parts) >= 3 else summary.get("schedule", ""),
        "distribution": parts[-2] if len(parts) >= 2 else "",
    }
    tag = run_dir.name
    if "seed" in tag:
        try:
            meta["seed"] = int(tag.rsplit("seed", 1)[-1])
        except ValueError:
            pass
    return meta


def _infer_sir_meta(run_dir: Path, summary: dict) -> dict:
    return {
        "experiment":  summary.get("experiment", run_dir.name),
        "progressions": summary.get("progressions", []),
    }


def _normalise_rounds(rounds: list[dict]) -> list[dict]:
    """Safe-cast all float values in each round row."""
    return [_safe_dict(r) for r in rounds]


# ── Public API ────────────────────────────────────────────────────────────────

def load_results(root: str | Path) -> list[dict]:
    """
    Walk root recursively for (summary.json, round_metrics.json) pairs.
    Returns one dict per run, sorted by path.

    Handles both:
      • Scheduled / fast-ablation layout  (summary["mode"] == "generate")
      • SIR-based FL layout               (summary["mode"] in {"federated",
                                           "local_only", "centralized"})
    """
    root = Path(root)
    runs: list[dict] = []

    for summary_path in sorted(root.rglob("summary.json")):
        run_dir      = summary_path.parent
        metrics_path = run_dir / "round_metrics.json"
        if not metrics_path.exists():
            continue

        try:
            summary = _load_json(summary_path)
            rounds  = _load_json(metrics_path)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue

        mode = summary.get("mode", "")

        run: dict = {
            "path":             str(run_dir.resolve()),
            "mode":             mode,
            "n_silos":          int(summary.get("n_silos", 0)),
            "n_rounds":         int(summary.get("n_rounds", len(rounds))),
            "seed":             summary.get("seed"),
            "final_diag_acc":   _safe(summary.get("final_diag_acc")),
            "final_triage_acc": _safe(summary.get("final_triage_acc")),
            "wall_seconds":     _safe(summary.get("wall_seconds")),
            "rounds":           _normalise_rounds(rounds),
        }

        if mode == "generate":
            run.update(_infer_scheduled_meta(run_dir, summary))
        else:
            run.update(_infer_sir_meta(run_dir, summary))

        runs.append(run)

    return runs


def filter_runs(
    runs:   list[dict],
    **kwargs,
) -> list[dict]:
    """
    Filter runs by exact field match.

    Example::

        iid_template = filter_runs(runs, generator="template", distribution="iid")
    """
    out = runs
    for key, val in kwargs.items():
        out = [r for r in out if r.get(key) == val]
    return out


def group_runs(runs: list[dict], by: str) -> dict[str, list[dict]]:
    """
    Partition runs into {value → [run, …]} by a metadata key.
    Order of insertion matches sorted key order.
    """
    groups: dict[str, list[dict]] = {}
    for r in runs:
        k = str(r.get(by, "?"))
        groups.setdefault(k, []).append(r)
    return dict(sorted(groups.items()))
