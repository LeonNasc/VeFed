#!/usr/bin/env python3
"""
Post-hoc cross-silo holdout evaluation.

Evaluates the final FL model (and optionally a local-only comparison) on the
pooled holdout from all silos, without re-running training.

Usage
-----
    # Evaluate a federated run
    python eval_cross_silo.py datasets/20260614_222108

    # Evaluate two runs (fed + local-only) and compare
    python eval_cross_silo.py datasets/20260614_222108 --local-dir datasets/20260614_local

    # Evaluate the most recent run in a parent dir
    python eval_cross_silo.py datasets/noniid_10silo
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _load_holdout_records(dataset_dir: Path) -> list[dict]:
    """Pool holdout.jsonl from all silo_* subdirectories."""
    records: list[dict] = []
    for silo_dir in sorted(dataset_dir.glob("silo_*")):
        h = silo_dir / "holdout.jsonl"
        if h.exists():
            for line in h.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def _newest_run_dir(parent: Path) -> Path | None:
    """Return the most-recently modified timestamped subdir, if any."""
    subdirs = sorted(
        [d for d in parent.iterdir() if d.is_dir() and not d.name.startswith("silo")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def _resolve_dataset_dir(path: Path) -> Path:
    """If path has timestamped subdirs, pick the newest; otherwise use path directly."""
    if not (path / "silo_0").exists():
        newest = _newest_run_dir(path)
        if newest and (newest / "silo_0").exists():
            return newest
    return path


def _evaluate_on_holdout(weights_npz: Path, records: list[dict],
                         device: str = "cpu", label_space: str = "disease") -> dict:
    """Load weights, build model, evaluate on records. Returns acc metrics."""
    from fl.lora import LoRAConfig, set_lora_weights, npz_to_weights
    from fl.learner import FLLearner
    import torch

    # FLLearner.__init__ sets the correct num_labels for the label_space
    learner = FLLearner(LoRAConfig(), device=device, label_space=label_space)
    weights = npz_to_weights(np.load(str(weights_npz), allow_pickle=False))
    set_lora_weights(learner.model, weights)   # builds model with correct labels
    result = learner.evaluate(records)
    learner.release()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _disease_breakdown(records: list[dict]) -> dict:
    counts = Counter(r.get("gt_disease", "?") for r in records)
    total  = len(records)
    return {d: {"n": n, "pct": round(100 * n / total, 1)} for d, n in counts.most_common()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc cross-silo holdout evaluation")
    ap.add_argument("dataset_dir",  help="Dataset directory (or parent with timestamped subdirs)")
    ap.add_argument("--local-dir",  help="Local-only dataset dir for comparison (optional)")
    ap.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--write-json", action="store_true",
                    help="Write cross_silo_eval.json alongside the dataset dir")
    args = ap.parse_args()

    fed_dir = _resolve_dataset_dir(Path(args.dataset_dir))
    print(f"Fed dataset dir : {fed_dir}")

    records = _load_holdout_records(fed_dir)
    if not records:
        print("ERROR: no holdout.jsonl found under", fed_dir)
        return

    print(f"\nPooled holdout  : {len(records)} records")
    breakdown = _disease_breakdown(records)
    for disease, info in breakdown.items():
        print(f"  {disease:<30} n={info['n']:>4}  ({info['pct']:.1f}%)")

    # Per-silo breakdown
    print("\nPer-silo holdout disease distribution:")
    for silo_dir in sorted(fed_dir.glob("silo_*")):
        h = silo_dir / "holdout.jsonl"
        if not h.exists():
            continue
        recs = [json.loads(l) for l in h.read_text().splitlines() if l.strip()]
        counts = Counter(r.get("gt_disease", "?") for r in recs)
        top2 = dict(counts.most_common(2))
        print(f"  {silo_dir.name}: n={len(recs)}  {top2}")

    # Federated model evaluation
    fed_weights_npz = fed_dir / "weights" / "fl_final.npz"
    if not fed_weights_npz.exists():
        print(f"\nWARN: {fed_weights_npz} not found — cannot evaluate federated model")
        fed_result = {}
    else:
        print(f"\nEvaluating federated model ({fed_weights_npz.name}) on pooled holdout…")
        fed_result = _evaluate_on_holdout(fed_weights_npz, records, device=args.device)
        print(f"  Fed diag acc  : {fed_result.get('diag_acc', float('nan')):.3f}")
        print(f"  Fed triage acc: {fed_result.get('triage_acc', float('nan')):.3f}")

    # Local-only comparison (optional)
    local_result = {}
    if args.local_dir:
        local_dir = _resolve_dataset_dir(Path(args.local_dir))
        print(f"\nLocal dataset dir: {local_dir}")
        local_records = _load_holdout_records(local_dir)
        # Also evaluate per-silo local models if available
        # (For now, just evaluate a "local" fl_final which would be silo_0's trained model)
        local_npz = local_dir / "weights" / "fl_final.npz"
        if local_npz.exists():
            print(f"Evaluating local model on cross-silo holdout (pooled from fed dir)…")
            local_result = _evaluate_on_holdout(local_npz, records, device=args.device)
            print(f"  Local diag acc  : {local_result.get('diag_acc', float('nan')):.3f}")

    # Summary
    fed_acc   = fed_result.get("diag_acc", float("nan"))
    local_acc = local_result.get("diag_acc", float("nan"))

    print("\n" + "─" * 50)
    print("CROSS-SILO HOLDOUT SUMMARY")
    print("─" * 50)
    if fed_acc == fed_acc:
        print(f"Federated global diag acc : {fed_acc:.3f}")
    if local_acc == local_acc:
        print(f"Local (silo 0) diag acc   : {local_acc:.3f}")
        if fed_acc == fed_acc:
            print(f"Federated gain            : {fed_acc - local_acc:+.3f}")
    print("─" * 50)

    if args.write_json:
        result = {
            "n_holdout":       len(records),
            "disease_dist":    breakdown,
            "fed_diag_acc":    None if fed_acc != fed_acc else fed_acc,
            "fed_triage_acc":  fed_result.get("triage_acc"),
            "local_diag_acc":  None if local_acc != local_acc else local_acc,
        }
        out = fed_dir / "cross_silo_eval.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
