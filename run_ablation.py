#!/usr/bin/env python3
"""
Overnight 4-way ablation: Centralized → Local-only → Fed-IID → Fed-nonIID

Design rationale
----------------
All four runs share identical world parameters (5 silos, 300 agents each,
horizon=200 sim-days, beta=2.0, no Ollama) so the only variable is the
training mode.  The ordering deliberately increases federation complexity:

  1. Centralized  — oracle upper bound: all events from 5 non-IID silos
                    pooled into one shared model.
  2. Local-only   — lower bound: each silo trains privately, no sharing.
  3. Fed-IID      — federation where every silo sees the same 50/50 disease
                    mix; the easy case.  Should approach centralized.
  4. Fed-nonIID   — federation across the full flu→pneumonia gradient; the
                    hard case.  The gap vs. local-only is the FL payoff.

Key metrics to inspect on waking up
-------------------------------------
  triage_acc   — severity classification (mild/moderate/severe/non-infectious).
                 Can be inflated by non-infectious cases if the model learns a
                 trivial heuristic.  Compare against danger_rate.
  danger_rate  — fraction of truly-severe cases the model missed.  This is the
                 safety-critical metric; low is good.  If triage_acc is high
                 but danger_rate is also high, world noise is masking failure.
  diag_acc     — disease identification (flu vs pneumonia vs non-infectious).
                 Meaningless in single-disease runs; the signal here.
  fl_gain      — (fed − local) triage_acc for fed runs; 0 for local-only.
                 Positive and growing means federation is adding value.

Usage
-----
    python run_ablation.py                          # foreground
    nohup python run_ablation.py > ablation.log 2>&1 &   # overnight
    tail -f ablation.log                            # follow progress
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


def _banner(title: str) -> None:
    line = "═" * 66
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _run_one(
    mode:     str,
    preset:   str,
    run_name: str,
    label:    str,
) -> dict | None:
    """
    Run one training mode and return a summary dict, or None on failure.

    Failures are caught and printed so the remaining runs still execute.
    """
    _banner(label)
    print(f"  preset={preset}  mode={mode}  run_name={run_name}")
    print(f"  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    from run import _make_presets, _make_fl_cfg
    cfg          = _make_presets()[preset]
    cfg.mode     = mode
    cfg.wandb_run_name = run_name
    fl_cfg       = _make_fl_cfg(cfg)

    t0 = time.time()
    try:
        if mode == "centralized":
            from fl.train import run_centralized_training
            run_centralized_training(fl_cfg)
        elif mode == "local_only":
            from fl.train import run_local_only_training
            run_local_only_training(fl_cfg)
        elif mode == "fl":
            from fl.train import run_federated_training
            run_federated_training(fl_cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        elapsed = time.time() - t0
        print(f"\n  Finished in {_hms(elapsed)}", flush=True)
        return {"label": label, "elapsed": elapsed, "ok": True}

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  *** FAILED after {_hms(elapsed)}: {exc} ***", flush=True)
        traceback.print_exc()
        return {"label": label, "elapsed": elapsed, "ok": False, "error": str(exc)}


def main() -> None:
    Path("reports").mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # Each tuple: (mode, preset, wandb_run_name, human_label)
    RUNS = [
        (
            "centralized",
            "ablation-noniid-5s",
            f"{ts}_cen_noniid",
            "1/4 — Centralized oracle   (5 non-IID silos · cases pooled)",
        ),
        (
            "local_only",
            "ablation-noniid-5s",
            f"{ts}_local_noniid",
            "2/4 — Local-only baseline  (5 non-IID silos · no sharing)",
        ),
        (
            "fl",
            "ablation-iid-5s",
            f"{ts}_fl_iid",
            "3/4 — Federated IID        (5 silos · 50/50 mix each)",
        ),
        (
            "fl",
            "ablation-noniid-5s",
            f"{ts}_fl_noniid",
            "4/4 — Federated non-IID    (5 silos · flu→pneumo gradient)",
        ),
    ]

    _banner("FedWorld Overnight Ablation")
    print(f"  {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Runs: {len(RUNS)}  |  training_device=cpu  |  wandb=offline")
    print(f"  Reports → reports/  |  W&B sync: wandb sync wandb/\n")
    print(f"  Metrics to watch on waking up:")
    print(f"    danger_rate  — missed severe cases (primary safety metric)")
    print(f"    triage_acc   — severity accuracy (can be noise-inflated)")
    print(f"    diag_acc     — disease ID accuracy (key non-IID signal)")
    print(f"    fl_gain      — (fed − local) triage improvement\n", flush=True)

    t_total = time.time()
    results = []
    for mode, preset, run_name, label in RUNS:
        r = _run_one(mode, preset, run_name, label)
        results.append(r or {"label": label, "elapsed": 0, "ok": False, "error": "unknown"})

    # ── Final summary ─────────────────────────────────────────────────────────
    _banner("ABLATION COMPLETE")
    print(f"  Wall time: {_hms(time.time() - t_total)}")
    print(f"  Finished:  {datetime.now().isoformat(timespec='seconds')}\n")
    all_ok = True
    for r in results:
        icon = "✓" if r["ok"] else "✗"
        t    = _hms(r["elapsed"])
        print(f"  {icon}  {r['label']:<52}  {t}")
        if not r["ok"]:
            all_ok = False
            print(f"       error: {r.get('error', '?')}")

    print(f"\n  Reports:   {Path('reports').resolve()}/")
    print(f"  W&B sync:  wandb sync wandb/\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
