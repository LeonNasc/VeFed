#!/usr/bin/env python3
"""
Redesigned 6-way ablation: (Centralized, Local-only, Federated) × (IID, non-IID)

Key insight: centralized and local-only don't need their own simulations.
  - FL IID/non-IID runs collect per-silo JSONL datasets on disk.
  - Local-only IID/non-IID re-train from those same datasets (no Ollama, ~10 min each).
  - Centralized non-IID trains from datasets/merged_noniid/ (pre-built, no Ollama).
  - Centralized IID uses run_static_training (sim + Ollama once, then trains).

Stages
------
  1. Federated IID       — full sim+FL; saves dataset dir as ablation_state.json#iid
  2. Federated non-IID   — full sim+FL; saves dataset dir as ablation_state.json#noniid
  3. Centralized non-IID — static train on datasets/merged_noniid/ (fast, no Ollama)
  4. Centralized IID     — run_static_training with IID preset (sim+Ollama → centralized)
  5. Local-only IID      — static train per-silo from stage 1 dataset (fast, no Ollama)
  6. Local-only non-IID  — static train per-silo from stage 2 dataset (fast, no Ollama)

Prerequisite
------------
  datasets/merged_noniid/ must exist.  Build it once with:
      python -c "
from pathlib import Path
srcs = ['datasets/20260610_225808', 'datasets/20260611_014946']
out  = Path('datasets/merged_noniid')
out.mkdir(exist_ok=True)
for s in range(3):
    d = out / f'silo_{s}'
    d.mkdir(exist_ok=True)
    for split in ('train.jsonl', 'holdout.jsonl'):
        lines = []
        for src in srcs:
            p = Path(src) / f'silo_{s}' / split
            if p.exists(): lines += [l for l in p.read_text().splitlines() if l.strip()]
        (d / split).write_text(chr(10).join(lines) + chr(10))
"

Usage
-----
    python run_ablation.py                               # full run
    python run_ablation.py --start 3                     # skip stages 1-2
    nohup python run_ablation.py > ablation.log 2>&1 &   # overnight
    tail -f ablation.log
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

MERGED_NONIID = Path("datasets/merged_noniid")


def _state_file(rep: int) -> Path:
    return Path(f"datasets/.ablation_state_rep{rep}.json")


def _banner(title: str) -> None:
    line = "═" * 66
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _gc() -> None:
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _newest_dataset_dir() -> Path | None:
    """Return the most recently modified silo dataset dir (excluding merged_noniid)."""
    base = Path("datasets")
    candidates = [
        d for d in base.iterdir()
        if d.is_dir() and d.name not in ("merged_noniid",) and not d.name.startswith(".")
        and (d / "silo_0").exists()
    ]
    return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


def _load_state(rep: int) -> dict:
    f = _state_file(rep)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {}


def _save_state(rep: int, key: str, value: str) -> None:
    state = _load_state(rep)
    state[key] = value
    _state_file(rep).write_text(json.dumps(state, indent=2))


# ── Stage runners ─────────────────────────────────────────────────────────────

def _run_fl(mode: str, preset: str, run_name: str, label: str, seed: int = 42) -> dict:
    """Full simulation + FL round loop. Returns dict with ok/elapsed/dataset_dir."""
    _banner(label)
    print(f"  preset={preset}  mode={mode}  run_name={run_name}  seed={seed}")
    print(f"  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    from run import _make_presets, _make_fl_cfg
    cfg          = _make_presets()[preset]
    cfg.mode     = mode
    cfg.seed     = seed
    cfg.wandb_run_name = run_name
    fl_cfg       = _make_fl_cfg(cfg)
    fl_cfg.batch_size = 4  # halve gradient-buffer peak to avoid OOM-killing llama-server

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

        dataset_dir = _newest_dataset_dir()
        elapsed = time.time() - t0
        print(f"\n  Finished in {_hms(elapsed)}", flush=True)
        print(f"  Dataset: {dataset_dir}", flush=True)
        return {"label": label, "elapsed": elapsed, "ok": True,
                "dataset_dir": str(dataset_dir) if dataset_dir else None}

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  *** FAILED after {_hms(elapsed)}: {exc} ***", flush=True)
        traceback.print_exc()
        return {"label": label, "elapsed": elapsed, "ok": False, "error": str(exc),
                "dataset_dir": None}
    finally:
        _gc()


def _run_static_centralized(preset: str, run_name: str, label: str, seed: int = 42) -> dict:
    """Run world simulation + centralized static training (Ollama required)."""
    _banner(label)
    print(f"  preset={preset}  mode=static_centralized  run_name={run_name}  seed={seed}")
    print(f"  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    from run import _make_presets, _make_fl_cfg
    cfg = _make_presets()[preset]
    cfg.mode = "static_centralized"
    cfg.seed = seed
    cfg.wandb_run_name = run_name
    fl_cfg = _make_fl_cfg(cfg)
    fl_cfg.batch_size   = 4
    fl_cfg.local_epochs = max(fl_cfg.local_epochs * 5, 15)

    t0 = time.time()
    try:
        from fl.train import run_static_training
        run_static_training(fl_cfg, mode="centralized")
        elapsed = time.time() - t0
        print(f"\n  Finished in {_hms(elapsed)}", flush=True)
        return {"label": label, "elapsed": elapsed, "ok": True}
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  *** FAILED after {_hms(elapsed)}: {exc} ***", flush=True)
        traceback.print_exc()
        return {"label": label, "elapsed": elapsed, "ok": False, "error": str(exc)}
    finally:
        _gc()


def _run_from_dataset(dataset_dir: str, mode: str, run_name: str, label: str,
                      preset: str = "ablation-iid-5s", seed: int = 42) -> dict:
    """Fast static training on an existing JSONL dataset (no Ollama)."""
    _banner(label)
    print(f"  dataset={dataset_dir}  mode={mode}  run_name={run_name}  seed={seed}")
    print(f"  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    from run import _make_presets, _make_fl_cfg
    cfg = _make_presets()[preset]
    cfg.seed = seed
    cfg.wandb_run_name = run_name
    cfg.use_ollama = False
    fl_cfg = _make_fl_cfg(cfg)

    t0 = time.time()
    try:
        from fl.train import run_training_from_dataset
        run_training_from_dataset(dataset_dir, mode=mode, cfg=fl_cfg)
        elapsed = time.time() - t0
        print(f"\n  Finished in {_hms(elapsed)}", flush=True)
        return {"label": label, "elapsed": elapsed, "ok": True}
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  *** FAILED after {_hms(elapsed)}: {exc} ***", flush=True)
        traceback.print_exc()
        return {"label": label, "elapsed": elapsed, "ok": False, "error": str(exc)}
    finally:
        _gc()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, metavar="N",
                        help="Start from stage N (1-6, default 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default 42; vary per replica)")
    parser.add_argument("--rep", type=int, default=0,
                        help="Replica index — used for run naming and state file (default 0)")
    args = parser.parse_args()

    seed = args.seed
    rep  = args.rep

    if not MERGED_NONIID.exists():
        print(f"ERROR: {MERGED_NONIID} not found. Run the merge script first.", flush=True)
        sys.exit(1)

    # Pre-flight: refuse to start if available RAM < 4 GB (would cause swap thrash)
    try:
        with open("/proc/meminfo") as f:
            meminfo = {l.split()[0].rstrip(":"): int(l.split()[1])
                       for l in f if len(l.split()) >= 2}
        avail_gb = meminfo.get("MemAvailable", 0) / (1024 ** 2)
        if avail_gb < 4.0:
            print(f"ERROR: only {avail_gb:.1f} GB RAM available (need ≥ 4 GB).", flush=True)
            print(f"  Restart Ollama to free memory: sudo systemctl restart ollama", flush=True)
            sys.exit(1)
        print(f"  RAM available: {avail_gb:.1f} GB ✓", flush=True)
    except Exception:
        pass  # don't block on unexpected /proc layout

    Path("reports").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    tag = f"rep{rep}_s{seed}"

    _banner(f"FedWorld Ablation — rep={rep}  seed={seed}")
    print(f"  {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Stages 1-2: FL (sim+training, Ollama required)")
    print(f"  Stage  3:   Centralized non-IID from datasets/merged_noniid/ (fast)")
    print(f"  Stage  4:   Centralized IID via run_static_training (sim+Ollama)")
    print(f"  Stages 5-6: Local-only from FL datasets (fast, no Ollama)\n", flush=True)

    state      = _load_state(rep)
    t_total    = time.time()
    results    = []
    start      = args.start

    # ── Stage 1: Federated IID ────────────────────────────────────────────────
    if start <= 1:
        r = _run_fl("fl", "ablation-iid-5s", f"{ts}_{tag}_fl_iid",
                    "1/6 — Federated IID  (3 silos · 50/50 flu+pneu · 75 agents · 20 rounds)",
                    seed=seed)
        results.append(r)
        if r["dataset_dir"]:
            _save_state(rep, "iid_dataset_dir", r["dataset_dir"])
            state["iid_dataset_dir"] = r["dataset_dir"]
    else:
        results.append({"label": "1/6 — Federated IID", "elapsed": 0, "ok": True,
                         "skipped": True})

    # ── Stage 2: Federated non-IID ───────────────────────────────────────────
    if start <= 2:
        r = _run_fl("fl", "ablation-noniid-5s", f"{ts}_{tag}_fl_noniid",
                    "2/6 — Federated non-IID  (3 silos · flu→pneu gradient · 75 agents · 20 rounds)",
                    seed=seed)
        results.append(r)
        if r["dataset_dir"]:
            _save_state(rep, "noniid_dataset_dir", r["dataset_dir"])
            state["noniid_dataset_dir"] = r["dataset_dir"]
    else:
        results.append({"label": "2/6 — Federated non-IID", "elapsed": 0, "ok": True,
                         "skipped": True})

    # ── Stage 3: Centralized non-IID from merged dataset ─────────────────────
    if start <= 3:
        r = _run_from_dataset(
            str(MERGED_NONIID), mode="centralized",
            run_name=f"{ts}_{tag}_cen_noniid",
            label="3/6 — Centralized non-IID  (from merged_noniid dataset · no Ollama)",
            preset="ablation-noniid-5s",
            seed=seed,
        )
        results.append(r)
    else:
        results.append({"label": "3/6 — Centralized non-IID", "elapsed": 0, "ok": True,
                         "skipped": True})

    # ── Stage 4: Centralized IID (run world + static centralized training) ────
    if start <= 4:
        r = _run_static_centralized(
            "ablation-iid-5s",
            run_name=f"{ts}_{tag}_cen_iid",
            label="4/6 — Centralized IID  (world sim + Ollama → centralized train)",
            seed=seed,
        )
        results.append(r)
    else:
        results.append({"label": "4/6 — Centralized IID", "elapsed": 0, "ok": True,
                         "skipped": True})

    # ── Resolve FL dataset dirs for stages 5-6 ───────────────────────────────
    iid_dir    = state.get("iid_dataset_dir")
    noniid_dir = state.get("noniid_dataset_dir")

    if start >= 5 and (not iid_dir or not noniid_dir):
        state      = _load_state(rep)
        iid_dir    = state.get("iid_dataset_dir",    iid_dir)
        noniid_dir = state.get("noniid_dataset_dir", noniid_dir)

    # ── Stage 5: Local-only IID ───────────────────────────────────────────────
    if start <= 5:
        if iid_dir:
            r = _run_from_dataset(
                iid_dir, mode="local",
                run_name=f"{ts}_{tag}_local_iid",
                label="5/6 — Local-only IID  (per-silo from FL IID dataset · no Ollama)",
                preset="ablation-iid-5s",
                seed=seed,
            )
        else:
            print("  WARN: iid_dataset_dir not found — stage 5 skipped.", flush=True)
            r = {"label": "5/6 — Local-only IID", "elapsed": 0, "ok": False,
                 "error": "iid_dataset_dir not found in state"}
        results.append(r)
    else:
        results.append({"label": "5/6 — Local-only IID", "elapsed": 0, "ok": True,
                         "skipped": True})

    # ── Stage 6: Local-only non-IID ──────────────────────────────────────────
    if start <= 6:
        if noniid_dir:
            r = _run_from_dataset(
                noniid_dir, mode="local",
                run_name=f"{ts}_{tag}_local_noniid",
                label="6/6 — Local-only non-IID  (per-silo from FL non-IID dataset · no Ollama)",
                preset="ablation-noniid-5s",
                seed=seed,
            )
        else:
            print("  WARN: noniid_dataset_dir not found — stage 6 skipped.", flush=True)
            r = {"label": "6/6 — Local-only non-IID", "elapsed": 0, "ok": False,
                 "error": "noniid_dataset_dir not found in state"}
        results.append(r)
    else:
        results.append({"label": "6/6 — Local-only non-IID", "elapsed": 0, "ok": True,
                         "skipped": True})

    # ── Final summary ─────────────────────────────────────────────────────────
    _banner("ABLATION COMPLETE")
    print(f"  Wall time: {_hms(time.time() - t_total)}")
    print(f"  Finished:  {datetime.now().isoformat(timespec='seconds')}\n")
    all_ok = True
    for r in results:
        if r.get("skipped"):
            icon = "↷"
            t    = "skipped"
        else:
            icon = "✓" if r["ok"] else "✗"
            t    = _hms(r["elapsed"])
        print(f"  {icon}  {r['label']:<60}  {t}")
        if not r.get("ok") and not r.get("skipped"):
            all_ok = False
            print(f"       error: {r.get('error', '?')}")

    print(f"\n  Reports:   {Path('reports').resolve()}/")
    print(f"  W&B sync:  wandb sync wandb/\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
