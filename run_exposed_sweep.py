#!/usr/bin/env python3
"""
Exposed-silo sweep for unknown disease detection.

Varies the number of silos that receive Morven injection across a 10-silo
federation, testing how detection silhouette and round degrade as fewer silos
are exposed.

Conditions: n_exposed ∈ {1, 2, 3, 5} × seeds {42, 43, 44}
  - 1/10 silos exposed  (current default — heavy dilution)
  - 2/10 silos exposed
  - 3/10 silos exposed  (equivalent fraction to the 3-silo experiment: 1/3 ≈ 33%)
  - 5/10 silos exposed  (50% — upper bound)

Results → results/scalability_10silo/exposed_sweep/n{k}/seed{s}/

Usage:
    python run_exposed_sweep.py                   # all conditions, CPU
    python run_exposed_sweep.py --device cuda     # GPU
    python run_exposed_sweep.py --dry-run         # print plan, no execution
    python run_exposed_sweep.py --n-exposed 1 3   # run only k=1,3
    python run_exposed_sweep.py --seeds 42        # single seed
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

N_SILOS     = 10
N_ROUNDS    = 20
EVENTS_PER_SILO = 160
INJECTION_ROUND = 10
INJECTION_PER_ROUND = 8

SWEEP_N_EXPOSED = [1, 2, 3, 5]
SWEEP_SEEDS     = [42, 43, 44]

OUT_BASE = Path("results/scalability_10silo/exposed_sweep")


def run_one(n_exposed: int, seed: int, device: str) -> dict:
    from run_unknown_disease import UnknownDiseaseConfig, run_unknown_disease

    out_dir = OUT_BASE / f"n{n_exposed}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "round_metrics.json"
    if metrics_path.exists():
        print(f"  [skip] n_exposed={n_exposed} seed={seed} (cached)")
        return {"ok": True, "cached": True}

    print(f"\n{'═'*60}")
    print(f"  Exposed-silo sweep — n_exposed={n_exposed}/{N_SILOS}  seed={seed}")
    print(f"  ({n_exposed/N_SILOS:.0%} of silos see Morven, injected at R{INJECTION_ROUND})")
    print(f"{'═'*60}\n")

    inner_dir = str(out_dir / "_inner")
    run_name  = f"exposed_n{n_exposed}_seed{seed}"

    cfg = UnknownDiseaseConfig(
        schedule            = "gaussian",
        n_silos             = N_SILOS,
        n_exposed_silos     = n_exposed,
        events_per_silo     = EVENTS_PER_SILO,
        n_rounds            = N_ROUNDS,
        injection_round     = INJECTION_ROUND,
        injection_per_round = INJECTION_PER_ROUND,
        do_inject           = True,
        local_only          = False,
        per_silo_snaps      = False,
        training_device     = device,
        seed                = seed,
        results_dir         = inner_dir,
        run_name            = run_name,
        snap_rounds         = [5, 10, 12, 15, 18, 20],
    )

    t0 = time.time()
    try:
        result = run_unknown_disease(cfg)
        elapsed = time.time() - t0

        inner = Path(inner_dir) / run_name
        for fname in ("round_metrics.json", "silhouette.json",
                      "umap_evolution.png", "silhouette_curve.png",
                      "prob_unknown_panel.png"):
            src = inner / fname
            if src.exists():
                (out_dir / fname).write_bytes(src.read_bytes())

        # Write compact summary
        summary = result.get("summary", {})
        summary["n_exposed_silos"] = n_exposed
        summary["n_silos"]         = N_SILOS
        summary["seed"]            = seed
        summary["elapsed_s"]       = round(elapsed, 1)

        # Derive scalar fields from silhouette_curve for easy table display
        curve = summary.get("silhouette_curve", [])
        inj_r = summary.get("injection_round", INJECTION_ROUND)
        sil_r15 = next((e["silhouette"] for e in curve if e["round"] == 15), None)
        detect_r = next(
            (e["round"] for e in curve
             if e["round"] > inj_r and e["silhouette"] > 0.3),
            None,
        )
        if sil_r15 is not None:
            summary["silhouette_at_r15"] = round(sil_r15, 4)
        summary["detection_round"] = detect_r

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print(f"\n  Done in {elapsed:.0f}s")
        return {"ok": True, "elapsed": elapsed, "summary": summary}

    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        print(f"  *** FAILED n_exposed={n_exposed} seed={seed}: {exc}")
        traceback.print_exc()
        return {"ok": False, "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--n-exposed",  type=int, nargs="+", default=SWEEP_N_EXPOSED)
    ap.add_argument("--seeds",      type=int, nargs="+", default=SWEEP_SEEDS)
    args = ap.parse_args()

    conditions = [(k, s) for k in sorted(args.n_exposed) for s in sorted(args.seeds)]

    print(f"Exposed-silo sweep: {len(conditions)} conditions")
    print(f"  n_exposed: {args.n_exposed}  ×  seeds: {args.seeds}")
    print(f"  n_silos={N_SILOS}, events/silo={EVENTS_PER_SILO}, "
          f"injection R{INJECTION_ROUND}, device={args.device}")
    print(f"  Output: {OUT_BASE}/\n")

    if args.dry_run:
        for k, s in conditions:
            out = OUT_BASE / f"n{k}" / f"seed{s}"
            cached = (out / "round_metrics.json").exists()
            print(f"  n_exposed={k} seed={s}  →  {out}  {'[cached]' if cached else ''}")
        return

    results = {}
    t_total = time.time()
    for k, s in conditions:
        results[(k, s)] = run_one(k, s, args.device)

    elapsed_total = time.time() - t_total
    ok  = sum(1 for r in results.values() if r.get("ok"))
    print(f"\n{'═'*60}")
    print(f"  Sweep complete — {ok}/{len(conditions)} succeeded in {elapsed_total:.0f}s")

    # Print quick summary table
    print("\nn_exposed | seed | detect_R | sil@R15 | known_acc")
    print("----------|------|----------|---------|----------")
    for k, s in conditions:
        summ_path = OUT_BASE / f"n{k}" / f"seed{s}" / "summary.json"
        if summ_path.exists():
            sm = json.loads(summ_path.read_text())
            dr  = sm.get("detection_round", "—")
            s15 = sm.get("silhouette_at_r15", "—")
            acc = sm.get("final_diag_acc", "—")
            print(f"  {k:>3}     |  {s}  |  {str(dr):>6}  | {str(round(s15,3) if isinstance(s15,float) else s15):>7} | {str(round(acc,3) if isinstance(acc,float) else acc):>8}")


if __name__ == "__main__":
    main()
