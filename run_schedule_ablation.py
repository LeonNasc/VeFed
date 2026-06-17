"""
Schedule ablation entry point.

Quick smoke test (generate mode, phrase library):
  python run_schedule_ablation.py --schedule gaussian --mode generate

Replay on existing rep1 FL IID dataset:
  python run_schedule_ablation.py --schedule burst --mode replay \
      --dataset-dir datasets/20260612_084630

Run all five schedules sequentially:
  python run_schedule_ablation.py --all-schedules --mode generate

Compare two schedules on existing non-IID data:
  python run_schedule_ablation.py --all-schedules --mode replay \
      --dataset-dir datasets/20260612_104939 --rounds 20
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fl.scheduled_runner import ScheduledFLConfig, run_scheduled_fl
from fl.schedules import SCHEDULES


def _build_noniid_dists(n_silos: int) -> list:
    """Simple non-IID: silo i is biased toward influenza or pneumonia."""
    base = [
        ("influenza",      "mild",     "neutral", 1.0),
        ("influenza",      "moderate", "neutral", 1.0),
        ("influenza",      "severe",   "neutral", 0.5),
        ("pneumonia",      "mild",     "neutral", 1.0),
        ("pneumonia",      "moderate", "neutral", 1.0),
        ("pneumonia",      "severe",   "neutral", 0.5),
        ("non_infectious", "mild",     "anxious", 0.5),
    ]
    dists = []
    for i in range(n_silos):
        if i % 2 == 0:
            # influenza-heavy
            d = [(dis, sev, per, w * (3.0 if dis == "influenza" else 1.0))
                 for dis, sev, per, w in base]
        else:
            # pneumonia-heavy
            d = [(dis, sev, per, w * (3.0 if dis == "pneumonia" else 1.0))
                 for dis, sev, per, w in base]
        dists.append(d)
    return dists


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", default="gaussian",
                   choices=list(SCHEDULES), help="Schedule name")
    p.add_argument("--all-schedules", action="store_true",
                   help="Run all five schedules sequentially")
    p.add_argument("--mode", default="generate", choices=["generate", "replay"])
    p.add_argument("--dataset-dir", default="",
                   help="Path to existing dataset dir (replay mode)")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--n-silos", type=int, default=3)
    p.add_argument("--events-per-silo", type=int, default=200,
                   help="Pool size per silo (generate mode)")
    p.add_argument("--noniid", action="store_true",
                   help="Use non-IID silo distributions (generate mode)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results/scheduled")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    schedules_to_run = list(SCHEDULES.keys()) if args.all_schedules else [args.schedule]
    silo_dists = _build_noniid_dists(args.n_silos) if args.noniid else None

    all_summaries = {}

    for sched in schedules_to_run:
        tag = f"{sched}_{'noniid' if args.noniid else 'iid'}_{args.mode}"
        print(f"\n{'━'*60}")
        print(f"  Running: {tag}")
        print(f"{'━'*60}")

        cfg = ScheduledFLConfig(
            mode                = args.mode,
            dataset_dir         = args.dataset_dir,
            n_silos             = args.n_silos,
            events_per_silo     = args.events_per_silo,
            silo_distributions  = silo_dists,
            schedule_name       = sched,
            n_rounds            = args.rounds,
            seed                = args.seed,
            training_device     = args.device,
            run_name            = tag,
            results_dir         = args.results_dir,
        )

        result = run_scheduled_fl(cfg)
        all_summaries[tag] = {
            "final_diag":   result["final_diag"],
            "final_triage": result["final_triage"],
        }

    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'schedule':<30}  diag    triage")
    for tag, s in all_summaries.items():
        print(f"  {tag:<30}  {s['final_diag']:.3f}   {s['final_triage']:.3f}")

    # Save combined summary
    out = Path(args.results_dir) / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  Summary written to {out}")


if __name__ == "__main__":
    main()
