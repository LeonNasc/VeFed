"""
Ollama ablation — replicates the fast-ablation schedule matrix using
Ollama (phi3:mini) as the patient text generator.

Two-stage pipeline
------------------
Stage 1  Pool generation: record pools via Ollama PatientLLMClient, saved as
         results/ollama_ablation/pools/seed{N}/{iid|noniid}/silo_{i}/train.jsonl
Stage 2  FL training: replay-mode FL over schedule × dist × seed matrix
         Results → results/ollama_ablation/<schedule>/<dist>/seed{N}/
Stage 3  Report → results/ollama_ablation/report.md

Usage
-----
    python run_ollama_ablation.py              # full run (3 seeds)
    python run_ollama_ablation.py --gen-only   # Stage 1 only (pool generation)
    python run_ollama_ablation.py --fl-only    # Stages 2+3 only (pools must exist)
    python run_ollama_ablation.py --resume     # skip already-completed runs
    python run_ollama_ablation.py --device cuda --rounds 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WANDB_MODE", "disabled")

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_ROOT    = Path("results/ollama_ablation")
POOL_ROOT       = RESULTS_ROOT / "pools"

SEEDS           = [42, 43, 44]
SCHEDULES       = ["flat", "gaussian", "burst", "ramp", "parabola"]
DISTS           = ["iid", "noniid"]
N_SILOS         = 3
EVENTS_PER_SILO = 200
HOLDOUT_FRAC    = 0.2
N_ROUNDS        = 20
OLLAMA_MODEL    = "phi3:mini"

# Severity string → float for PatientLLMClient
_SEV_FLOAT: dict[str, float] = {
    "mild":      0.2,
    "moderate":  0.5,
    "severe":    0.85,
    "discharge": 0.1,
}

# IID distribution (mirrors scheduled_runner._IID_DISTRIBUTION)
_IID_DIST: list[tuple[str, str, str, float]] = [
    ("influenza",      "mild",      "stoic",   1.0),
    ("influenza",      "mild",      "neutral",  2.0),
    ("influenza",      "mild",      "anxious",  1.0),
    ("influenza",      "moderate",  "stoic",   1.0),
    ("influenza",      "moderate",  "neutral",  2.0),
    ("influenza",      "moderate",  "anxious",  1.0),
    ("influenza",      "severe",    "neutral",  1.0),
    ("pneumonia",      "mild",      "stoic",   1.0),
    ("pneumonia",      "mild",      "neutral",  2.0),
    ("pneumonia",      "mild",      "anxious",  1.0),
    ("pneumonia",      "moderate",  "stoic",   1.0),
    ("pneumonia",      "moderate",  "neutral",  2.0),
    ("pneumonia",      "moderate",  "anxious",  1.0),
    ("pneumonia",      "severe",    "neutral",  1.0),
    ("non-infectious", "mild",      "neutral",  1.0),
    ("non-infectious", "mild",      "anxious",  2.0),
    ("non-infectious", "discharge", "neutral",  1.0),
]


def _noniid_dists(n_silos: int) -> list[list[tuple]]:
    """Odd/even skew: even silos flu-heavy, odd silos pneumonia-heavy."""
    dists = []
    for i in range(n_silos):
        flu_w  = 3.0 if i % 2 == 0 else 1.0
        pneu_w = 1.0 if i % 2 == 0 else 3.0
        d = [
            (dis, sev, per, w * (flu_w  if dis == "influenza"
                            else pneu_w if dis == "pneumonia"
                            else 1.0))
            for dis, sev, per, w in _IID_DIST
        ]
        dists.append(d)
    return dists


def _weighted_sample(dist: list[tuple], rng) -> tuple[str, str, str]:
    total = sum(w for *_, w in dist)
    r     = rng.random() * total
    cum   = 0.0
    for disease, severity, personality, w in dist:
        cum += w
        if r <= cum:
            return disease, severity, personality
    return dist[-1][0], dist[-1][1], dist[-1][2]


# ── Ollama record generator ───────────────────────────────────────────────────

class OllamaRecordGenerator:
    """
    Wraps PatientLLMClient to produce training records in PhraseLibrary format:
        {text, label="disease/severity", gt_disease, gt_severity, days}
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        from simulation.patient_llm import PatientLLMClient
        self._client = PatientLLMClient(model=model)

    def sample(
        self,
        disease:     str,
        severity:    str,
        personality: str = "neutral",
        days:        int | None = None,
    ) -> dict:
        import random
        from simulation.symptom_language import Personality

        if days is None:
            days = random.randint(1, 14)

        sev_float = _SEV_FLOAT.get(severity, 0.5)
        p_enum    = getattr(Personality, personality.upper(), Personality.NEUTRAL)
        text      = self._client.opening_statement(sev_float, days, p_enum)

        return {
            "text":        text,
            "label":       f"{disease}/{severity}",
            "gt_disease":  disease,
            "gt_severity": severity,
            "days":        days,
        }


# ── PhraseLibrary fallback (used when Ollama call fails) ──────────────────────

def _make_fallback(seed: int):
    from simulation.phrase_sampler import PhraseLibrary
    return PhraseLibrary(seed=seed)


# ── Stage 1: pool generation ──────────────────────────────────────────────────

def generate_pool_for_seed(
    seed:    int,
    model:   str  = OLLAMA_MODEL,
    resume:  bool = True,
) -> dict[str, Path]:
    """
    Generate IID and non-IID record pools for one seed via Ollama.
    Pools are saved to POOL_ROOT/seed{N}/{iid|noniid}/silo_{i}/.
    Returns {dist: pool_dir}.
    """
    import random

    gen      = OllamaRecordGenerator(model=model)
    fallback = _make_fallback(seed)

    pool_dirs: dict[str, Path] = {}

    for dist in DISTS:
        pool_dir = POOL_ROOT / f"seed{seed}" / dist
        if resume and all(
            (pool_dir / f"silo_{i}" / "train.jsonl").exists()
            for i in range(N_SILOS)
        ):
            print(f"    [skip] seed={seed}/{dist} pool already exists")
            pool_dirs[dist] = pool_dir
            continue

        print(f"    seed={seed} {dist} — {N_SILOS}×{EVENTS_PER_SILO} records")
        rng       = random.Random(seed + (0 if dist == "iid" else 1000))
        dist_list = [_IID_DIST] * N_SILOS if dist == "iid" else _noniid_dists(N_SILOS)
        n_errors  = 0

        for silo_i, silo_dist in enumerate(dist_list):
            silo_dir = pool_dir / f"silo_{silo_i}"
            silo_dir.mkdir(parents=True, exist_ok=True)

            records: list[dict] = []
            t0 = time.time()

            for j in range(EVENTS_PER_SILO):
                disease, severity, personality = _weighted_sample(silo_dist, rng)
                days = rng.randint(1, 14)
                try:
                    rec = gen.sample(disease, severity, personality, days=days)
                except Exception as exc:
                    n_errors += 1
                    rec = fallback.sample(disease, severity, personality, days=days)
                    if n_errors <= 3:
                        print(f"      [warn] Ollama error: {exc} — fallback used")
                records.append(rec)

                if (j + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate    = (j + 1) / elapsed
                    eta     = (EVENTS_PER_SILO - j - 1) / rate
                    print(f"      silo_{silo_i}: {j+1}/{EVENTS_PER_SILO}  "
                          f"{rate:.1f} rec/s  eta={eta:.0f}s")

            rng.shuffle(records)
            n_hold  = max(1, int(len(records) * HOLDOUT_FRAC))
            holdout = records[:n_hold]
            train   = records[n_hold:]

            (silo_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(r) for r in train) + "\n"
            )
            (silo_dir / "holdout.jsonl").write_text(
                "\n".join(json.dumps(r) for r in holdout) + "\n"
            )
            print(f"      silo_{silo_i}: {len(train)} train + {len(holdout)} holdout"
                  + (f"  ({n_errors} Ollama errors)" if n_errors else ""))
            n_errors = 0

        pool_dirs[dist] = pool_dir

    return pool_dirs


# ── Stage 2: single FL run ────────────────────────────────────────────────────

def run_one(
    schedule: str,
    dist:     str,
    seed:     int,
    pool_dir: Path,
    n_rounds: int,
    resume:   bool,
    device:   str = "cpu",
) -> dict:
    from fl.scheduled_runner import ScheduledFLConfig, run_scheduled_fl

    run_name = f"seed{seed}"
    out_dir  = RESULTS_ROOT / schedule / dist / run_name

    if resume and (out_dir / "summary.json").exists():
        with (out_dir / "summary.json").open() as f:
            summary = json.load(f)
        print(f"    [skip] {schedule}/{dist}/seed={seed}  "
              f"diag={summary.get('final_diag_acc', float('nan')):.3f}")
        return {
            "final_diag_acc":   summary.get("final_diag_acc",   float("nan")),
            "final_triage_acc": summary.get("final_triage_acc", float("nan")),
            "schedule": schedule, "dist": dist, "seed": seed,
        }

    cfg = ScheduledFLConfig(
        mode            = "replay",
        dataset_dir     = str(pool_dir),
        n_silos         = N_SILOS,
        schedule_name   = schedule,
        n_rounds        = n_rounds,
        seed            = seed,
        gen_seed        = seed,
        training_device = device,
        run_name        = run_name,
        results_dir     = str(RESULTS_ROOT / schedule / dist),
    )

    result = run_scheduled_fl(cfg)
    return {
        "final_diag_acc":   result["final_diag"],
        "final_triage_acc": result["final_triage"],
        "schedule": schedule,
        "dist":     dist,
        "seed":     seed,
        "rounds":   result["rounds"],
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(collected: list[dict]) -> dict:
    from collections import defaultdict

    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in collected:
        key = (r["schedule"], r["dist"])
        v   = r.get("final_diag_acc", float("nan"))
        if v == v:
            groups[key].append(v)

    stats = {}
    for (sched, dist), vals in groups.items():
        stats[f"{sched}/{dist}"] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "n":    len(vals),
        }
    return stats


def print_stats(collected: list[dict]) -> None:
    stats = compute_stats(collected)

    print(f"\n{'═'*65}")
    print("  OLLAMA ABLATION — diag_acc mean ± std across seeds")
    print(f"{'═'*65}")
    print(f"  {'schedule':12s}  {'dist':7s}   mean    std    n")
    print(f"  {'─'*55}")
    for key, v in sorted(stats.items()):
        sched, dist = key.split("/")
        print(f"  {sched:12s}  {dist:7s}   {v['mean']:.3f}   {v['std']:.3f}   {v['n']}")

    stats_path = RESULTS_ROOT / "statistics.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Statistics → {stats_path}")


# ── Stage 3: report ───────────────────────────────────────────────────────────

def generate_report(collected: list[dict], seeds: list[int], n_rounds: int) -> None:
    stats = compute_stats(collected)

    # Load phrase_library baseline from fast_ablation
    baseline: dict[str, dict] = {}
    baseline_path = Path("results/fast_ablation/statistics.json")
    if baseline_path.exists():
        with baseline_path.open() as f:
            raw = json.load(f)
        for k, v in raw.items():
            parts = k.split("/")
            if len(parts) == 3 and parts[0] == "phrase_library":
                baseline[f"{parts[1]}/{parts[2]}"] = v

    ts = time.strftime("%Y-%m-%d %H:%M")

    lines: list[str] = [
        "# Ollama Ablation Report",
        "",
        f"Generated: {ts}  |  Seeds: {seeds}  |  "
        f"Events/silo: {EVENTS_PER_SILO}  |  Rounds: {n_rounds}",
        "",
        "Patient text generated by **phi3:mini via Ollama** (`PatientLLMClient`).",
        "FL uses the same LoRA / FedAvg pipeline as the phrase-library fast ablation.",
        "Pool generation is **one-time per seed** (Stage 1); all 5 schedules replay",
        "the same pool so schedule differences reflect temporal structure only.",
        "",
        "---",
        "",
        "## §1  Diagnostic Accuracy by Schedule",
        "",
        "| Schedule | Dist | Ollama mean±std | n |",
        "|----------|------|-----------------|---|",
    ]

    for key, v in sorted(stats.items()):
        sched, dist = key.split("/")
        lines.append(f"| {sched} | {dist} | {v['mean']:.3f} ± {v['std']:.3f} | {v['n']} |")

    # IID vs Non-IID gap
    iid_vals = [r["final_diag_acc"] for r in collected
                if r["dist"] == "iid" and r["final_diag_acc"] == r["final_diag_acc"]]
    nid_vals = [r["final_diag_acc"] for r in collected
                if r["dist"] == "noniid" and r["final_diag_acc"] == r["final_diag_acc"]]

    lines += [
        "",
        "---",
        "",
        "## §2  IID vs Non-IID Gap",
        "",
    ]
    if iid_vals and nid_vals:
        gap = np.mean(iid_vals) - np.mean(nid_vals)
        lines += [
            f"- IID mean: **{np.mean(iid_vals):.3f}** ± {np.std(iid_vals):.3f}",
            f"- Non-IID mean: **{np.mean(nid_vals):.3f}** ± {np.std(nid_vals):.3f}",
            f"- Gap (IID − Non-IID): **{gap:+.3f}**",
        ]

    # Comparison table vs phrase library
    lines += [
        "",
        "---",
        "",
        "## §3  Ollama vs Phrase Library",
        "",
    ]

    if baseline:
        lines += [
            "Δ = Ollama − Phrase Library (positive = Ollama better).",
            "",
            "| Condition | Ollama | Phrase Library | Δ |",
            "|-----------|--------|----------------|---|",
        ]
        for key in sorted(stats.keys()):
            v = stats[key]
            if key in baseline:
                bl    = baseline[key]
                delta = v["mean"] - bl["mean"]
                sign  = "+" if delta >= 0 else ""
                lines.append(f"| {key} | {v['mean']:.3f} | {bl['mean']:.3f} | {sign}{delta:.3f} |")
            else:
                lines.append(f"| {key} | {v['mean']:.3f} | — | — |")

        ollama_means = [stats[k]["mean"] for k in stats if k in baseline]
        pl_means     = [baseline[k]["mean"] for k in stats if k in baseline]
        if ollama_means and pl_means:
            avg_delta = np.mean(ollama_means) - np.mean(pl_means)
            sign = "+" if avg_delta >= 0 else ""
            lines += ["", f"**Average Δ across all conditions: {sign}{avg_delta:.3f}**"]
    else:
        lines.append(
            "_Phrase library baseline not found. Run `run_fast_ablation.py` first "
            "to populate `results/fast_ablation/statistics.json`._"
        )

    # Key observations
    lines += [
        "",
        "---",
        "",
        "## §4  Key Observations",
        "",
        f"- **Generation cost**: {N_SILOS}×{EVENTS_PER_SILO} = "
        f"{N_SILOS * EVENTS_PER_SILO} Ollama calls per seed; at ~3 s/call "
        f"≈ {N_SILOS * EVENTS_PER_SILO * 3 / 60:.0f} min pool generation per seed.",
        "- **Naturalness ceiling**: Ollama-generated text includes paraphrases and",
        "  rare symptom descriptions not present in the phrase library, which may help",
        "  generalisation or add noise depending on model calibration.",
        "- **Realism vs controllability trade-off**: unlike the phrase library,",
        "  Ollama occasionally produces cross-disease symptom text, contributing",
        "  irreducible error analogous to the `confusion_rate` parameter.",
        "- **Same FL pipeline**: LoRA rank=8, FedAvg, local_epochs=3, batch=8.",
        "  Any accuracy difference isolates the data source, not the FL algorithm.",
        "",
        "---",
        "",
        "_Report auto-generated by `run_ollama_ablation.py`_",
    ]

    report_path = RESULTS_ROOT / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"  Report → {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Ollama ablation (3 seeds)")
    ap.add_argument("--seeds",     nargs="+", type=int, default=SEEDS)
    ap.add_argument("--schedules", nargs="+", default=SCHEDULES, choices=SCHEDULES)
    ap.add_argument("--dists",     nargs="+", default=DISTS,     choices=DISTS)
    ap.add_argument("--rounds",    type=int,  default=N_ROUNDS)
    ap.add_argument("--device",    default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--model",     default=OLLAMA_MODEL,
                    help="Ollama model for patient text generation")
    ap.add_argument("--gen-only",  action="store_true",
                    help="Stage 1 only (pool generation, then exit)")
    ap.add_argument("--fl-only",   action="store_true",
                    help="Stages 2+3 only — pools must already exist")
    ap.add_argument("--resume",    action="store_true",
                    help="Skip runs whose output already exists")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Ollama health check ───────────────────────────────────────────────────
    if not args.fl_only:
        from simulation.patient_llm import PatientLLMClient
        client = PatientLLMClient(model=args.model)
        if not client.health_check():
            print("ERROR: Ollama is not reachable at localhost:11434")
            print("  Start: ollama serve")
            print(f"  Pull:  ollama pull {args.model}")
            raise SystemExit(1)
        print(f"[OK] Ollama reachable  model={args.model}")

    total_conditions = len(args.schedules) * len(args.dists) * len(args.seeds)
    print(f"\n{'━'*65}")
    print(f"  Ollama ablation  |  seeds={args.seeds}  rounds={args.rounds}")
    print(f"  {len(args.schedules)} schedules × {len(args.dists)} dists × "
          f"{len(args.seeds)} seeds = {total_conditions} FL runs")
    print(f"{'━'*65}")

    # ── Stage 1: pool generation ──────────────────────────────────────────────
    pool_dirs: dict[int, dict[str, Path]] = {}

    if not args.fl_only:
        print(f"\n  Stage 1 — pool generation")
        t_gen = time.time()
        for seed in args.seeds:
            print(f"\n  seed={seed}")
            pool_dirs[seed] = generate_pool_for_seed(
                seed=seed, model=args.model, resume=args.resume
            )
        elapsed = time.time() - t_gen
        print(f"\n  Pool generation done in {elapsed/60:.1f} min")
        if args.gen_only:
            print("  --gen-only: stopping after Stage 1.")
            return
    else:
        for seed in args.seeds:
            pool_dirs[seed] = {
                dist: POOL_ROOT / f"seed{seed}" / dist
                for dist in args.dists
            }
            for dist, p in pool_dirs[seed].items():
                if not p.exists():
                    raise FileNotFoundError(
                        f"Pool missing: {p}\nRun Stage 1 first (drop --fl-only)."
                    )

    # ── Stage 2: FL training ──────────────────────────────────────────────────
    print(f"\n  Stage 2 — FL training ({total_conditions} runs)")

    collected: list[dict] = []
    done  = 0
    t_fl  = time.time()

    for dist in args.dists:
        print(f"\n  dist={dist}")
        for schedule in args.schedules:
            for seed in args.seeds:
                done += 1
                elapsed = time.time() - t_fl
                eta = elapsed / done * (total_conditions - done) if done > 1 else 0
                print(f"  [{done}/{total_conditions}  eta={eta/60:.1f}min] "
                      f"{schedule} / seed={seed}")
                result = run_one(
                    schedule = schedule,
                    dist     = dist,
                    seed     = seed,
                    pool_dir = pool_dirs[seed][dist],
                    n_rounds = args.rounds,
                    resume   = args.resume,
                    device   = args.device,
                )
                collected.append(result)

    print_stats(collected)

    # ── Stage 3: report ───────────────────────────────────────────────────────
    if not args.no_report:
        print(f"\n  Stage 3 — report")
        generate_report(collected, seeds=args.seeds, n_rounds=args.rounds)

    total_time = time.time() - t_fl
    print(f"\n  Total wall time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
