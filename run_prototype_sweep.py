"""
Prototype-bank sweep — all 5 schedule shapes × 3 seeds.

Runs 15 configurations of run_prototype and produces a comparison plot showing
how quickly each epidemic curve shape allows DBSCAN to detect the Morven cluster
and how accurately the prototype bank classifies Morven probes after attribution.

Usage:
    python run_prototype_sweep.py [--device cuda] [--plot-only]

Outputs (results/prototype/sweep/)
    sweep_results.json          — all 15 runs' per-round metrics
    proto_morven_by_shape.png   — mean ± std proto accuracy (Morven) per shape
    detection_round_box.png     — boxplot: DBSCAN detection round per shape
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SHAPES  = ["flat", "ramp", "gaussian", "burst", "parabola"]
SEEDS   = [42, 43, 44]
OUT_DIR = Path("results/prototype/sweep")

_SHAPE_COLORS = {
    "flat":     "#6c757d",
    "ramp":     "#4361ee",
    "gaussian": "#2a9d8f",
    "burst":    "#e63946",
    "parabola": "#f4a261",
}

_SNAP_ROUNDS = [5, 8, 10, 12, 15, 18, 20]


# ── Run all configs ───────────────────────────────────────────────────────────

def run_sweep(device: str = "cuda") -> None:
    from run_prototype import PrototypeConfig, run_prototype

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[list[dict]]] = {s: [] for s in SHAPES}

    total = len(SHAPES) * len(SEEDS)
    done  = 0
    for shape in SHAPES:
        for seed in SEEDS:
            run_name = f"{shape}_seed{seed}"
            run_dir  = Path("results/prototype") / run_name

            # Skip if already complete
            metrics_path = run_dir / "round_metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                all_results[shape].append(metrics)
                done += 1
                print(f"[{done}/{total}] skip {run_name} (cached)")
                continue

            done += 1
            print(f"\n[{done}/{total}] {shape}  seed={seed}")
            cfg = PrototypeConfig(
                schedule        = shape,
                seed            = seed,
                events_per_silo = 160,
                training_device = device,
                snap_rounds     = _SNAP_ROUNDS,
                run_name        = run_name,
            )
            run_prototype(cfg)
            metrics = json.loads(metrics_path.read_text())
            all_results[shape].append(metrics)

    # Save aggregated results
    (OUT_DIR / "sweep_results.json").write_text(
        json.dumps(all_results, indent=2)
    )
    print(f"\n  Saved: {OUT_DIR}/sweep_results.json")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _extract_series(
    runs: list[list[dict]], key: str, snap_rounds: list[int]
) -> np.ndarray:
    """
    Return array of shape [n_seeds, n_snaps] with values for `key` at each snap round.
    Missing snap rounds → nan.
    """
    out = np.full((len(runs), len(snap_rounds)), float("nan"))
    for r_idx, run in enumerate(runs):
        rnd_to_metric = {m["round"]: m.get(key) for m in run if m.get(key) is not None}
        for s_idx, rnd in enumerate(snap_rounds):
            if rnd in rnd_to_metric:
                out[r_idx, s_idx] = rnd_to_metric[rnd]
    return out


def _detection_rounds(runs: list[list[dict]]) -> list[int | None]:
    """Return first round where n_unknown_clusters >= 1 for each seed run."""
    results = []
    for run in runs:
        detected = None
        for m in run:
            if m.get("n_unknown_clusters", 0) >= 1 and m.get("morven_injected"):
                detected = m["round"]
                break
        results.append(detected)
    return results


def plot_sweep() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep_path = OUT_DIR / "sweep_results.json"
    if not sweep_path.exists():
        print("  No sweep_results.json — run the sweep first.")
        return
    all_results: dict[str, list[list[dict]]] = json.loads(sweep_path.read_text())

    snap_rounds = _SNAP_ROUNDS

    # ── Plot 1: proto_morven_exact over rounds, mean ± std per shape ──────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da"); sp.set_linewidth(0.5)

    ax_acc, ax_det = axes

    for shape in SHAPES:
        runs  = all_results.get(shape, [])
        if not runs:
            continue
        color = _SHAPE_COLORS[shape]

        # Proto morven accuracy
        arr = _extract_series(runs, "proto_morven_exact", snap_rounds)
        mu  = np.nanmean(arr, axis=0)
        std = np.nanstd(arr,  axis=0)
        ax_acc.plot(snap_rounds, mu, marker="o", color=color, linewidth=2.0,
                    markersize=5, label=shape)
        ax_acc.fill_between(snap_rounds, mu - std, mu + std,
                            color=color, alpha=0.15)

    ax_acc.axvline(10, color="#adb5bd", linewidth=1.0, linestyle=":", label="injection R10")
    ax_acc.axvline(15, color="#888", linewidth=1.0, linestyle="--", label="attribution R15")
    ax_acc.set_title("P(morven | Morven probe) — prototype classifier", fontsize=10)
    ax_acc.set_xlabel("FL Round"); ax_acc.set_ylabel("Mean accuracy (3 seeds)")
    ax_acc.set_ylim(-0.05, 1.05); ax_acc.set_xticks(snap_rounds)
    ax_acc.legend(fontsize=9, framealpha=0.92)

    # ── Plot 2: detection round boxplot ───────────────────────────────────────
    det_data   = [_detection_rounds(all_results.get(s, [])) for s in SHAPES]
    det_finite = [[v for v in d if v is not None] for d in det_data]

    # Scatter points + mean bar (cleaner than box for n=3)
    for x_idx, (shape, vals) in enumerate(zip(SHAPES, det_finite)):
        color = _SHAPE_COLORS[shape]
        jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(vals))
        ax_det.scatter([x_idx + jitter[j] for j in range(len(vals))], vals,
                       color=color, s=60, zorder=3)
        if vals:
            ax_det.hlines(np.mean(vals), x_idx - 0.25, x_idx + 0.25,
                          color=color, linewidth=2.5, zorder=4)

    ax_det.set_xticks(range(len(SHAPES))); ax_det.set_xticklabels(SHAPES)
    ax_det.axhline(10, color="#adb5bd", linewidth=1.0, linestyle=":", label="injection round")
    ax_det.set_title("DBSCAN detection round (first coherent unknown cluster)", fontsize=10)
    ax_det.set_ylabel("Round"); ax_det.set_ylim(0, 22)
    ax_det.legend(fontsize=9)

    fig.suptitle(
        "Prototype-bank sweep — 5 schedule shapes × 3 seeds × 160 events/silo\n"
        "Injection R10 (Morven → 'unknown')  |  Attribution R15 (rename → 'morven')",
        fontsize=10, color="#212529",
    )

    out = OUT_DIR / "proto_sweep.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n  Detection round (mean ± std across seeds):")
    for shape, vals in zip(SHAPES, det_finite):
        if vals:
            print(f"    {shape:10s}  {np.mean(vals):.1f} ± {np.std(vals):.1f}  "
                  f"  (raw: {vals})")
        else:
            print(f"    {shape:10s}  never detected")

    print("\n  Proto accuracy at R20 (mean ± std):")
    for shape in SHAPES:
        runs = all_results.get(shape, [])
        arr  = _extract_series(runs, "proto_morven_exact", snap_rounds)
        col  = arr[:, snap_rounds.index(20)] if 20 in snap_rounds else arr[:, -1]
        print(f"    {shape:10s}  {np.nanmean(col):.3f} ± {np.nanstd(col):.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",    default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        run_sweep(device=args.device)
    plot_sweep()


if __name__ == "__main__":
    main()
