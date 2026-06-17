"""
Hypothesis test: does FedAvg propagate novel-disease detection to silos
that never saw Morven in training?

Three conditions
----------------
  federated_inject   — FedAvg + Morven injected at silo_0 from round 10
                       Already run; loads saved global logits.
  confound_inject    — FedAvg + Morven injected at ALL silos from round 10,
                       but only silo_0 labels it "unknown"; silos 1+2 mislabel
                       as velarex/sornathis. Tests FedAvg against counter-gradient.
  local_inject       — No FedAvg + same injection as federated_inject.
                       Runs fresh; saves per-silo local logits.

Key measurement
---------------
P(novel | Morven probe) at each snapshot round, per silo / condition.

  - Federated (clean):  global model P(novel) rises after injection
  - Confound:           global model P(novel) rises MORE slowly (counter-gradient)
  - Local-only:         silo_0 P(novel) rises; silos 1+2 stay flat forever

Usage
-----
    python run_hypothesis_test.py           # runs both missing conditions, then plots
    python run_hypothesis_test.py --device cuda
    python run_hypothesis_test.py --plot-only   # skip training, just plot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SNAP_ROUNDS       = [5, 8, 10, 15, 20]
INJECTION_ROUND   = 10
SEED              = 42
FED_DIR           = Path("results/unknown_disease/gauss_inject_r10_seed42")
CONFOUND_DIR      = Path("results/unknown_disease/confound_inject_r10_seed42")
LOCAL_DIR         = Path("results/unknown_disease/local_inject_r10_seed42")
OUT_DIR           = Path("results/unknown_disease")

_DISEASE_COLORS = {
    "velarex":   "#e63946",
    "sornathis": "#4361ee",
    "morven":    "#2a9d8f",
}
_IDX = {"non-infectious": 0, "velarex": 1, "sornathis": 2, "unknown": 3}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _p_novel_morven(logits: np.ndarray, labels: list[str]) -> float:
    """Mean P(unknown class) for Morven probes only."""
    probs      = _softmax(logits)
    p_unk      = probs[:, _IDX["unknown"]]
    morven_idx = [i for i, l in enumerate(labels) if l.split("/")[0] == "morven"]
    return float(p_unk[morven_idx].mean()) if morven_idx else float("nan")


# ── Stage 1a: run local-only inject ──────────────────────────────────────────

def run_local_inject(device: str = "cpu") -> None:
    from run_unknown_disease import UnknownDiseaseConfig, run_unknown_disease

    cfg = UnknownDiseaseConfig(
        schedule            = "gaussian",
        n_silos             = 3,
        events_per_silo     = 200,
        n_rounds            = 20,
        injection_round     = INJECTION_ROUND,
        injection_per_round = 8,
        do_inject           = True,
        local_only          = True,    # no FedAvg
        per_silo_snaps      = True,    # save per-silo logits at snap rounds
        snap_rounds         = SNAP_ROUNDS,
        training_device     = device,
        seed                = SEED,
        results_dir         = str(OUT_DIR),
        run_name            = "local_inject_r10_seed42",
    )
    print("\n" + "━" * 60)
    print("  LOCAL-ONLY INJECT  (no FedAvg)")
    print("━" * 60)
    run_unknown_disease(cfg)


# ── Stage 1b: run confound inject ────────────────────────────────────────────

def run_confound_inject(device: str = "cpu") -> None:
    from run_unknown_disease import UnknownDiseaseConfig, run_unknown_disease

    cfg = UnknownDiseaseConfig(
        schedule            = "gaussian",
        n_silos             = 3,
        events_per_silo     = 200,
        n_rounds            = 20,
        injection_round     = INJECTION_ROUND,
        injection_per_round = 8,
        do_inject           = True,
        confound_silos      = True,    # all silos see Morven; silos 1+2 mislabel
        snap_rounds         = SNAP_ROUNDS,
        training_device     = device,
        seed                = SEED,
        results_dir         = str(OUT_DIR),
        run_name            = "confound_inject_r10_seed42",
    )
    print("\n" + "━" * 60)
    print("  CONFOUND INJECT  (all silos see Morven, only silo_0 labels correctly)")
    print("━" * 60)
    run_unknown_disease(cfg)


# ── Stage 2: load logits from all conditions ──────────────────────────────────

def load_global_logits(run_dir: Path, rounds: list[int]) -> dict[int, np.ndarray]:
    """Load global model logits saved by run_unknown_disease."""
    snaps = {}
    for rnd in rounds:
        p = run_dir / f"logits_r{rnd:02d}.npz"
        if p.exists():
            snaps[rnd] = np.load(p)["logits"]
    return snaps


def load_local_silo(silo: int, rounds: list[int]) -> dict[int, np.ndarray]:
    """Load silo-specific local model logits from the local inject run."""
    snaps = {}
    for rnd in rounds:
        p = LOCAL_DIR / f"logits_r{rnd:02d}_silo{silo}.npz"
        if p.exists():
            snaps[rnd] = np.load(p)["logits"]
    return snaps


# ── Stage 3: comparison plot ──────────────────────────────────────────────────

def plot_hypothesis(labels: list[str]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = SNAP_ROUNDS

    # Load all conditions
    fed_global      = load_global_logits(FED_DIR,      rounds)
    confound_global = load_global_logits(CONFOUND_DIR, rounds)
    local_silos     = {i: load_local_silo(i, rounds) for i in range(3)}

    def series(snaps: dict[int, np.ndarray]) -> list[float]:
        return [_p_novel_morven(snaps[r], labels) if r in snaps else float("nan")
                for r in rounds]

    fed_vals      = series(fed_global)
    confound_vals = series(confound_global)
    local_vals    = {i: series(local_silos[i]) for i in range(3)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True,
                             constrained_layout=True)
    fig.patch.set_facecolor("white")

    silo_styles = [
        ("Silo 0 (injected, novel label)",  "#2a9d8f", "-",  "D", 7),
        ("Silo 1 (no Morven)",              "#6c757d", "--", "o", 5),
        ("Silo 2 (no Morven)",              "#adb5bd", "--", "s", 5),
    ]

    panels = [
        (
            "Federated — clean\n(only silo_0 sees Morven)",
            {"Global model (all silos receive)": (fed_vals, "#e76f51", "-", "D", 8)},
        ),
        (
            "Federated — confound\n(all silos see Morven; 2/3 mislabel)",
            {"Global model (confounded FedAvg)": (confound_vals, "#9b5de5", "-", "^", 8)},
        ),
        (
            "Local-only\n(no FedAvg, silo_0 only)",
            {f"Silo {i} local model": (local_vals[i], col, ls, mk, ms)
             for i, (_, col, ls, mk, ms) in enumerate(silo_styles)},
        ),
    ]

    for ax, (title, lines) in zip(axes, panels):
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da"); sp.set_linewidth(0.5)

        for label, (ys, color, ls, mk, ms) in lines.items():
            ax.plot(rounds, ys, marker=mk, color=color, linestyle=ls,
                    linewidth=2.0, markersize=ms, label=label, zorder=3)

        ax.axvline(INJECTION_ROUND, color="#2a9d8f", linewidth=1.0,
                   linestyle=":", label=f"injection (R{INJECTION_ROUND})", zorder=2)
        ax.axhline(0.25, color="#ced4da", linewidth=0.7, linestyle="--",
                   label="chance (4 classes)", zorder=1)

        ax.set_title(title, fontsize=10, color="#212529", pad=8)
        ax.set_xlabel("FL Round", fontsize=10)
        ax.set_xticks(rounds)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8.5, framealpha=0.92, loc="upper left")

    axes[0].set_ylabel("Mean P(novel class | Morven probe)", fontsize=10)

    fig.suptitle(
        "Hypothesis test: FedAvg propagation of novel-disease detection\n"
        "Clean injection vs confounded labeling vs local-only baseline",
        fontsize=11, color="#212529",
    )

    out_path = OUT_DIR / "hypothesis_test.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"\n  Saved: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",    default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip training, load existing logits and plot")
    args = ap.parse_args()

    from run_unknown_disease import generate_fictional_probe_events
    probe_events = generate_fictional_probe_events(n_per_band=12, seed=999)
    labels       = [ev.ground_truth for ev in probe_events]

    if not args.plot_only:
        # Federated logits must already exist (from prior run_unknown_disease.py run)
        missing = [r for r in SNAP_ROUNDS
                   if not (FED_DIR / f"logits_r{r:02d}.npz").exists()]
        if missing:
            print(f"  [warn] Federated logits missing for rounds {missing}.")
            print(f"  Run:  python run_unknown_disease.py --run-name gauss_inject_r10_seed42")
            raise SystemExit(1)

        # Run confound inject if needed
        confound_missing = [r for r in SNAP_ROUNDS
                            if not (CONFOUND_DIR / f"logits_r{r:02d}.npz").exists()]
        if confound_missing:
            run_confound_inject(device=args.device)

        # Run local inject if needed
        local_missing = [r for r in SNAP_ROUNDS
                         if not (LOCAL_DIR / f"logits_r{r:02d}_silo0.npz").exists()]
        if local_missing:
            run_local_inject(device=args.device)

    out = plot_hypothesis(labels)
    print(f"  Plot: {out}")


if __name__ == "__main__":
    main()
