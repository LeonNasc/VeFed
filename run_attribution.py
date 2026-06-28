"""
Attribution experiment — Phase 1 → Phase 2 federated novel-disease identification.

Protocol
--------
  R1–9   (pre-injection):   All silos train on velarex + sornathis.
  R10–14 (detection):       All silos see Morven, label it "unknown".
                             FedAvg should push P(unknown | Morven) up.
  R15–20 (attribution):     Silo_0 is told the disease is Morven → switches
                             its label to "morven". Silos 1+2 still label "unknown".
                             Does P(morven | Morven) propagate via FedAvg?

5-class label space: [non-infectious=0, velarex=1, sornathis=2, unknown=3, morven=4]

Outputs (results/unknown_disease/attribution_r10_r15_seed42/)
  logits_r{N:02d}.npz   — global model logits at snap rounds
  attribution_curve.png — P(unknown) and P(morven) for Morven probes across rounds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SNAP_ROUNDS      = [5, 8, 10, 12, 15, 18, 20]
INJECTION_ROUND  = 10
ATTRIBUTION_ROUND = 15
SEED             = 42
OUT_DIR          = Path("results/unknown_disease")
RUN_NAME         = "attribution_localhead_r10_r15_seed42"
NAIVE_RELABEL    = False

_IDX = {"non-infectious": 0, "velarex": 1, "sornathis": 2, "unknown": 3, "morven": 4}
_COLORS = {
    "velarex":   "#e63946",
    "sornathis": "#4361ee",
    "morven":    "#2a9d8f",
    "unknown":   "#f4a261",
}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_attribution(device: str = "cuda") -> None:
    from run_unknown_disease import UnknownDiseaseConfig, run_unknown_disease

    cfg = UnknownDiseaseConfig(
        schedule            = "gaussian",
        n_silos             = 3,
        events_per_silo     = 200,
        n_rounds            = 20,
        injection_round     = INJECTION_ROUND,
        injection_per_round = 16,
        do_inject           = True,
        all_silos_unknown   = True,    # Phase 1: everyone labels Morven "unknown"
        attribution_round   = ATTRIBUTION_ROUND,  # Phase 2: silo_0 switches to "morven"
        label_space         = "fictional_disease_5",
        snap_rounds         = SNAP_ROUNDS,
        training_device     = device,
        seed                = SEED,
        results_dir         = str(OUT_DIR),
        run_name            = RUN_NAME,
        naive_relabel       = NAIVE_RELABEL,
    )
    print("\n" + "━" * 60)
    print("  ATTRIBUTION EXPERIMENT")
    print(f"  Detection R{INJECTION_ROUND}  →  Attribution R{ATTRIBUTION_ROUND}")
    print("━" * 60)
    run_unknown_disease(cfg)


def plot_attribution() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_unknown_disease import generate_fictional_probe_events

    run_dir = OUT_DIR / RUN_NAME
    probe_events = generate_fictional_probe_events(n_per_band=12, seed=999)
    labels       = [ev.ground_truth for ev in probe_events]

    morven_idx   = [i for i, l in enumerate(labels) if l.split("/")[0] == "morven"]
    velarex_idx  = [i for i, l in enumerate(labels) if l.split("/")[0] == "velarex"]
    sornathis_idx= [i for i, l in enumerate(labels) if l.split("/")[0] == "sornathis"]

    rounds_found = sorted(
        int(p.stem.split("_r")[1]) for p in run_dir.glob("logits_r??.npz")
    )
    if not rounds_found:
        print("  No logit snapshots found — run the experiment first.")
        return OUT_DIR / "attribution_curve.png"

    def _load_logits(r: int, silo0_local: bool) -> np.ndarray | None:
        """Load logits for round r. In Phase 2, prefer silo_0 local logits if available."""
        if silo0_local and r >= ATTRIBUTION_ROUND:
            p = run_dir / f"logits_r{r:02d}_silo0.npz"
            if p.exists():
                return np.load(p)["logits"]
        p = run_dir / f"logits_r{r:02d}.npz"
        return np.load(p)["logits"] if p.exists() else None

    def series(idx_group: list[int], cls: str, silo0_local: bool = False) -> list[float]:
        vals = []
        for r in rounds_found:
            logits = _load_logits(r, silo0_local)
            if logits is not None:
                probs = _softmax(logits)
                vals.append(float(probs[np.array(idx_group), _IDX[cls]].mean()))
            else:
                vals.append(float("nan"))
        return vals

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    disease_groups = [
        ("Morven probes (silo_0 local in P2)", morven_idx,    True),
        ("Velarex probes",                      velarex_idx,   False),
        ("Sornathis probes",                    sornathis_idx, False),
    ]

    for ax, (group_name, idx, is_morven) in zip(axes, disease_groups):
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da"); sp.set_linewidth(0.5)

        # In Phase 2, use silo_0's local logits (for Morven panel only)
        use_local = is_morven
        p_unknown = series(idx, "unknown", silo0_local=use_local)
        p_morven  = series(idx, "morven",  silo0_local=use_local)

        if is_morven:
            ax.plot(rounds_found, p_unknown, marker="o", color=_COLORS["unknown"],
                    linewidth=2.2, markersize=6, label='P("unknown")', zorder=3)
            ax.plot(rounds_found, p_morven,  marker="D", color=_COLORS["morven"],
                    linewidth=2.2, markersize=7, label='P("morven") ← silo_0 local', zorder=3)
        else:
            ax.plot(rounds_found, p_unknown, marker="o", color=_COLORS["unknown"],
                    linewidth=1.6, markersize=5, label='P("unknown")', zorder=3)
            ax.plot(rounds_found, p_morven,  marker="D", color=_COLORS["morven"],
                    linewidth=1.6, markersize=5, label='P("morven")', zorder=3)

        ax.axvline(INJECTION_ROUND,  color="#adb5bd", linewidth=1.0, linestyle=":",
                   label=f"detection R{INJECTION_ROUND}", zorder=2)
        ax.axvline(ATTRIBUTION_ROUND, color=_COLORS["morven"], linewidth=1.2, linestyle="--",
                   label=f"attribution R{ATTRIBUTION_ROUND}", zorder=2)
        ax.axhline(0.2, color="#dee2e6", linewidth=0.7, linestyle="--",
                   label="chance (5 classes)", zorder=1)

        ax.set_title(group_name, fontsize=10, color="#212529", pad=8)
        ax.set_xlabel("FL Round", fontsize=10)
        ax.set_xticks(rounds_found)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8.5, framealpha=0.92, loc="upper left")

    axes[0].set_ylabel("Probability", fontsize=10)
    fig.suptitle(
        "Attribution experiment — frozen backbone + warm-start head (local attribution)\n"
        f"P1 (detection, R{INJECTION_ROUND}): all silos label Morven 'unknown'   |   "
        f"P2 (attribution, R{ATTRIBUTION_ROUND}): silo_0 trains head-only (excluded from FedAvg)",
        fontsize=10, color="#212529",
    )

    out_path = OUT_DIR / "attribution_curve.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as _plt; _plt.close(fig)
    print(f"\n  Saved: {out_path}")
    return out_path


def main() -> None:
    global SEED, RUN_NAME, NAIVE_RELABEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--device",    default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--seed",      type=int, default=SEED)
    ap.add_argument("--run-name",  default=None)
    ap.add_argument("--naive-relabel", action="store_true")
    args = ap.parse_args()

    SEED          = args.seed
    NAIVE_RELABEL = args.naive_relabel
    RUN_NAME      = args.run_name or RUN_NAME

    run_dir = OUT_DIR / RUN_NAME
    missing = [r for r in SNAP_ROUNDS if not (run_dir / f"logits_r{r:02d}.npz").exists()]
    if not args.plot_only and missing:
        run_attribution(device=args.device)

    out = plot_attribution()
    print(f"  Plot: {out}")


if __name__ == "__main__":
    main()
