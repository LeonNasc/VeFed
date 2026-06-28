"""
Embedding & logit-probability scatter plots for the unknown disease experiment.

Produces two figures:
  1. logit_proba_scatter.png  — probability simplex: P(velarex) vs P(sornathis),
                                dot size ∝ P(unknown); split by round
  2. cls_umap_scatter.png     — UMAP of CLS [CLS] token embeddings, same layout

Usage:
    python show_embeddings.py
    python show_embeddings.py --run gauss_inject_r10_seed42 --rounds 5 10 15 20
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.special import softmax

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_RUN    = "gauss_inject_r10_seed42_inj32"
DEFAULT_ROUNDS = [5, 10, 15, 20]
RESULTS_DIR    = pathlib.Path("results/unknown_disease")

# Class order from build_fictional_disease_map(): non-infect=0, velarex=1, sornathis=2, unknown=3
IDX = {"non_infect": 0, "velarex": 1, "sornathis": 2, "unknown": 3}

PROBE_DISEASES = ["velarex", "sornathis", "morven"]   # order in probe set
N_PER_BAND     = 12   # probe_per_band default
N_BANDS        = 3    # mild / moderate / severe

COLORS = {
    "velarex":   "#e63946",
    "sornathis": "#4361ee",
    "morven":    "#2a9d8f",
}
SEV_LABELS = ["mild", "moderate", "severe"]
SEV_ALPHA  = [0.40, 0.70, 1.00]
SEV_SIZE   = [30,   55,   90  ]


def _probe_labels() -> list[tuple[str, str]]:
    """Return (disease, severity) for every probe event, in generation order."""
    labs = []
    for disease in PROBE_DISEASES:
        for sev in SEV_LABELS:
            for _ in range(N_PER_BAND):
                labs.append((disease, sev))
    return labs


def load_round(run_dir: pathlib.Path, rnd: int):
    """Load (logits, cls) arrays for a given round number."""
    path = run_dir / f"logits_r{rnd:02d}.npz"
    if not path.exists():
        return None, None
    d = np.load(path)
    return d["logits"], d["cls"]


# ── Figure 1: logit probability scatter ───────────────────────────────────────

def plot_proba_scatter(run_dir: pathlib.Path, rounds: list[int], out: pathlib.Path):
    labels = _probe_labels()
    n      = len(rounds)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.8, 4.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for ax, rnd in zip(axes, rounds):
        logits, _ = load_round(run_dir, rnd)
        if logits is None:
            ax.set_title(f"R{rnd} (missing)")
            continue

        proba = softmax(logits, axis=1)   # (N, 4)
        p_vel = proba[:, IDX["velarex"]]
        p_sor = proba[:, IDX["sornathis"]]
        p_unk = proba[:, IDX["unknown"]]

        for i, (disease, sev) in enumerate(labels):
            sev_i = SEV_LABELS.index(sev)
            ax.scatter(
                p_vel[i], p_sor[i],
                color     = COLORS[disease],
                alpha     = SEV_ALPHA[sev_i],
                s         = SEV_SIZE[sev_i] + p_unk[i] * 120,   # size grows with P(unknown)
                marker    = "D" if disease == "morven" else "o",
                linewidths= 0.6 if disease == "morven" else 0,
                edgecolors= "black" if disease == "morven" else "none",
                zorder    = 3 if disease == "morven" else 2,
            )

        ax.set_xlabel("P(Velarex)", fontsize=9)
        ax.set_ylabel("P(Sornathis)", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Round {rnd}" + (" ★ injection" if rnd == 10 else ""), fontsize=9)
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da")
            sp.set_linewidth(0.5)
        ax.tick_params(labelsize=7)

        # Annotate mean P(unknown) for Morven
        morven_mask = [d == "morven" for d, _ in labels]
        mean_unk = float(np.mean(p_unk[morven_mask]))
        ax.text(0.02, 0.97, f"Morven P(unk)={mean_unk:.2f}",
                transform=ax.transAxes, fontsize=7, va="top", color=COLORS["morven"])

    # Legend
    patches = [
        mpatches.Patch(color=COLORS["velarex"],   label="Velarex"),
        mpatches.Patch(color=COLORS["sornathis"], label="Sornathis"),
        mpatches.Patch(color=COLORS["morven"],    label="Morven ◆ (novel)"),
    ]
    size_handles = [
        plt.scatter([], [], s=30,  color="#888", alpha=0.5, label="mild"),
        plt.scatter([], [], s=55,  color="#888", alpha=0.75, label="moderate"),
        plt.scatter([], [], s=90,  color="#888", alpha=1.0,  label="severe"),
    ]
    fig.legend(handles=patches + size_handles,
               loc="lower center", ncol=6, fontsize=8,
               framealpha=0.95, facecolor="white", edgecolor="#ced4da",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Logit probability scatter  ·  P(Velarex) vs P(Sornathis)\n"
        "Dot size grows with P(Unknown) · ◆ = Morven (novel disease)",
        fontsize=10,
    )

    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: CLS embedding UMAP ─────────────────────────────────────────────

def _umap_or_pca(data: np.ndarray, seed: int = 42) -> np.ndarray:
    try:
        import umap as umap_lib
        return umap_lib.UMAP(
            n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1
        ).fit_transform(data)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(data)


def plot_cls_umap(run_dir: pathlib.Path, rounds: list[int], out: pathlib.Path):
    labels = _probe_labels()
    n      = len(rounds)

    # Fit UMAP on the union of all rounds for a stable layout
    all_cls = []
    available = []
    for rnd in rounds:
        _, cls = load_round(run_dir, rnd)
        if cls is not None:
            all_cls.append(cls)
            available.append(rnd)
    if not all_cls:
        print("  No CLS data found.")
        return

    combined  = np.vstack(all_cls)
    coords_2d = _umap_or_pca(combined, seed=42)
    per_round = np.split(coords_2d, len(all_cls))

    fig, axes = plt.subplots(1, len(available), figsize=(len(available) * 3.8, 4.0),
                              constrained_layout=True)
    fig.patch.set_facecolor("white")
    if len(available) == 1:
        axes = [axes]

    for ax, rnd, coords in zip(axes, available, per_round):
        for i, (disease, sev) in enumerate(labels):
            sev_i = SEV_LABELS.index(sev)
            ax.scatter(
                coords[i, 0], coords[i, 1],
                color     = COLORS[disease],
                alpha     = SEV_ALPHA[sev_i],
                s         = SEV_SIZE[sev_i],
                marker    = "D" if disease == "morven" else "o",
                linewidths= 0.6 if disease == "morven" else 0,
                edgecolors= "black" if disease == "morven" else "none",
                zorder    = 3 if disease == "morven" else 2,
            )
        ax.set_title(f"Round {rnd}" + (" ★" if rnd == 10 else ""), fontsize=9)
        ax.set_facecolor("#f8f9fa")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_color("#ced4da")
            sp.set_linewidth(0.5)

    patches = [
        mpatches.Patch(color=COLORS["velarex"],   label="Velarex"),
        mpatches.Patch(color=COLORS["sornathis"], label="Sornathis"),
        mpatches.Patch(color=COLORS["morven"],    label="Morven ◆ (novel)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.95, facecolor="white", edgecolor="#ced4da",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "CLS embedding UMAP  ·  ◆ = Morven (novel disease)\n"
        "UMAP fitted on union of all displayed rounds for stable layout",
        fontsize=10,
    )

    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",    default=DEFAULT_RUN)
    ap.add_argument("--rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: results/unknown_disease/<run>/)")
    args = ap.parse_args()

    run_dir = RESULTS_DIR / args.run
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun: {args.run}")
    print(f"Rounds: {args.rounds}")

    plot_proba_scatter(run_dir, args.rounds, out_dir / "logit_proba_scatter.png")
    plot_cls_umap(run_dir, args.rounds, out_dir / "cls_umap_scatter.png")


if __name__ == "__main__":
    main()
