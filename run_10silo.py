#!/usr/bin/env python3
"""
10-silo scalability experiments.

Three experiments, each with 3-silo reference + 10-silo scaled run:

  1. SIR-Gaussian IID      — all silos see the same 50/50 flu+pneumonia mix
  2. SIR-Gaussian Non-IID  — disease gradient (flu-only → pneumo-only) across silos
  3. Unknown disease        — controlled Gaussian schedule, Morven injected at R10

All runs use --no-ollama (template text, no LLM) for reproducibility.
Results written to results/scalability_10silo/.

Usage
-----
    python run_10silo.py                    # all six runs
    python run_10silo.py --skip-sir         # only unknown disease
    python run_10silo.py --skip-unknown     # only SIR runs
    python run_10silo.py --plot-only        # regenerate plots from saved results
    python run_10silo.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("results/scalability_10silo")

# Reference 3-silo numbers from the existing report (§7-8 and §9) used when
# 3-silo reference runs are not re-executed (--skip-sir-ref / --skip-unk-ref).
# These match the results already documented in fast_ablation/report.md.
_BASELINE_3S = {
    "iid":     {"final_diag_acc": 0.874, "peak_diag_acc": 0.967, "note": "sir-cal-2x seed=42"},
    "noniid":  {"final_diag_acc": 0.855, "peak_diag_acc": 0.948, "note": "sir-cal-2x non-IID §8"},
    "unknown": {
        "silhouette_at_r15": 0.87,
        "detection_round":   12,
        "final_diag_acc":    1.00,
        "note": "§9 gaussian+injection seed=42",
    },
}

# ── WorldConfig helper ────────────────────────────────────────────────────────

def _wc(n_agents: int, disease_weights, beta: float = 2.0):
    """Return a WorldConfig dict compatible with FLTrainConfig.world_configs."""
    from fl.train import WorldConfig
    return WorldConfig(
        num_agents       = n_agents,
        progressions     = ["Influenza", "Bacterial Pneumonia"],
        disease_weights  = disease_weights,
        disease_strategy = "Influenza",
        beta_scale       = beta,
        initial_seeds    = 8,
        contact_rate_sigma = 0.5,
        confusion_rate   = 0.10,
        end_condition    = "horizon",
        end_condition_param = 40,
    )


def _make_iid_worldcfgs(n_silos: int, n_agents: int = 150):
    """All silos see 50/50 flu+pneumonia."""
    return [_wc(n_agents, [0.5, 0.5]) for _ in range(n_silos)]


def _make_noniid_worldcfgs(n_silos: int, n_agents: int = 150):
    """Linear disease gradient: silo 0 = pure flu, silo N-1 = pure pneumo."""
    cfgs = []
    for i in range(n_silos):
        if n_silos == 1:
            w_flu = 0.5
        else:
            w_flu = 1.0 - i / (n_silos - 1)
        w_flu = max(0.05, min(0.95, w_flu))   # avoid degenerate pure-class
        cfgs.append(_wc(n_agents, [w_flu, 1 - w_flu]))
    return cfgs


# ── FL runner (SIR-based) ─────────────────────────────────────────────────────

def run_sir_experiment(
    label:        str,
    n_silos:      int,
    world_cfgs,
    out_sub:      str,
    device:       str = "cpu",
    seed:         int = 42,
) -> dict:
    """Run one SIR-based FL experiment. Returns summary dict."""
    from fl.train import FLTrainConfig, run_federated_training

    out = OUT_DIR / out_sub
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "round_metrics.json"
    if metrics_path.exists():
        print(f"  [skip]  {out_sub}  (cached)")
        return {"label": label, "ok": True, "cached": True,
                "round_metrics": json.loads(metrics_path.read_text())}

    print(f"\n{'═'*60}")
    print(f"  SIR-Gaussian  {label}")
    print(f"  silos={n_silos}  agents/silo={world_cfgs[0].num_agents}  seed={seed}")
    print(f"{'═'*60}\n")

    cfg = FLTrainConfig(
        num_silos           = n_silos,
        max_rounds          = 20,
        sim_days            = 2,
        min_events_to_train = 3,
        fedavg_min_examples = 0,
        local_epochs        = 3,
        end_condition       = "horizon",
        end_condition_param = 40,
        seed                = seed,
        use_ollama          = False,
        wandb_offline       = True,
        training_device     = device,
        world_configs       = world_cfgs,
        dataset_dir         = str(out),
        track_embeddings    = False,
    )

    t0 = time.time()
    try:
        run_federated_training(cfg)
        elapsed = time.time() - t0

        # Find the round_metrics.json written by run_federated_training inside
        # the timestamped subdirectory and copy it to out/round_metrics.json.
        # (run_federated_training writes to dataset_dir/<run_id>/round_metrics.json)
        sub_dirs = sorted(
            [d for d in out.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        copied = False
        for sub in reversed(sub_dirs):
            src = sub / "round_metrics.json"
            if src.exists():
                metrics_path.write_text(src.read_text())
                # Also copy summary.json
                ss = sub / "summary.json"
                if ss.exists():
                    (out / "summary.json").write_text(ss.read_text())
                copied = True
                break

        if not copied:
            print(f"  WARN: could not find round_metrics.json in {out}")
            return {"label": label, "ok": False, "elapsed": elapsed}

        metrics = json.loads(metrics_path.read_text())
        print(f"\n  Done in {elapsed:.0f}s — {len(metrics)} rounds saved")
        return {"label": label, "ok": True, "elapsed": elapsed, "round_metrics": metrics}

    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        print(f"  *** FAILED: {exc}")
        traceback.print_exc()
        return {"label": label, "ok": False, "elapsed": elapsed, "error": str(exc)}


# ── Unknown disease runner ────────────────────────────────────────────────────

def run_unknown_experiment(
    n_silos:    int,
    out_sub:    str,
    device:     str  = "cpu",
    seed:       int  = 42,
    local_only: bool = False,
) -> dict:
    """Run unknown disease detection experiment. Returns summary dict."""
    from run_unknown_disease import UnknownDiseaseConfig, run_unknown_disease

    out = OUT_DIR / out_sub
    out.mkdir(parents=True, exist_ok=True)
    run_name = out_sub

    metrics_path = OUT_DIR / out_sub / "round_metrics.json"
    if metrics_path.exists():
        print(f"  [skip]  {out_sub}  (cached)")
        return {
            "ok": True, "cached": True,
            "round_metrics": json.loads(metrics_path.read_text()),
        }

    mode_str = "local-only" if local_only else "federated"
    print(f"\n{'═'*60}")
    print(f"  Unknown Disease — {n_silos} silos  [{mode_str}]  inject R10")
    print(f"  seed={seed}")
    print(f"{'═'*60}\n")

    cfg = UnknownDiseaseConfig(
        schedule            = "gaussian",
        n_silos             = n_silos,
        events_per_silo     = 160,
        n_rounds            = 20,
        injection_round     = 10,
        injection_per_round = 8,
        do_inject           = True,
        local_only          = local_only,
        per_silo_snaps      = local_only,   # need per-silo logits to compute sil post-hoc
        training_device     = device,
        seed                = seed,
        results_dir         = str(OUT_DIR / out_sub / "_inner"),
        run_name            = run_name,
        snap_rounds         = [2, 5, 8, 10, 12, 15, 20],
    )

    t0 = time.time()
    try:
        result = run_unknown_disease(cfg)
        elapsed = time.time() - t0

        inner = OUT_DIR / out_sub / "_inner" / run_name

        # Copy per-round metrics and plots
        for fname in ("round_metrics.json", "silhouette.json",
                      "umap_evolution.png", "silhouette_curve.png",
                      "prob_unknown_panel.png"):
            src = inner / fname
            if src.exists():
                (out / fname).write_bytes(src.read_bytes())

        # For local-only runs, compute per-silo silhouette from saved logit files
        # (run_unknown_disease skips the global silhouette curve in local_only mode)
        if local_only:
            sil_by_silo = _compute_silo_silhouettes(inner, n_silos, cfg.snap_rounds)
            (out / "silhouette_per_silo.json").write_text(
                json.dumps(sil_by_silo, indent=2)
            )

        print(f"\n  Done in {elapsed:.0f}s")
        return {
            "ok": True, "elapsed": elapsed,
            "round_metrics": result["round_metrics"],
            "summary": result["summary"],
        }

    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        print(f"  *** FAILED: {exc}")
        traceback.print_exc()
        return {"ok": False, "elapsed": elapsed, "error": str(exc)}


def _compute_silo_silhouettes(
    inner_dir: Path,
    n_silos:   int,
    snap_rounds: list[int],
) -> dict[str, list[dict]]:
    """
    Compute Morven silhouette per silo from saved logit_r{rnd:02d}_silo{i}.npz files.

    Returns {silo_key: [{round, silhouette}, ...]} where silo_key is e.g. "silo_0".
    Used to show that unexposed silos (1..N-1) cannot detect Morven without federation.
    """
    from run_unknown_disease import (
        generate_fictional_probe_events, _silhouette_morven, _project_umap
    )

    probe_events = generate_fictional_probe_events(n_per_band=12, seed=999)
    probe_labels = [ev.ground_truth for ev in probe_events]

    result: dict[str, list[dict]] = {}
    for i in range(n_silos):
        silo_key = f"silo_{i}"
        curve = []
        for rnd in snap_rounds:
            logit_path = inner_dir / f"logits_r{rnd:02d}_silo{i}.npz"
            if not logit_path.exists():
                continue
            data   = np.load(logit_path)
            logits = data["logits"]
            coords = _project_umap(logits, seed=42)
            sil    = _silhouette_morven(coords, probe_labels)
            curve.append({"round": rnd, "silhouette": float(sil)})
        result[silo_key] = curve
    return result


# ── Comparison plots ──────────────────────────────────────────────────────────

def _load_metrics(out_sub: str) -> list[dict] | None:
    p = OUT_DIR / out_sub / "round_metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _load_sil(out_sub: str) -> list[dict] | None:
    p = OUT_DIR / out_sub / "silhouette.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _diag_curve(metrics: list[dict]) -> tuple[list[int], list[float]]:
    xs = [m["round"] for m in metrics]
    ys = [m.get("agg_diag_acc", float("nan")) for m in metrics]
    return xs, ys


def _loss_curve(metrics: list[dict]) -> tuple[list[int], list[float]]:
    xs = [m["round"] for m in metrics]
    ys = [m.get("mean_loss", float("nan")) for m in metrics]
    return xs, ys


def plot_sir_comparison() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da"); sp.set_linewidth(0.5)

    ax_iid_acc, ax_iid_loss, ax_noniid_acc, ax_noniid_loss = axes.flat

    _C3  = "#4361ee"   # 3-silo colour
    _C10 = "#e63946"   # 10-silo colour

    # IID accuracy
    m3  = _load_metrics("iid_3silo")
    m10 = _load_metrics("iid_10silo")

    ax_iid_acc.set_title("IID — diagnostic accuracy", fontsize=10)
    ax_iid_acc.set_ylabel("Aggregated diagnostic accuracy")
    ax_iid_acc.set_xlabel("FL Round")
    ax_iid_acc.set_ylim(-0.05, 1.05)
    if m3:
        xs, ys = _diag_curve(m3)
        ax_iid_acc.plot(xs, ys, color=_C3, linewidth=2.0, marker="o",
                        markersize=4, label="3 silos")
    else:
        ax_iid_acc.axhline(_BASELINE_3S["iid"]["peak_diag_acc"], color=_C3,
                            linestyle="--", linewidth=1.5, label=f"3 silos (ref §8)")
    if m10:
        xs, ys = _diag_curve(m10)
        ax_iid_acc.plot(xs, ys, color=_C10, linewidth=2.0, marker="s",
                        markersize=4, label="10 silos")
    ax_iid_acc.legend(fontsize=9)

    # IID loss
    ax_iid_loss.set_title("IID — training loss", fontsize=10)
    ax_iid_loss.set_ylabel("Mean training loss"); ax_iid_loss.set_xlabel("FL Round")
    if m3:
        xs, ys = _loss_curve(m3)
        ax_iid_loss.plot(xs, ys, color=_C3, linewidth=2.0, marker="o", markersize=4, label="3 silos")
    if m10:
        xs, ys = _loss_curve(m10)
        ax_iid_loss.plot(xs, ys, color=_C10, linewidth=2.0, marker="s", markersize=4, label="10 silos")
    ax_iid_loss.legend(fontsize=9)

    # Non-IID accuracy
    m3n  = _load_metrics("noniid_3silo")
    m10n = _load_metrics("noniid_10silo")

    ax_noniid_acc.set_title("Non-IID — diagnostic accuracy", fontsize=10)
    ax_noniid_acc.set_ylabel("Aggregated diagnostic accuracy")
    ax_noniid_acc.set_xlabel("FL Round")
    ax_noniid_acc.set_ylim(-0.05, 1.05)
    if m3n:
        xs, ys = _diag_curve(m3n)
        ax_noniid_acc.plot(xs, ys, color=_C3, linewidth=2.0, marker="o", markersize=4, label="3 silos")
    else:
        ax_noniid_acc.axhline(_BASELINE_3S["noniid"]["peak_diag_acc"], color=_C3,
                               linestyle="--", linewidth=1.5, label="3 silos (ref §8)")
    if m10n:
        xs, ys = _diag_curve(m10n)
        ax_noniid_acc.plot(xs, ys, color=_C10, linewidth=2.0, marker="s", markersize=4, label="10 silos")
    ax_noniid_acc.legend(fontsize=9)

    # Non-IID loss
    ax_noniid_loss.set_title("Non-IID — training loss", fontsize=10)
    ax_noniid_loss.set_ylabel("Mean training loss"); ax_noniid_loss.set_xlabel("FL Round")
    if m3n:
        xs, ys = _loss_curve(m3n)
        ax_noniid_loss.plot(xs, ys, color=_C3, linewidth=2.0, marker="o", markersize=4, label="3 silos")
    if m10n:
        xs, ys = _loss_curve(m10n)
        ax_noniid_loss.plot(xs, ys, color=_C10, linewidth=2.0, marker="s", markersize=4, label="10 silos")
    ax_noniid_loss.legend(fontsize=9)

    fig.suptitle(
        "Silo scalability: 3 vs 10 silos — SIR-Gaussian epidemic\n"
        "150 agents/silo · β=2.0 · horizon=40 sim-days · 20 rounds · no Ollama (templates)",
        fontsize=10, color="#212529",
    )
    out = OUT_DIR / "sir_comparison.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_silo_accuracy_heatmap(n_silos: int, out_sub: str, title: str) -> None:
    """Per-silo accuracy heatmap (rounds × silos)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _load_metrics(out_sub)
    if not metrics:
        return

    # Build matrix [rounds × silos]
    rounds = [m["round"] for m in metrics]
    silo_accs = []
    for m in metrics:
        row = m.get("silo_diag", [float("nan")] * n_silos)
        # Pad or trim to n_silos
        row = list(row) + [float("nan")] * max(0, n_silos - len(row))
        silo_accs.append(row[:n_silos])

    mat = np.array(silo_accs)  # (n_rounds, n_silos)

    fig, ax = plt.subplots(figsize=(max(8, n_silos * 0.9), 5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    im = ax.imshow(mat.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   origin="upper")
    fig.colorbar(im, ax=ax, label="Diagnostic accuracy")
    ax.set_xlabel("FL Round"); ax.set_ylabel("Silo")
    ax.set_xticks(range(len(rounds))); ax.set_xticklabels(rounds, fontsize=7)
    ax.set_yticks(range(n_silos)); ax.set_yticklabels([f"S{i}" for i in range(n_silos)], fontsize=8)
    ax.set_title(title, fontsize=10, color="#212529")

    out = OUT_DIR / f"{out_sub}_silo_heatmap.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_unknown_comparison() -> None:
    """Silhouette curve + P(unknown) comparison: 3-silo vs 10-silo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da"); sp.set_linewidth(0.5)

    _C3  = "#4361ee"
    _C10 = "#e63946"

    ax_sil, ax_acc = axes

    # Silhouette
    sil3  = _load_sil("unknown_3silo")
    sil10 = _load_sil("unknown_10silo")

    ax_sil.axhline(0, color="#6c757d", linewidth=0.8, linestyle="--", label="sil=0")
    ax_sil.axvline(10, color="#888", linewidth=0.8, linestyle=":", label="injection R10")
    if sil3:
        xs = [d["round"] for d in sil3]; ys = [d["silhouette"] for d in sil3]
        ax_sil.plot(xs, ys, color=_C3, marker="o", markersize=5, linewidth=2.0, label="3 silos")
    else:
        ax_sil.scatter([15], [0.87], color=_C3, s=80, zorder=4)
        ax_sil.annotate("ref §9: 0.87 @R15", (15, 0.87), fontsize=8,
                         xytext=(13, 0.75), color=_C3)
    if sil10:
        xs = [d["round"] for d in sil10]; ys = [d["silhouette"] for d in sil10]
        ax_sil.plot(xs, ys, color=_C10, marker="s", markersize=5, linewidth=2.0, label="10 silos")

    ax_sil.set_xlabel("FL Round"); ax_sil.set_ylabel("Morven silhouette score")
    ax_sil.set_title("Novel disease separability (Morven silhouette)", fontsize=10)
    ax_sil.set_ylim(-0.35, 1.05)
    ax_sil.legend(fontsize=9)

    # Diagnostic accuracy for known diseases
    m3  = _load_metrics("unknown_3silo")
    m10 = _load_metrics("unknown_10silo")

    ax_acc.axvline(10, color="#888", linewidth=0.8, linestyle=":", label="injection R10")
    if m3:
        xs, ys = _diag_curve(m3)
        ax_acc.plot(xs, ys, color=_C3, marker="o", markersize=5, linewidth=2.0, label="3 silos")
    if m10:
        xs, ys = _diag_curve(m10)
        ax_acc.plot(xs, ys, color=_C10, marker="s", markersize=5, linewidth=2.0, label="10 silos")

    ax_acc.set_xlabel("FL Round"); ax_acc.set_ylabel("Known-disease diagnostic accuracy")
    ax_acc.set_title("Known-disease accuracy (Velarex+Sornathis) under novel injection", fontsize=10)
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend(fontsize=9)

    fig.suptitle(
        "Unknown disease detection — silo scalability (3 vs 10 silos)\n"
        "Gaussian schedule · 160 events/silo · Morven injected at R10 into silo_0",
        fontsize=10, color="#212529",
    )
    out = OUT_DIR / "unknown_comparison.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_diffusion_claim() -> None:
    """
    The key plot for claim (iii): federated vs local-only Morven silhouette.

    Left panel  — 10-silo: global (fed) vs silo_0 (local, exposed) vs silo_1 (local, unexposed)
    Right panel — 3-silo:  same breakdown

    The critical comparison: silo_1's local-only silhouette stays ≈ 0 throughout
    (unexposed silo cannot detect Morven alone), while the federated global model
    achieves high silhouette because silo_0's signal propagates via FedAvg.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da"); sp.set_linewidth(0.5)

    _C_FED      = "#2a9d8f"   # federated global — teal
    _C_EXPOSED  = "#f4a261"   # silo_0 (exposed, local) — orange
    _C_HIDDEN   = "#e63946"   # silo_1 (unexposed, local) — red

    for ax, n_silos, fed_sub, local_sub, title in [
        (axes[0], 10, "unknown_10silo", "unknown_10silo_local", "10 silos"),
        (axes[1],  3, "unknown_3silo",  "unknown_3silo_local",  "3 silos"),
    ]:
        ax.axhline(0,  color="#6c757d", linewidth=0.8, linestyle="--")
        ax.axvline(10, color="#888",    linewidth=0.8, linestyle=":",
                   label="injection R10")

        # Federated global silhouette
        sil_fed = _load_sil(fed_sub)
        if sil_fed:
            xs = [d["round"] for d in sil_fed]
            ys = [d["silhouette"] for d in sil_fed]
            ax.plot(xs, ys, color=_C_FED, linewidth=2.2, marker="o",
                    markersize=5, label="federated (global model)", zorder=4)

        # Local-only per-silo silhouettes
        p = OUT_DIR / local_sub / "silhouette_per_silo.json"
        if p.exists():
            per_silo = json.loads(p.read_text())

            # silo_0 — the one that sees Morven
            s0 = per_silo.get("silo_0", [])
            if s0:
                xs0 = [d["round"] for d in s0]
                ys0 = [d["silhouette"] for d in s0]
                ax.plot(xs0, ys0, color=_C_EXPOSED, linewidth=1.6, marker="s",
                        markersize=4, linestyle="--",
                        label="local-only silo_0 (Morven-exposed)", zorder=3)

            # silo_1 — unexposed; representative of all non-exposed silos
            s1 = per_silo.get("silo_1", [])
            if s1:
                xs1 = [d["round"] for d in s1]
                ys1 = [d["silhouette"] for d in s1]
                ax.plot(xs1, ys1, color=_C_HIDDEN, linewidth=1.6, marker="^",
                        markersize=4, linestyle="--",
                        label="local-only silo_1 (never sees Morven)", zorder=3)

        ax.set_title(f"{title} — federated diffusion vs local-only", fontsize=10)
        ax.set_xlabel("FL Round"); ax.set_ylabel("Morven silhouette score")
        ax.set_ylim(-0.45, 1.05)
        ax.legend(fontsize=8, framealpha=0.92)

    fig.suptitle(
        "Claim (iii): cross-silo diffusion enables detection in unexposed silos\n"
        "Without federation, silo_1 silhouette ≈ 0 throughout. "
        "With FedAvg, silo_0's Morven signal propagates → global model detects.",
        fontsize=10, color="#212529",
    )
    out = OUT_DIR / "diffusion_claim.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_zombie_silos(n_silos: int, out_sub: str, title: str) -> None:
    """Stacked bar of train_sizes per silo per round — shows zombie silo pattern."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    metrics = _load_metrics(out_sub)
    if not metrics:
        return

    rounds = [m["round"] for m in metrics]
    train_mat = np.array([
        (m.get("train_sizes") or [0] * n_silos)[:n_silos]
        for m in metrics
    ], dtype=float)  # (n_rounds, n_silos)

    colors = get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    bottom = np.zeros(len(rounds))
    for i in range(n_silos):
        ax.bar(rounds, train_mat[:, i], bottom=bottom,
               color=colors[i % 10], label=f"S{i}", alpha=0.85, width=0.7)
        bottom += train_mat[:, i]

    ax.set_xlabel("FL Round"); ax.set_ylabel("Training examples contributed")
    ax.set_title(title, fontsize=10, color="#212529")
    ax.legend(fontsize=7, ncol=max(1, n_silos // 3),
              loc="upper right", framealpha=0.9)
    for sp in ax.spines.values():
        sp.set_color("#ced4da"); sp.set_linewidth(0.5)

    out = OUT_DIR / f"{out_sub}_zombie.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Summary JSON ──────────────────────────────────────────────────────────────

def _summarise_sir(out_sub: str) -> dict:
    m = _load_metrics(out_sub)
    if not m:
        return {}
    diag = [x.get("agg_diag_acc", float("nan")) for x in m]
    loss = [x.get("mean_loss", float("nan")) for x in m]
    valid_diag = [x for x in diag if x == x]
    return {
        "peak_diag_acc":  round(max(valid_diag), 4) if valid_diag else None,
        "final_diag_acc": round(valid_diag[-1], 4)  if valid_diag else None,
        "mean_loss_r1":   round(loss[0], 4) if loss and loss[0] == loss[0] else None,
        "mean_loss_final":round(loss[-1], 4) if loss and loss[-1] == loss[-1] else None,
        "n_rounds":       len(m),
    }


def _summarise_unknown(out_sub: str) -> dict:
    m  = _load_metrics(out_sub)
    sl = _load_sil(out_sub)
    if not m:
        return {}

    diag = [x.get("agg_diag_acc", float("nan")) for x in m]
    valid_diag = [x for x in diag if x == x]

    sil_vals = {d["round"]: d["silhouette"] for d in sl} if sl else {}
    detect_round = None
    for rnd in sorted(sil_vals):
        if sil_vals[rnd] > 0:
            detect_round = rnd
            break

    return {
        "final_diag_acc":    round(valid_diag[-1], 4) if valid_diag else None,
        "silhouette_at_r15": round(sil_vals.get(15, float("nan")), 4),
        "silhouette_at_r20": round(sil_vals.get(20, float("nan")), 4),
        "detection_round":   detect_round,
    }


def print_summary_table(summary: dict) -> None:
    print("\n" + "═" * 60)
    print("  SCALABILITY SUMMARY")
    print("═" * 60)

    for key, label in [
        ("iid_3silo",     "IID 3-silo (ref)  "),
        ("iid_10silo",    "IID 10-silo        "),
        ("noniid_3silo",  "Non-IID 3-silo (ref)"),
        ("noniid_10silo", "Non-IID 10-silo    "),
    ]:
        d = summary.get(key, {})
        peak  = d.get("peak_diag_acc",  "—")
        final = d.get("final_diag_acc", "—")
        print(f"  {label}  peak={peak}  final={final}")

    print()
    for key, label in [
        ("unknown_3silo",  "Unknown 3-silo (ref)  "),
        ("unknown_10silo", "Unknown 10-silo        "),
    ]:
        d = summary.get(key, {})
        sil15 = d.get("silhouette_at_r15", "—")
        det   = d.get("detection_round",   "—")
        fin   = d.get("final_diag_acc",    "—")
        print(f"  {label}  sil@R15={sil15}  detect_R={det}  final={fin}")

    print("═" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",       default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--out-dir",      default=None,
                    help="Override output directory (default: results/scalability_10silo)")
    ap.add_argument("--skip-sir",     action="store_true", help="Skip SIR experiments")
    ap.add_argument("--skip-unknown", action="store_true", help="Skip unknown disease experiment")
    ap.add_argument("--plot-only",    action="store_true", help="Only regenerate plots")
    ap.add_argument("--no-ref",       action="store_true",
                    help="Skip ALL 3-silo reference re-runs (use hardcoded baseline numbers)")
    ap.add_argument("--no-sir-ref",   action="store_true",
                    help="Skip 3-silo SIR reference re-runs only (still runs 3-silo unknown)")
    args = ap.parse_args()
    # --no-ref implies --no-sir-ref
    _skip_sir_ref = args.no_ref or args.no_sir_ref

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    summary: dict = {}

    # ── SIR experiments ───────────────────────────────────────────────────────
    if not args.skip_sir and not args.plot_only:
        # 3-silo IID reference
        if not _skip_sir_ref:
            r = run_sir_experiment(
                "3-silo IID (reference)",
                n_silos    = 3,
                world_cfgs = _make_iid_worldcfgs(3),
                out_sub    = "iid_3silo",
                device     = args.device,
                seed       = args.seed,
            )
            if r.get("ok") and r.get("round_metrics"):
                summary["iid_3silo"] = _summarise_sir("iid_3silo")

        # 10-silo IID
        r = run_sir_experiment(
            "10-silo IID",
            n_silos    = 10,
            world_cfgs = _make_iid_worldcfgs(10),
            out_sub    = "iid_10silo",
            device     = args.device,
            seed       = args.seed,
        )
        if r.get("ok") and r.get("round_metrics"):
            summary["iid_10silo"] = _summarise_sir("iid_10silo")

        # 3-silo Non-IID reference
        if not _skip_sir_ref:
            r = run_sir_experiment(
                "3-silo Non-IID (reference)",
                n_silos    = 3,
                world_cfgs = _make_noniid_worldcfgs(3),
                out_sub    = "noniid_3silo",
                device     = args.device,
                seed       = args.seed,
            )
            if r.get("ok") and r.get("round_metrics"):
                summary["noniid_3silo"] = _summarise_sir("noniid_3silo")

        # 10-silo Non-IID
        r = run_sir_experiment(
            "10-silo Non-IID",
            n_silos    = 10,
            world_cfgs = _make_noniid_worldcfgs(10),
            out_sub    = "noniid_10silo",
            device     = args.device,
            seed       = args.seed,
        )
        if r.get("ok") and r.get("round_metrics"):
            summary["noniid_10silo"] = _summarise_sir("noniid_10silo")

    # ── Unknown disease experiments ───────────────────────────────────────────
    if not args.skip_unknown and not args.plot_only:
        # 3-silo federated reference
        if not args.no_ref:
            r = run_unknown_experiment(
                n_silos = 3, out_sub = "unknown_3silo",
                device  = args.device, seed = args.seed,
            )
            if r.get("ok"):
                summary["unknown_3silo"] = _summarise_unknown("unknown_3silo")

        # 10-silo federated
        r = run_unknown_experiment(
            n_silos = 10, out_sub = "unknown_10silo",
            device  = args.device, seed = args.seed,
        )
        if r.get("ok"):
            summary["unknown_10silo"] = _summarise_unknown("unknown_10silo")

        # Local-only baselines (claim iii: unexposed silos can't detect without federation)
        if not args.no_ref:
            r = run_unknown_experiment(
                n_silos = 3, out_sub = "unknown_3silo_local",
                device  = args.device, seed = args.seed, local_only = True,
            )
        r = run_unknown_experiment(
            n_silos = 10, out_sub = "unknown_10silo_local",
            device  = args.device, seed = args.seed, local_only = True,
        )

    # ── Fill in reference baselines from constants if not re-run ─────────────
    for key in ("iid_3silo", "noniid_3silo"):
        if key not in summary:
            kind = key.replace("_3silo", "")
            summary[key] = _BASELINE_3S.get(kind, {})
    if "unknown_3silo" not in summary:
        summary["unknown_3silo"] = _BASELINE_3S["unknown"]

    # ── Load any cached metrics we might have missed ──────────────────────────
    for out_sub in ("iid_3silo", "iid_10silo", "noniid_3silo", "noniid_10silo"):
        if out_sub not in summary:
            s = _summarise_sir(out_sub)
            if s:
                summary[out_sub] = s
    for out_sub in ("unknown_3silo", "unknown_10silo"):
        if out_sub not in summary:
            s = _summarise_unknown(out_sub)
            if s:
                summary[out_sub] = s

    # ── Save summary JSON ─────────────────────────────────────────────────────
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_sir_comparison()
    plot_unknown_comparison()
    plot_diffusion_claim()

    # Per-silo heatmaps and zombie plots
    for (n, sub, lbl) in [
        (3,  "iid_3silo",    "IID 3-silo — per-silo accuracy"),
        (10, "iid_10silo",   "IID 10-silo — per-silo accuracy"),
        (3,  "noniid_3silo", "Non-IID 3-silo — per-silo accuracy"),
        (10, "noniid_10silo","Non-IID 10-silo — per-silo accuracy"),
    ]:
        plot_silo_accuracy_heatmap(n, sub, lbl)
        plot_zombie_silos(n, sub, f"{lbl} — training examples per silo")

    print_summary_table(summary)

    wall = time.time() - t_wall
    h, rem = divmod(int(wall), 3600); m, s = divmod(rem, 60)
    print(f"\n  Total wall time: {h}h{m:02d}m{s:02d}s")
    print(f"  Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
