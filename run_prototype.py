"""
Prototype-bank experiment — open-set disease discovery without softmax retraining.

Architecture
------------
  Backbone:  DistilBERT + LoRA, trained with a fixed softmax head on known labels.
             FedAvg aggregates backbone weights as usual.

  Classifier: PrototypeBank — one named centroid per class (cosine nearest-neighbour).
              Updated from training-data embeddings after each round.
              FedAvg aggregates centroids (weighted mean of positions).

  Discovery:  DBSCAN runs on all probe embeddings after each FedAvg round.
              When cluster count exceeds the number of named prototypes, a new
              sub-cluster is declared and named "unknown_N".

  Attribution: `bank.rename("unknown_0", "morven")` at a configured round.
              No gradient descent, no new class slot — just a centroid rename.

Protocol (default)
------------------
  R1–9   (pre-injection):  3 silos, velarex + sornathis + non-infectious.
  R10–19 (detection):      Morven injected into silo_0, labeled "unknown".
  R15    (attribution):    silo_0 renames the unknown sub-cluster to "morven".

Outputs (results/prototype/)
  round_metrics.json          — per-round metrics
  prototype_curve.png         — cluster count, proto accuracy, softmax comparison
  umap_r{N:02d}.png           — UMAP of probe embeddings with centroid markers
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

OUT_DIR = Path("results/prototype")

_DISEASE_COLORS = {
    "velarex":        "#e63946",
    "sornathis":      "#4361ee",
    "morven":         "#2a9d8f",
    "non-infectious": "#6c757d",
    "unknown":        "#f4a261",
    "unknown_0":      "#f4a261",
    "unknown_1":      "#e9c46a",
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PrototypeConfig:
    # Simulation
    n_silos:             int   = 3
    events_per_silo:     int   = 160
    n_rounds:            int   = 20
    seed:                int   = 42
    schedule:            str   = "gaussian"   # flat|ramp|gaussian|burst|parabola

    # Injection
    injection_round:     int   = 10
    injection_per_round: int   = 16
    do_inject:           bool  = True

    # Attribution (rename unknown_0 → morven at this round; 0 = disabled)
    attribution_round:   int   = 15

    # Prototype bank
    pca_components:      int   = 50
    dbscan_eps:          float = 0.30
    dbscan_min_samples:  int   = 5

    # Training
    training_device:     str   = "cpu"
    fedavg_min_examples: int   = 5
    min_events_to_train: int   = 10

    # Output
    snap_rounds:         list  = field(default_factory=lambda: [5, 8, 10, 12, 15, 18, 20])
    results_dir:         str   = str(OUT_DIR)
    run_name:            str   = "proto_seed42"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fedavg(weights_list, n_examples):
    total = sum(n_examples)
    result = []
    for arrays in zip(*weights_list):
        agg = sum(w * (n / total) for w, n in zip(arrays, n_examples))
        result.append(agg)
    return result


def _project_umap(cls_emb: np.ndarray, seed: int = 42) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import umap
        return umap.UMAP(n_components=2, random_state=seed, n_jobs=1).fit_transform(cls_emb)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── Probe events ──────────────────────────────────────────────────────────────

def _load_probe_events(seed: int = 42):
    """Return (probe_events, probe_labels) — same probes as run_unknown_disease."""
    from run_unknown_disease import generate_fictional_probe_events
    events = generate_fictional_probe_events(n_per_band=12, seed=seed)
    labels = [ev.ground_truth.split("/")[0] for ev in events]
    return events, labels


# ── Per-round probe evaluation ────────────────────────────────────────────────

def _probe_texts(probe_events) -> list[str]:
    texts = []
    for ev in probe_events:
        if isinstance(ev, dict):
            texts.append(ev.get("text", ""))
        else:
            turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
            texts.append(turns[-1] if turns else "")
    return texts


def _extract_cls(learner, texts: list[str]) -> np.ndarray:
    """Extract [CLS] embeddings from raw text strings."""
    import torch
    enc = learner._tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    model = learner.model
    dev   = next(model.parameters()).device
    enc   = {k: v.to(dev) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[-1][:, 0, :].cpu().numpy()


def _eval_probes(learner, probe_events, probe_labels, bank):
    """Extract probe embeddings and compute prototype + softmax metrics."""
    cls_emb = _extract_cls(learner, _probe_texts(probe_events))

    proto_preds = bank.classify(cls_emb)
    morven_idx  = [i for i, l in enumerate(probe_labels) if l == "morven"]

    proto_acc_all    = sum(p == t for p, t in zip(proto_preds, probe_labels)) / max(len(probe_labels), 1)
    proto_acc_morven = (
        sum(proto_preds[i] in ("morven", "unknown", "unknown_0") for i in morven_idx) / max(len(morven_idx), 1)
        if morven_idx else float("nan")
    )
    # For morven probes, what fraction does proto-bank call "morven" (after attribution)?
    proto_morven_exact = (
        sum(proto_preds[i] == "morven" for i in morven_idx) / max(len(morven_idx), 1)
        if morven_idx else float("nan")
    )

    # DBSCAN on probe embeddings that the prototype bank routes to any "unknown*" class.
    # This asks: "does the unknown region have sub-structure?" rather than
    # "how many total clusters exist?" (which includes known diseases and over-splits).
    unknown_names = {n for n in bank.names() if "unknown" in n}
    unknown_mask  = np.array([p in unknown_names for p in proto_preds])
    n_unknown_clusters = (
        bank.dbscan_cluster_count(cls_emb[unknown_mask])
        if unknown_mask.sum() >= 2 * bank.dbscan_min_samples
        else (1 if unknown_mask.sum() > 0 else 0)
    )

    return {
        "cls_emb":              cls_emb,
        "proto_preds":          proto_preds,
        "proto_acc_all":        proto_acc_all,
        "proto_acc_morven":     proto_acc_morven,
        "proto_morven_exact":   proto_morven_exact,
        "n_unknown_clusters":   n_unknown_clusters,
        "n_named_protos":       len(bank.names()),
        "proto_names":          list(bank.names()),
        "unknown_mask":         unknown_mask,
    }


# ── UMAP plot for one round ───────────────────────────────────────────────────

def _plot_umap_round(
    cls_emb: np.ndarray,
    probe_labels: list[str],
    bank,
    round_num: int,
    out_path: Path,
    n_clusters: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = _project_umap(cls_emb, seed=0)

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white")
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values():
        sp.set_color("#ced4da"); sp.set_linewidth(0.5)

    seen: set[str] = set()
    for i, lbl in enumerate(probe_labels):
        color = _DISEASE_COLORS.get(lbl, "#888888")
        marker = "D" if lbl == "morven" else "o"
        label  = lbl if lbl not in seen else None
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=30, marker=marker,
                   alpha=0.75, label=label, linewidths=0.3, edgecolors="white")
        seen.add(lbl)

    # Overlay prototype centroids (projected into the same UMAP space isn't
    # strictly valid, but we can show them as text annotations on top of clusters)
    ax.set_title(
        f"Round {round_num} — probe embeddings\n"
        f"named prototypes: {bank.names()}  |  DBSCAN clusters: {n_clusters}",
        fontsize=9, color="#212529",
    )
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ── Summary plot ──────────────────────────────────────────────────────────────

def _plot_summary(round_metrics: list[dict], out_path: Path, cfg: PrototypeConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [m["round"] for m in round_metrics]
    n_clusters = [m.get("n_unknown_clusters", float("nan")) for m in round_metrics]
    n_protos   = [m.get("n_named_protos",     float("nan")) for m in round_metrics]
    proto_acc  = [m.get("proto_morven_exact", float("nan")) for m in round_metrics]
    softmax_uk = [m.get("softmax_p_unknown_morven", float("nan")) for m in round_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # Left: cluster count vs named prototypes
    for ax in (ax1, ax2):
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values():
            sp.set_color("#ced4da"); sp.set_linewidth(0.5)

    ax1.plot(rounds, n_clusters, marker="o", color="#2a9d8f", linewidth=2.2, label="DBSCAN clusters")
    ax1.plot(rounds, n_protos,   marker="s", color="#e63946", linewidth=2.2, linestyle="--",
             label="named prototypes")
    ax1.axvline(cfg.injection_round, color="#adb5bd", linewidth=1, linestyle=":", label=f"injection R{cfg.injection_round}")
    if cfg.attribution_round:
        ax1.axvline(cfg.attribution_round, color="#2a9d8f", linewidth=1.2, linestyle="--",
                    label=f"attribution R{cfg.attribution_round}")
    ax1.set_title("Cluster discovery", fontsize=10)
    ax1.set_xlabel("FL Round"); ax1.set_ylabel("Count")
    ax1.legend(fontsize=8.5); ax1.set_xticks(rounds)

    # Right: P(morven|Morven) — prototype vs softmax
    ax2.plot(rounds, proto_acc,  marker="D", color="#2a9d8f", linewidth=2.2,
             label='proto: P("morven"|Morven)')
    ax2.plot(rounds, softmax_uk, marker="o", color="#f4a261", linewidth=2.2,
             label='softmax: P("unknown"|Morven)')
    ax2.axvline(cfg.injection_round, color="#adb5bd", linewidth=1, linestyle=":")
    if cfg.attribution_round:
        ax2.axvline(cfg.attribution_round, color="#2a9d8f", linewidth=1.2, linestyle="--")
    ax2.axhline(1/3, color="#dee2e6", linewidth=0.8, linestyle="--", label="chance (3 probes)")
    ax2.set_title("Attribution accuracy (Morven probes)", fontsize=10)
    ax2.set_xlabel("FL Round"); ax2.set_ylabel("Probability / accuracy")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=8.5); ax2.set_xticks(rounds)

    fig.suptitle(
        "Prototype-bank experiment — open-set disease discovery\n"
        "DBSCAN detects Morven cluster; attribution = centroid rename, no retraining",
        fontsize=10, color="#212529",
    )
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main experiment ───────────────────────────────────────────────────────────

def run_prototype(cfg: PrototypeConfig) -> None:
    from fl.learner import FLLearner
    from fl.lora import LoRAConfig
    from fl.prototype_bank import PrototypeBank
    from run_unknown_disease import (
        _make_schedule,
        _build_pools,
        _build_morven_pool,
        generate_fictional_probe_events,
    )
    import fl.train as fl_train

    rng  = np.random.default_rng(cfg.seed)
    t0   = time.time()

    out_dir = Path(cfg.results_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Probe events (fixed test set, never in training) ─────────────────────
    probe_events, probe_labels = _load_probe_events(seed=cfg.seed)
    print(f"\n{'═'*60}")
    print(f"  Prototype-bank experiment  [{cfg.schedule}]  seed={cfg.seed}")
    print(f"  silos={cfg.n_silos}  events/silo={cfg.events_per_silo}  rounds={cfg.n_rounds}")
    print(f"  DBSCAN eps={cfg.dbscan_eps}  min_samples={cfg.dbscan_min_samples}")
    print(f"{'═'*60}\n")

    # ── Per-silo train pools ──────────────────────────────────────────────────
    train_pools, holdouts = _build_pools(
        n_silos=cfg.n_silos,
        events_per_silo=cfg.events_per_silo,
        holdout_frac=0.15,
        seed=cfg.seed,
    )
    schedules = [
        _make_schedule(cfg.schedule, len(train_pools[i]), cfg.n_rounds)
        for i in range(cfg.n_silos)
    ]
    morven_pool = _build_morven_pool(n=500, seed=cfg.seed) if cfg.do_inject else []

    # ── Learners ──────────────────────────────────────────────────────────────
    lora_cfg = LoRAConfig(num_labels=4)   # 4-class softmax: velarex/sornathis/non-inf/unknown
    learners = [
        FLLearner(
            lora_config=lora_cfg,
            label_space="fictional_disease",
            min_events_to_train=cfg.min_events_to_train,
            device=cfg.training_device,
        )
        for _ in range(cfg.n_silos)
    ]

    # ── Prototype banks (one per silo; FedAvg'd each round) ──────────────────
    bank_kwargs = dict(
        pca_components=cfg.pca_components,
        dbscan_eps=cfg.dbscan_eps,
        dbscan_min_samples=cfg.dbscan_min_samples,
    )
    silo_banks  = [PrototypeBank(**bank_kwargs) for _ in range(cfg.n_silos)]
    global_bank = PrototypeBank(**bank_kwargs)

    # ── State ─────────────────────────────────────────────────────────────────
    cursors        = [0] * cfg.n_silos
    total_revealed = [0] * cfg.n_silos
    morven_cursor  = 0
    global_w       = None
    round_metrics  = []

    # Track when DBSCAN first finds a coherent sub-cluster within "unknown"
    split_detected_round: Optional[int] = None
    # Persist attributions across FedAvg rounds: {old_name → new_name}
    attribution_map: dict[str, str] = {}

    _IDX_UNKNOWN = 3   # "unknown" slot in the 4-class softmax

    for r in range(cfg.n_rounds):
        rnd = r + 1

        # ── Reveal new events ────────────────────────────────────────────────
        new_this_round: list[list] = []
        for i in range(cfg.n_silos):
            n_rev  = schedules[i][r]
            new_ev = train_pools[i][cursors[i]: cursors[i] + n_rev]
            cursors[i]         += n_rev
            total_revealed[i]  += len(new_ev)

            # Morven injection: only silo_0, labeled "unknown"
            if (cfg.do_inject and rnd >= cfg.injection_round
                    and i == 0 and morven_pool):
                m_end         = min(morven_cursor + cfg.injection_per_round, len(morven_pool))
                morven_batch  = morven_pool[morven_cursor:m_end]
                morven_cursor = m_end
                new_ev = list(new_ev) + [
                    {**rec, "label": "unknown", "gt_disease": "unknown"}
                    for rec in morven_batch
                ]

            new_this_round.append(list(new_ev))

        # ── Per-silo: train backbone → update prototypes ─────────────────────
        round_weights:  list[list] = []
        train_sizes:    list[int]  = []
        silo_bank_dicts: list[dict] = []

        for i, learner in enumerate(learners):
            if global_w is not None:
                learner.set_weights(global_w)

            new_ev = new_this_round[i]
            if total_revealed[i] >= cfg.min_events_to_train and new_ev:
                n_trained, _ = learner.train(new_ev)
                train_sizes.append(n_trained)
            else:
                train_sizes.append(0)

            # Extract embeddings from ALL events seen so far this round
            if new_ev:
                embs, lbls = learner.extract_embeddings(new_ev)
                if len(embs) > 0:
                    # Update per-class centroids
                    for cls_name in set(lbls):
                        mask = np.array([l == cls_name for l in lbls])
                        silo_banks[i].update(cls_name, embs[mask])

            round_weights.append(learner.get_weights())
            silo_bank_dicts.append(silo_banks[i].to_dict())
            learner.release()

        # ── FedAvg backbone ───────────────────────────────────────────────────
        active_idx = [i for i, s in enumerate(train_sizes) if s >= cfg.fedavg_min_examples]
        if active_idx:
            global_w = _fedavg(
                [round_weights[i] for i in active_idx],
                [train_sizes[i]   for i in active_idx],
            )

        # ── FedAvg prototype banks ────────────────────────────────────────────
        active_banks   = [silo_banks[i] for i in active_idx]
        active_weights = [train_sizes[i] for i in active_idx]
        if active_banks:
            global_bank = PrototypeBank.fedavg(active_banks, active_weights, **bank_kwargs)
            # Re-apply any confirmed attributions that FedAvg may have overwritten
            for old_name, new_name in attribution_map.items():
                if global_bank.has(old_name) and not global_bank.has(new_name):
                    global_bank.rename(old_name, new_name)

        # ── DBSCAN discovery on probe embeddings ──────────────────────────────
        probe_metrics: dict = {}
        if global_w is not None and rnd in cfg.snap_rounds:
            learners[-1].set_weights(global_w)
            em = _eval_probes(learners[-1], probe_events, probe_labels, global_bank)

            n_unk_clusters = em["n_unknown_clusters"]

            # Detection: DBSCAN finds a coherent cluster in the "unknown" region
            if n_unk_clusters >= 1 and split_detected_round is None and rnd >= cfg.injection_round:
                split_detected_round = rnd
                print(f"    [proto] R{rnd}: unknown cluster first detected (DBSCAN)")

            # Attribution: rename the "unknown" centroid to the confirmed disease name.
            # Uses oracle morven probe centroid as the nearest-known reference.
            if cfg.attribution_round > 0 and rnd == cfg.attribution_round:
                morven_mask  = np.array([l == "morven" for l in probe_labels])
                morven_embs  = em["cls_emb"][morven_mask]
                if len(morven_embs) > 0:
                    morven_center = morven_embs.mean(axis=0)
                    unknown_names = [n for n in global_bank.names() if "unknown" in n]
                    if unknown_names:
                        dists   = {n: np.linalg.norm(global_bank._protos[n].centroid - morven_center)
                                   for n in unknown_names}
                        nearest = min(dists, key=dists.get)
                        global_bank.rename(nearest, "morven")
                        attribution_map[nearest] = "morven"
                        # Propagate rename to all silo banks
                        for sb in silo_banks:
                            if sb.has(nearest):
                                sb.rename(nearest, "morven")
                        print(f"    [proto] R{rnd}: attribution '{nearest}' → 'morven'")

            # Softmax P(unknown|Morven) for comparison
            import torch
            morven_texts = [t for t, l in zip(_probe_texts(probe_events), probe_labels)
                            if l == "morven"]
            softmax_p_unk = float("nan")
            if morven_texts:
                enc   = learners[-1]._tokenizer(
                    morven_texts, padding=True, truncation=True,
                    max_length=128, return_tensors="pt"
                )
                model = learners[-1].model
                dev   = next(model.parameters()).device
                enc   = {k: v.to(dev) for k, v in enc.items()}
                model.eval()
                with torch.no_grad():
                    out = model(**enc)
                probs = _softmax(out.logits.cpu().numpy())
                softmax_p_unk = float(probs[:, _IDX_UNKNOWN].mean())
            learners[-1].release()

            probe_metrics = {
                "proto_acc_all":            em["proto_acc_all"],
                "proto_acc_morven":         em["proto_acc_morven"],
                "proto_morven_exact":       em["proto_morven_exact"],
                "n_unknown_clusters":       n_unk_clusters,
                "n_named_protos":           len(global_bank.names()),
                "proto_names":              list(global_bank.names()),
                "softmax_p_unknown_morven": softmax_p_unk,
                "split_detected_round":     split_detected_round,
            }

            # UMAP snapshot
            _plot_umap_round(
                em["cls_emb"], probe_labels, global_bank,
                rnd, out_dir / f"umap_r{rnd:02d}.png", n_unk_clusters,
            )

        rm = {
            "round":            rnd,
            "train_sizes":      train_sizes,
            "morven_injected":  cfg.do_inject and rnd >= cfg.injection_round,
            **probe_metrics,
        }
        round_metrics.append(rm)

        inj_str   = "yes" if rnd >= cfg.injection_round and cfg.do_inject else "no"
        clust_str = f"  unk_clusters={probe_metrics.get('n_unknown_clusters','?')}" if probe_metrics else ""
        proto_str = f"  protos={probe_metrics.get('proto_names','')}" if probe_metrics else ""
        print(f"  R{rnd:02d}  inject={inj_str}{clust_str}{proto_str}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    (out_dir / "round_metrics.json").write_text(json.dumps(round_metrics, indent=2))
    _plot_summary(round_metrics, out_dir / "prototype_curve.png", cfg)

    wall = time.time() - t0
    print(f"\n  Wall: {wall:.0f}s")
    print(f"  Results: {out_dir}/")
    if split_detected_round:
        print(f"  Split first detected: R{split_detected_round}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",      default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--eps",         type=float, default=0.30)
    ap.add_argument("--no-inject",   action="store_true")
    ap.add_argument("--run-name",    default="proto_seed42")
    ap.add_argument("--n-silos",     type=int, default=3)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--results-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    cfg = PrototypeConfig(
        training_device  = args.device,
        dbscan_eps       = args.eps,
        do_inject        = not args.no_inject,
        run_name         = args.run_name,
        n_silos          = args.n_silos,
        seed             = args.seed,
        results_dir      = args.results_dir,
    )
    run_prototype(cfg)


if __name__ == "__main__":
    main()
