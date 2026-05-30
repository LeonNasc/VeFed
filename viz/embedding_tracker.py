"""
Embedding tracker for federated learning runs.

After each FL round this module snapshots CLS embeddings from:
  - The global (FedAvg) model
  - Each silo's federated model (local weights, pre-global-update)
  - Each silo's local-only shadow model (never receives global weights)

All snapshots are taken against a fixed benchmark probe set — a collection of
DiagnosticEvents drawn from all known diseases + BenchmarkFever (ICD A99.0).
Because the probe set is identical every round, embedding shifts are purely
attributable to model weight changes, not data variation.

Outputs written to <output_dir>/
  round_NNN/
    global.npz          — CLS embeddings + labels for FedAvg model
    silo_N_fed.npz      — silo N federated (local) model
    silo_N_local.npz    — silo N shadow local-only model
  evolution_global.png  — global model at each round (shared PCA space)
  final_all_models.png  — final round, all models side-by-side
  fl_gain_final.png     — final round, fed vs local per silo

NPZ schema (all files):
  raw    (N, H)  — raw CLS hidden states (H = 768 for DistilBERT)
  labels (N,)    — ground-truth strings  e.g. "A99.0 / treat"
"""
from __future__ import annotations

import random as _random
from pathlib import Path
from typing import Optional

import numpy as np


# ── Probe event generation ────────────────────────────────────────────────────

def generate_probe_events(
    n_per_tier: int = 15,
    seed: int = 999,
) -> list:
    """
    Build a fixed set of DiagnosticEvents spanning all diseases × management tiers.

    Runs a mini WorldEngine (no LLM, SymptomNarrator templates only) for each
    disease to collect natural-language opening statements.  Uses a fixed seed
    so the same probe set is generated every run.

    Parameters
    ----------
    n_per_tier : int
        Target events per management tier per disease (home rest / treat / hospitalise).
        Actual counts may be lower for very mild diseases (e.g. Mild Corona rarely
        reaches the hospitalise tier organically — those slots are filled with the
        best available approximations).
    seed : int
        Fixed RNG seed — do not change between runs or PCA coordinates shift.

    Returns
    -------
    List of DiagnosticEvents with non-empty conversation[0] (patient opening).
    """
    from simulation.world import WorldEngine
    from simulation.progression import PROGRESSION_STRATEGIES, BenchmarkFeverProgression

    rng = _random.Random(seed)
    all_events: list = []

    probes = {name: s for name, s in PROGRESSION_STRATEGIES.items() if name != "MIMIC"}
    probes["Benchmark Fever"] = BenchmarkFeverProgression()

    for name, strategy in probes.items():
        # Run a small world long enough to populate all severity tiers
        world = WorldEngine(
            num_agents=60,
            seed=rng.randint(0, 99999),
            progression_strategy=strategy,
        )
        for _ in range(40 * world.TICKS_PER_DAY):
            world.step_tick()

        by_tier: dict[str, list] = {"home rest": [], "treat": [], "hospitalise": []}
        for ev in world.clinic_queue.processed:
            if not ev.ground_truth or not ev.conversation:
                continue
            tier = ev.ground_truth.rsplit(" / ", 1)[-1]
            if tier in by_tier:
                by_tier[tier].append(ev)

        for tier_events in by_tier.values():
            rng.shuffle(tier_events)
            all_events.extend(tier_events[:n_per_tier])

    return all_events


# ── CLS extraction (re-exported from disease_viz for convenience) ─────────────

def _extract_cls(model, events: list, tokenizer) -> np.ndarray:
    """Forward events through model, return CLS hidden states (N, H)."""
    import torch

    texts = []
    for ev in events:
        turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        if turns:
            texts.append(turns[-1])
        else:
            texts.append("")

    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=128, return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
    return out.hidden_states[-1][:, 0, :].cpu().numpy()


# ── Tracker ───────────────────────────────────────────────────────────────────

class EmbeddingTracker:
    """
    Attaches to a federated training run and snapshots embeddings each round.

    Usage in train.py
    -----------------
        tracker = EmbeddingTracker("viz_output/embeddings/run_<id>", lora_cfg)
        # inside FL loop, after _fedavg():
        tracker.snapshot(round_num, global_weights, silos)
        # after loop:
        tracker.save_plots()
    """

    def __init__(
        self,
        output_dir: str | Path,
        lora_config,
        probe_events: Optional[list] = None,
        n_per_tier: int = 15,
        probe_seed: int = 999,
    ):
        self.output_dir   = Path(output_dir)
        self.lora_config  = lora_config
        self.probe_events = probe_events or generate_probe_events(n_per_tier, probe_seed)
        self._probe_labels = [ev.ground_truth or "unknown" for ev in self.probe_events]

        # Raw storage: snapshots[round][model_name] = np.ndarray (N, H)
        self._snapshots: dict[int, dict[str, np.ndarray]] = {}

        self._probe_model  = None   # lazy-built; lives on CPU
        self._tokenizer    = None

    # ── Lazy model + tokenizer ────────────────────────────────────────────────

    @property
    def _model(self):
        if self._probe_model is None:
            from fl.lora import build_model
            self._probe_model = build_model(self.lora_config)
            self._probe_model = self._probe_model.cpu()
        return self._probe_model

    @property
    def _tok(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.lora_config.model_name_or_path
            )
        return self._tokenizer

    # ── Core extraction ───────────────────────────────────────────────────────

    def _extract(self, weights: list) -> np.ndarray:
        from fl.lora import set_lora_weights
        set_lora_weights(self._model, weights)
        return _extract_cls(self._model, self.probe_events, self._tok)

    # ── Public API ────────────────────────────────────────────────────────────

    def snapshot(self, round_num: int, global_weights: list, silos: list) -> None:
        """
        Extract and save embeddings for the global model and all silos.

        Parameters
        ----------
        round_num      : FL round index (1-based).
        global_weights : FedAvg-aggregated weights.
        silos          : list[WorldFLClient].
        """
        from fl.lora import get_lora_weights

        snap: dict[str, np.ndarray] = {}

        snap["global"] = self._extract(global_weights)
        for i, silo in enumerate(silos):
            snap[f"silo_{i}_fed"]   = self._extract(silo.get_weights())
            snap[f"silo_{i}_local"] = self._extract(get_lora_weights(silo.local_model))

        self._snapshots[round_num] = snap
        self._save_round_npz(round_num, snap)

        n_models = len(snap)
        n_events = len(self.probe_events)
        print(f"  [embed] round {round_num} — {n_models} models × {n_events} probes saved")

    def _save_round_npz(self, round_num: int, snap: dict) -> None:
        rd = self.output_dir / f"round_{round_num:03d}"
        rd.mkdir(parents=True, exist_ok=True)
        labels = np.array(self._probe_labels)
        for name, raw in snap.items():
            np.savez(rd / f"{name}.npz", raw=raw, labels=labels)

    # ── Plot generation ───────────────────────────────────────────────────────

    def save_plots(self) -> list[Path]:
        """
        Generate the three summary plots using a single shared PCA space.
        Returns list of saved PNG paths.
        """
        if not self._snapshots:
            return []

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError("pip install matplotlib scikit-learn") from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        rounds   = sorted(self._snapshots)
        n_silos  = sum(1 for k in self._snapshots[rounds[0]] if k.startswith("silo_") and k.endswith("_fed"))
        labels   = np.array(self._probe_labels)

        # ── Fit shared PCA on ALL embeddings combined ─────────────────────────
        all_raw = np.vstack([
            raw
            for snap in self._snapshots.values()
            for raw in snap.values()
        ])
        pca = PCA(n_components=2)
        pca.fit(all_raw)

        def _proj(raw: np.ndarray) -> np.ndarray:
            return pca.transform(raw)

        # Disease colour map: colour by ICD (first 3 chars) ignoring tier
        icd3_list = [lbl.split(" / ")[0][:3] if " / " in lbl else lbl[:3] for lbl in labels]
        unique_icd3 = sorted(set(icd3_list))
        cmap = plt.cm.get_cmap("tab10", len(unique_icd3))
        icd_color = {icd: cmap(i) for i, icd in enumerate(unique_icd3)}
        colors = [icd_color[icd] for icd in icd3_list]

        saved: list[Path] = []

        # ── 1. Global model evolution ─────────────────────────────────────────
        saved.append(self._plot_evolution_global(rounds, pca, colors, unique_icd3, icd_color, plt))

        # ── 2. Final round — all models ───────────────────────────────────────
        saved.append(self._plot_final_all(rounds[-1], n_silos, pca, colors, unique_icd3, icd_color, plt))

        # ── 3. FL gain — fed vs local per silo ───────────────────────────────
        saved.append(self._plot_fl_gain(rounds[-1], n_silos, pca, colors, unique_icd3, icd_color, plt))

        return [p for p in saved if p is not None]

    def _plot_evolution_global(self, rounds, pca, colors, unique_icd3, icd_color, plt) -> Optional[Path]:
        """Rows/cols grid: global model at each round."""
        n = len(rounds)
        cols = min(5, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2),
                                 constrained_layout=True)
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for ax_i, rnd in enumerate(rounds):
            ax = axes_flat[ax_i]
            raw = self._snapshots[rnd]["global"]
            coords = pca.transform(raw)
            for icd in unique_icd3:
                mask = self._icd3_mask(icd)
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           color=icd_color[icd], alpha=0.65, s=18, linewidths=0)
            ax.set_title(f"Round {rnd}", fontsize=8)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for ax in axes_flat[len(rounds):]:
            ax.set_visible(False)

        _add_legend(fig, unique_icd3, icd_color)
        var = pca.explained_variance_ratio_
        fig.suptitle(f"Global model — embedding evolution\nPC1={var[0]:.1%}  PC2={var[1]:.1%}",
                     fontsize=10)
        path = self.output_dir / "evolution_global.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def _icd3_mask(self, icd: str) -> list[bool]:
        """Boolean mask for probe events whose ICD 3-char prefix matches icd."""
        return [
            (lbl.split(" / ")[0][:3] if " / " in lbl else lbl[:3]) == icd
            for lbl in self._probe_labels
        ]

    def _plot_final_all(self, last_round, n_silos, pca, colors, unique_icd3, icd_color, plt) -> Optional[Path]:
        """Final round: global + each silo fed model."""
        snap = self._snapshots[last_round]
        model_keys = ["global"] + [f"silo_{i}_fed" for i in range(n_silos)]
        model_keys = [k for k in model_keys if k in snap]

        cols = len(model_keys)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 3.2, 3.5), constrained_layout=True)
        if cols == 1:
            axes = [axes]

        for ax, key in zip(axes, model_keys):
            coords = pca.transform(snap[key])
            for icd in unique_icd3:
                mask = self._icd3_mask(icd)
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           color=icd_color[icd], alpha=0.65, s=18, linewidths=0)
            label = "Global" if key == "global" else f"Silo {key.split('_')[1]} (fed)"
            ax.set_title(label, fontsize=8)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        _add_legend(fig, unique_icd3, icd_color)
        var = pca.explained_variance_ratio_
        fig.suptitle(f"Round {last_round} — all federated models\nPC1={var[0]:.1%}  PC2={var[1]:.1%}",
                     fontsize=10)
        path = self.output_dir / "final_all_models.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def _plot_fl_gain(self, last_round, n_silos, pca, colors, unique_icd3, icd_color, plt) -> Optional[Path]:
        """Final round: fed vs local overlay per silo (2 rows × n_silos cols)."""
        snap = self._snapshots[last_round]

        fig, axes = plt.subplots(2, n_silos, figsize=(n_silos * 3.0, 6.5),
                                 constrained_layout=True)
        if n_silos == 1:
            axes = axes.reshape(2, 1)

        row_labels = ["Federated", "Local-only"]

        for col, i in enumerate(range(n_silos)):
            for row, suffix in enumerate(["_fed", "_local"]):
                key = f"silo_{i}{suffix}"
                if key not in snap:
                    axes[row, col].set_visible(False)
                    continue
                ax = axes[row, col]
                coords = pca.transform(snap[key])
                for icd in unique_icd3:
                    mask = self._icd3_mask(icd)
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                               color=icd_color[icd], alpha=0.65, s=18, linewidths=0)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=8)
                if row == 0:
                    ax.set_title(f"Silo {i}", fontsize=8)

        _add_legend(fig, unique_icd3, icd_color)
        var = pca.explained_variance_ratio_
        fig.suptitle(f"Round {last_round} — federated vs local-only per silo\nPC1={var[0]:.1%}  PC2={var[1]:.1%}",
                     fontsize=10)
        path = self.output_dir / "fl_gain_final.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


# ── Legend helper ─────────────────────────────────────────────────────────────

def _add_legend(fig, unique_icd3: list, icd_color: dict) -> None:
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=icd_color[icd], label=icd) for icd in unique_icd3]
    fig.legend(handles=patches, loc="lower center", ncol=min(7, len(unique_icd3)),
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
