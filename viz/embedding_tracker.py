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
  labels (N,)    — ground-truth strings  e.g. "influenza/severe", "non-infectious"
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
    from simulation.world_config import WorldConfig as SimWorldConfig, AgentConfig, EpidemicConfig
    from simulation.progression import PROGRESSION_STRATEGIES, BenchmarkFeverProgression

    rng = _random.Random(seed)
    all_events: list = []

    probes = {name: s for name, s in PROGRESSION_STRATEGIES.items() if name != "MIMIC"}
    probes["Benchmark Fever"] = BenchmarkFeverProgression()

    for name, strategy in probes.items():
        # Run a small world long enough to populate all severity tiers
        _wc = SimWorldConfig(
            agents   = AgentConfig(num_agents=60),
            epidemic = EpidemicConfig(progressions=[name], disease_strategy=name),
        )
        world = WorldEngine(_wc, seed=rng.randint(0, 99999))
        for _ in range(40 * world.TICKS_PER_DAY):
            world.step_tick()

        by_severity: dict[str, list] = {"mild": [], "moderate": [], "severe": [], "non-infectious": []}
        for ev in world.clinic_queue.processed:
            if not ev.ground_truth or not ev.conversation:
                continue
            # New format: "disease/severity" or "non-infectious"
            gt = ev.ground_truth
            if "/" in gt:
                sev = gt.split("/", 1)[1]
            else:
                sev = gt  # "non-infectious"
            if sev in by_severity:
                by_severity[sev].append(ev)

        for sev_events in by_severity.values():
            rng.shuffle(sev_events)
            all_events.extend(sev_events[:n_per_tier])

    return all_events


# ── Forward pass: CLS hidden states + logits ─────────────────────────────────

def _forward(model, events: list, tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """
    Single forward pass through model for all probe events.

    Returns
    -------
    cls    (N, H)   — CLS token from the last hidden layer
    logits (N, C)   — raw classifier logits (C = num_labels)
    """
    import torch

    texts = []
    for ev in events:
        turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        texts.append(turns[-1] if turns else "")

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
    cls    = out.hidden_states[-1][:, 0, :].cpu().numpy()
    logits = out.logits.cpu().numpy()
    return cls, logits


def _project(data: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    """
    2-D projection of high-dimensional embeddings.

    method : "umap" (preferred) or "pca" (fallback).
    """
    if method == "umap":
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(n_components=2, random_state=seed,
                                    n_neighbors=15, min_dist=0.1)
            return reducer.fit_transform(data)
        except Exception:
            pass
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(data)


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

    def _extract(self, weights: list) -> tuple[np.ndarray, np.ndarray]:
        """Load weights, return (cls (N,H), logits (N,C))."""
        from fl.lora import set_lora_weights
        set_lora_weights(self._model, weights)
        return _forward(self._model, self.probe_events, self._tok)

    # ── Public API ────────────────────────────────────────────────────────────

    def snapshot(self, round_num: int, global_weights: list,
                 silo_fed_weights: list | None = None,
                 silo_local_weights: list | None = None,
                 extra_weights: dict | None = None) -> None:
        """
        Extract and save CLS embeddings + logits for all models.

        Accepts pre-computed weight lists so the caller controls when models are
        built and released.  Never calls silo.model itself — no GPU rebuilds here.

        Parameters
        ----------
        global_weights : list[np.ndarray]
            FedAvg global model adapter weights.
        silo_fed_weights : list[list[np.ndarray]]
            Per-silo federated weights, collected during the round loop while
            models are already built (before release_model()).
        silo_local_weights : list[list[np.ndarray]]
            Per-silo local-only shadow model weights (same ordering as fed).
        extra_weights : dict[str, list[np.ndarray]] | None
            Additional named models, e.g. {"centralized": weights}.
        """
        snap: dict[str, dict] = {}

        def _record(name, weights):
            cls, logits = self._extract(weights)
            snap[name] = {"cls": cls, "logits": logits}

        _record("global", global_weights)

        if silo_fed_weights:
            for i, fw in enumerate(silo_fed_weights):
                _record(f"silo_{i}_fed", fw)
        if silo_local_weights:
            for i, lw in enumerate(silo_local_weights):
                _record(f"silo_{i}_local", lw)

        if extra_weights:
            for name, weights in extra_weights.items():
                _record(name, weights)

        self._snapshots[round_num] = snap
        self._save_round_npz(round_num, snap)

        n_models = len(snap)
        n_events = len(self.probe_events)
        print(f"  [embed] round {round_num} — {n_models} models × {n_events} probes saved")

    def _save_round_npz(self, round_num: int, snap: dict) -> None:
        rd = self.output_dir / f"round_{round_num:03d}"
        rd.mkdir(parents=True, exist_ok=True)
        labels = np.array(self._probe_labels)
        for name, d in snap.items():
            np.savez(rd / f"{name}.npz", cls=d["cls"], logits=d["logits"], labels=labels)

    # ── Plot generation ───────────────────────────────────────────────────────

    def save_plots(self) -> list[Path]:
        """
        Generate four summary plots. Uses UMAP (PCA fallback) per-plot.

        1. evolution_global_cls.png   — global CLS embeddings, one subplot per round
        2. evolution_global_logits.png — same but in logit space (more discriminative)
        3. final_all_models.png       — final round, all models, logit-space UMAP
        4. fl_gain_final.png          — final round, fed vs local per silo, logit-space
        """
        if not self._snapshots:
            return []

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("pip install matplotlib") from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        rounds  = sorted(self._snapshots)
        n_silos = sum(1 for k in self._snapshots[rounds[0]]
                      if k.startswith("silo_") and k.endswith("_fed"))
        labels  = np.array(self._probe_labels)

        # Colour maps — disease (warm) vs non-infectious (cool) split palettes
        # New format: "disease/severity" or "non-infectious"
        icd3_list   = [(lbl.split("/")[0] if "/" in lbl else lbl)
                       for lbl in self._probe_labels]
        tier_list   = [(lbl.split("/", 1)[1] if "/" in lbl else "non-infectious")
                       for lbl in self._probe_labels]

        unique_icd3 = sorted(set(icd3_list))
        unique_tier = ["mild", "moderate", "severe", "non-infectious"]
        icd_color   = _build_icd_color_map(unique_icd3)
        cmap_tier   = plt.cm.get_cmap("Set2", 4)
        tier_color  = {tier: cmap_tier(i) for i, tier in enumerate(unique_tier)}

        saved: list[Path] = []

        # ── 1 & 2: Global model evolution in CLS and logit space ─────────────
        for space in ("cls", "logits"):
            saved.append(
                self._plot_evolution(rounds, space, icd3_list, unique_icd3, icd_color, plt,
                                     fname=f"evolution_global_{space}.png",
                                     title_suffix="CLS hidden states" if space == "cls" else "logit space")
            )

        # ── 3: Final round — all models in logit space, coloured by disease ──
        saved.append(self._plot_final_all(rounds[-1], n_silos,
                                          icd3_list, unique_icd3, icd_color,
                                          tier_list, tier_color, plt))

        # ── 4: FL gain — fed vs local per silo, logit space ──────────────────
        saved.append(self._plot_fl_gain(rounds[-1], n_silos,
                                        icd3_list, unique_icd3, icd_color, plt))

        # ── 5: Federated vs Centralized comparison (only when both present) ──
        if "centralized" in self._snapshots.get(rounds[-1], {}):
            saved.append(self._plot_comparison(rounds, icd3_list, unique_icd3,
                                               icd_color, tier_list, tier_color, plt))

        return [p for p in saved if p is not None]

    def as_html_imgs(self) -> dict[str, str]:
        """
        Generate all embedding plots and return as {plot_name: <img> tag} dict.
        Plots are encoded as base64 PNGs — no files written.
        Used by RunReport to embed embedding visualisations inline.
        """
        if not self._snapshots:
            return {}

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {}

        rounds  = sorted(self._snapshots)
        n_silos = sum(1 for k in self._snapshots[rounds[0]]
                      if k.startswith("silo_") and k.endswith("_fed"))
        icd3_list = [(lbl.split("/")[0] if "/" in lbl else lbl)
                     for lbl in self._probe_labels]
        tier_list = [(lbl.split("/", 1)[1] if "/" in lbl else "non-infectious")
                     for lbl in self._probe_labels]
        unique_icd3 = sorted(set(icd3_list))
        unique_tier = ["mild", "moderate", "severe", "non-infectious"]
        icd_color   = _build_icd_color_map(unique_icd3)
        cmap_tier   = plt.cm.get_cmap("Set2", 4)
        tier_color  = {tier: cmap_tier(i) for i, tier in enumerate(unique_tier)}

        result: dict[str, str] = {}

        # Evolution plots (CLS and logit space)
        for space, label in [("cls", "CLS hidden states"), ("logits", "logit space")]:
            n    = len(rounds)
            cols = min(5, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0),
                                     constrained_layout=True)
            axes_flat = np.array(axes).flatten() if n > 1 else [axes]
            _style_fig(fig, axes_flat)
            for ax_i, rnd in enumerate(rounds):
                ax = axes_flat[ax_i]
                data = self._snapshots[rnd]["global"][space]
                coords = _project(data)
                self._scatter_panel(ax, coords, icd3_list, unique_icd3, icd_color, plt,
                                    annotate=(rnd >= 3))
                ax.set_title(f"Round {rnd}", fontsize=8, color="#212529")
            for ax in axes_flat[len(rounds):]:
                ax.set_visible(False)
            _add_legend(fig, unique_icd3, icd_color)
            fig.suptitle(f"Global model evolution — {label}\n(UMAP per round)",
                         fontsize=10, color="#212529")
            result[f"evolution_{space}"] = _fig_to_img_tag(fig, plt)

        # Final all models
        snap     = self._snapshots[rounds[-1]]
        mod_keys = [k for k in ["global"] + [f"silo_{i}_fed" for i in range(n_silos)]
                    if k in snap]
        cols     = len(mod_keys)
        if cols > 0:
            fig, axes = plt.subplots(2, cols, figsize=(cols * 3.0, 6.5),
                                     constrained_layout=True)
            if cols == 1:
                axes = axes.reshape(2, 1)
            _style_fig(fig, axes.flat)
            row_labels = ["by disease (ICD)", "by management tier"]
            for col, key in enumerate(mod_keys):
                coords = _project(snap[key]["logits"])
                for row, (groups, cmap) in enumerate([(unique_icd3, icd_color),
                                                       (unique_tier, tier_color)]):
                    ax = axes[row, col]
                    self._scatter_panel(ax, coords,
                                        icd3_list if row == 0 else tier_list,
                                        groups, cmap, plt, annotate=(row == 0))
                    if col == 0:
                        ax.set_ylabel(row_labels[row], fontsize=8, color="#212529")
                    if row == 0:
                        lbl = "Global" if key == "global" else f"Silo {key.split('_')[1]}"
                        ax.set_title(lbl, fontsize=8, color="#212529")
            _add_legend(fig, unique_icd3, icd_color, ncol=4)
            fig.suptitle(f"Round {rounds[-1]} — all federated models (logit-space UMAP)",
                         fontsize=10, color="#212529")
            result["final_all_models"] = _fig_to_img_tag(fig, plt)

        # FL gain: fed vs local per silo
        if n_silos:
            fig, axes = plt.subplots(2, n_silos, figsize=(n_silos * 3.0, 6.5),
                                     constrained_layout=True)
            if n_silos == 1:
                axes = axes.reshape(2, 1)
            _style_fig(fig, axes.flat)
            for col, i in enumerate(range(n_silos)):
                for row, suffix in enumerate(["_fed", "_local"]):
                    key = f"silo_{i}{suffix}"
                    if key not in snap:
                        axes[row, col].set_visible(False)
                        continue
                    ax = axes[row, col]
                    self._scatter_panel(ax, _project(snap[key]["logits"]),
                                        icd3_list, unique_icd3, icd_color, plt, annotate=True)
                    if col == 0:
                        ax.set_ylabel("Federated" if row == 0 else "Local-only",
                                      fontsize=8, color="#212529")
                    if row == 0:
                        ax.set_title(f"Silo {i}", fontsize=8, color="#212529")
            _add_legend(fig, unique_icd3, icd_color, ncol=4)
            fig.suptitle(f"Round {rounds[-1]} — federated vs local-only (logit-space UMAP)",
                         fontsize=10, color="#212529")
            result["fl_gain"] = _fig_to_img_tag(fig, plt)

        return result

    def _scatter_panel(self, ax, coords, group_list, unique_groups, color_map, plt,
                       annotate: bool = False):
        """
        Plot one scatter panel from pre-projected 2D coords.

        Each ICD group uses its own marker shape (circle = infectious,
        triangle = non-infectious) so the split is readable without colour alone.
        When annotate=True, a readable name is drawn at each cluster centroid.
        """
        import numpy as np
        group_arr = np.array(group_list)
        for grp in unique_groups:
            mask = group_arr == grp
            marker = ICD3_MARKER.get(grp, "o")
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=color_map[grp], alpha=0.75, s=35,
                       marker=marker, linewidths=0, label=grp)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if annotate:
            _annotate_centroids(ax, coords, group_list, unique_groups, color_map)

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

    def _plot_evolution(self, rounds, space, group_list, unique_groups, color_map,
                        plt, fname, title_suffix) -> Optional[Path]:
        """Small-multiples grid: global model at each round in a given embedding space."""
        n    = len(rounds)
        cols = min(5, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0),
                                 constrained_layout=True)
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        _style_fig(fig, axes_flat)

        for ax_i, rnd in enumerate(rounds):
            ax   = axes_flat[ax_i]
            data = self._snapshots[rnd]["global"][space]
            coords = _project(data)
            self._scatter_panel(ax, coords, group_list, unique_groups, color_map, plt,
                                annotate=(rnd >= 3))
            ax.set_title(f"Round {rnd}", fontsize=8, color="#212529")

        for ax in axes_flat[len(rounds):]:
            ax.set_visible(False)

        _add_legend(fig, unique_groups, color_map)
        fig.suptitle(f"Global model evolution — {title_suffix}\n(UMAP per round)",
                     fontsize=10, color="#212529")
        path = self.output_dir / fname
        fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def _plot_final_all(self, last_round, n_silos, icd3_list, unique_icd3, icd_color,
                        tier_list, tier_color, plt) -> Optional[Path]:
        """Final round: global + each silo's fed model. Two rows: by disease / by tier."""
        snap      = self._snapshots[last_round]
        mod_keys  = ["global"] + [f"silo_{i}_fed" for i in range(n_silos)]
        mod_keys  = [k for k in mod_keys if k in snap]
        cols      = len(mod_keys)
        if cols == 0:
            return None

        fig, axes = plt.subplots(2, cols, figsize=(cols * 3.0, 6.5), constrained_layout=True)
        if cols == 1:
            axes = axes.reshape(2, 1)
        _style_fig(fig, axes.flat)

        row_labels = ["by disease (ICD)", "by management tier"]

        for col, key in enumerate(mod_keys):
            data   = snap[key]["logits"]
            coords = _project(data)
            for row, (groups, cmap) in enumerate([
                (unique_icd3, icd_color),
                (list(tier_color), tier_color),
            ]):
                ax = axes[row, col]
                group_list = icd3_list if row == 0 else tier_list
                self._scatter_panel(ax, coords, group_list, groups, cmap, plt,
                                    annotate=(row == 0))
                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=8, color="#212529")
                if row == 0:
                    label = "Global" if key == "global" else f"Silo {key.split('_')[1]}"
                    ax.set_title(label, fontsize=8, color="#212529")

        _add_legend(fig, unique_icd3, icd_color, ncol=4)
        fig.suptitle(f"Round {last_round} — all federated models (logit-space UMAP)",
                     fontsize=10, color="#212529")
        path = self.output_dir / "final_all_models.png"
        fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def _plot_fl_gain(self, last_round, n_silos, icd3_list, unique_icd3,
                      icd_color, plt) -> Optional[Path]:
        """Final round: federated vs local-only per silo (logit-space UMAP)."""
        if n_silos == 0:
            return None
        snap = self._snapshots[last_round]

        fig, axes = plt.subplots(2, n_silos, figsize=(n_silos * 3.0, 6.5),
                                 constrained_layout=True)
        if n_silos == 1:
            axes = axes.reshape(2, 1)
        _style_fig(fig, axes.flat)

        for col, i in enumerate(range(n_silos)):
            for row, suffix in enumerate(["_fed", "_local"]):
                key = f"silo_{i}{suffix}"
                if key not in snap:
                    axes[row, col].set_visible(False)
                    continue
                ax     = axes[row, col]
                coords = _project(snap[key]["logits"])
                self._scatter_panel(ax, coords, icd3_list, unique_icd3, icd_color, plt,
                                    annotate=True)
                if col == 0:
                    ax.set_ylabel("Federated" if row == 0 else "Local-only",
                                  fontsize=8, color="#212529")
                if row == 0:
                    ax.set_title(f"Silo {i}", fontsize=8, color="#212529")

        _add_legend(fig, unique_icd3, icd_color, ncol=4)
        fig.suptitle(f"Round {last_round} — federated vs local-only (logit-space UMAP)",
                     fontsize=10, color="#212529")
        path = self.output_dir / "fl_gain_final.png"
        fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path


    def _plot_comparison(self, rounds, icd3_list, unique_icd3, icd_color,
                         tier_list, tier_color, plt) -> Optional[Path]:
        """
        Two-row grid: federated global vs centralized model, one column per round.
        Row 0 coloured by disease, Row 1 coloured by management tier.
        Allows direct visual comparison of learned geometry across rounds.
        """
        last = rounds[-1]
        # Use rounds where both models have snapshots
        cmp_rounds = [r for r in rounds if "centralized" in self._snapshots.get(r, {})]
        if not cmp_rounds:
            return None

        cols = len(cmp_rounds)
        fig, axes = plt.subplots(2, cols, figsize=(cols * 3.0, 6.5),
                                 constrained_layout=True)
        if cols == 1:
            axes = axes.reshape(2, 1)
        _style_fig(fig, axes.flat)

        row_labels = ["Federated (global)", "Centralized (oracle)"]
        for col, rnd in enumerate(cmp_rounds):
            snap = self._snapshots[rnd]
            for row, key in enumerate(["global", "centralized"]):
                ax     = axes[row, col]
                coords = _project(snap[key]["logits"])
                self._scatter_panel(ax, coords, icd3_list, unique_icd3, icd_color, plt,
                                    annotate=(rnd == cmp_rounds[-1]))
                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=8, color="#212529")
                if row == 0:
                    ax.set_title(f"Round {rnd}", fontsize=8, color="#212529")

        _add_legend(fig, unique_icd3, icd_color, ncol=4)
        fig.suptitle("Federated vs Centralized — logit-space UMAP per round",
                     fontsize=10, color="#212529")
        path = self.output_dir / "comparison_fed_vs_centralized.png"
        fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path


# ── Label helpers ─────────────────────────────────────────────────────────────

ICD3_NAMES: dict[str, str] = {
    "A41": "Sepsis",
    "A99": "Benchmark fever",
    "F41": "Anxiety",
    "J11": "Flu",
    "M54": "Back pain",
    "R51": "Headache",
    "R53": "Fatigue",
    "U07": "Corona",
    "Z87": "Hypertension f/u",
}

# Infectious diseases use filled circles; non-infectious use triangles.
ICD3_MARKER: dict[str, str] = {
    "A41": "o", "A99": "o", "J11": "o", "U07": "o",   # infectious
    "F41": "^", "M54": "^", "R51": "^", "R53": "^", "Z87": "^",  # non-infectious
}

# Warm palette for infectious diseases; cool palette for non-infectious.
# Different hue families make the split instantly readable.
_INFECTIOUS_COLORS: dict[str, str] = {
    "J11": "#e63946",   # Flu            — vivid red
    "U07": "#f4a261",   # Corona         — amber orange
    "A41": "#9b2226",   # Sepsis         — deep crimson
    "A99": "#ff6b9d",   # Benchmark fever — hot pink
}
_NON_INFECTIOUS_COLORS: dict[str, str] = {
    "F41": "#2a9d8f",   # Anxiety        — teal
    "M54": "#52b788",   # Back pain      — sage green
    "R51": "#4361ee",   # Headache       — indigo
    "R53": "#118ab2",   # Fatigue        — steel blue
    "Z87": "#7b2d8b",   # Hypertension   — violet
}


def _build_icd_color_map(unique_icd3: list) -> dict:
    """Return per-ICD3 color using warm palette for infectious, cool for non-infectious."""
    base = {**_INFECTIOUS_COLORS, **_NON_INFECTIOUS_COLORS}
    # Fallback for any unexpected codes: cycle through tab10
    import matplotlib.pyplot as plt
    tab = plt.cm.get_cmap("tab10")
    extras = [icd for icd in unique_icd3 if icd not in base]
    for i, icd in enumerate(extras):
        base[icd] = tab(i % 10)
    return {icd: base[icd] for icd in unique_icd3}


def _style_fig(fig, axes_iter) -> None:
    """Apply light-mode styling to a figure and its axes."""
    fig.patch.set_facecolor("white")
    for ax in axes_iter:
        ax.set_facecolor("#f8f9fa")
        ax.tick_params(colors="#495057", labelsize=7)
        ax.spines[:].set_color("#ced4da")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)


def _add_legend(fig, unique_groups: list, color_map: dict, ncol: int = 7) -> None:
    import matplotlib.patches as mpatches
    infectious     = [g for g in unique_groups if g in _INFECTIOUS_COLORS]
    non_infectious = [g for g in unique_groups if g in _NON_INFECTIOUS_COLORS]
    other          = [g for g in unique_groups if g not in _INFECTIOUS_COLORS
                      and g not in _NON_INFECTIOUS_COLORS]
    ordered = infectious + non_infectious + other

    patches = []
    for g in ordered:
        marker = ICD3_MARKER.get(g, "o")
        label  = ICD3_NAMES.get(g, g)
        # Add ○/△ marker hint to label
        hint = " ●" if marker == "o" else " ▲"
        patches.append(mpatches.Patch(color=color_map[g], label=label + hint))

    fig.legend(handles=patches, loc="lower center", ncol=min(ncol, len(patches)),
               fontsize=8, framealpha=0.95, labelcolor="#212529",
               facecolor="white", edgecolor="#ced4da",
               bbox_to_anchor=(0.5, -0.02))


def _annotate_centroids(ax, coords: "np.ndarray", group_list: list,
                        unique_groups: list, color_map: dict) -> None:
    """Draw name at each ICD cluster centroid, styled for light background."""
    import numpy as np
    for grp in unique_groups:
        mask = np.array([g == grp for g in group_list])
        if not mask.any():
            continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        name = ICD3_NAMES.get(grp, grp)
        ax.text(
            cx, cy, name,
            fontsize=6, ha="center", va="center", fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc=color_map[grp],
                      ec="none", alpha=0.88),
        )


def _fig_to_img_tag(fig, plt, dpi: int = 130) -> str:
    """Encode a matplotlib figure as a base64 <img> tag and close it."""
    import base64, io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;margin:.5em 0">'
