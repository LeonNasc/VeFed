"""
Visual storage of network representations for sample diseases.

Artefacts written to disk
─────────────────────────
trajectories/<strategy>_curves.png
    Severity s(t) and symptom σ(t) curves for n_samples individual
    trajectories sampled from each DiseaseProgressionStrategy.
    Triage thresholds (hospitalise / treatment) are overlaid.

embeddings/embeddings_<tag>.png
    2-D PCA scatter of the DistilBERT [CLS] token embedding from the
    last hidden layer, one point per DiagnosticEvent, coloured by
    ground-truth triage label.

embeddings/embeddings_<tag>.npz
    Raw data: coords (N×2 PCA), raw (N×H hidden-state), labels (N,).
    Reload with np.load(..., allow_pickle=False).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from simulation.progression import PROGRESSION_STRATEGIES


# ── Trajectory curve visualisation ───────────────────────────────────────────

def plot_progression_curves(
    output_dir: str | Path = "viz_output/trajectories",
    n_samples: int = 8,
    days: int = 40,
    seed: int = 42,
) -> list[Path]:
    """
    Save one PNG per disease strategy showing severity + symptom curves.

    Parameters
    ----------
    n_samples : int
        Individual trajectories drawn per strategy (shows variance).
    days : int
        Simulation days to plot.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    List of saved PNG paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("pip install matplotlib") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    saved: list[Path] = []

    for name, strategy in PROGRESSION_STRATEGIES.items():
        fig, (ax_sev, ax_sym) = plt.subplots(
            2, 1, figsize=(8, 5), sharex=True, constrained_layout=True
        )
        fig.suptitle(f"{name}", fontsize=11, fontweight="bold")

        for _ in range(n_samples):
            traj = strategy.sample_trajectory(rng)
            sevs, syms = [], []
            for _ in range(days):
                s, sigma = traj.step()
                sevs.append(s)
                syms.append(sigma)
            t = range(days)
            ax_sev.plot(t, sevs, alpha=0.55, linewidth=1.4)
            ax_sym.plot(t, syms, alpha=0.55, linewidth=1.4)

        ax_sev.axhline(0.70, color="#d62728", linestyle="--", linewidth=0.9,
                       label="hospitalise ≥ 0.70")
        ax_sev.axhline(0.40, color="#ff7f0e", linestyle="--", linewidth=0.9,
                       label="treat ≥ 0.40")
        ax_sev.set_ylabel("Severity  s(t)", fontsize=9)
        ax_sev.set_ylim(0, 1.05)
        ax_sev.legend(fontsize=7, loc="upper right")

        ax_sym.set_ylabel("Symptoms  σ(t)", fontsize=9)
        ax_sym.set_ylim(0, 1.05)
        ax_sym.set_xlabel("Day", fontsize=9)

        slug = name.replace(" ", "_").lower()
        path = out / f"{slug}_curves.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(path)

    return saved


# ── Embedding visualisation ───────────────────────────────────────────────────

def _extract_cls_embeddings(
    events: list,
    model,
    tokenizer,
) -> tuple:
    """
    Forward events through model, extract [CLS] from last hidden layer.

    Returns (embeddings: np.ndarray shape (N, H), labels: list[str]).
    """
    import numpy as np
    import torch

    texts: list[str] = []
    labels: list[str] = []
    for ev in events:
        patient_turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
        if not patient_turns:
            continue
        texts.append(patient_turns[-1])
        labels.append(ev.ground_truth or "unknown")

    if not texts:
        return np.array([]), []

    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=128, return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        # output_hidden_states works for both bare and PEFT-wrapped models
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
    # hidden_states[-1] shape: (B, seq_len, H) — take [CLS] token at position 0
    cls = out.hidden_states[-1][:, 0, :].cpu().numpy()
    return cls, labels


def save_embedding_plot(
    events: list,
    model,
    lora_config,
    output_dir: str | Path = "viz_output/embeddings",
    tag: str = "round",
) -> Optional[Path]:
    """
    PCA scatter of DistilBERT [CLS] embeddings coloured by disease label.
    Saves <tag>.png and <tag>.npz (raw embeddings + coords + labels).

    Parameters
    ----------
    events : list[DiagnosticEvent]
        Processed diagnostic events from one or more simulation rounds.
    model : PEFT model
        The LoRA-adapted DistilBERT returned by fl.lora.build_model().
    lora_config : LoRAConfig
        Needed to load the matching tokenizer.
    tag : str
        Filename suffix, e.g. "round_05" or "baseline".

    Returns
    -------
    Path to saved PNG, or None if no valid events.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.decomposition import PCA
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "pip install matplotlib scikit-learn transformers"
        ) from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(lora_config.model_name_or_path)
    embeddings, labels = _extract_cls_embeddings(events, model, tokenizer)
    if embeddings.size == 0:
        return None

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    unique = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for lbl in unique:
        mask = [l == lbl for l in labels]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=lbl, color=color_map[lbl],
            alpha=0.72, s=45, linewidths=0,
        )
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({var[0]:.1%} var)", fontsize=9)
    ax.set_ylabel(f"PC2  ({var[1]:.1%} var)", fontsize=9)
    ax.set_title(f"Disease embeddings — {tag}", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)

    png_path = out / f"embeddings_{tag}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    np.savez(
        out / f"embeddings_{tag}.npz",
        coords=coords,
        raw=embeddings,
        labels=np.array(labels),
    )
    return png_path


# ── Full static atlas (no model required) ────────────────────────────────────

def generate_disease_atlas(
    output_dir: str | Path = "viz_output",
    n_samples: int = 8,
    days: int = 40,
) -> dict[str, list[Path]]:
    """
    Generate the complete static disease visualization atlas.
    No trained model needed — trajectory curves only.

    Returns {"trajectories": [list of PNG paths]}.
    """
    paths = plot_progression_curves(
        output_dir=Path(output_dir) / "trajectories",
        n_samples=n_samples,
        days=days,
    )
    return {"trajectories": paths}


# ── Synthetic events for bootstrap embedding plots ───────────────────────────

def sample_disease_events(
    n_per_disease: int = 20,
    seed: int = 0,
) -> list:
    """
    Generate synthetic DiagnosticEvents (one batch per progression strategy)
    without running a full WorldEngine simulation.  Useful for visualising the
    embedding space before any real FL training has occurred.
    """
    import random as _random
    from simulation.world import WorldEngine
    from simulation.progression import PROGRESSION_STRATEGIES

    all_events = []
    rng = _random.Random(seed)

    for strategy_name, strategy in PROGRESSION_STRATEGIES.items():
        world = WorldEngine(
            num_agents=n_per_disease + 5,
            seed=rng.randint(0, 9999),
            progression_strategy=strategy,
        )
        # Advance until we have enough processed events
        for _ in range(14 * world.TICKS_PER_DAY):
            world.step_tick()

        batch = world.clinic_queue.processed[:n_per_disease]
        # Tag with strategy name for label override if ground_truth is missing
        for ev in batch:
            if not ev.ground_truth:
                ev.ground_truth = strategy_name
        all_events.extend(batch)

    return all_events
