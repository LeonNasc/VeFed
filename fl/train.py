"""
Standalone federated training loop.

Runs N WorldEngine silos in-process — no Flower server or Ray required.
FedAvg aggregation is implemented directly (3-line weighted average).
W&B receives per-silo and aggregated metrics every round.

Quick start
-----------
    from fl.train import run_federated_training
    run_federated_training(num_silos=3, num_rounds=10)

    # Or from the command line:
    python -m fl.train --silos 3 --rounds 10 --agents 30

Distributed mode
----------------
Once results look good here, switch to real Flower:
  1. Server: fl.server.run_flower_server(num_rounds, num_clients, wandb_run)
  2. Clients: fl.flwr_client.start_flower_client(WorldFLClient(...), server_addr)

The WandBFedAvg strategy in fl/server.py expects the same metric keys that
WorldFLClient.run_round() returns, so the W&B dashboards will be identical.

W&B logging structure (all under one run, step = FL round)
----------------------------------------------------------
  round
  aggregated/loss, accuracy, num_examples, num_trained
  silo_{i}/loss, accuracy, num_examples
  silo_{i}/num_events, trained_on, train_loss_final
  silo_{i}/sir_s, sir_i, sir_r
  silo_{i}/epoch_losses  (table)
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from fl.lora import LoRAConfig
from fl.silo import FLSilo
from simulation.world import WorldEngine
from simulation.progression import PROGRESSION_STRATEGIES, DiseaseProgressionStrategy
from simulation.end_conditions import EndCondition, from_config as end_condition_from_config


# ── Per-silo world configuration ─────────────────────────────────────────────

@dataclass
class SiloPresetConfig:
    """
    Per-silo world configuration for non-IID federated learning experiments.

    When a list of SiloPresetConfig objects is passed to FLTrainConfig.world_configs,
    each silo gets its own disease mix, population size, spread rate, and end
    condition — enabling asymmetric (non-IID) federation.

    Example — urban vs. rural silos:
        SiloPresetConfig(num_agents=150, progressions=["Influenza", "Bacterial Pneumonia"],
                         disease_strategy="Influenza", beta_scale=1.3)
        SiloPresetConfig(num_agents=60,  progressions=["Bacterial Pneumonia"],
                         disease_strategy="Influenza",  beta_scale=0.7)

    Note: simulation.world_config.WorldConfig is the *nested* config object passed to
    WorldEngine.  This class is the flat *preset* format used in run.py and FLTrainConfig.
    They are different classes that serve different abstraction layers.
    """
    num_agents:            int         = 30
    progressions:          list[str]   = field(default_factory=lambda: ["Influenza"])
    disease_weights:       Optional[list[float]] = None  # per-disease prevalence; None = uniform
    disease_strategy:      str         = "Influenza"
    beta_scale:            float       = 1.0
    background_visit_rate: float       = 0.025
    end_condition:         str         = "extinction"
    end_condition_param:   Optional[int] = None
    seed_offset:           int         = 0
    reveal_incubating_icd: bool        = True
    initial_seeds:         int         = 3
    contact_rate_sigma:    float       = 0.0   # individual sociability heterogeneity (0 = off)
    confusion_rate:        float       = 0.0   # fraction of visits with wrong-disease text (label stays correct)
    # Static mode — bypasses SIR; set infectious_fraction as the "slider"
    static_mode:           bool        = False
    infectious_fraction:   float       = 0.5   # fraction of visits that are infectious
    cases_per_day:         int         = 20    # clinic throughput per simulated day


# Backward-compat alias — run.py and external scripts may still import WorldConfig from here.
# simulation.world_config.WorldConfig is the *engine* config; this is the flat *preset*.
WorldConfig = SiloPresetConfig


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class FLTrainConfig:
    """All hyperparameters for one federated training run."""
    num_silos:           int         = 3
    max_rounds:          int         = 100     # safety cap; run ends earlier when all silos done
    num_agents:          int         = 30      # agents per silo (overridden by world_configs)
    sim_days:            int         = 7       # simulated days per FL round
    min_events_to_train: int         = 10      # skip LoRA update if fewer events this round
    local_epochs:        int         = 10
    batch_size:          int         = 8
    lr:                  float       = 1e-4
    lora_rank:           int         = 8
    lora_alpha:          float       = 16.0
    lora_dropout:        float       = 0.05
    progressions:        list[str]   = field(default_factory=lambda: ["Influenza"])
    disease_strategy:    str         = "Influenza"
    seed:                int         = 42
    # Per-silo non-IID config (overrides global params when provided)
    world_configs:       Optional[list[SiloPresetConfig]] = None
    # Global IID params (used when world_configs is None)
    beta_scale:          float       = 1.0
    initial_seeds:       int         = 3
    contact_rate_sigma:  float       = 0.0
    # End condition (applied per silo; default: run until epidemic extinct)
    end_condition:       str         = "extinction"
    end_condition_param: Optional[int] = None
    # Non-IID disease distribution — Dirichlet(α) per silo over the disease list
    # α → 0: each silo dominated by one disease; α → ∞: IID mix
    dirichlet_alpha:     float       = 0.3
    # Training device: "cpu" (default, safe) or "cuda" (faster but risks VRAM clash with Ollama)
    training_device:     str         = "cpu"
    # On-disk dataset per silo — replaces in-memory replay buffer
    dataset_dir:         str         = "datasets"
    holdout_fraction:    float        = 0.2   # fraction of events withheld for testing
    train_sample_cap:    int         = 512    # max records sampled per round (stratified)
    # W&B
    wandb_project:       str         = "fedworld"
    wandb_run_name:      Optional[str] = None
    wandb_entity:        Optional[str] = None
    wandb_offline:       bool        = False
    use_ollama:          bool        = True
    patient_model:       str         = "phi3:mini"
    # FL backend: "inprocess" (default, no dependencies) or "flower" (uses Flower protocol)
    fl_backend:          str         = "inprocess"
    # Fictional disease experiment — set to a list of fictional disease names (e.g.
    # ["velarex", "sornathis"]) to activate fictional prompts and label spaces.
    # None = standard mode (influenza/pneumonia).
    disease_glossary:    Optional[list] = None
    # Label space for the doctor classifier; "fictional_disease" when disease_glossary set.
    doctor_label_space:  str         = "disease"
    # Ablation: explicitly tell LLM that influenza/pneumonia don't exist in this world.
    # Only meaningful when disease_glossary is also set.
    explicit_real_disease_exclusion: bool = False
    # Weighted FedAvg: silos with fewer training examples than this are excluded from
    # the aggregate (they still receive the global model). 0 = include all that trained.
    fedavg_min_examples: int = 0

    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(
            rank         = self.lora_rank,
            lora_alpha   = self.lora_alpha,
            lora_dropout = self.lora_dropout,
        )


# ── Shared-world config ───────────────────────────────────────────────────────

@dataclass
class SharedWorldConfig:
    """
    Configuration for shared-world FL: one SIR world, N hospital districts.

    Non-IID arises from geographic disease seeding: hospital_disease_weights[i]
    is the disease probability vector for agents assigned to hospital i.
    The single epidemic curve eliminates the phase-asynchrony noise that
    plagues independent-silo runs.

    Mirrors the real-world MIMIC/eICU structure where hospitals share a city's
    epidemic but serve different patient populations.
    """
    # World size
    n_hospitals:        int         = 2
    n_agents:           int         = 800   # total agents, divided round-robin across hospitals
    # Disease mix — per-hospital for non-IID; all hospitals see the same mix for IID
    progressions:       list[str]   = field(default_factory=lambda: ["Influenza", "Bacterial Pneumonia"])
    # hospital_disease_weights[i] = [p_flu, p_pneumo, ...] for hospital i.
    # None = IID (all hospitals get the global mix).
    hospital_disease_weights: Optional[list[list[float]]] = None
    beta_scale:         float       = 2.0
    initial_seeds:      int         = 10
    # "horizon" recommended — single world avoids stochastic extinction asymmetry
    end_condition:      str         = "horizon"
    end_condition_param: int        = 200
    contact_rate_sigma: float       = 0.0
    # FL training hyperparams (shared across all hospital learners)
    sim_days:           int         = 4
    min_events_to_train: int        = 10
    local_epochs:       int         = 3
    batch_size:         int         = 8
    lr:                 float       = 1e-4
    max_rounds:         int         = 50
    # Infrastructure
    seed:               int         = 42
    training_device:    str         = "cpu"
    dataset_dir:        str         = "datasets"
    holdout_fraction:   float       = 0.2
    train_sample_cap:   int         = 512
    lora_rank:          int         = 8
    lora_alpha:         float       = 16.0
    lora_dropout:       float       = 0.05
    wandb_project:      str         = "fedworld"
    wandb_run_name:     Optional[str] = None
    wandb_entity:       Optional[str] = None
    wandb_offline:      bool        = False
    use_ollama:         bool        = False
    patient_model:      str         = "phi3:mini"

    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(
            rank         = self.lora_rank,
            lora_alpha   = self.lora_alpha,
            lora_dropout = self.lora_dropout,
        )


# ── FedAvg weight aggregation ─────────────────────────────────────────────────

def _fedavg(
    weights_list: list[list[np.ndarray]],
    n_examples:   list[int],
) -> list[np.ndarray]:
    """Weighted average of LoRA weight arrays (FedAvg eq. 1)."""
    total = sum(n_examples)
    if total == 0:
        return weights_list[0]
    return [
        sum(w[i] * (n / total) for w, n in zip(weights_list, n_examples))
        for i in range(len(weights_list[0]))
    ]


# ── World / silo factory helpers ─────────────────────────────────────────────

def _build_sim_world_config(
    silo_idx:      int,
    wc,            # WorldConfig | None  (flat, from fl/train.py)
    cfg:           "FLTrainConfig",
    rng_dirichlet,
    data_source=None,
):
    """
    Convert the flat per-silo WorldConfig → simulation.world_config.WorldConfig.

    Centralises all the if-wc-else-cfg dispatch that was duplicated in every
    _make_world() closure.  data_source overrides the TemplateDataSource default
    (pass an OllamaDataSource when the patient LLM is active).
    """
    from simulation.world_config import (
        WorldConfig as SimWorldConfig, AgentConfig, EpidemicConfig,
    )
    from simulation.data_sources import TemplateDataSource

    if wc is not None:
        progs               = wc.progressions
        disease_strategy    = wc.disease_strategy
        end_cond_str        = wc.end_condition
        end_cond_param      = wc.end_condition_param
        n_agents            = wc.num_agents
        bg_rate             = wc.background_visit_rate
        bscale              = wc.beta_scale
        s_offset            = wc.seed_offset
        rev_icd             = wc.reveal_incubating_icd
        n_seeds             = wc.initial_seeds
        sociability_sigma   = wc.contact_rate_sigma
        confusion_rate      = wc.confusion_rate
        static_mode         = wc.static_mode
        infectious_fraction = wc.infectious_fraction
        cases_per_day       = wc.cases_per_day
        disease_weights_raw = wc.disease_weights
    else:
        unknown = [p for p in cfg.progressions if p not in PROGRESSION_STRATEGIES]
        if unknown:
            raise ValueError(
                f"Unknown progression(s): {unknown}. "
                f"Available: {list(PROGRESSION_STRATEGIES)}"
            )
        progs               = cfg.progressions
        disease_strategy    = cfg.disease_strategy
        end_cond_str        = cfg.end_condition
        end_cond_param      = cfg.end_condition_param
        n_agents            = cfg.num_agents
        bg_rate             = 0.025
        bscale              = cfg.beta_scale
        s_offset            = 0
        rev_icd             = True
        n_seeds             = cfg.initial_seeds
        sociability_sigma   = cfg.contact_rate_sigma
        confusion_rate      = getattr(cfg, "confusion_rate", 0.0)
        static_mode         = False
        infectious_fraction = 0.5
        cases_per_day       = 20
        disease_weights_raw = None

    # Disease weights: explicit → Dirichlet → uniform
    if disease_weights_raw is not None:
        d_weights = disease_weights_raw
    elif len(progs) > 1 and getattr(cfg, "dirichlet_alpha", 0) > 0:
        d_weights = rng_dirichlet.dirichlet(
            [cfg.dirichlet_alpha] * len(progs)
        ).tolist()
    else:
        d_weights = None

    if data_source is None:
        data_source = TemplateDataSource(
            seed           = cfg.seed + silo_idx,
            confusion_rate = confusion_rate,
        )

    return SimWorldConfig(
        agents   = AgentConfig(
            num_agents            = n_agents,
            data_source           = data_source,
            background_visit_rate = bg_rate,
        ),
        epidemic = EpidemicConfig(
            progressions        = progs,
            disease_strategy    = disease_strategy,
            disease_weights     = d_weights,
            beta_scale          = bscale,
            initial_seeds       = n_seeds,
            contact_rate_sigma  = sociability_sigma,
            static_mode         = static_mode,
            infectious_fraction = infectious_fraction,
            cases_per_day       = cases_per_day,
        ),
        seed_offset           = s_offset,
        reveal_incubating_icd = rev_icd,
    )


def _build_fl_silo(
    silo_idx:      int,
    wc,            # WorldConfig | None  (flat, from fl/train.py)
    cfg:           "FLTrainConfig",
    rng_dirichlet,
    dataset,
    data_source=None,
) -> "FLSilo":
    """Build one FLSilo: world + doctor learner + nurse learner + end condition."""
    from fl.silo import FLSilo
    from fl.learner import FLLearner

    sim_wc   = _build_sim_world_config(silo_idx, wc, cfg, rng_dirichlet, data_source)
    end_cond = end_condition_from_config(
        wc.end_condition if wc else cfg.end_condition,
        wc.end_condition_param if wc else cfg.end_condition_param,
    )

    world = WorldEngine(sim_wc, seed=cfg.seed + silo_idx)

    learner_kwargs = dict(
        min_events_to_train = cfg.min_events_to_train,
        local_epochs        = cfg.local_epochs,
        batch_size          = cfg.batch_size,
        lr                  = cfg.lr,
        dataset             = dataset,
        train_sample_cap    = cfg.train_sample_cap,
        device              = cfg.training_device,
    )
    doctor = FLLearner(
        lora_config = cfg.lora_config(),
        label_space = cfg.doctor_label_space,
        **learner_kwargs,
    )
    nurse = FLLearner(
        lora_config = cfg.lora_config(),
        label_space = "severity",
        **learner_kwargs,
    )
    return FLSilo(
        world               = world,
        learner             = doctor,
        nurse_learner       = nurse,
        sim_days            = cfg.sim_days,
        min_events_to_train = cfg.min_events_to_train,
        end_condition       = end_cond,
    )


def _build_bare_world(
    silo_idx:      int,
    wc,            # WorldConfig | None
    cfg:           "FLTrainConfig",
    rng_dirichlet,
    data_source=None,
) -> "WorldEngine":
    """Build a WorldEngine only (no learners) for static / centralized modes."""
    sim_wc   = _build_sim_world_config(silo_idx, wc, cfg, rng_dirichlet, data_source)
    end_cond = end_condition_from_config(
        wc.end_condition if wc else cfg.end_condition,
        wc.end_condition_param if wc else cfg.end_condition_param,
    )
    return WorldEngine(sim_wc, seed=cfg.seed + silo_idx, end_condition=end_cond)


# ── Eval / JSON export helpers ────────────────────────────────────────────────

def _nan_to_none(x):
    """Convert float NaN/inf to None so dicts are JSON-serialisable."""
    if isinstance(x, float) and (x != x or x == float("inf") or x == float("-inf")):
        return None
    return x


def _extract_round_metric(round_num: int, log: dict, n_silos: int) -> dict:
    """Distil the W&B log dict into a compact, JSON-safe round row."""
    return {
        "round":           round_num,
        "agg_diag_acc":    _nan_to_none(log.get("aggregated/diag_acc")),
        "agg_triage_acc":  _nan_to_none(log.get("aggregated/triage_acc")),
        "mean_loss":       _nan_to_none(log.get("aggregated/loss")),
        "n_silos_trained": int(log.get("aggregated/silos_trained", 0)),
        "silo_diag":   [_nan_to_none(log.get(f"silo_{i}/diag_acc"))   for i in range(n_silos)],
        "silo_triage": [_nan_to_none(log.get(f"silo_{i}/triage_acc")) for i in range(n_silos)],
        "sir_i":       [int(log.get(f"silo_{i}/sir_i", 0))            for i in range(n_silos)],
    }


def _flush_round_metrics(out_dir, metrics: list[dict]) -> None:
    """Rewrite round_metrics.json after every round so a crash doesn't lose prior rounds."""
    import json as _json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "round_metrics.json").write_text(_json.dumps(metrics, indent=2))


def _write_summary(
    out_dir,
    cfg,
    run_id: str,
    mode:   str,
    metrics: list[dict],
    *,
    diag_key:   str = "agg_diag_acc",
    triage_key: str = "agg_triage_acc",
) -> None:
    """Write summary.json (final-accuracy snapshot) at the end of a run."""
    import json as _json

    final_diag = final_triage = None
    for m in reversed(metrics):
        if m.get(diag_key) is not None:
            final_diag   = m[diag_key]
            final_triage = m.get(triage_key)
            break

    summary = {
        "experiment":       getattr(cfg, "wandb_run_name", None) or run_id,
        "mode":             mode,
        "n_silos":          getattr(cfg, "num_silos", 0),
        "n_rounds":         len(metrics),
        "final_diag_acc":   final_diag,
        "final_triage_acc": final_triage,
        "seed":             getattr(cfg, "seed", None),
        "progressions":     getattr(cfg, "progressions", []),
        "doctor_label_space": getattr(cfg, "doctor_label_space", "disease"),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(_json.dumps(summary, indent=2))
    print(f"  [eval]  {out_dir}/{{round_metrics,summary}}.json")


# ── Main training loop ────────────────────────────────────────────────────────

def run_federated_training(cfg: FLTrainConfig | None = None,
                           _shared_tracker=None, **kwargs) -> list[WorldEngine]:
    """
    Run federated LoRA training across N WorldEngine silos.

    Parameters
    ----------
    cfg : FLTrainConfig | None
        If None, a default config is built from **kwargs.
    **kwargs
        Any FLTrainConfig field (e.g. num_silos=5, num_rounds=20).

    Returns
    -------
    List of trained WorldEngine instances (weights updated to final global model).
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    if getattr(cfg, "fl_backend", "inprocess") == "flower":
        return run_flower_federated_training(cfg, _shared_tracker=_shared_tracker)

    import wandb

    # ── W&B initialisation ────────────────────────────────────────────────────
    if cfg.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    wandb_cfg = {k: v for k, v in asdict(cfg).items()
                 if not k.startswith("wandb_") and k != "world_configs"}
    run = wandb.init(
        project = cfg.wandb_project,
        name    = cfg.wandb_run_name,
        entity  = cfg.wandb_entity,
        config  = wandb_cfg,
        reinit  = "finish_previous",
    )

    # ── Per-silo on-disk datasets ─────────────────────────────────────────────
    # Created here (before worlds) so each run gets its own subdirectory.
    _run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    _dataset_dir = Path(cfg.dataset_dir) / _run_id
    from fl.dataset import SiloDataset
    _silo_datasets = [
        SiloDataset(_dataset_dir, i, cfg.holdout_fraction, cfg.seed)
        for i in range(cfg.num_silos)
    ]
    print(f"  [dataset] {_dataset_dir}/silo_{{0..{cfg.num_silos-1}}}/  "
          f"holdout={cfg.holdout_fraction:.0%}  cap={cfg.train_sample_cap}")

    # ── Create silos ──────────────────────────────────────────────────────────
    rng_dirichlet = np.random.default_rng(cfg.seed)
    silos: list[FLSilo] = [
        _build_fl_silo(
            i,
            cfg.world_configs[i] if cfg.world_configs else None,
            cfg,
            rng_dirichlet,
            _silo_datasets[i],
        )
        for i in range(cfg.num_silos)
    ]

    # ── Prototype libraries — one per silo ────────────────────────────────────
    from fl.prototype_library import PrototypeLibrary, federate_libraries
    proto_libs = [PrototypeLibrary(silo_id=i) for i in range(cfg.num_silos)]

    # ── Stereotype libraries — one per silo (federated centroid averaging) ────
    from fl.stereotype_library import StereotypeLibrary, aggregate_stereotypes
    stereo_libs = [StereotypeLibrary() for _ in range(cfg.num_silos)]

    # ── Wire LLM clients — single shared Ollama server, all silos call it ────
    ollama_client  = None
    ollama_active  = False
    ollama_status  = "disabled"
    patient_status = "disabled"
    if cfg.use_ollama:
        from simulation.ollama_client import OllamaDiagnosticClient
        from simulation.patient_llm   import PatientLLMClient

        # Patient LLM — small frozen model, generates opening statements + answers
        patient_llm = PatientLLMClient(model=cfg.patient_model)
        if patient_llm.health_check():
            for silo in silos:
                silo.set_patient_llm(patient_llm)
            patient_status = f"active ({patient_llm.model})"
        else:
            patient_llm    = None
            patient_status = f"unreachable — pull: ollama pull {cfg.patient_model}"

        # Doctor LLM — multi-turn diagnostic model, receives formatted vitals
        ollama_client = OllamaDiagnosticClient(
            patient_llm       = patient_llm,
            disease_glossary  = cfg.disease_glossary,
            explicit_exclusion= getattr(cfg, "explicit_real_disease_exclusion", False),
        )
        if ollama_client.health_check():
            for i, silo in enumerate(silos):
                q = silo.clinic_queue
                # Each silo gets its own diagnostic fn capturing its proto lib
                _lib = proto_libs[i]
                silo.register_diagnostic_fn(
                    lambda ev, _q=q, _lib=_lib: ollama_client.diagnose(
                        ev, _queue=_q, _proto_lib=_lib)
                )
            # Dedicated CPU encoder for prototype retrieval — kept separate from
            # the FL models so it's never released mid-run and always on CPU.
            from transformers import AutoTokenizer
            from fl.lora import build_model as _build_model
            _enc_tokenizer = AutoTokenizer.from_pretrained(cfg.lora_config().model_name_or_path)
            _enc_model     = _build_model(cfg.lora_config()).cpu()
            ollama_client.set_prototype_library(
                library       = None,     # per-silo lib passed at diagnose() time
                encoder_model = _enc_model,
                tokenizer     = _enc_tokenizer,
                k             = 3,
            )
            ollama_active = True
            ollama_status = f"active ({ollama_client.model})"
        else:
            ollama_status = "unreachable — using stub oracle"

    prog_summary = ", ".join(cfg.progressions) if not cfg.world_configs else "per-silo (WorldConfig)"
    print(
        f"\n{'─'*60}\n"
        f"  FedWorld FL Training\n"
        f"  silos={cfg.num_silos}  max_rounds={cfg.max_rounds}  "
        f"agents/silo={cfg.world_configs[0].num_agents if cfg.world_configs else cfg.num_agents}\n"
        f"  sim_days={cfg.sim_days}  min_events={cfg.min_events_to_train}  "
        f"local_epochs={cfg.local_epochs}  lr={cfg.lr}\n"
        f"  progressions={prog_summary}  seed={cfg.seed}\n"
        f"  end_condition={cfg.end_condition}  (simulation-guided)\n"
        f"  LLM doctor:  {ollama_status}\n"
        f"  LLM patient: {patient_status}\n"
        f"{'─'*60}\n"
    )

    # ── Report ────────────────────────────────────────────────────────────────
    from simulation.report import RunReport
    _report    = RunReport(cfg, _run_id)
    _report_path = Path(f"reports/run_{_run_id}.html")

    # ── Embedding tracker ─────────────────────────────────────────────────────
    from viz.embedding_tracker import EmbeddingTracker
    if _shared_tracker is not None:
        _tracker      = _shared_tracker
        _own_tracker  = False
    else:
        print("  [embed] Building benchmark probe set…")
        _embed_dir   = Path(f"viz_output/embeddings/run_{_run_id}")
        _tracker     = EmbeddingTracker(_embed_dir, cfg.lora_config())
        _own_tracker = True
        print(f"  [embed] {len(_tracker.probe_events)} probe events ready "
              f"({len(set(e.ground_truth for e in _tracker.probe_events))} unique labels)\n")

    # ── Initialise global weights from silo 0 (doctor + nurse) ───────────────
    global_weights       = silos[0].get_weights()
    global_nurse_weights = silos[0].get_nurse_weights()
    silos[0].release_model()   # free immediately — rebuilt in round 1 with set_weights

    _known_stereo_labels: set[str] = set()   # tracks growth across rounds
    _run_metrics:         list[dict] = []    # per-round rows for JSON export

    # ── FL rounds — simulation-guided (run until all silos done or max_rounds) ─
    for round_num in range(1, cfg.max_rounds + 1):
        t0 = time.time()

        round_weights:       list[list[np.ndarray]] = []
        round_nurse_weights: list[list[np.ndarray]] = []
        round_nexamples:     list[int]             = []
        round_trained:       list[bool]            = []
        log: dict = {"round": round_num}

        agg_loss = agg_acc = agg_n = agg_trained = 0.0

        # Collect weights for embedding snapshot while model is still built,
        # before release_model(). This avoids rebuilding N models during snapshot.
        snap_fed_weights:   list = []
        snap_local_weights: list = []

        for i, silo in enumerate(silos):
            if silo.is_done:
                silo.try_accept_global(global_weights)
                silo.set_nurse_weights(global_nurse_weights)
                m = silo.run_round(round_num=round_num)
                fedavg_n = 0
            else:
                silo.set_weights(global_weights)
                silo.set_nurse_weights(global_nurse_weights)
                m = silo.run_round(round_num=round_num)
                fedavg_n = m.get("num_examples", 0)
                if cfg.fedavg_min_examples > 0 and fedavg_n < cfg.fedavg_min_examples:
                    fedavg_n = 0  # too few examples — receives global model, excluded from aggregate

            fed_w   = silo.get_weights()
            nurse_w = silo.get_nurse_weights()
            from fl.lora import get_lora_weights as _glw
            local_w = _glw(silo.learner.local_model)
            snap_fed_weights.append(fed_w)
            snap_local_weights.append(local_w)

            round_weights.append(fed_w)
            round_nurse_weights.append(nurse_w)
            silo.release_model()   # free VRAM — also releases nurse model
            n         = m.get("num_examples", 0)
            did_train = bool(m.get("trained", 0))
            round_nexamples.append(fedavg_n)
            round_trained.append(did_train)

            # Per-silo metrics
            epoch_losses = m.pop("epoch_losses", [])
            for key, val in m.items():
                if isinstance(val, (int, float)):
                    log[f"silo_{i}/{key}"] = val

            if epoch_losses:
                log[f"silo_{i}/train_loss_final"] = epoch_losses[-1]
                for ep_i, ep_loss in enumerate(epoch_losses):
                    log[f"silo_{i}/epoch_{ep_i}_loss"] = ep_loss

            if silo.is_done:
                log[f"silo_{i}/done"] = 1
                log[f"silo_{i}/stop_reason"] = silo.stop_reason or "done"

            if n > 0 and did_train:
                agg_loss    += m.get("loss",       0.0) * n
                agg_n       += n
                agg_trained += m.get("trained_on", 0)
                for sub in ("triage_acc", "diag_acc", "icd_exact_acc",
                            "combined_acc", "triage_macro_f1", "danger_rate",
                            "avg_turns", "first_turn_rate", "action_accuracy",
                            "fp_escalation_rate", "local_triage_acc", "local_diag_acc",
                            "fl_gain", "fl_diag_gain"):
                    v = m.get(sub, float("nan"))
                    if v == v:
                        agg_key = f"aggregated/{sub}"
                        log[agg_key] = log.get(agg_key, 0.0) + v * n

        # ── FedAvg — doctor + nurse aggregated separately ──────────────────
        if any(n > 0 for n in round_nexamples):
            global_weights       = _fedavg(round_weights,       round_nexamples)
            global_nurse_weights = _fedavg(round_nurse_weights, round_nexamples)
        log["aggregated/silos_trained"]    = sum(round_trained)
        log["aggregated/silos_in_fedavg"]  = sum(1 for n in round_nexamples if n > 0)

        # ── Embedding snapshot — weights already collected, no model rebuilds ──
        _tracker.snapshot(round_num, global_weights,
                          silo_fed_weights=snap_fed_weights,
                          silo_local_weights=snap_local_weights)

        # ── Federated few-shot update — refresh Ollama doctor's example bank ──
        if ollama_active and ollama_client is not None:
            round_events = [ev for silo in silos for ev in silo.last_round_events]
            ollama_client.update_examples(round_events)
            log["aggregated/llm_few_shot_examples"] = ollama_client.num_examples()

        # ── Prototype library — ingest + federate ─────────────────────────────
        if ollama_active and ollama_client is not None and ollama_client._proto_encoder is not None:
            enc_model, enc_tok = ollama_client._proto_encoder
            enc_device = "cpu"
            for i, silo in enumerate(silos):
                added = proto_libs[i].add_from_events(
                    silo.last_round_events, enc_model, enc_tok,
                    round_num, device=enc_device,
                )
                log[f"silo_{i}/proto_added"] = added
                log[f"silo_{i}/proto_size"]  = proto_libs[i].size()

            # Federate: merge all silos' subsets into each silo's library
            global_protos = federate_libraries(proto_libs, max_per_label=10)
            for lib in proto_libs:
                lib.merge(global_protos)
            log["aggregated/proto_total"] = sum(l.size() for l in proto_libs)

            # ── Stereotype federation: update centroids, aggregate, broadcast ─
            for i, silo in enumerate(silos):
                stereo_libs[i].update_from_events(
                    silo.last_round_events, enc_model, enc_tok, device=enc_device,
                )
            global_stereos = aggregate_stereotypes(
                [l.get_state() for l in stereo_libs]
            )
            for lib in stereo_libs:
                lib.set_global(global_stereos)
            ollama_client.update_global_stereotypes(global_stereos,
                                                    n_silos=cfg.num_silos)
            _new_labels = [l for l in global_stereos if l not in _known_stereo_labels]
            _known_stereo_labels.update(global_stereos.keys())
            log["aggregated/stereo_classes"]     = len(global_stereos)
            log["aggregated/stereo_new_classes"] = len(_new_labels)
            if _new_labels:
                print(f"  [stereo] New class centroid(s) federated: {sorted(_new_labels)}")

        # ── Early exit if all silos finished ──────────────────────────────────
        if all(silo.is_done for silo in silos):
            log["aggregated/all_silos_done"] = 1
            run.log(log, step=round_num)
            _run_metrics.append(_extract_round_metric(round_num, log, cfg.num_silos))
            _flush_round_metrics(_dataset_dir, _run_metrics)
            elapsed = time.time() - t0
            _print_round(round_num, cfg.max_rounds, log, elapsed)
            _report.add_round(round_num, log, [s.last_round_events for s in silos])
            reasons = [silo.stop_reason or "done" for silo in silos]
            print(f"\n  All silos finished: {reasons[0]}")
            break

        # ── Aggregated log ────────────────────────────────────────────────────
        if agg_n > 0:
            log["aggregated/loss"]         = agg_loss / agg_n
            log["aggregated/num_examples"] = int(agg_n)
            log["aggregated/num_trained"]  = int(agg_trained)
            for sub in ("triage_acc", "diag_acc", "icd_exact_acc",
                        "combined_acc", "triage_macro_f1", "danger_rate",
                        "avg_turns", "first_turn_rate", "action_accuracy",
                        "fp_escalation_rate", "local_triage_acc", "local_diag_acc",
                        "fl_gain", "fl_diag_gain"):
                k = f"aggregated/{sub}"
                if k in log:
                    log[k] /= agg_n

        run.log(log, step=round_num)
        _run_metrics.append(_extract_round_metric(round_num, log, cfg.num_silos))
        _flush_round_metrics(_dataset_dir, _run_metrics)
        _report.add_round(round_num, log, [s.last_round_events for s in silos])

        elapsed = time.time() - t0
        _print_round(round_num, cfg.max_rounds, log, elapsed)

    # ── Distribute final global models (doctor + nurse) to all silos ─────────
    for silo in silos:
        silo.set_weights(global_weights)
        silo.set_nurse_weights(global_nurse_weights)

    # ── Save final federated weights for post-hoc eval ────────────────────────
    from fl.lora import weights_to_npz
    weights_dir = _dataset_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    np.savez(str(weights_dir / "fl_final.npz"),      **weights_to_npz(global_weights))
    np.savez(str(weights_dir / "fl_final_nurse.npz"), **weights_to_npz(global_nurse_weights))
    print(f"  [weights] {weights_dir}/fl_final{{,_nurse}}.npz")

    _write_summary(_dataset_dir, cfg, _run_id, "federated", _run_metrics)

    run.finish()

    # ── Embedding plots — generate before writing report so they embed inline ──
    if _own_tracker:
        print("  [embed] Generating evolution plots…")
        embed_plots = _tracker.save_plots()
        for p in embed_plots:
            print(f"  [embed] {p.resolve()}")

    _report.write(_report_path, tracker=_tracker if _own_tracker else None)

    run_ref = run.url or f"offline — sync with: wandb sync {run.dir}"
    print(f"\nTraining complete. W&B: {run_ref}")
    print(f"Report:           {_report_path.resolve()}\n")
    return silos


# ── Shared-world FL training ─────────────────────────────────────────────────

def run_shared_world_training(cfg: SharedWorldConfig,
                               _shared_tracker=None) -> dict:
    """
    Shared-world federated training: one SIR epidemic, N hospital districts.

    Unlike run_federated_training (N independent WorldEngines), here all
    hospitals co-exist in the same simulation world.  This eliminates the
    phase-asynchrony noise from independent SIR dynamics while preserving
    non-IID disease distributions via geographic seeding:
    hospital_disease_weights[i] skews which disease agents in district i
    are most likely to contract.

    Structural correspondence to run_federated_training:
      · One WorldEngine  ↔  N WorldEngines (silos)
      · N ClinicQueue[i] ↔  silo.clinic_queue
      · N FLLearner[i]   ↔  silo.learner
      · FedAvg unchanged

    Paper framing: "hospitals in the same city share an epidemic arc but
    serve different patient demographics" (MIMIC/eICU-like).
    """
    import wandb
    from simulation.strategies import STRATEGIES
    from simulation.progression import PROGRESSION_STRATEGIES
    from simulation.end_conditions import from_config as end_condition_from_config
    from fl.learner import FLLearner
    from fl.dataset import SiloDataset

    if cfg.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    from dataclasses import asdict
    run = wandb.init(
        project = cfg.wandb_project,
        name    = cfg.wandb_run_name,
        entity  = cfg.wandb_entity,
        config  = {k: v for k, v in asdict(cfg).items() if not k.startswith("wandb_")},
        reinit  = "finish_previous",
    )

    _run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    _dataset_dir = Path(cfg.dataset_dir) / f"shared_{_run_id}"

    # ── Build the single shared world ────────────────────────────────────────
    from simulation.world_config import (
        WorldConfig as SimWorldConfig, AgentConfig, EpidemicConfig,
    )
    from simulation.data_sources import TemplateDataSource

    end_cond = end_condition_from_config(cfg.end_condition, cfg.end_condition_param)

    _shared_wc = SimWorldConfig(
        agents=AgentConfig(
            num_agents=cfg.n_agents,
            data_source=TemplateDataSource(
                seed=cfg.seed,
                confusion_rate=getattr(cfg, "confusion_rate", 0.0),
            ),
        ),
        epidemic=EpidemicConfig(
            progressions=cfg.progressions,
            disease_strategy=getattr(cfg, "disease_strategy",
                                     cfg.progressions[0] if cfg.progressions else "Influenza"),
            beta_scale=cfg.beta_scale,
            initial_seeds=cfg.initial_seeds,
            contact_rate_sigma=cfg.contact_rate_sigma,
        ),
    )
    world = WorldEngine(
        _shared_wc,
        seed=cfg.seed,
        n_hospitals=cfg.n_hospitals,
        hospital_disease_weights=cfg.hospital_disease_weights,
        end_condition=end_cond,
    )

    # ── Wire Ollama (optional) ────────────────────────────────────────────────
    patient_status = "disabled"
    if cfg.use_ollama:
        from simulation.ollama_client import OllamaDiagnosticClient
        from simulation.patient_llm   import PatientLLMClient
        patient_llm = PatientLLMClient(model=cfg.patient_model)
        if patient_llm.health_check():
            world.set_patient_llm(patient_llm)
            patient_status = f"active ({patient_llm.model})"
        else:
            patient_llm = None
            patient_status = f"unreachable — pull: ollama pull {cfg.patient_model}"

        ollama_client = OllamaDiagnosticClient(patient_llm=patient_llm)
        if ollama_client.health_check():
            world.register_diagnostic_fn(
                lambda ev: ollama_client.diagnose(ev, _queue=world.clinic_queue)
            )

    # ── Create N FLLearner pairs (doctor + nurse), one per hospital ───────────
    learner_kwargs = dict(
        min_events_to_train = cfg.min_events_to_train,
        local_epochs        = cfg.local_epochs,
        batch_size          = cfg.batch_size,
        lr                  = cfg.lr,
        train_sample_cap    = cfg.train_sample_cap,
        device              = cfg.training_device,
    )
    learners:       list[FLLearner] = []
    nurse_learners: list[FLLearner] = []
    for i in range(cfg.n_hospitals):
        ds      = SiloDataset(_dataset_dir, i, cfg.holdout_fraction, cfg.seed)
        nds     = SiloDataset(_dataset_dir / "nurse", i, cfg.holdout_fraction, cfg.seed)
        learners.append(FLLearner(
            lora_config  = cfg.lora_config(),
            label_space  = "disease",
            dataset      = ds,
            **learner_kwargs,
        ))
        nurse_learners.append(FLLearner(
            lora_config  = cfg.lora_config(),
            label_space  = "severity",
            dataset      = nds,
            **learner_kwargs,
        ))

    print(
        f"\n{'─'*60}\n"
        f"  FedWorld Shared-World Training\n"
        f"  hospitals={cfg.n_hospitals}  n_agents={cfg.n_agents}  "
        f"agents/hospital≈{cfg.n_agents // cfg.n_hospitals}\n"
        f"  progressions={cfg.progressions}  seed={cfg.seed}\n"
        f"  hospital_disease_weights={cfg.hospital_disease_weights}\n"
        f"  max_rounds={cfg.max_rounds}  sim_days={cfg.sim_days}  "
        f"local_epochs={cfg.local_epochs}\n"
        f"  LLM patient: {patient_status}\n"
        f"{'─'*60}\n"
    )

    global_weights       = learners[0].get_weights()
    global_nurse_weights = nurse_learners[0].get_weights()

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for round_num in range(1, cfg.max_rounds + 1):
        t0 = time.time()
        log: dict = {"round": round_num}

        # Advance the shared world for sim_days
        start_sizes = [len(q.processed) for q in world.clinic_queues]
        for _ in range(cfg.sim_days * WorldEngine.TICKS_PER_DAY):
            world.step_tick()
            if world.is_done:
                break

        # Collect per-hospital events from the processed archive
        events_per_hosp: list[list] = []
        for i, q in enumerate(world.clinic_queues):
            new_events = q.processed[start_sizes[i]:]
            events_per_hosp.append(list(new_events))
            del q.processed[:start_sizes[i]]   # trim to keep archive bounded

        # Per-hospital: set global weights → evaluate → train → collect
        round_weights:       list[list[np.ndarray]] = []
        round_nurse_weights: list[list[np.ndarray]] = []
        round_nexamples:     list[int]             = []

        agg_loss = agg_n = agg_trained = 0.0
        sir = world.sir_model

        for i in range(cfg.n_hospitals):
            events = events_per_hosp[i]
            learners[i].set_weights(global_weights)
            nurse_learners[i].set_weights(global_nurse_weights)

            metrics       = learners[i].evaluate(events)
            nurse_metrics = nurse_learners[i].evaluate(events)
            n_ev          = len(events)

            if n_ev >= cfg.min_events_to_train:
                n_tr, epoch_losses = learners[i].train(events, round_num=round_num)
                learners[i].train_local(events, round_num=round_num)
                nurse_learners[i].train(events, round_num=round_num)
                nurse_learners[i].train_local(events, round_num=round_num)
                trained = 1
            else:
                n_tr, epoch_losses, trained = 0, [], 0

            fed_w  = learners[i].get_weights()
            nurse_w = nurse_learners[i].get_weights()
            round_weights.append(fed_w)
            round_nurse_weights.append(nurse_w)
            round_nexamples.append(n_tr)

            # Per-hospital metrics
            log[f"hosp_{i}/num_events"]  = n_ev
            log[f"hosp_{i}/trained"]     = trained
            log[f"hosp_{i}/trained_on"]  = n_tr
            log[f"hosp_{i}/diag_acc"]    = metrics.get("diag_acc",  float("nan"))
            log[f"hosp_{i}/triage_acc"]  = nurse_metrics.get("triage_acc", float("nan"))
            log[f"hosp_{i}/sir_i"]       = sir.I
            if epoch_losses:
                log[f"hosp_{i}/train_loss_final"] = epoch_losses[-1]

            if n_tr > 0:
                agg_loss    += metrics.get("loss", 0.0) * n_tr
                agg_n       += n_tr
                agg_trained += n_tr

        # FedAvg
        if any(n > 0 for n in round_nexamples):
            global_weights       = _fedavg(round_weights, round_nexamples)
            global_nurse_weights = _fedavg(round_nurse_weights, round_nexamples)

        if agg_n > 0:
            log["aggregated/loss"]        = agg_loss / agg_n
            log["aggregated/num_trained"] = int(agg_n)
            for sub in ("diag_acc", "triage_acc"):
                vals = [log[f"hosp_{i}/{sub}"] for i in range(cfg.n_hospitals)]
                valid = [v for v in vals if v == v]
                if valid:
                    log[f"aggregated/{sub}"] = sum(valid) / len(valid)

        log["world/sir_s"] = sir.S
        log["world/sir_i"] = sir.I
        log["world/sir_r"] = sir.R
        log["world/day"]   = world.current_day

        run.log(log, step=round_num)
        elapsed = time.time() - t0

        diag = log.get("aggregated/diag_acc", float("nan"))
        tr   = log.get("aggregated/triage_acc", float("nan"))
        n    = int(log.get("aggregated/num_trained", 0))
        print(
            f"  Round {round_num:>3}/~{cfg.max_rounds} | "
            f"diag={diag:.3f}  triage={tr:.3f}  "
            f"S={sir.S} I={sir.I} R={sir.R}  n={n}  "
            f"{elapsed:.1f}s"
        )

        if world.is_done:
            print(f"\n  World finished: {world.stop_reason}")
            break

    # ── Save final weights ────────────────────────────────────────────────────
    from fl.lora import weights_to_npz
    weights_dir = _dataset_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(weights_dir / "fl_final.npz"),       **weights_to_npz(global_weights))
    np.savez(str(weights_dir / "fl_final_nurse.npz"), **weights_to_npz(global_nurse_weights))
    print(f"  [weights] {weights_dir}/fl_final{{,_nurse}}.npz")

    run.finish()
    return {"world": world, "learners": learners, "nurse_learners": nurse_learners}


# ── Console progress helper ───────────────────────────────────────────────────

def _print_round(round_num: int, max_rounds: int, log: dict, elapsed: float) -> None:
    nan = float("nan")
    agg_loss    = log.get("aggregated/loss",              nan)
    triage      = log.get("aggregated/triage_acc",        nan)
    local_tr    = log.get("aggregated/local_triage_acc",  nan)
    fl_gain     = log.get("aggregated/fl_gain",           nan)
    diag        = log.get("aggregated/diag_acc",          nan)
    local_diag  = log.get("aggregated/local_diag_acc",    nan)
    fl_diag_gain= log.get("aggregated/fl_diag_gain",      nan)
    macro_f1    = log.get("aggregated/triage_macro_f1",   nan)
    danger      = log.get("aggregated/danger_rate",       nan)
    fp_esc      = log.get("aggregated/fp_escalation_rate",nan)
    avg_turns   = log.get("aggregated/avg_turns",         nan)
    act_acc     = log.get("aggregated/action_accuracy",   nan)
    n_trained   = int(log.get("aggregated/num_trained",   0))
    silos_tr    = int(log.get("aggregated/silos_trained", 0))

    def _fmt(v: float, dec: int = 2) -> str:
        return f"{v:.{dec}f}" if v == v else "—"

    silo_parts = []
    i = 0
    while f"silo_{i}/sir_i" in log:
        sir_i = int(log[f"silo_{i}/sir_i"])
        n_ev  = int(log.get(f"silo_{i}/num_events", 0))
        if log.get(f"silo_{i}/done"):
            silo_parts.append(f"s{i}[I={sir_i} ~done]")
        else:
            did  = "✓" if log.get(f"silo_{i}/trained", 0) else "✗"
            tr_i = _fmt(log.get(f"silo_{i}/triage_acc", nan))
            dg_i = _fmt(log.get(f"silo_{i}/diag_acc",   nan))
            silo_parts.append(f"s{i}[I={sir_i} ev={n_ev} {did}tr={tr_i} dg={dg_i}]")
        i += 1

    print(
        f"  Round {round_num:>3}/~{max_rounds} | "
        f"loss={_fmt(agg_loss,4)} "
        f"tr={_fmt(triage)} local_tr={_fmt(local_tr)} +Δtr={_fmt(fl_gain,3)} "
        f"diag={_fmt(diag)} local_dg={_fmt(local_diag)} +Δdg={_fmt(fl_diag_gain,3)} "
        f"f1={_fmt(macro_f1)} "
        f"danger={_fmt(danger)} fp_esc={_fmt(fp_esc)} "
        f"dr_turns={_fmt(avg_turns,1)} dr_act={_fmt(act_acc)} "
        f"n={n_trained:>4} silos={silos_tr} | "
        f"{' '.join(silo_parts) or '—'} | "
        f"{elapsed:.1f}s"
    )


# ── Flower backend ────────────────────────────────────────────────────────────

def run_flower_federated_training(
    cfg: FLTrainConfig | None = None,
    _shared_tracker=None,
    **kwargs,
) -> list[WorldEngine]:
    """
    Flower-protocol federated training (in-process, no gRPC).

    Uses Flower's NumPyClient.fit() + WandBFedAvg.aggregate_fit() for the
    round protocol, driving all clients and aggregation in-process.
    World state (SIR model, LoRA adapters) persists between rounds.

    Callable via run_federated_training(cfg) when cfg.fl_backend == "flower",
    or directly.  W&B keys are identical to the inprocess backend so dashboards
    are directly comparable.
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    try:
        from flwr.common import (
            ndarrays_to_parameters, parameters_to_ndarrays,
            FitRes, Status, Code,
        )
    except ImportError as exc:
        raise ImportError(
            "Flower not installed. Run: pip install 'flwr[simulation]>=1.8'"
        ) from exc

    import wandb
    from fl.server import WandBFedAvg, InProcessClientProxy
    from fl.flwr_client import make_flower_client
    from fl.lora import get_lora_weights as _glw

    # ── W&B initialisation ────────────────────────────────────────────────────
    if cfg.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    wandb_cfg = {k: v for k, v in asdict(cfg).items()
                 if not k.startswith("wandb_") and k != "world_configs"}
    run = wandb.init(
        project = cfg.wandb_project,
        name    = cfg.wandb_run_name,
        entity  = cfg.wandb_entity,
        config  = wandb_cfg,
        reinit  = "finish_previous",
    )

    # ── Per-silo on-disk datasets ─────────────────────────────────────────────
    _run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    _dataset_dir = Path(cfg.dataset_dir) / _run_id
    from fl.dataset import SiloDataset
    _silo_datasets = [
        SiloDataset(_dataset_dir, i, cfg.holdout_fraction, cfg.seed)
        for i in range(cfg.num_silos)
    ]
    print(f"  [dataset] {_dataset_dir}/silo_{{0..{cfg.num_silos-1}}}/  "
          f"holdout={cfg.holdout_fraction:.0%}  cap={cfg.train_sample_cap}")

    # ── Create silos ──────────────────────────────────────────────────────────
    rng_dirichlet = np.random.default_rng(cfg.seed)
    silos: list[FLSilo] = [
        _build_fl_silo(
            i,
            cfg.world_configs[i] if cfg.world_configs else None,
            cfg,
            rng_dirichlet,
            _silo_datasets[i],
        )
        for i in range(cfg.num_silos)
    ]

    # ── Flower clients + strategy ─────────────────────────────────────────────
    flower_clients = [make_flower_client(silos[i], cid=str(i))
                      for i in range(cfg.num_silos)]

    # WandBFedAvg handles FedAvg math; W&B logging done in our round loop
    # so the log structure is identical to the inprocess backend.
    strategy = WandBFedAvg(
        wandb_run            = None,
        fraction_fit         = 1.0,
        fraction_evaluate    = 0.0,
        min_fit_clients      = 1,
        min_evaluate_clients = 0,
        min_available_clients= 1,
        initial_parameters   = ndarrays_to_parameters(silos[0].get_weights()),
    )

    # ── Prototype libraries ───────────────────────────────────────────────────
    from fl.prototype_library import PrototypeLibrary, federate_libraries
    proto_libs = [PrototypeLibrary(silo_id=i) for i in range(cfg.num_silos)]

    # ── Stereotype libraries ──────────────────────────────────────────────────
    from fl.stereotype_library import StereotypeLibrary, aggregate_stereotypes
    stereo_libs = [StereotypeLibrary() for _ in range(cfg.num_silos)]

    # ── Wire LLM clients ──────────────────────────────────────────────────────
    ollama_client  = None
    ollama_active  = False
    ollama_status  = "disabled"
    patient_status = "disabled"
    if cfg.use_ollama:
        from simulation.ollama_client import OllamaDiagnosticClient
        from simulation.patient_llm   import PatientLLMClient

        patient_llm = PatientLLMClient(model=cfg.patient_model)
        if patient_llm.health_check():
            for silo in silos:
                silo.set_patient_llm(patient_llm)
            patient_status = f"active ({patient_llm.model})"
        else:
            patient_llm    = None
            patient_status = f"unreachable — pull: ollama pull {cfg.patient_model}"

        ollama_client = OllamaDiagnosticClient(
            patient_llm       = patient_llm,
            disease_glossary  = cfg.disease_glossary,
            explicit_exclusion= getattr(cfg, "explicit_real_disease_exclusion", False),
        )
        if ollama_client.health_check():
            for i, silo in enumerate(silos):
                q    = silo.clinic_queue
                _lib = proto_libs[i]
                silo.register_diagnostic_fn(
                    lambda ev, _q=q, _lib=_lib: ollama_client.diagnose(
                        ev, _queue=_q, _proto_lib=_lib)
                )
            from transformers import AutoTokenizer
            from fl.lora import build_model as _build_model
            _enc_tokenizer = AutoTokenizer.from_pretrained(cfg.lora_config().model_name_or_path)
            _enc_model     = _build_model(cfg.lora_config()).cpu()
            ollama_client.set_prototype_library(
                library       = None,
                encoder_model = _enc_model,
                tokenizer     = _enc_tokenizer,
                k             = 3,
            )
            ollama_active = True
            ollama_status = f"active ({ollama_client.model})"
        else:
            ollama_status = "unreachable — using stub oracle"

    prog_summary = ", ".join(cfg.progressions) if not cfg.world_configs else "per-silo (WorldConfig)"
    print(
        f"\n{'─'*60}\n"
        f"  FedWorld FL Training (Flower backend)\n"
        f"  silos={cfg.num_silos}  max_rounds={cfg.max_rounds}  "
        f"agents/silo={cfg.world_configs[0].num_agents if cfg.world_configs else cfg.num_agents}\n"
        f"  sim_days={cfg.sim_days}  min_events={cfg.min_events_to_train}  "
        f"local_epochs={cfg.local_epochs}  lr={cfg.lr}\n"
        f"  progressions={prog_summary}  seed={cfg.seed}\n"
        f"  end_condition={cfg.end_condition}  (simulation-guided)\n"
        f"  LLM doctor:  {ollama_status}\n"
        f"  LLM patient: {patient_status}\n"
        f"{'─'*60}\n"
    )

    # ── Report ────────────────────────────────────────────────────────────────
    from simulation.report import RunReport
    _report      = RunReport(cfg, _run_id)
    _report_path = Path(f"reports/run_{_run_id}.html")

    # ── Embedding tracker ─────────────────────────────────────────────────────
    from viz.embedding_tracker import EmbeddingTracker
    if _shared_tracker is not None:
        _tracker      = _shared_tracker
        _own_tracker  = False
    else:
        print("  [embed] Building benchmark probe set…")
        _embed_dir   = Path(f"viz_output/embeddings/run_{_run_id}")
        _tracker     = EmbeddingTracker(_embed_dir, cfg.lora_config())
        _own_tracker = True
        print(f"  [embed] {len(_tracker.probe_events)} probe events ready "
              f"({len(set(e.ground_truth for e in _tracker.probe_events))} unique labels)\n")

    # ── Initialise global weights (doctor + nurse) ────────────────────────────
    global_weights       = silos[0].get_weights()
    global_nurse_weights = silos[0].get_nurse_weights()
    silos[0].release_model()

    _known_stereo_labels: set[str] = set()   # tracks growth across rounds

    # ── FL rounds ────────────────────────────────────────────────────────────
    for round_num in range(1, cfg.max_rounds + 1):
        t0 = time.time()

        fit_results:   list[tuple]  = []
        log:           dict         = {"round": round_num}
        agg_loss = agg_n = agg_trained = 0.0
        snap_fed_weights:   list = []
        snap_local_weights: list = []
        round_trained: list[bool] = []

        for i, (silo, client) in enumerate(zip(silos, flower_clients)):
            # Pass weights as List[np.ndarray] — NumPyClient.fit() expects ndarrays
            # (Flower's NumPyClientWrapper does this conversion in real deployment;
            # here we call fit() directly so we skip the wrapper).
            weights_out, n_examples, metrics = client.fit(
                global_weights, {"round_num": round_num}
            )

            # Capture local weights before release (Bug 2 fix: saved in learner)
            local_w = _glw(silo.learner.local_model)
            snap_fed_weights.append(weights_out)
            snap_local_weights.append(local_w)
            silo.release_model()

            proxy   = InProcessClientProxy(str(i))
            fit_res = FitRes(
                status      = Status(code=Code.OK, message=""),
                parameters  = ndarrays_to_parameters(weights_out),
                num_examples= n_examples,
                metrics     = metrics,
            )
            fit_results.append((proxy, fit_res))

            did_train = bool(metrics.get("trained", 0))
            round_trained.append(did_train)

            for key, val in metrics.items():
                if key == "silo_id":
                    continue
                log[f"silo_{i}/{key}"] = val

            if silo.is_done:
                log[f"silo_{i}/done"]        = 1
                log[f"silo_{i}/stop_reason"] = silo.stop_reason or "done"

            n = n_examples
            if n > 0 and did_train:
                loss_v = metrics.get("loss", -1.0)
                if loss_v != -1.0:
                    agg_loss += loss_v * n
                agg_n       += n
                agg_trained += metrics.get("trained_on", 0)
                for sub in ("triage_acc", "diag_acc", "icd_exact_acc",
                            "combined_acc", "triage_macro_f1", "danger_rate",
                            "avg_turns", "first_turn_rate", "action_accuracy",
                            "fp_escalation_rate", "local_triage_acc", "local_diag_acc",
                            "fl_gain", "fl_diag_gain"):
                    v = metrics.get(sub, -1.0)
                    if v != -1.0:
                        agg_key = f"aggregated/{sub}"
                        log[agg_key] = log.get(agg_key, 0.0) + v * n

        # ── Flower FedAvg aggregation ─────────────────────────────────────────
        # Done silos pass n_examples=0, so aggregate_fit naturally excludes them.
        active_fit_results = [(p, r) for p, r in fit_results if r.num_examples > 0]
        if active_fit_results:
            agg_params, _ = strategy.aggregate_fit(round_num, active_fit_results, [])
            if agg_params is not None:
                global_weights = parameters_to_ndarrays(agg_params)

        log["aggregated/silos_trained"] = sum(round_trained)

        # ── Embedding snapshot ────────────────────────────────────────────────
        _tracker.snapshot(round_num, global_weights,
                          silo_fed_weights=snap_fed_weights,
                          silo_local_weights=snap_local_weights)

        # ── Federated few-shot update ─────────────────────────────────────────
        if ollama_active and ollama_client is not None:
            round_events = [ev for silo in silos for ev in silo.last_round_events]
            ollama_client.update_examples(round_events)
            log["aggregated/llm_few_shot_examples"] = ollama_client.num_examples()

        # ── Prototype library ─────────────────────────────────────────────────
        if ollama_active and ollama_client is not None and ollama_client._proto_encoder is not None:
            enc_model, enc_tok = ollama_client._proto_encoder
            for i, silo in enumerate(silos):
                added = proto_libs[i].add_from_events(
                    silo.last_round_events, enc_model, enc_tok,
                    round_num, device="cpu",
                )
                log[f"silo_{i}/proto_added"] = added
                log[f"silo_{i}/proto_size"]  = proto_libs[i].size()
            global_protos = federate_libraries(proto_libs, max_per_label=10)
            for lib in proto_libs:
                lib.merge(global_protos)
            log["aggregated/proto_total"] = sum(l.size() for l in proto_libs)

            # ── Stereotype federation ─────────────────────────────────────────
            for i, silo in enumerate(silos):
                stereo_libs[i].update_from_events(
                    silo.last_round_events, enc_model, enc_tok, device="cpu",
                )
            global_stereos = aggregate_stereotypes(
                [l.get_state() for l in stereo_libs]
            )
            for lib in stereo_libs:
                lib.set_global(global_stereos)
            ollama_client.update_global_stereotypes(global_stereos,
                                                    n_silos=cfg.num_silos)
            _new_labels = [l for l in global_stereos if l not in _known_stereo_labels]
            _known_stereo_labels.update(global_stereos.keys())
            log["aggregated/stereo_classes"]     = len(global_stereos)
            log["aggregated/stereo_new_classes"] = len(_new_labels)
            if _new_labels:
                print(f"  [stereo] New class centroid(s) federated: {sorted(_new_labels)}")

        # ── Early exit ────────────────────────────────────────────────────────
        if all(silo.is_done for silo in silos):
            log["aggregated/all_silos_done"] = 1
            run.log(log, step=round_num)
            elapsed = time.time() - t0
            _print_round(round_num, cfg.max_rounds, log, elapsed)
            _report.add_round(round_num, log, [s.last_round_events for s in silos])
            reasons = [silo.stop_reason or "done" for silo in silos]
            print(f"\n  All silos finished: {reasons[0]}")
            break

        # ── Aggregated log ────────────────────────────────────────────────────
        if agg_n > 0:
            log["aggregated/loss"]         = agg_loss / agg_n
            log["aggregated/num_examples"] = int(agg_n)
            log["aggregated/num_trained"]  = int(agg_trained)
            for sub in ("triage_acc", "diag_acc", "icd_exact_acc",
                        "combined_acc", "triage_macro_f1", "danger_rate",
                        "avg_turns", "first_turn_rate", "action_accuracy",
                        "fp_escalation_rate", "local_triage_acc", "local_diag_acc",
                        "fl_gain", "fl_diag_gain"):
                k = f"aggregated/{sub}"
                if k in log:
                    log[k] /= agg_n

        run.log(log, step=round_num)
        _report.add_round(round_num, log, [s.last_round_events for s in silos])

        elapsed = time.time() - t0
        _print_round(round_num, cfg.max_rounds, log, elapsed)

    # ── Distribute final global models (doctor + nurse) ─────────────────────
    for silo in silos:
        silo.set_weights(global_weights)
        silo.set_nurse_weights(global_nurse_weights)

    run.finish()

    if _own_tracker:
        print("  [embed] Generating evolution plots…")
        embed_plots = _tracker.save_plots()
        for p in embed_plots:
            print(f"  [embed] {p.resolve()}")

    _report.write(_report_path, tracker=_tracker if _own_tracker else None)

    run_ref = run.url or f"offline — sync with: wandb sync {run.dir}"
    print(f"\nTraining complete. W&B: {run_ref}")
    print(f"Report:           {_report_path.resolve()}\n")
    return silos


# ── Local-only baseline ───────────────────────────────────────────────────────

def run_local_only_training(cfg: FLTrainConfig | None = None, **kwargs) -> list[WorldEngine]:
    """
    Local-only baseline: each silo trains its own LoRA adapter independently.
    No FedAvg, no weight sharing.  Reports the local shadow model's metrics as
    the primary output so results are directly comparable to the federated run.

    The local shadow model is already trained inside run_round() via train_local();
    this loop simply skips set_weights() / FedAvg and promotes local_* metrics.

    W&B keys mirror the federated run exactly so W&B overlay comparison works:
      aggregated/triage_acc  ← local model severity accuracy (not fed model)
      aggregated/diag_acc    ← local model disease accuracy
      aggregated/fl_gain     ← always 0.0 (no federation; useful as a flat baseline line)
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    import wandb

    if cfg.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    wandb_cfg = {k: v for k, v in asdict(cfg).items()
                 if not k.startswith("wandb_") and k != "world_configs"}
    run = wandb.init(
        project = cfg.wandb_project,
        name    = cfg.wandb_run_name or "local_only",
        entity  = cfg.wandb_entity,
        config  = wandb_cfg,
        reinit  = "finish_previous",
    )

    _run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    _dataset_dir = Path(cfg.dataset_dir) / _run_id
    from fl.dataset import SiloDataset
    _silo_datasets = [
        SiloDataset(_dataset_dir, i, cfg.holdout_fraction, cfg.seed)
        for i in range(cfg.num_silos)
    ]
    rng_dirichlet = np.random.default_rng(cfg.seed)

    silos: list[FLSilo] = [
        _build_fl_silo(
            i,
            cfg.world_configs[i] if cfg.world_configs else None,
            cfg,
            rng_dirichlet,
            _silo_datasets[i],
        )
        for i in range(cfg.num_silos)
    ]

    from fl.prototype_library import PrototypeLibrary
    proto_libs = [PrototypeLibrary(silo_id=i) for i in range(cfg.num_silos)]

    ollama_client  = None
    ollama_status  = "disabled"
    patient_status = "disabled"
    if cfg.use_ollama:
        from simulation.ollama_client import OllamaDiagnosticClient
        from simulation.patient_llm   import PatientLLMClient

        patient_llm = PatientLLMClient(model=cfg.patient_model)
        if patient_llm.health_check():
            for silo in silos:
                silo.set_patient_llm(patient_llm)
            patient_status = f"active ({patient_llm.model})"
        else:
            patient_llm    = None
            patient_status = f"unreachable — pull: ollama pull {cfg.patient_model}"

        ollama_client = OllamaDiagnosticClient(
            patient_llm       = patient_llm,
            disease_glossary  = cfg.disease_glossary,
            explicit_exclusion= getattr(cfg, "explicit_real_disease_exclusion", False),
        )
        if ollama_client.health_check():
            for i, silo in enumerate(silos):
                q    = silo.clinic_queue
                _lib = proto_libs[i]
                silo.register_diagnostic_fn(
                    lambda ev, _q=q, _lib=_lib: ollama_client.diagnose(
                        ev, _queue=_q, _proto_lib=_lib)
                )
            from transformers import AutoTokenizer
            from fl.lora import build_model as _build_model
            _enc_tokenizer = AutoTokenizer.from_pretrained(cfg.lora_config().model_name_or_path)
            _enc_model     = _build_model(cfg.lora_config()).cpu()
            ollama_client.set_prototype_library(
                library       = None,
                encoder_model = _enc_model,
                tokenizer     = _enc_tokenizer,
                k             = 3,
            )
            ollama_status = f"active ({ollama_client.model})"
        else:
            ollama_status = "unreachable — using stub oracle"

    prog_summary = ", ".join(cfg.progressions) if not cfg.world_configs else "per-silo (WorldConfig)"
    print(
        f"\n{'─'*60}\n"
        f"  FedWorld Local-Only Training (no federation)\n"
        f"  silos={cfg.num_silos}  max_rounds={cfg.max_rounds}  agents/silo={cfg.world_configs[0].num_agents if cfg.world_configs else cfg.num_agents}\n"
        f"  sim_days={cfg.sim_days}  min_events={cfg.min_events_to_train}  "
        f"local_epochs={cfg.local_epochs}\n"
        f"  progressions={prog_summary}  seed={cfg.seed}\n"
        f"  [dataset] {_dataset_dir}/  holdout={cfg.holdout_fraction:.0%}  cap={cfg.train_sample_cap}\n"
        f"  LLM doctor:  {ollama_status}\n"
        f"  LLM patient: {patient_status}\n"
        f"{'─'*60}\n"
    )

    _run_metrics: list[dict] = []

    for round_num in range(1, cfg.max_rounds + 1):
        t0 = time.time()
        log: dict = {"round": round_num}
        agg_triage = agg_diag = agg_n = agg_trained = agg_loss = 0.0

        for i, silo in enumerate(silos):
            # run_round() trains both the federated model (ignored here) and the
            # local shadow model.  We deliberately do NOT call set_weights() so
            # each silo's federated model also never receives global weights.
            m = silo.run_round(round_num=round_num)
            silo.release_model()

            n        = m.get("num_examples", 0)
            did_train = bool(m.get("trained", 0))

            # Promote local metrics to primary slot so the W&B key names align
            # with the federated run for easy overlay.
            local_tr  = m.get("local_triage_acc", float("nan"))
            local_dg  = m.get("local_diag_acc",   float("nan"))
            m["triage_acc"] = local_tr
            m["diag_acc"]   = local_dg
            m["fl_gain"]    = 0.0   # flat baseline reference line

            epoch_losses = m.pop("epoch_losses", [])
            for key, val in m.items():
                if isinstance(val, (int, float)):
                    log[f"silo_{i}/{key}"] = val
            if epoch_losses:
                log[f"silo_{i}/train_loss_final"] = epoch_losses[-1]

            if n > 0 and did_train and local_tr == local_tr:
                agg_loss   += m.get("loss", 0.0) * n
                agg_triage += local_tr * n
                agg_diag   += local_dg * n if local_dg == local_dg else 0.0
                agg_trained += m.get("trained_on", 0)
                agg_n       += n

        if all(s.is_done for s in silos):
            log["aggregated/all_silos_done"] = 1
            run.log(log, step=round_num)
            _run_metrics.append(_extract_round_metric(round_num, log, cfg.num_silos))
            _flush_round_metrics(_dataset_dir, _run_metrics)
            print(f"  Round {round_num:>3} | all silos done | {time.time()-t0:.1f}s")
            break

        if agg_n > 0:
            log["aggregated/loss"]        = agg_loss   / agg_n
            log["aggregated/triage_acc"]  = agg_triage / agg_n
            log["aggregated/diag_acc"]    = agg_diag   / agg_n
            log["aggregated/fl_gain"]     = 0.0
            log["aggregated/num_examples"] = int(agg_n)
            log["aggregated/num_trained"]  = int(agg_trained)

        run.log(log, step=round_num)
        _run_metrics.append(_extract_round_metric(round_num, log, cfg.num_silos))
        _flush_round_metrics(_dataset_dir, _run_metrics)
        elapsed = time.time() - t0
        nan = float("nan")
        print(
            f"  Round {round_num:>3}/~{cfg.max_rounds} | "
            f"loss={log.get('aggregated/loss', nan):.4f} "
            f"local_tr={log.get('aggregated/triage_acc', nan):.2f} "
            f"local_dg={log.get('aggregated/diag_acc', nan):.2f} "
            f"n={int(agg_trained):>4} | {elapsed:.1f}s"
        )

    _write_summary(_dataset_dir, cfg, _run_id, "local_only", _run_metrics)
    run.finish()
    print(f"\nLocal-only training complete. W&B: {run.url or 'offline'}\n")
    return silos


# ── Centralized baseline ──────────────────────────────────────────────────────

def run_centralized_training(
    cfg: FLTrainConfig | None = None,
    shared_tracker=None,
    **kwargs,
) -> "CentralizedTrainer":
    """
    Centralized oracle: N worlds, all events pooled, one shared model trained.

    Uses identical seeds and world configs as run_federated_training so the
    comparison is fair.  W&B keys are prefixed 'centralized/' to sit alongside
    federated metrics in the same dashboard.

    Parameters
    ----------
    shared_tracker : EmbeddingTracker | None
        When provided (run_comparison mode), snapshots the centralized model
        into the same tracker as the federated run so both appear in the same
        UMAP space and the comparison plot is generated automatically.
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    import wandb
    from fl.centralized import CentralizedTrainer

    if cfg.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    wandb_cfg = {k: v for k, v in __import__("dataclasses").asdict(cfg).items()
                 if not k.startswith("wandb_") and k != "world_configs"}
    run = wandb.init(
        project  = cfg.wandb_project,
        name     = (cfg.wandb_run_name or "centralized"),
        entity   = cfg.wandb_entity,
        config   = wandb_cfg,
        reinit   = "finish_previous",
    )

    rng_dirichlet = np.random.default_rng(cfg.seed)

    # Build worlds for centralized training (models managed by CentralizedTrainer)
    silos: list[WorldEngine] = [
        _build_bare_world(
            i,
            cfg.world_configs[i] if cfg.world_configs else None,
            cfg,
            rng_dirichlet,
        )
        for i in range(cfg.num_silos)
    ]

    from fl.prototype_library import PrototypeLibrary
    proto_libs = [PrototypeLibrary(silo_id=i) for i in range(cfg.num_silos)]

    ollama_status  = "disabled"
    patient_status = "disabled"
    if cfg.use_ollama:
        from simulation.ollama_client import OllamaDiagnosticClient
        from simulation.patient_llm   import PatientLLMClient

        patient_llm = PatientLLMClient(model=cfg.patient_model)
        if patient_llm.health_check():
            for silo in silos:
                silo.set_patient_llm(patient_llm)
            patient_status = f"active ({patient_llm.model})"
        else:
            patient_llm    = None
            patient_status = f"unreachable — pull: ollama pull {cfg.patient_model}"

        ollama_client = OllamaDiagnosticClient(
            patient_llm       = patient_llm,
            disease_glossary  = cfg.disease_glossary,
            explicit_exclusion= getattr(cfg, "explicit_real_disease_exclusion", False),
        )
        if ollama_client.health_check():
            for i, silo in enumerate(silos):
                q    = silo.clinic_queue
                _lib = proto_libs[i]
                silo.register_diagnostic_fn(
                    lambda ev, _q=q, _lib=_lib: ollama_client.diagnose(
                        ev, _queue=_q, _proto_lib=_lib)
                )
            from transformers import AutoTokenizer
            from fl.lora import build_model as _build_model
            _enc_tokenizer = AutoTokenizer.from_pretrained(cfg.lora_config().model_name_or_path)
            _enc_model     = _build_model(cfg.lora_config()).cpu()
            ollama_client.set_prototype_library(
                library       = None,
                encoder_model = _enc_model,
                tokenizer     = _enc_tokenizer,
                k             = 3,
            )
            ollama_status = f"active ({ollama_client.model})"
        else:
            ollama_status = "unreachable — using stub oracle"

    trainer = CentralizedTrainer(
        silos               = silos,
        lora_config         = cfg.lora_config(),
        local_epochs        = cfg.local_epochs,
        batch_size          = cfg.batch_size,
        lr                  = cfg.lr,
        min_events_to_train = cfg.min_events_to_train,
        sim_days            = cfg.sim_days,
        train_sample_cap    = cfg.train_sample_cap,
        device              = cfg.training_device,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    from simulation.report import RunReport
    _run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    _dataset_dir = Path(cfg.dataset_dir) / _run_id
    _run_metrics: list[dict] = []
    _report      = RunReport(cfg, _run_id)
    _report_path = Path(f"reports/run_{_run_id}_centralized.html")

    # ── Embedding tracker (standalone or shared with a federated run) ─────────
    from viz.embedding_tracker import EmbeddingTracker
    if shared_tracker is not None:
        _tracker      = shared_tracker
        _own_tracker  = False
    else:
        _embed_dir    = Path(f"viz_output/embeddings/run_{_run_id}_centralized")
        _tracker      = EmbeddingTracker(_embed_dir, cfg.lora_config())
        _own_tracker  = True
        print(f"  [embed] {len(_tracker.probe_events)} probe events ready\n")

    print(
        f"\n{'─'*60}\n"
        f"  FedWorld Centralized Training (oracle baseline)\n"
        f"  silos={cfg.num_silos}  max_rounds={cfg.max_rounds}  agents/silo={cfg.world_configs[0].num_agents if cfg.world_configs else cfg.num_agents}\n"
        f"  sim_days={cfg.sim_days}  local_epochs={cfg.local_epochs}  lr={cfg.lr}\n"
        f"  LLM doctor:  {ollama_status}\n"
        f"  LLM patient: {patient_status}\n"
        f"{'─'*60}\n"
    )

    # Dummy global weights for snapshot — use the centralized trainer's model,
    # not silos[0] (centralized worlds have no lora_config).
    _dummy_global = trainer.get_weights()

    for round_num in range(1, cfg.max_rounds + 1):
        m = trainer.run_round()
        m.round_num = round_num

        # Snapshot centralized model into tracker each round
        cen_weights = trainer.get_weights()
        if _own_tracker:
            # Standalone centralized run: snapshot with empty silo list,
            # centralized model stored as "global" for evolution plots
            _tracker.snapshot(round_num, cen_weights, [],
                              extra_weights={})
        else:
            # Shared tracker: add centralized weights alongside federated ones
            # (federated run already called snapshot this round — append only
            # the centralized key to the existing snap dict)
            existing = _tracker._snapshots.get(round_num, {})
            if existing:
                cls, logits = _tracker._extract(cen_weights)
                existing["centralized"] = {"cls": cls, "logits": logits}
                _tracker._save_round_npz(round_num, existing)
        # Model is NOT released here — it must persist so trained weights
        # carry over to the next round. release_model() is called once after
        # the loop ends (see below).

        log = {
            "round":                    round_num,
            "centralized/triage_acc":   m.triage_acc,
            "centralized/diag_acc":     m.diag_acc,
            "centralized/danger_rate":  m.danger_rate,
            "centralized/num_events":   m.num_events,
            "centralized/trained_on":   m.trained_on,
            "centralized/loss":         m.loss,
        }
        run.log(log, step=round_num)
        _run_metrics.append({
            "round":          round_num,
            "agg_diag_acc":   _nan_to_none(log.get("centralized/diag_acc")),
            "agg_triage_acc": _nan_to_none(log.get("centralized/triage_acc")),
            "mean_loss":      _nan_to_none(log.get("centralized/loss")),
        })
        _flush_round_metrics(_dataset_dir, _run_metrics)
        _report.add_round(round_num, log, [s.last_round_events for s in silos])

        def _f(v): return f"{v:.2f}" if v == v else "—"
        print(
            f"  Round {round_num:>3}/~{cfg.max_rounds} | "
            f"loss={m.loss:.4f} tr={_f(m.triage_acc)} diag={_f(m.diag_acc)} "
            f"danger={_f(m.danger_rate)} "
            f"ev={m.num_events} trained_on={m.trained_on} | {m.elapsed:.1f}s"
        )

        if m.all_done:
            print(f"\n  All silos finished (centralized).")
            break

    trainer.release_model()   # free once, after all rounds complete
    _write_summary(_dataset_dir, cfg, _run_id, "centralized", _run_metrics)
    run.finish()

    if _own_tracker:
        plots = _tracker.save_plots()
        for p in plots:
            print(f"  [embed] {p.resolve()}")

    _report.write(_report_path, tracker=_tracker if _own_tracker else None)
    print(f"Report: {_report_path.resolve()}")

    return trainer


def run_static_training(
    cfg: FLTrainConfig | None = None,
    mode: str = "centralized",
    **kwargs,
) -> dict:
    """
    Two-phase static baseline: simulate first, train after.

    Phase 1 — run all N worlds to completion via step_tick() (pure SIR +
               Ollama diagnosis, no LoRA, no weight communication).
    Phase 2 — train on the collected events:
        mode='centralized' — pool all silo events, train one shared model.
        mode='local'       — train one model per silo on its own events only.

    Each silo's events are randomly split 80/20 (train/holdout) before
    training begins, so evaluation is always on unseen data.

    Returns dict with keys: mode, silo_results (list[dict]), log (W&B dict).
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    import random
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    import wandb
    from transformers import AutoTokenizer
    from fl.lora import build_model
    from fl.learner import build_label_map, build_fictional_label_map, _split_label

    if getattr(cfg, "disease_glossary", None):
        label2id, id2label = build_fictional_label_map()
    else:
        label2id, id2label = build_label_map()
    prefix = "static_centralized" if mode == "centralized" else "static_local"

    if cfg.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    wandb_cfg = {k: v for k, v in asdict(cfg).items()
                 if not k.startswith("wandb_") and k != "world_configs"}
    wandb_cfg["static_mode"] = mode
    run = wandb.init(
        project = cfg.wandb_project,
        name    = cfg.wandb_run_name or prefix,
        entity  = cfg.wandb_entity,
        config  = wandb_cfg,
        reinit  = "finish_previous",
    )

    # ── World builder (no learners — sim only) ───────────────────────────────
    rng_dirichlet = np.random.default_rng(cfg.seed)
    silos: list[WorldEngine] = [
        _build_bare_world(
            i,
            cfg.world_configs[i] if cfg.world_configs else None,
            cfg,
            rng_dirichlet,
        )
        for i in range(cfg.num_silos)
    ]

    # ── Ollama setup ──────────────────────────────────────────────────────────
    ollama_status  = "disabled"
    patient_status = "disabled"
    if cfg.use_ollama:
        from simulation.ollama_client import OllamaDiagnosticClient
        from simulation.patient_llm   import PatientLLMClient
        patient_llm = PatientLLMClient(model=cfg.patient_model)
        if patient_llm.health_check():
            for silo in silos:
                silo.set_patient_llm(patient_llm)
            patient_status = f"active ({patient_llm.model})"
        else:
            patient_llm    = None
            patient_status = f"unreachable — pull: ollama pull {cfg.patient_model}"
        ollama_client = OllamaDiagnosticClient(
            patient_llm       = patient_llm,
            disease_glossary  = cfg.disease_glossary,
            explicit_exclusion= getattr(cfg, "explicit_real_disease_exclusion", False),
        )
        if ollama_client.health_check():
            for silo in silos:
                q = silo.clinic_queue
                silo.register_diagnostic_fn(
                    lambda ev, _q=q: ollama_client.diagnose(ev, _queue=_q))
            ollama_status = f"active ({ollama_client.model})"
        else:
            ollama_status = "unreachable — using stub oracle"

    prog_summary = ", ".join(cfg.progressions) if not cfg.world_configs else "per-silo (WorldConfig)"
    label = "Centralized Static" if mode == "centralized" else "Local Static"
    print(
        f"\n{'─'*60}\n"
        f"  FedWorld {label} Training\n"
        f"  silos={cfg.num_silos}  agents/silo={cfg.num_agents}  "
        f"end={cfg.end_condition}\n"
        f"  epochs={cfg.local_epochs}  lr={cfg.lr}  batch={cfg.batch_size}  "
        f"holdout=20%\n"
        f"  progressions={prog_summary}  seed={cfg.seed}\n"
        f"  LLM doctor: {ollama_status}  patient: {patient_status}\n"
        f"{'─'*60}\n"
    )

    # ── Phase 1: simulate all worlds to completion ────────────────────────────
    print("  Phase 1 — running simulations…")
    all_silo_events: list[list] = []
    for i, silo in enumerate(silos):
        t0 = time.time()
        while not silo.is_done:
            silo.step_tick()
        events = list(silo.clinic_queue.processed)
        all_silo_events.append(events)
        sir = silo.sir_model
        print(f"    silo {i:>2}: {len(events):>4} events  "
              f"S={sir.S:>3} I={sir.I:>3} R={sir.R:>3}  "
              f"stop={silo.stop_reason or 'done'}  {time.time()-t0:.1f}s")

    total_events = sum(len(e) for e in all_silo_events)
    print(f"  Total events: {total_events}\n")

    # ── Shared tokenizer and helpers ──────────────────────────────────────────
    lora_cfg = cfg.lora_config()
    tokenizer = AutoTokenizer.from_pretrained(lora_cfg.model_name_or_path)

    def _extract(events: list) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        for ev in events:
            lid = label2id.get(ev.ground_truth or "")
            if lid is None:
                continue
            turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
            if not turns:
                continue
            texts.append(turns[-1])
            labels.append(lid)
        return texts, labels

    def _train_eval(train_events: list, eval_events: list, tag: str) -> dict:
        train_texts, train_labels = _extract(train_events)
        eval_texts,  eval_labels  = _extract(eval_events)

        model       = build_model(lora_cfg).to(cfg.training_device)
        epoch_losses: list[float] = []

        if len(train_texts) >= cfg.min_events_to_train:
            enc     = tokenizer(train_texts, padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
            label_t = torch.tensor(train_labels, dtype=torch.long)
            loader  = DataLoader(
                TensorDataset(enc["input_ids"], enc["attention_mask"], label_t),
                batch_size=cfg.batch_size, shuffle=True,
            )
            model.train()
            opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
            for _ in range(cfg.local_epochs):
                batch_losses = []
                for iids, amask, blabels in loader:
                    iids    = iids.to(cfg.training_device)
                    amask   = amask.to(cfg.training_device)
                    blabels = blabels.to(cfg.training_device)
                    opt.zero_grad()
                    out = model(input_ids=iids, attention_mask=amask, labels=blabels)
                    out.loss.backward()
                    opt.step()
                    batch_losses.append(out.loss.detach().item())
                epoch_losses.append(sum(batch_losses) / len(batch_losses))

        m = {"triage_acc": float("nan"), "diag_acc": float("nan"),
             "danger_rate": float("nan"), "n_eval": 0,
             "n_train": len(train_texts), "epoch_losses": epoch_losses}

        if eval_texts:
            enc     = tokenizer(eval_texts, padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
            label_t = torch.tensor(eval_labels, dtype=torch.long).to(cfg.training_device)
            model.eval()
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"].to(cfg.training_device),
                            attention_mask=enc["attention_mask"].to(cfg.training_device),
                            labels=label_t)
            preds = out.logits.argmax(dim=-1).cpu().tolist()
            n = len(preds)
            sev_ok = diag_ok = danger = n_severe = 0
            for pred_idx, true_idx in zip(preds, eval_labels):
                pred_d, pred_s = _split_label(id2label[pred_idx])
                true_d, true_s = _split_label(id2label[true_idx])
                sev_ok  += pred_s == true_s
                diag_ok += pred_d == true_d
                if true_s == "severe":
                    n_severe += 1
                    if pred_s != "severe" or id2label[pred_idx] == "non-infectious":
                        danger += 1
            m.update({
                "triage_acc":  sev_ok  / n,
                "diag_acc":    diag_ok / n,
                "danger_rate": danger  / n_severe if n_severe > 0 else float("nan"),
                "n_eval":      n,
            })

        _f = lambda v: f"{v:.3f}" if v == v else "—"
        loss_curve = " → ".join(f"{l:.4f}" for l in epoch_losses) or "—"
        print(f"    {tag:>20} | train={m['n_train']:>4}  eval={m['n_eval']:>3}  "
              f"loss=[{loss_curve}]  "
              f"triage={_f(m['triage_acc'])}  diag={_f(m['diag_acc'])}  "
              f"danger={_f(m['danger_rate'])}")
        return m

    # ── Phase 2: train ────────────────────────────────────────────────────────
    print("  Phase 2 — training…\n")
    rng_split = random.Random(cfg.seed)

    def _split(events: list) -> tuple[list, list]:
        shuffled = list(events)
        rng_split.shuffle(shuffled)
        n_holdout = max(1, int(len(shuffled) * 0.2))
        return shuffled[n_holdout:], shuffled[:n_holdout]

    log: dict  = {}
    silo_results: list[dict] = []

    t_train = time.time()
    if mode == "centralized":
        train_pool, eval_pool = [], []
        for events in all_silo_events:
            tr, ev = _split(events)
            train_pool.extend(tr)
            eval_pool.extend(ev)
        m = _train_eval(train_pool, eval_pool, "pooled (all silos)")
        silo_results.append(m)
        for k in ("triage_acc", "diag_acc", "danger_rate", "n_train", "n_eval"):
            log[f"{prefix}/{k}"] = m[k]
        for ep_i, loss in enumerate(m["epoch_losses"]):
            log[f"{prefix}/epoch_{ep_i}_loss"] = loss
        if m["epoch_losses"]:
            log[f"{prefix}/final_loss"] = m["epoch_losses"][-1]
    else:
        for i, events in enumerate(all_silo_events):
            train_ev, eval_ev = _split(events)
            m = _train_eval(train_ev, eval_ev, f"silo {i}")
            silo_results.append(m)
            for k in ("triage_acc", "diag_acc", "danger_rate", "n_train", "n_eval"):
                log[f"silo_{i}/{k}"] = m[k]
            for ep_i, loss in enumerate(m["epoch_losses"]):
                log[f"silo_{i}/epoch_{ep_i}_loss"] = loss
            if m["epoch_losses"]:
                log[f"silo_{i}/final_loss"] = m["epoch_losses"][-1]

        valid    = [m for m in silo_results if m["n_eval"] > 0 and m["triage_acc"] == m["triage_acc"]]
        dr_valid = [m for m in valid if m["danger_rate"] == m["danger_rate"]]
        if valid:
            log[f"{prefix}/macro_triage_acc"] = sum(m["triage_acc"] for m in valid) / len(valid)
            log[f"{prefix}/macro_diag_acc"]   = sum(m["diag_acc"]   for m in valid) / len(valid)
        if dr_valid:
            log[f"{prefix}/macro_danger_rate"] = sum(m["danger_rate"] for m in dr_valid) / len(dr_valid)

    log[f"{prefix}/total_events"] = total_events
    run.log(log)
    run.finish()

    # ── Summary ───────────────────────────────────────────────────────────────
    _f = lambda v: f"{v:.3f}" if v == v else "—"
    elapsed = time.time() - t_train
    print(f"\n{'─'*60}")
    print(f"  {label} — done in {elapsed:.1f}s")
    if mode == "centralized":
        m = silo_results[0]
        print(f"  triage={_f(m['triage_acc'])}  diag={_f(m['diag_acc'])}  "
              f"danger={_f(m['danger_rate'])}  "
              f"(train={m['n_train']}  eval={m['n_eval']})")
    else:
        ma_tr = log.get(f"{prefix}/macro_triage_acc", float("nan"))
        ma_dg = log.get(f"{prefix}/macro_diag_acc",   float("nan"))
        ma_dr = log.get(f"{prefix}/macro_danger_rate", float("nan"))
        print(f"  macro triage={_f(ma_tr)}  diag={_f(ma_dg)}  danger={_f(ma_dr)}")
    print(f"{'─'*60}\n")

    return {"mode": mode, "silo_results": silo_results, "log": log}


def run_training_from_dataset(
    dataset_dir: "str | Path",
    mode: str = "centralized",
    cfg: "FLTrainConfig | None" = None,
    n_epochs: "int | None" = None,
    **kwargs,
) -> dict:
    """
    Train (centralized or local) from an existing on-disk JSONL dataset.
    No simulation or Ollama needed — a fast static training pass on saved records.

    mode='centralized' — pool all silos, train one shared model.
    mode='local'       — train one model per silo independently.
    n_epochs           — override training epochs (default: cfg.local_epochs * 5).
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    import random, torch, wandb
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer
    from fl.lora import build_model
    from fl.learner import build_label_map, _split_label
    from fl.dataset import SiloDataset

    dataset_dir = Path(dataset_dir)
    label2id, id2label = build_label_map()
    prefix = "from_dataset_centralized" if mode == "centralized" else "from_dataset_local"
    epochs = n_epochs if n_epochs is not None else max(cfg.local_epochs * 5, 10)

    n_silos = sum(1 for d in dataset_dir.iterdir()
                  if d.is_dir() and d.name.startswith("silo_"))

    if cfg.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    wandb_cfg = {k: v for k, v in asdict(cfg).items()
                 if not k.startswith("wandb_") and k != "world_configs"}
    wandb_cfg.update({"static_mode": mode, "source_dataset": str(dataset_dir),
                      "n_epochs": epochs})
    run = wandb.init(
        project = cfg.wandb_project,
        name    = cfg.wandb_run_name or prefix,
        entity  = cfg.wandb_entity,
        config  = wandb_cfg,
        reinit  = "finish_previous",
    )

    # ── Load per-silo records ─────────────────────────────────────────────────
    silo_train:   list[list[dict]] = []
    silo_holdout: list[list[dict]] = []
    for i in range(n_silos):
        ds = SiloDataset(dataset_dir, i, cfg.holdout_fraction, cfg.seed)
        silo_train.append(ds.sample_train())
        silo_holdout.append(ds.load_holdout())

    total = sum(len(t) + len(h) for t, h in zip(silo_train, silo_holdout))
    label_str = "Centralized from Dataset" if mode == "centralized" else "Local from Dataset"
    print(
        f"\n{'─'*60}\n"
        f"  FedWorld {label_str}\n"
        f"  source={dataset_dir}  silos={n_silos}  total={total}\n"
        f"  epochs={epochs}  lr={cfg.lr}  batch={cfg.batch_size}\n"
        f"{'─'*60}\n"
    )
    for i, (t, h) in enumerate(zip(silo_train, silo_holdout)):
        print(f"    silo {i}: train={len(t)}  holdout={len(h)}")
    print()

    # ── Training helpers ──────────────────────────────────────────────────────
    lora_cfg  = cfg.lora_config()
    tokenizer = AutoTokenizer.from_pretrained(lora_cfg.model_name_or_path)

    def _extract(records: list[dict]) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        for r in records:
            lid = label2id.get(r.get("label", ""))
            if lid is None:
                continue
            texts.append(r["text"])
            labels.append(lid)
        return texts, labels

    def _train_eval(train_recs: list[dict], eval_recs: list[dict], tag: str) -> dict:
        train_texts, train_labels = _extract(train_recs)
        eval_texts,  eval_labels  = _extract(eval_recs)

        model = build_model(lora_cfg).to(cfg.training_device)
        epoch_losses: list[float] = []

        if len(train_texts) >= cfg.min_events_to_train:
            enc = tokenizer(train_texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            label_t = torch.tensor(train_labels, dtype=torch.long)
            loader  = DataLoader(
                TensorDataset(enc["input_ids"], enc["attention_mask"], label_t),
                batch_size=cfg.batch_size, shuffle=True,
            )
            model.train()
            opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
            for _ in range(epochs):
                batch_losses = []
                for iids, amask, blabels in loader:
                    iids    = iids.to(cfg.training_device)
                    amask   = amask.to(cfg.training_device)
                    blabels = blabels.to(cfg.training_device)
                    opt.zero_grad()
                    out = model(input_ids=iids, attention_mask=amask, labels=blabels)
                    out.loss.backward()
                    opt.step()
                    batch_losses.append(out.loss.detach().item())
                epoch_losses.append(sum(batch_losses) / len(batch_losses))

        m = {"triage_acc": float("nan"), "diag_acc": float("nan"),
             "danger_rate": float("nan"), "n_eval": 0,
             "n_train": len(train_texts), "epoch_losses": epoch_losses}

        if eval_texts:
            enc = tokenizer(eval_texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            label_t = torch.tensor(eval_labels, dtype=torch.long).to(cfg.training_device)
            model.eval()
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"].to(cfg.training_device),
                            attention_mask=enc["attention_mask"].to(cfg.training_device),
                            labels=label_t)
            preds = out.logits.argmax(dim=-1).cpu().tolist()
            n = len(preds)
            sev_ok = diag_ok = danger = n_severe = 0
            for pred_idx, true_idx in zip(preds, eval_labels):
                pred_d, pred_s = _split_label(id2label[pred_idx])
                true_d, true_s = _split_label(id2label[true_idx])
                sev_ok  += pred_s == true_s
                diag_ok += pred_d == true_d
                if true_s == "severe":
                    n_severe += 1
                    if pred_s != "severe" or id2label[pred_idx] == "non-infectious":
                        danger += 1
            m.update({
                "triage_acc":  sev_ok  / n,
                "diag_acc":    diag_ok / n,
                "danger_rate": danger  / n_severe if n_severe > 0 else float("nan"),
                "n_eval":      n,
            })

        from fl.lora import get_lora_weights
        m["lora_weights"] = get_lora_weights(model)

        _f = lambda v: f"{v:.3f}" if v == v else "—"
        lc  = " → ".join(f"{l:.4f}" for l in epoch_losses) or "—"
        print(f"    {tag:>20} | train={m['n_train']:>4}  eval={m['n_eval']:>3}  "
              f"loss=[{lc}]  "
              f"triage={_f(m['triage_acc'])}  diag={_f(m['diag_acc'])}  "
              f"danger={_f(m['danger_rate'])}")
        return m

    # ── Train ─────────────────────────────────────────────────────────────────
    log: dict         = {}
    silo_results: list[dict] = []
    t_train = time.time()

    if mode == "centralized":
        train_all = [r for t in silo_train   for r in t]
        eval_all  = [r for h in silo_holdout for r in h]
        random.Random(cfg.seed).shuffle(train_all)
        m = _train_eval(train_all, eval_all, "pooled (all silos)")
        silo_results.append(m)
        for k in ("triage_acc", "diag_acc", "danger_rate", "n_train", "n_eval"):
            log[f"{prefix}/{k}"] = m[k]
        for ep_i, loss in enumerate(m["epoch_losses"]):
            log[f"{prefix}/epoch_{ep_i}_loss"] = loss
        if m["epoch_losses"]:
            log[f"{prefix}/final_loss"] = m["epoch_losses"][-1]
    else:
        for i, (tr, ev) in enumerate(zip(silo_train, silo_holdout)):
            m = _train_eval(tr, ev, f"silo {i}")
            silo_results.append(m)
            for k in ("triage_acc", "diag_acc", "danger_rate", "n_train", "n_eval"):
                log[f"silo_{i}/{k}"] = m[k]
            for ep_i, loss in enumerate(m["epoch_losses"]):
                log[f"silo_{i}/epoch_{ep_i}_loss"] = loss
            if m["epoch_losses"]:
                log[f"silo_{i}/final_loss"] = m["epoch_losses"][-1]
        valid    = [m for m in silo_results if m["n_eval"] > 0 and m["triage_acc"] == m["triage_acc"]]
        dr_valid = [m for m in valid if m["danger_rate"] == m["danger_rate"]]
        if valid:
            log[f"{prefix}/macro_triage_acc"] = sum(m["triage_acc"] for m in valid) / len(valid)
            log[f"{prefix}/macro_diag_acc"]   = sum(m["diag_acc"]   for m in valid) / len(valid)
        if dr_valid:
            log[f"{prefix}/macro_danger_rate"] = sum(m["danger_rate"] for m in dr_valid) / len(dr_valid)

    log[f"{prefix}/total_records"] = total
    run.log(log)
    run.finish()

    # ── Save final LoRA weights for post-hoc eval ─────────────────────────────
    from fl.lora import weights_to_npz
    weights_dir = dataset_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    if mode == "centralized" and silo_results and "lora_weights" in silo_results[0]:
        npz_path = weights_dir / "centralized.npz"
        np.savez(str(npz_path), **weights_to_npz(silo_results[0]["lora_weights"]))
        print(f"  [weights] {npz_path}")
    elif mode == "local":
        for i, m in enumerate(silo_results):
            if "lora_weights" in m:
                npz_path = weights_dir / f"silo_{i}.npz"
                np.savez(str(npz_path), **weights_to_npz(m["lora_weights"]))
        print(f"  [weights] {weights_dir}/silo_{{0..{len(silo_results)-1}}}.npz")

    # ── Summary ───────────────────────────────────────────────────────────────
    _f = lambda v: f"{v:.3f}" if v == v else "—"
    elapsed = time.time() - t_train
    print(f"\n{'─'*60}")
    print(f"  {label_str} — done in {elapsed:.1f}s")
    if mode == "centralized" and silo_results:
        m = silo_results[0]
        print(f"  triage={_f(m['triage_acc'])}  diag={_f(m['diag_acc'])}  "
              f"danger={_f(m['danger_rate'])}  (train={m['n_train']}  eval={m['n_eval']})")
    else:
        ma_tr = log.get(f"{prefix}/macro_triage_acc", float("nan"))
        ma_dg = log.get(f"{prefix}/macro_diag_acc",   float("nan"))
        ma_dr = log.get(f"{prefix}/macro_danger_rate", float("nan"))
        print(f"  macro triage={_f(ma_tr)}  diag={_f(ma_dg)}  danger={_f(ma_dr)}")
    print(f"{'─'*60}\n")

    return {"mode": mode, "silo_results": silo_results, "log": log}


def run_comparison(cfg: FLTrainConfig | None = None, **kwargs) -> dict:
    """
    Run federated and centralized training back-to-back with the same config
    and a shared EmbeddingTracker so both models appear in the same UMAP space.

    Produces all standard federated plots plus comparison_fed_vs_centralized.png.
    Returns dict with keys 'federated' (list[WorldEngine]) and
    'centralized' (CentralizedTrainer) for downstream analysis.
    """
    import dataclasses
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    # One shared tracker so federated and centralized land in the same UMAP
    from viz.embedding_tracker import EmbeddingTracker
    from pathlib import Path
    _run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    _embed_dir = Path(f"viz_output/embeddings/run_{_run_id}_comparison")
    _tracker   = EmbeddingTracker(_embed_dir, cfg.lora_config())
    print(f"  [embed] Shared tracker — {len(_tracker.probe_events)} probe events\n")

    print("═" * 60)
    print("  COMPARISON RUN — Federated")
    print("═" * 60)
    fed_cfg = FLTrainConfig(**{**dataclasses.asdict(cfg),
                               "wandb_run_name": (cfg.wandb_run_name or "comparison") + "_federated",
                               "world_configs": cfg.world_configs})
    fed_silos = run_federated_training(fed_cfg, _shared_tracker=_tracker)

    print("═" * 60)
    print("  COMPARISON RUN — Centralized")
    print("═" * 60)
    cen_cfg = FLTrainConfig(**{**dataclasses.asdict(cfg),
                               "wandb_run_name": (cfg.wandb_run_name or "comparison") + "_centralized",
                               "world_configs": cfg.world_configs})
    cen_trainer = run_centralized_training(cen_cfg, shared_tracker=_tracker)

    # Generate all plots now that both models are in the tracker
    print("  [embed] Generating comparison plots…")
    plots = _tracker.save_plots()
    for p in plots:
        print(f"  [embed] {p.resolve()}")

    return {"federated": fed_silos, "centralized": cen_trainer}


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> FLTrainConfig:
    p = argparse.ArgumentParser(description="FedWorld FL training loop")
    p.add_argument("--silos",       type=int,   default=3)
    p.add_argument("--max-rounds",  type=int,   default=100,
                   dest="max_rounds",
                   help="Safety cap on FL rounds; run ends earlier when all silos done")
    p.add_argument("--agents",      type=int,   default=30)
    p.add_argument("--sim-days",    type=int,   default=7,
                   help="Simulated days per FL round")
    p.add_argument("--min-events",  type=int,   default=10,
                   dest="min_events_to_train",
                   help="Skip LoRA training if fewer events collected this round")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lora-rank",   type=int,   default=8)
    p.add_argument("--progressions", nargs="+", default=["Influenza"],
                   help="One or more disease progressions circulating in all silos")
    p.add_argument("--disease-strategy", type=str, default="Influenza",
                   dest="disease_strategy")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--beta-scale",          type=float, default=1.0,
                   dest="beta_scale",
                   help="Multiply BASE_BETA; 1.0 ≈ R0 1.5, 1.2 ≈ R0 1.8")
    p.add_argument("--initial-seeds",       type=int,   default=3,
                   dest="initial_seeds",
                   help="Number of initially infected agents per silo")
    p.add_argument("--end-condition",       type=str,   default="extinction",
                   choices=["horizon", "extinction", "no_susceptibles", "budget"])
    p.add_argument("--end-condition-param", type=int,   default=None,
                   help="max_days / consecutive_days / target_events")
    p.add_argument("--project",     type=str,   default="fedworld")
    p.add_argument("--run-name",    type=str,   default=None)
    p.add_argument("--entity",      type=str,   default=None)
    p.add_argument("--offline",     action="store_true")
    p.add_argument("--no-ollama",   action="store_true",
                   help="Skip Ollama doctor; use rule-based stub (faster, no LLM calls)")
    a = p.parse_args()
    return FLTrainConfig(
        num_silos           = a.silos,
        max_rounds          = a.max_rounds,
        num_agents          = a.agents,
        sim_days            = a.sim_days,
        local_epochs        = a.epochs,
        batch_size          = a.batch_size,
        lr                  = a.lr,
        lora_rank           = a.lora_rank,
        progressions        = a.progressions,
        disease_strategy    = a.disease_strategy,
        seed                = a.seed,
        min_events_to_train = a.min_events_to_train,
        end_condition       = a.end_condition,
        end_condition_param = a.end_condition_param,
        beta_scale          = a.beta_scale,
        initial_seeds       = a.initial_seeds,
        wandb_project       = a.project,
        wandb_run_name      = a.run_name,
        wandb_entity        = a.entity,
        wandb_offline       = a.offline,
        use_ollama          = not a.no_ollama,
    )


if __name__ == "__main__":
    run_federated_training(_parse_args())
