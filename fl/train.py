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
from fl.base_client import WorldFLClient
from simulation.world import WorldEngine
from simulation.progression import PROGRESSION_STRATEGIES, DiseaseProgressionStrategy
from simulation.end_conditions import EndCondition, from_config as end_condition_from_config


# ── Per-silo world configuration ─────────────────────────────────────────────

@dataclass
class WorldConfig:
    """
    Per-silo world configuration for non-IID federated learning experiments.

    When a list of WorldConfig objects is passed to FLTrainConfig.world_configs,
    each silo gets its own disease mix, population size, spread rate, and end
    condition — enabling asymmetric (non-IID) federation.

    Example — urban vs. rural silos:
        WorldConfig(num_agents=150, progressions=["Aggressive Flu", "Mild Corona"],
                    disease_strategy="Aggressive Flu", beta_scale=1.3)
        WorldConfig(num_agents=60,  progressions=["Standard Flu"],
                    disease_strategy="Standard Flu",   beta_scale=0.7)
    """
    num_agents:            int         = 30
    progressions:          list[str]   = field(default_factory=lambda: ["Standard Flu"])
    disease_strategy:      str         = "Standard Flu"
    beta_scale:            float       = 1.0
    background_visit_rate: float       = 0.025
    end_condition:         str         = "extinction"
    end_condition_param:   Optional[int] = None
    seed_offset:           int         = 0
    reveal_incubating_icd: bool        = True
    initial_seeds:         int         = 3


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
    progressions:        list[str]   = field(default_factory=lambda: ["Standard Flu"])
    disease_strategy:    str         = "Standard Flu"
    seed:                int         = 42
    # Per-silo non-IID config (overrides global params when provided)
    world_configs:       Optional[list[WorldConfig]] = None
    # Global IID params (used when world_configs is None)
    beta_scale:          float       = 1.0
    initial_seeds:       int         = 3
    # End condition (applied per silo; default: run until epidemic extinct)
    end_condition:       str         = "extinction"
    end_condition_param: Optional[int] = None
    # Non-IID disease distribution — Dirichlet(α) per silo over the disease list
    # α → 0: each silo dominated by one disease; α → ∞: IID mix
    dirichlet_alpha:     float       = 0.3
    # Replay buffer: events accumulated across rounds for local LoRA training
    replay_buffer_size:  int         = 2048
    # W&B
    wandb_project:       str         = "fedworld"
    wandb_run_name:      Optional[str] = None
    wandb_entity:        Optional[str] = None
    wandb_offline:       bool        = False
    use_ollama:          bool        = True
    patient_model:       str         = "tinyllama"

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


# ── Main training loop ────────────────────────────────────────────────────────

def run_federated_training(cfg: FLTrainConfig | None = None, **kwargs) -> list[WorldFLClient]:
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
    List of trained WorldFLClient instances (weights updated to final global model).
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    import wandb
    from simulation.strategies import STRATEGIES

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

    # ── World builder — respects per-silo WorldConfig when provided ───────────
    rng_dirichlet = np.random.default_rng(cfg.seed)

    def _make_world(silo_idx: int) -> WorldEngine:
        wc = cfg.world_configs[silo_idx] if cfg.world_configs else None

        if wc is not None:
            progs          = [PROGRESSION_STRATEGIES[p] for p in wc.progressions]
            strategy       = STRATEGIES.get(wc.disease_strategy)
            end_cond       = end_condition_from_config(wc.end_condition, wc.end_condition_param)
            n_agents       = wc.num_agents
            bg_rate        = wc.background_visit_rate
            bscale         = wc.beta_scale
            s_offset       = wc.seed_offset
            rev_incub_icd  = wc.reveal_incubating_icd
            n_seeds        = wc.initial_seeds
        else:
            unknown = [p for p in cfg.progressions if p not in PROGRESSION_STRATEGIES]
            if unknown:
                raise ValueError(
                    f"Unknown progression(s): {unknown}. "
                    f"Available: {list(PROGRESSION_STRATEGIES)}"
                )
            progs         = [PROGRESSION_STRATEGIES[p] for p in cfg.progressions]
            strategy      = STRATEGIES.get(cfg.disease_strategy)
            end_cond      = end_condition_from_config(cfg.end_condition, cfg.end_condition_param)
            n_agents      = cfg.num_agents
            bg_rate       = 0.025
            bscale        = cfg.beta_scale
            s_offset      = 0
            rev_incub_icd = True
            n_seeds       = cfg.initial_seeds

        # Dirichlet per-silo disease weights (reproducible: seed + silo index)
        if len(progs) > 1 and cfg.dirichlet_alpha > 0:
            d_weights = rng_dirichlet.dirichlet(
                [cfg.dirichlet_alpha] * len(progs)
            ).tolist()
        else:
            d_weights = None  # single disease or α=0 → uniform

        return WorldEngine(
            num_agents             = n_agents,
            seed                   = cfg.seed + silo_idx + s_offset,
            progression_strategies = progs,
            disease_strategy       = strategy,
            end_condition          = end_cond,
            background_visit_rate  = bg_rate,
            beta_scale             = bscale,
            reveal_incubating_icd  = rev_incub_icd,
            initial_seeds          = n_seeds,
            disease_weights        = d_weights,
        )

    # ── Create silos ──────────────────────────────────────────────────────────
    lora_cfg = cfg.lora_config()
    silos: list[WorldFLClient] = []
    for i in range(cfg.num_silos):
        world = _make_world(i)
        silos.append(
            WorldFLClient(
                world                = world,
                lora_config          = lora_cfg,
                sim_days             = cfg.sim_days,
                min_events_to_train  = cfg.min_events_to_train,
                local_epochs         = cfg.local_epochs,
                batch_size           = cfg.batch_size,
                lr                   = cfg.lr,
                replay_buffer_size   = cfg.replay_buffer_size,
            )
        )

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
                silo.world.set_patient_llm(patient_llm)
            patient_status = f"active ({patient_llm.model})"
        else:
            patient_llm    = None
            patient_status = f"unreachable — pull: ollama pull {cfg.patient_model}"

        # Doctor LLM — multi-turn diagnostic model, receives formatted vitals
        ollama_client = OllamaDiagnosticClient(patient_llm=patient_llm)
        if ollama_client.health_check():
            for silo in silos:
                q = silo.world.clinic_queue
                silo.world.register_diagnostic_fn(
                    lambda ev, _q=q: ollama_client.diagnose(ev, _queue=_q)
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
        f"agents/silo={cfg.num_agents}\n"
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
    _run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    _report    = RunReport(cfg, _run_id)
    _report_path = Path(f"reports/run_{_run_id}.html")

    # ── Embedding tracker ─────────────────────────────────────────────────────
    from viz.embedding_tracker import EmbeddingTracker
    print("  [embed] Building benchmark probe set…")
    _embed_dir = Path(f"viz_output/embeddings/run_{_run_id}")
    _tracker = EmbeddingTracker(_embed_dir, cfg.lora_config())
    print(f"  [embed] {len(_tracker.probe_events)} probe events ready "
          f"({len(set(e.ground_truth for e in _tracker.probe_events))} unique labels)\n")

    # ── Initialise global weights from silo 0 ─────────────────────────────────
    global_weights = silos[0].get_weights()
    silos[0].release_model()   # free immediately — rebuilt in round 1 with set_weights

    # ── FL rounds — simulation-guided (run until all silos done or max_rounds) ─
    for round_num in range(1, cfg.max_rounds + 1):
        t0 = time.time()

        round_weights:   list[list[np.ndarray]] = []
        round_nexamples: list[int]             = []
        round_trained:   list[bool]            = []
        log: dict = {"round": round_num}

        agg_loss = agg_acc = agg_n = agg_trained = 0.0

        for i, silo in enumerate(silos):
            if silo.world.is_done:
                # Done silo: conditionally accept global model, then re-upload weights
                silo.try_accept_global(global_weights)
                m = silo.run_round()  # frozen path — no simulation
                fedavg_n = silo._frozen_n  # fixed weight for FedAvg
            else:
                silo.set_weights(global_weights)
                m = silo.run_round()
                fedavg_n = m.get("num_examples", 0)

            round_weights.append(silo.get_weights())
            silo.release_model()   # free VRAM immediately — only one silo active at a time
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

            if silo.world.is_done:
                log[f"silo_{i}/done"] = 1
                log[f"silo_{i}/stop_reason"] = silo.world.stop_reason or "done"

            if n > 0 and did_train:
                agg_loss    += m.get("loss",       0.0) * n
                agg_n       += n
                agg_trained += m.get("trained_on", 0)
                for sub in ("triage_acc", "diag_acc", "icd_exact_acc",
                            "combined_acc", "triage_macro_f1", "danger_rate",
                            "avg_turns", "first_turn_rate", "action_accuracy",
                            "fp_escalation_rate", "local_triage_acc", "fl_gain"):
                    v = m.get(sub, float("nan"))
                    if v == v:
                        agg_key = f"aggregated/{sub}"
                        log[agg_key] = log.get(agg_key, 0.0) + v * n

        # ── FedAvg — include all silos (done silos use _frozen_n weight) ──────
        if any(n > 0 for n in round_nexamples):
            global_weights = _fedavg(round_weights, round_nexamples)
        log["aggregated/silos_trained"] = sum(round_trained)

        # ── Embedding snapshot (global + per-silo fed + per-silo local) ───────
        _tracker.snapshot(round_num, global_weights, silos)

        # ── Federated few-shot update — refresh Ollama doctor's example bank ──
        if ollama_active and ollama_client is not None:
            round_events = [ev for silo in silos for ev in silo.last_round_events]
            ollama_client.update_examples(round_events)
            log["aggregated/llm_few_shot_examples"] = ollama_client.num_examples()

        # ── Early exit if all silos finished ──────────────────────────────────
        if all(silo.world.is_done for silo in silos):
            log["aggregated/all_silos_done"] = 1
            run.log(log, step=round_num)
            elapsed = time.time() - t0
            _print_round(round_num, cfg.max_rounds, log, elapsed)
            _report.add_round(round_num, log, [s.last_round_events for s in silos])
            reasons = [silo.world.stop_reason or "done" for silo in silos]
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
                        "fp_escalation_rate", "local_triage_acc", "fl_gain"):
                k = f"aggregated/{sub}"
                if k in log:
                    log[k] /= agg_n

        run.log(log, step=round_num)
        _report.add_round(round_num, log, [s.last_round_events for s in silos])

        elapsed = time.time() - t0
        _print_round(round_num, cfg.max_rounds, log, elapsed)

    # ── Distribute final global model to all silos ────────────────────────────
    for silo in silos:
        silo.set_weights(global_weights)

    run.finish()
    _report.write(_report_path)

    # ── Save embedding evolution plots ────────────────────────────────────────
    print("  [embed] Generating evolution plots…")
    embed_plots = _tracker.save_plots()
    for p in embed_plots:
        print(f"  [embed] {p.resolve()}")

    run_ref = run.url or f"offline — sync with: wandb sync {run.dir}"
    print(f"\nTraining complete. W&B: {run_ref}")
    print(f"Report:           {_report_path.resolve()}\n")
    return silos


# ── Console progress helper ───────────────────────────────────────────────────

def _print_round(round_num: int, max_rounds: int, log: dict, elapsed: float) -> None:
    nan = float("nan")
    agg_loss    = log.get("aggregated/loss",              nan)
    triage      = log.get("aggregated/triage_acc",        nan)
    local_tr    = log.get("aggregated/local_triage_acc",  nan)
    fl_gain     = log.get("aggregated/fl_gain",           nan)
    diag        = log.get("aggregated/diag_acc",          nan)
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
        f"fed={_fmt(triage)} local={_fmt(local_tr)} +Δ={_fmt(fl_gain,3)} "
        f"diag={_fmt(diag)} f1={_fmt(macro_f1)} "
        f"danger={_fmt(danger)} fp_esc={_fmt(fp_esc)} "
        f"dr_turns={_fmt(avg_turns,1)} dr_act={_fmt(act_acc)} "
        f"n={n_trained:>4} silos={silos_tr} | "
        f"{' '.join(silo_parts) or '—'} | "
        f"{elapsed:.1f}s"
    )


# ── Centralized baseline ──────────────────────────────────────────────────────

def run_centralized_training(
    cfg: FLTrainConfig | None = None,
    **kwargs,
) -> "CentralizedTrainer":
    """
    Centralized oracle: N worlds, all events pooled, one shared model trained.

    Uses identical seeds and world configs as run_federated_training so the
    comparison is fair.  W&B keys are prefixed 'centralized/' to sit alongside
    federated metrics in the same dashboard.
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})

    import wandb
    from simulation.strategies import STRATEGIES
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

    def _make_world(silo_idx: int) -> "WorldEngine":
        wc = cfg.world_configs[silo_idx] if cfg.world_configs else None
        if wc is not None:
            progs         = [PROGRESSION_STRATEGIES[p] for p in wc.progressions]
            strategy      = STRATEGIES.get(wc.disease_strategy)
            end_cond      = end_condition_from_config(wc.end_condition, wc.end_condition_param)
            n_agents, bg_rate, bscale = wc.num_agents, wc.background_visit_rate, wc.beta_scale
            s_offset, rev_icd, n_seeds = wc.seed_offset, wc.reveal_incubating_icd, wc.initial_seeds
        else:
            progs    = [PROGRESSION_STRATEGIES[p] for p in cfg.progressions]
            strategy = STRATEGIES.get(cfg.disease_strategy)
            end_cond = end_condition_from_config(cfg.end_condition, cfg.end_condition_param)
            n_agents, bg_rate, bscale = cfg.num_agents, 0.025, cfg.beta_scale
            s_offset, rev_icd, n_seeds = 0, True, cfg.initial_seeds

        d_weights = (
            rng_dirichlet.dirichlet([cfg.dirichlet_alpha] * len(progs)).tolist()
            if len(progs) > 1 and cfg.dirichlet_alpha > 0 else None
        )
        return WorldEngine(
            num_agents=n_agents, seed=cfg.seed + silo_idx + s_offset,
            progression_strategies=progs, disease_strategy=strategy,
            end_condition=end_cond, background_visit_rate=bg_rate,
            beta_scale=bscale, reveal_incubating_icd=rev_icd,
            initial_seeds=n_seeds, disease_weights=d_weights,
        )

    # Build silo clients (worlds only — we ignore their individual models)
    lora_cfg = cfg.lora_config()
    silos: list[WorldFLClient] = []
    for i in range(cfg.num_silos):
        silos.append(WorldFLClient(
            world               = _make_world(i),
            lora_config         = lora_cfg,
            sim_days            = cfg.sim_days,
            min_events_to_train = cfg.min_events_to_train,
        ))

    trainer = CentralizedTrainer(
        silos               = silos,
        lora_config         = lora_cfg,
        local_epochs        = cfg.local_epochs,
        batch_size          = cfg.batch_size,
        lr                  = cfg.lr,
        min_events_to_train = cfg.min_events_to_train,
    )

    print(
        f"\n{'─'*60}\n"
        f"  FedWorld Centralized Training (oracle baseline)\n"
        f"  silos={cfg.num_silos}  max_rounds={cfg.max_rounds}  agents/silo={cfg.num_agents}\n"
        f"  sim_days={cfg.sim_days}  local_epochs={cfg.local_epochs}  lr={cfg.lr}\n"
        f"{'─'*60}\n"
    )

    nan = float("nan")
    for round_num in range(1, cfg.max_rounds + 1):
        m = trainer.run_round()
        m.round_num = round_num

        log = {
            "round":                        round_num,
            "centralized/triage_acc":       m.triage_acc,
            "centralized/diag_acc":         m.diag_acc,
            "centralized/combined_acc":     m.combined_acc,
            "centralized/triage_macro_f1":  m.triage_macro_f1,
            "centralized/danger_rate":      m.danger_rate,
            "centralized/num_events":       m.num_events,
            "centralized/trained_on":       m.trained_on,
            "centralized/loss":             m.loss,
        }
        run.log(log, step=round_num)

        def _f(v): return f"{v:.2f}" if v == v else "—"
        print(
            f"  Round {round_num:>3}/~{cfg.max_rounds} | "
            f"loss={m.loss:.4f} tr={_f(m.triage_acc)} diag={_f(m.diag_acc)} "
            f"f1={_f(m.triage_macro_f1)} danger={_f(m.danger_rate)} "
            f"ev={m.num_events} trained_on={m.trained_on} | {m.elapsed:.1f}s"
        )

        if m.all_done:
            print(f"\n  All silos finished (centralized).")
            break

    run.finish()
    return trainer


def run_comparison(cfg: FLTrainConfig | None = None, **kwargs) -> dict:
    """
    Run federated and centralized training back-to-back with the same config.

    Returns dict with keys 'federated' (list[WorldFLClient]) and
    'centralized' (CentralizedTrainer) for downstream analysis.
    """
    if cfg is None:
        cfg = FLTrainConfig(**{k: v for k, v in kwargs.items()
                                if k in FLTrainConfig.__dataclass_fields__})
    print("═" * 60)
    print("  COMPARISON RUN — Federated")
    print("═" * 60)
    fed_run_name  = (cfg.wandb_run_name or "comparison") + "_federated"
    fed_silos     = run_federated_training(
        FLTrainConfig(**{**__import__("dataclasses").asdict(cfg),
                         "wandb_run_name": fed_run_name,
                         "world_configs": cfg.world_configs})
    )

    print("═" * 60)
    print("  COMPARISON RUN — Centralized")
    print("═" * 60)
    cen_run_name  = (cfg.wandb_run_name or "comparison") + "_centralized"
    cen_trainer   = run_centralized_training(
        FLTrainConfig(**{**__import__("dataclasses").asdict(cfg),
                         "wandb_run_name": cen_run_name,
                         "world_configs": cfg.world_configs})
    )

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
    p.add_argument("--progressions", nargs="+", default=["Standard Flu"],
                   help="One or more disease progressions circulating in all silos")
    p.add_argument("--disease-strategy", type=str, default="Standard Flu",
                   dest="disease_strategy")
    p.add_argument("--seed",                type=int,   default=42)
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
        wandb_project       = a.project,
        wandb_run_name      = a.run_name,
        wandb_entity        = a.entity,
        wandb_offline       = a.offline,
        use_ollama          = not a.no_ollama,
    )


if __name__ == "__main__":
    run_federated_training(_parse_args())
