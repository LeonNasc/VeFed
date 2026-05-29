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
from typing import Optional

import numpy as np

from fl.lora import LoRAConfig
from fl.base_client import WorldFLClient
from simulation.world import WorldEngine
from simulation.progression import PROGRESSION_STRATEGIES, DiseaseProgressionStrategy
from simulation.end_conditions import EndCondition, from_config as end_condition_from_config


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class FLTrainConfig:
    """All hyperparameters for one federated training run."""
    num_silos:          int   = 3
    num_rounds:         int   = 10
    num_agents:         int   = 30       # agents per silo
    sim_days:           int   = 7        # simulated days per FL round
    min_events_to_train: int  = 10       # skip LoRA update if fewer events this round
    local_epochs:       int   = 3
    batch_size:         int   = 8
    lr:                 float = 1e-4
    lora_rank:          int   = 8
    lora_alpha:         float = 16.0
    lora_dropout:       float = 0.05
    progression:        str   = "Standard Flu"   # key in PROGRESSION_STRATEGIES
    seed:               int   = 42
    # End condition
    end_condition:      str   = "horizon"   # horizon | extinction | no_susceptibles | budget
    end_condition_param: Optional[int] = None  # max_days | consecutive_days | target_events
    # W&B
    wandb_project:      str   = "fedworld"
    wandb_run_name:     Optional[str] = None
    wandb_entity:       Optional[str] = None
    wandb_offline:      bool  = False    # True → no network, logs saved locally
    use_ollama:         bool  = True     # Wire Ollama doctor; falls back to stub if unreachable
    patient_model:      str   = "tinyllama"  # Ollama model for patient statement generation

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

    # ── Progression strategy ──────────────────────────────────────────────────
    if cfg.progression not in PROGRESSION_STRATEGIES:
        raise ValueError(
            f"Unknown progression strategy {cfg.progression!r}. "
            f"Available: {list(PROGRESSION_STRATEGIES)}"
        )
    progression = PROGRESSION_STRATEGIES[cfg.progression]

    # ── W&B initialisation ────────────────────────────────────────────────────
    if cfg.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        project = cfg.wandb_project,
        name    = cfg.wandb_run_name,
        entity  = cfg.wandb_entity,
        config  = {
            **{k: v for k, v in asdict(cfg).items()
               if not k.startswith("wandb_")},
        },
        reinit  = "finish_previous",
    )

    # ── End condition — one instance per silo (each has independent state) ────
    def _make_end_condition() -> EndCondition:
        return end_condition_from_config(cfg.end_condition, cfg.end_condition_param)

    # ── Create silos ──────────────────────────────────────────────────────────
    lora_cfg = cfg.lora_config()
    silos: list[WorldFLClient] = []
    for i in range(cfg.num_silos):
        world = WorldEngine(
            num_agents           = cfg.num_agents,
            seed                 = cfg.seed + i,
            progression_strategy = progression,
            end_condition        = _make_end_condition(),
        )
        silos.append(
            WorldFLClient(
                world                = world,
                lora_config          = lora_cfg,
                sim_days             = cfg.sim_days,
                min_events_to_train  = cfg.min_events_to_train,
                local_epochs         = cfg.local_epochs,
                batch_size           = cfg.batch_size,
                lr                   = cfg.lr,
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

    print(
        f"\n{'─'*60}\n"
        f"  FedWorld FL Training\n"
        f"  silos={cfg.num_silos}  rounds={cfg.num_rounds}  "
        f"agents/silo={cfg.num_agents}\n"
        f"  sim_days={cfg.sim_days}  min_events={cfg.min_events_to_train}  "
        f"local_epochs={cfg.local_epochs}  lr={cfg.lr}\n"
        f"  progression={cfg.progression}  seed={cfg.seed}\n"
        f"  LLM doctor:  {ollama_status}\n"
        f"  LLM patient: {patient_status}\n"
        f"{'─'*60}\n"
    )

    # ── Initialise global weights from silo 0 ─────────────────────────────────
    global_weights = silos[0].get_weights()

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for round_num in range(1, cfg.num_rounds + 1):
        t0 = time.time()

        round_weights:   list[list[np.ndarray]] = []
        round_nexamples: list[int]             = []
        round_trained:   list[bool]            = []   # which silos actually trained
        log: dict = {"round": round_num}

        agg_loss = agg_acc = agg_n = agg_trained = 0.0

        for i, silo in enumerate(silos):
            silo.set_weights(global_weights)
            m = silo.run_round()

            round_weights.append(silo.get_weights())
            n          = m.get("num_examples", 0)
            did_train  = bool(m.get("trained", 0))
            round_nexamples.append(n)
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
                agg_loss     += m.get("loss",             0.0) * n
                agg_acc      += m.get("accuracy",         0.0) * n
                agg_n        += n
                agg_trained  += m.get("trained_on",       0)
                # Also aggregate sub-metrics weighted by examples
                for sub in ("icd_exact_acc", "icd_category_acc", "mgmt_acc"):
                    agg_key = f"aggregated/{sub}"
                    log[agg_key] = log.get(agg_key, 0.0) + m.get(sub, 0.0) * n

        # ── FedAvg — only silos that trained contribute updated weights ────────
        trained_weights  = [w for w, t in zip(round_weights, round_trained) if t]
        trained_examples = [n for n, t in zip(round_nexamples, round_trained) if t]
        if trained_weights:
            global_weights = _fedavg(trained_weights, trained_examples)
        log["aggregated/silos_trained"] = sum(round_trained)

        # ── Federated few-shot update — refresh Ollama doctor's example bank ──
        # Harvest correct conversations from all silos this round and update
        # the shared Ollama client so the doctor improves each round.
        if ollama_active and ollama_client is not None:
            round_events = [ev for silo in silos for ev in silo.last_round_events]
            ollama_client.update_examples(round_events)
            log["aggregated/llm_few_shot_examples"] = ollama_client.num_examples()

        # ── Early exit if all silos finished ──────────────────────────────────
        if all(silo.world.is_done for silo in silos):
            log["aggregated/all_silos_done"] = 1
            run.log(log, step=round_num)
            elapsed = time.time() - t0
            _print_round(round_num, cfg.num_rounds, log, elapsed)
            reasons = [silo.world.stop_reason or "done" for silo in silos]
            print(f"\n  All silos finished: {reasons[0]}")
            break

        # ── Aggregated log ────────────────────────────────────────────────────
        if agg_n > 0:
            log["aggregated/loss"]             = agg_loss / agg_n
            log["aggregated/accuracy"]         = agg_acc  / agg_n
            log["aggregated/num_examples"]     = int(agg_n)
            log["aggregated/num_trained"]      = int(agg_trained)
            for sub in ("icd_exact_acc", "icd_category_acc", "mgmt_acc"):
                k = f"aggregated/{sub}"
                if k in log:
                    log[k] /= agg_n

        run.log(log, step=round_num)

        elapsed = time.time() - t0
        _print_round(round_num, cfg.num_rounds, log, elapsed)

    # ── Distribute final global model to all silos ────────────────────────────
    for silo in silos:
        silo.set_weights(global_weights)

    run.finish()
    run_ref = run.url or f"offline — sync with: wandb sync {run.dir}"
    print(f"\nTraining complete. W&B: {run_ref}\n")
    return silos


# ── Console progress helper ───────────────────────────────────────────────────

def _print_round(round_num: int, total: int, log: dict, elapsed: float) -> None:
    agg_loss     = log.get("aggregated/loss") or float("nan")
    agg_acc      = log.get("aggregated/accuracy") or float("nan")
    n_trained    = int(log.get("aggregated/num_trained", 0))
    silos_trained= int(log.get("aggregated/silos_trained", 0))

    silo_parts = []
    i = 0
    while f"silo_{i}/sir_i" in log:
        sir_i   = int(log[f"silo_{i}/sir_i"])
        n_ev    = int(log.get(f"silo_{i}/num_events", 0))
        did     = "✓" if log.get(f"silo_{i}/trained", 0) else "✗"
        acc_i   = log.get(f"silo_{i}/accuracy", float("nan"))
        silo_parts.append(f"s{i}[I={sir_i} ev={n_ev} {did}{acc_i:.2f}]")
        i += 1

    loss_str    = f"{agg_loss:.4f}" if agg_loss == agg_loss else "—"
    acc_str     = f"{agg_acc:.3f}"  if agg_acc  == agg_acc  else "—"
    icd_cat_str = f"{log.get('aggregated/icd_category_acc', float('nan')):.2f}"
    mgmt_str    = f"{log.get('aggregated/mgmt_acc', float('nan')):.2f}"
    print(
        f"  Round {round_num:>3}/{total} | "
        f"loss={loss_str} acc={acc_str} "
        f"icd={icd_cat_str} mgmt={mgmt_str} "
        f"n={n_trained:>4} silos={silos_trained} | "
        f"{' '.join(silo_parts) or '—'} | "
        f"{elapsed:.1f}s"
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> FLTrainConfig:
    p = argparse.ArgumentParser(description="FedWorld FL training loop")
    p.add_argument("--silos",       type=int,   default=3)
    p.add_argument("--rounds",      type=int,   default=10)
    p.add_argument("--agents",      type=int,   default=30)
    p.add_argument("--sim-days",    type=int,   default=7,
                   help="Simulated days per FL round")
    p.add_argument("--min-events",  type=int,   default=10,
                   dest="min_events_to_train",
                   help="Skip LoRA training if fewer events collected this round")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lora-rank",   type=int,   default=8)
    p.add_argument("--progression",         type=str,   default="Standard Flu")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--end-condition",       type=str,   default="horizon",
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
        num_silos       = a.silos,
        num_rounds      = a.rounds,
        num_agents      = a.agents,
        sim_days        = a.sim_days,
        local_epochs    = a.epochs,
        batch_size      = a.batch_size,
        lr              = a.lr,
        lora_rank       = a.lora_rank,
        progression     = a.progression,
        seed                = a.seed,
        min_events_to_train = a.min_events_to_train,
        end_condition       = a.end_condition,
        end_condition_param = a.end_condition_param,
        wandb_project       = a.project,
        wandb_run_name  = a.run_name,
        wandb_entity    = a.entity,
        wandb_offline   = a.offline,
        use_ollama      = not a.no_ollama,
    )


if __name__ == "__main__":
    run_federated_training(_parse_args())
