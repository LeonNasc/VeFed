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
        reinit  = True,
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
                world        = world,
                lora_config  = lora_cfg,
                sim_days     = cfg.sim_days,
                local_epochs = cfg.local_epochs,
                batch_size   = cfg.batch_size,
                lr           = cfg.lr,
            )
        )

    print(
        f"\n{'─'*60}\n"
        f"  FedWorld FL Training\n"
        f"  silos={cfg.num_silos}  rounds={cfg.num_rounds}  "
        f"agents/silo={cfg.num_agents}\n"
        f"  sim_days={cfg.sim_days}  local_epochs={cfg.local_epochs}  "
        f"lr={cfg.lr}\n"
        f"  progression={cfg.progression}  seed={cfg.seed}\n"
        f"{'─'*60}\n"
    )

    # ── Initialise global weights from silo 0 ─────────────────────────────────
    global_weights = silos[0].get_weights()

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for round_num in range(1, cfg.num_rounds + 1):
        t0 = time.time()

        round_weights:  list[list[np.ndarray]] = []
        round_nexamples: list[int]             = []
        log: dict = {"round": round_num}

        agg_loss = agg_acc = agg_n = agg_trained = 0.0

        for i, silo in enumerate(silos):
            silo.set_weights(global_weights)
            m = silo.run_round()

            round_weights.append(silo.get_weights())
            n = m.get("num_examples", 0)
            round_nexamples.append(n)

            # Per-silo metrics
            epoch_losses = m.pop("epoch_losses", [])
            for key, val in m.items():
                if isinstance(val, (int, float)):
                    log[f"silo_{i}/{key}"] = val

            if epoch_losses:
                log[f"silo_{i}/train_loss_final"] = epoch_losses[-1]
                for ep_i, ep_loss in enumerate(epoch_losses):
                    log[f"silo_{i}/epoch_{ep_i}_loss"] = ep_loss

            # Flag done silos
            if silo.world.is_done:
                log[f"silo_{i}/done"] = 1
                log[f"silo_{i}/stop_reason"] = silo.world.stop_reason or "done"

            # Accumulate weighted aggregates
            if n > 0:
                agg_loss    += m.get("loss",      0.0) * n
                agg_acc     += m.get("accuracy",  0.0) * n
                agg_n       += n
                agg_trained += m.get("trained_on", 0)

        # ── FedAvg ────────────────────────────────────────────────────────────
        if any(n > 0 for n in round_nexamples):
            global_weights = _fedavg(round_weights, round_nexamples)

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
            log["aggregated/loss"]        = agg_loss / agg_n
            log["aggregated/accuracy"]    = agg_acc  / agg_n
            log["aggregated/num_examples"]= int(agg_n)
            log["aggregated/num_trained"] = int(agg_trained)

        run.log(log, step=round_num)

        elapsed = time.time() - t0
        _print_round(round_num, cfg.num_rounds, log, elapsed)

    # ── Distribute final global model to all silos ────────────────────────────
    for silo in silos:
        silo.set_weights(global_weights)

    run.finish()
    print(f"\nTraining complete. W&B run: {run.url}\n")
    return silos


# ── Console progress helper ───────────────────────────────────────────────────

def _print_round(round_num: int, total: int, log: dict, elapsed: float) -> None:
    agg_loss = log.get("aggregated/loss", float("nan"))
    agg_acc  = log.get("aggregated/accuracy", float("nan"))
    n_train  = int(log.get("aggregated/num_trained", 0))

    silo_parts = []
    i = 0
    while f"silo_{i}/sir_i" in log:
        sir_i = int(log[f"silo_{i}/sir_i"])
        acc_i = log.get(f"silo_{i}/accuracy", float("nan"))
        silo_parts.append(f"s{i}[I={sir_i} acc={acc_i:.2f}]")
        i += 1

    print(
        f"  Round {round_num:>3}/{total} | "
        f"loss={agg_loss:.4f} acc={agg_acc:.3f} "
        f"n={n_train:>4} | "
        f"{' '.join(silo_parts) or '—'} | "
        f"{elapsed:.1f}s"
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> FLTrainConfig:
    p = argparse.ArgumentParser(description="FedWorld FL training loop")
    p.add_argument("--silos",       type=int,   default=3)
    p.add_argument("--rounds",      type=int,   default=10)
    p.add_argument("--agents",      type=int,   default=30)
    p.add_argument("--sim-days",    type=int,   default=7)
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
        end_condition       = a.end_condition,
        end_condition_param = a.end_condition_param,
        wandb_project       = a.project,
        wandb_run_name  = a.run_name,
        wandb_entity    = a.entity,
        wandb_offline   = a.offline,
    )


if __name__ == "__main__":
    run_federated_training(_parse_args())
