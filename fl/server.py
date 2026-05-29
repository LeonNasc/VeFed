"""
Flower server strategy with W&B logging.

WandBFedAvg subclasses FedAvg and hooks into aggregate_fit /
aggregate_evaluate to log per-silo and aggregated metrics to a W&B run.

Works in both modes:
  - Standalone simulation (fl/train.py) — wandb_run passed from the loop
  - Real Flower distributed deployment — initialise W&B on the server side
    before calling run_flower_server(), then pass the run handle

Logging structure (single W&B run):
  round                          — FL round index (step axis)
  aggregated/loss                — weighted-average loss across silos
  aggregated/accuracy            — weighted-average accuracy
  aggregated/num_examples        — total eval examples
  aggregated/num_trained         — total training examples this round
  silo_{cid}/loss                — per-silo post-train eval loss
  silo_{cid}/accuracy            — per-silo accuracy
  silo_{cid}/num_examples        — per-silo eval examples
  silo_{cid}/num_events          — raw events collected this round
  silo_{cid}/sir_s/i/r           — SIR state after simulation
  silo_{cid}/train_loss_final    — last-epoch mean train loss
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from flwr.common import (
    FitRes, EvaluateRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class WandBFedAvg(FedAvg):
    """
    FedAvg with Weights & Biases logging.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run | None
        Active W&B run to log into.  Pass None to disable W&B (strategy
        still aggregates normally).
    **kwargs
        Forwarded to FedAvg.__init__ (fraction_fit, min_fit_clients, …).
    """

    def __init__(self, wandb_run=None, **kwargs):
        super().__init__(**kwargs)
        self._run = wandb_run

    # ── Fit aggregation ────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if self._run is not None and results:
            self._run.log(
                self._build_fit_log(server_round, results),
                step=server_round,
            )

        return aggregated_params, aggregated_metrics

    # ── Evaluate aggregation ───────────────────────────────────────────────────

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if self._run is not None and results:
            self._run.log(
                self._build_eval_log(server_round, results),
                step=server_round,
            )

        return aggregated_loss, aggregated_metrics

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_fit_log(
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> dict:
        log: dict = {"round": server_round}
        total_n = 0
        w_loss = 0.0
        w_acc  = 0.0

        for proxy, fit_res in results:
            cid = proxy.cid
            m   = fit_res.metrics
            n   = fit_res.num_examples
            total_n += n

            for key, val in m.items():
                log[f"silo_{cid}/{key}"] = val

            if "loss" in m:
                w_loss += float(m["loss"]) * n
            if "accuracy" in m:
                w_acc  += float(m["accuracy"]) * n

        if total_n > 0:
            log["aggregated/loss"]         = w_loss / total_n
            log["aggregated/accuracy"]     = w_acc  / total_n
            log["aggregated/num_examples"] = total_n

        return log

    @staticmethod
    def _build_eval_log(
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
    ) -> dict:
        log: dict = {}
        for proxy, eval_res in results:
            cid = proxy.cid
            for key, val in eval_res.metrics.items():
                log[f"silo_{cid}/eval_{key}"] = val
        return log


# ── Distributed server launcher ───────────────────────────────────────────────

def run_flower_server(
    num_rounds: int = 10,
    num_clients: int = 3,
    wandb_run=None,
    server_address: str = "0.0.0.0:8080",
    initial_parameters: Optional[Parameters] = None,
) -> None:
    """
    Start a Flower server with WandBFedAvg for distributed training.

    Clients connect separately via fl/flwr_client.py:start_flower_client().
    W&B is logged server-side — initialise the run before calling this.

    Example
    -------
        import wandb, flwr as fl
        run = wandb.init(project="fedworld")
        run_flower_server(num_rounds=10, num_clients=3, wandb_run=run)
    """
    import flwr as fl

    strategy = WandBFedAvg(
        wandb_run           = wandb_run,
        fraction_fit        = 1.0,
        fraction_evaluate   = 1.0,
        min_fit_clients     = num_clients,
        min_evaluate_clients= num_clients,
        min_available_clients= num_clients,
        initial_parameters  = initial_parameters,
    )

    fl.server.start_server(
        server_address = server_address,
        config         = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy       = strategy,
    )
