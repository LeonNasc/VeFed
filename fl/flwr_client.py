"""
Flower (flwr) adapter for FLSilo participants.

Two use modes
-------------
In-process simulation (fl/train.py run_flower_federated_training):
    The orchestrator calls client.fit() directly in a round loop — no gRPC,
    no Ray.  Silo state persists between rounds because everything runs in
    the same process.

Real distributed deployment:
    start_flower_client(silo, cid, server_address) connects to a real Flower
    server via gRPC.  The server aggregates weights from all connected clients.

    Server side:
        from fl.server import run_flower_server, WandBFedAvg
        run_flower_server(num_rounds=20, num_clients=3, wandb_run=run)
"""
from __future__ import annotations

from fl.silo import FLSilo


def make_flower_client(silo: FLSilo, cid: str = "0"):
    """
    Wrap an FLSilo as a Flower NumPyClient.

    get_parameters : return current LoRA adapter weights (doctor model)
    fit            : global weights → simulate + train → return weights + metrics
    evaluate       : global weights → holdout eval → return loss + metrics

    The fit() method passes ``round_num`` from the config dict to
    silo.run_round() so simulation-guided termination works correctly.
    Done silos return 0 examples so FedAvg naturally excludes them.
    """
    try:
        import flwr as fl
    except ImportError as exc:
        raise ImportError(
            "Flower not installed. Run: pip install 'flwr[simulation]>=1.8'"
        ) from exc

    class _SiloFlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return silo.get_weights()

        def fit(self, parameters, config):
            round_num = int(config.get("round_num", 0))

            # Done silos are passive observers: accept global weights only if
            # they strictly improve on the frozen checkpoint (ratchet).
            if silo.is_done:
                silo.try_accept_global(parameters)
            else:
                silo.set_weights(parameters)

            m = silo.run_round(round_num=round_num)
            epoch_losses = m.pop("epoch_losses", [])

            # Coerce to Flower Scalar types; replace NaN with sentinel -1.0
            metrics: dict = {}
            for k, v in m.items():
                if isinstance(v, bool):
                    metrics[k] = int(v)
                elif isinstance(v, int):
                    metrics[k] = v
                elif isinstance(v, float):
                    metrics[k] = v if v == v else -1.0
                # dicts (per_class) and lists are not Flower-compatible — skip

            if epoch_losses:
                metrics["train_loss_final"] = float(epoch_losses[-1])
                for j, loss in enumerate(epoch_losses):
                    metrics[f"epoch_{j}_loss"] = float(loss)
            metrics["silo_id"] = int(cid)
            metrics["is_done"] = int(silo.is_done)

            # Done silos contribute 0 weight to FedAvg
            n_examples = 0 if silo.is_done else int(m.get("num_examples", 0))
            return silo.get_weights(), n_examples, metrics

        def evaluate(self, parameters, config):
            # silo.evaluate() sets weights then runs doctor holdout eval
            m = silo.evaluate(parameters=parameters)
            nurse_m = silo.nurse_learner.evaluate_holdout()
            sir = silo.sir_model
            extra = {
                "accuracy":   float(m.get("diag_acc",        0.0)),
                "diag_acc":   float(m.get("diag_acc",        0.0)),
                "triage_acc": float(nurse_m.get("triage_acc", 0.0)),
                "sir_s":      int(sir.S),
                "sir_i":      int(sir.I),
                "sir_r":      int(sir.R),
                "silo_id":    int(cid),
            }
            loss = m.get("loss", float("nan"))
            return (loss if loss == loss else -1.0), m.get("num_examples", 0), extra

    return _SiloFlowerClient()


def start_flower_client(
    silo: FLSilo,
    cid: str = "0",
    server_address: str = "localhost:8080",
) -> None:
    """Connect to a Flower server and begin federated LoRA training."""
    import flwr as fl

    fl.client.start_numpy_client(
        server_address=server_address,
        client=make_flower_client(silo, cid=cid),
    )
