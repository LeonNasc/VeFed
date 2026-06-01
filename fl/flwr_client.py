"""
Flower (flwr) adapter for WorldEngine FL silos.

Usage — client side:
    from simulation.world import WorldEngine
    from fl.lora import LoRAConfig
    from fl.flwr_client import start_flower_client

    world = WorldEngine(num_agents=50, seed=42, lora_config=LoRAConfig(), sim_days=7)
    start_flower_client(world, server_address="localhost:8080")

Usage — server side (vanilla FedAvg over LoRA weights):
    import flwr as fl
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
"""
from __future__ import annotations

from simulation.world import WorldEngine


def make_flower_client(world: WorldEngine, cid: str = "0"):
    """
    Wrap a WorldEngine FL silo as a Flower NumPyClient.

    get_parameters : return current LoRA adapter weights
    fit            : load global weights → simulate → train → return weights + metrics
    evaluate       : load global weights → evaluate → return loss + metrics
    """
    try:
        import flwr as fl
    except ImportError as exc:
        raise ImportError(
            "Flower not installed. Run: pip install flwr>=1.8"
        ) from exc

    class _WorldFlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return world.get_weights()

        def fit(self, parameters, config):
            world.set_weights(parameters)
            m = world.run_round()
            epoch_losses = m.pop("epoch_losses", [])
            metrics = {
                k: float(v) if isinstance(v, float) else int(v)
                for k, v in m.items()
                if isinstance(v, (int, float))
            }
            if epoch_losses:
                metrics["train_loss_final"] = float(epoch_losses[-1])
            metrics["silo_id"] = int(cid)
            return world.get_weights(), metrics.get("num_examples", 0), metrics

        def evaluate(self, parameters, config):
            world.set_weights(parameters)
            m = world.evaluate()
            sir = world.sir_model
            extra = {
                "accuracy": float(m.get("triage_acc", 0.0)),
                "sir_s":    int(sir.S),
                "sir_i":    int(sir.I),
                "sir_r":    int(sir.R),
                "silo_id":  int(cid),
            }
            return float(m.get("loss", float("nan"))), m.get("num_examples", 0), extra

    return _WorldFlowerClient()


def start_flower_client(
    world: WorldEngine,
    server_address: str = "localhost:8080",
) -> None:
    """Connect to a Flower server and begin federated LoRA training."""
    import flwr as fl

    fl.client.start_numpy_client(
        server_address=server_address,
        client=make_flower_client(world),
    )
