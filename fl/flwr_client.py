"""
Flower (flwr) adapter for WorldFLClient.

Usage — client side:
    from simulation.world import WorldEngine
    from fl.lora import LoRAConfig
    from fl.base_client import WorldFLClient
    from fl.flwr_client import start_flower_client

    world  = WorldEngine(num_agents=50, seed=42)
    client = WorldFLClient(world, LoRAConfig(), sim_days=7, local_epochs=3)
    start_flower_client(client, server_address="localhost:8080")

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

from fl.base_client import WorldFLClient


def make_flower_client(client: WorldFLClient):
    """
    Wrap a WorldFLClient as a Flower NumPyClient.

    get_parameters : return current LoRA adapter weights
    fit            : load global weights → simulate → train → return updated weights
    evaluate       : load global weights → evaluate → return loss + accuracy
    """
    try:
        import flwr as fl
    except ImportError as exc:
        raise ImportError(
            "Flower not installed. Run: pip install flwr>=1.8"
        ) from exc

    class _WorldFlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return client.get_weights()

        def fit(self, parameters, config):
            client.set_weights(parameters)
            events = client.run_simulation_round()
            n = client.train_on_events(events)
            return client.get_weights(), n, {}

        def evaluate(self, parameters, config):
            client.set_weights(parameters)
            m = client.evaluate()
            return m["loss"], m["num_examples"], {"accuracy": m["accuracy"]}

    return _WorldFlowerClient()


def start_flower_client(
    client: WorldFLClient,
    server_address: str = "localhost:8080",
) -> None:
    """Connect to a Flower server and begin federated LoRA training."""
    import flwr as fl

    fl.client.start_numpy_client(
        server_address=server_address,
        client=make_flower_client(client),
    )
