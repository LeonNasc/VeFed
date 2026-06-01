"""
FedN adapter for WorldEngine FL silos.

FedN uses a file-based model-exchange protocol:
  train(in_model_path, out_model_path) — load .npz, train, save .npz
  validate(in_model_path, out_data_path) — load .npz, eval, save JSON

Usage — register with FedN SDK:
    from simulation.world import WorldEngine
    from fl.lora import LoRAConfig
    from fl.fedn_client import make_fedn_callbacks, run_fedn_client

    world = WorldEngine(num_agents=50, seed=7, lora_config=LoRAConfig(), sim_days=7)
    run_fedn_client(world, api_url="http://localhost:8092", token="<jwt>")
"""
from __future__ import annotations

import json
from pathlib import Path

from simulation.world import WorldEngine
from fl.lora import load_weights, save_weights


def make_fedn_callbacks(world: WorldEngine):
    """Return (train_fn, validate_fn) conforming to FedN's entrypoint contract."""

    def train(in_model_path: str, out_model_path: str, **_) -> None:
        world.set_weights(load_weights(in_model_path))
        m = world.run_round()
        save_weights(world.get_weights(), out_model_path)
        print(f"[FedN] Trained on {m.get('trained_on', 0)} examples → {out_model_path}")

    def validate(in_model_path: str, out_data_path: str, **_) -> None:
        world.set_weights(load_weights(in_model_path))
        metrics = world.evaluate()
        Path(out_data_path).write_text(json.dumps(metrics))
        print(f"[FedN] Validation metrics: {metrics}")

    return train, validate


def run_fedn_client(
    world: WorldEngine,
    api_url: str,
    token: str,
    client_id: str | None = None,
) -> None:
    """
    Start a FedN client process that connects to a running FedN combiner.
    Requires: pip install fedn
    """
    try:
        from fedn.network.clients.fedn_client import FednClient
    except ImportError as exc:
        raise ImportError(
            "FedN not installed. Run: pip install fedn"
        ) from exc

    train_fn, validate_fn = make_fedn_callbacks(world)
    fedn = FednClient(train_callback=train_fn, validate_callback=validate_fn)

    fedn.set_url(api_url)
    fedn.set_token(token)
    if client_id:
        fedn.set_client_id(client_id)

    fedn.run()
