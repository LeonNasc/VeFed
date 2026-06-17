"""
FedN adapter for FLSilo participants.

FedN uses a file-based model-exchange protocol:
  train(in_model_path, out_model_path) — load .npz, train, save .npz
  validate(in_model_path, out_data_path) — load .npz, eval, save JSON

Usage — register with FedN SDK:
    from fl.silo import make_silo
    from fl.train import FLTrainConfig
    from fl.fedn_client import make_fedn_callbacks, run_fedn_client

    silo = make_silo(world_config, fl_cfg, silo_idx=0,
                     dataset=ds, end_condition=cond, seed=42)
    run_fedn_client(silo, api_url="http://localhost:8092", token="<jwt>")
"""
from __future__ import annotations

import json
from pathlib import Path

from fl.silo import FLSilo
from fl.lora import load_weights, save_weights


def make_fedn_callbacks(silo: FLSilo):
    """Return (train_fn, validate_fn) conforming to FedN's entrypoint contract."""

    def train(in_model_path: str, out_model_path: str, **_) -> None:
        silo.set_weights(load_weights(in_model_path))
        m = silo.run_round()
        save_weights(silo.get_weights(), out_model_path)
        print(f"[FedN] Trained on {m.get('trained_on', 0)} examples → {out_model_path}")

    def validate(in_model_path: str, out_data_path: str, **_) -> None:
        m = silo.evaluate(parameters=load_weights(in_model_path))
        Path(out_data_path).write_text(json.dumps(m))
        print(f"[FedN] Validation metrics: {m}")

    return train, validate


def run_fedn_client(
    silo: FLSilo,
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

    train_fn, validate_fn = make_fedn_callbacks(silo)
    fedn = FednClient(train_callback=train_fn, validate_callback=validate_fn)

    fedn.set_url(api_url)
    fedn.set_token(token)
    if client_id:
        fedn.set_client_id(client_id)

    fedn.run()
