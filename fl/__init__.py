"""
fl/ — Federated Learning package for the Virtual World simulator.

Each WorldEngine instance is one FL client. Supports both Flower (flwr)
and FedN backends. Training uses LoRA adapters on a DistilBERT-based
diagnostic classifier — only adapter weights are communicated per round.

Quick start (Flower):
    from simulation.world import WorldEngine
    from fl.lora import LoRAConfig
    from fl.base_client import WorldFLClient
    from fl.flwr_client import start_flower_client

    world = WorldEngine(num_agents=50, seed=0)
    client = WorldFLClient(world, LoRAConfig())
    start_flower_client(client, server_address="localhost:8080")

Quick start (FedN):
    from fl.fedn_client import run_fedn_client
    run_fedn_client(world, fedn_url="http://localhost:8092", token="...")
"""
