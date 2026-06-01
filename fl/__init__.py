"""
fl/ — Federated Learning package for the Virtual World simulator.

Each WorldEngine instance is one FL silo. Pass lora_config to WorldEngine
to enable FL training — no separate wrapper needed.

Quick start (standalone FL):
    from fl.train import run_federated_training, FLTrainConfig
    run_federated_training(FLTrainConfig(num_silos=3))

Quick start (Flower):
    from simulation.world import WorldEngine
    from fl.lora import LoRAConfig
    from fl.flwr_client import start_flower_client

    world = WorldEngine(num_agents=50, seed=0, lora_config=LoRAConfig())
    start_flower_client(world, server_address="localhost:8080")

Quick start (FedN):
    from simulation.world import WorldEngine
    from fl.lora import LoRAConfig
    from fl.fedn_client import run_fedn_client

    world = WorldEngine(num_agents=50, seed=0, lora_config=LoRAConfig())
    run_fedn_client(world, api_url="http://localhost:8092", token="...")
"""
