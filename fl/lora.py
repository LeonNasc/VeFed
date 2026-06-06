"""
LoRA configuration and weight I/O utilities.

Both Flower and FedN communicate model state as flat numpy arrays / .npz
files respectively.  The LoRA adapter is applied to DistilBERT's attention
projections (q_lin, v_lin) — the only trainable weights exchanged per round.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoRAConfig:
    model_name_or_path: str = "distilbert-base-uncased"
    rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    # DistilBERT uses q_lin / v_lin (not query / value like BERT)
    target_modules: list[str] = field(
        default_factory=lambda: ["q_lin", "v_lin"]
    )
    # 7 diagnostic labels: 2 diseases × 3 severities + non-infectious
    num_labels: int = 7

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.rank


def build_model(config: LoRAConfig):
    """
    Return a PEFT LoRA-adapted DistilBERT sequence classifier.
    Only LoRA adapter params are trainable; base weights are frozen.
    """
    from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, logging as hf_logging

    # Suppress expected warnings (UNEXPECTED keys = MLM head, MISSING keys = new
    # classifier head) and the per-parameter loading progress bar — both are
    # expected artifacts of loading a base model for LoRA fine-tuning.
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    base = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=config.num_labels,
    )

    hf_logging.enable_progress_bar()
    hf_logging.set_verbosity_warning()  # restore for other HF calls
    peft_cfg = PeftLoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        # Keep the classification head trainable and included in weight exchange.
        # Without this, PEFT freezes the randomly-initialised head and training
        # never propagates to the output layer.
        modules_to_save=["pre_classifier", "classifier"],
    )
    return get_peft_model(base, peft_cfg)


# ── Weight extraction / loading ───────────────────────────────────────────────

def get_lora_weights(model) -> list[np.ndarray]:
    """Return all trainable parameters (LoRA adapters + classifier head) as numpy arrays."""
    return [
        p.detach().cpu().numpy()
        for _, p in model.named_parameters()
        if p.requires_grad
    ]


def set_lora_weights(model, weights: list[np.ndarray]) -> None:
    """Write federated weights back into all trainable parameters in-place."""
    import torch

    idx = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = torch.as_tensor(weights[idx], dtype=p.dtype).to(p.device)
            idx += 1


# ── Serialisation helpers (for FedN .npz protocol) ───────────────────────────

def weights_to_npz(weights: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {f"w{i:04d}": w for i, w in enumerate(weights)}


def npz_to_weights(npz: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [npz[k] for k in sorted(npz, key=lambda k: int(k[1:]))]


def save_weights(weights: list[np.ndarray], path: str) -> None:
    np.savez(path, **weights_to_npz(weights))


def load_weights(path: str) -> list[np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return npz_to_weights(dict(data))
