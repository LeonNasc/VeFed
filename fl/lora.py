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
    # 8 canonical ICD codes × 3 management tiers (home rest / treat / hospitalise)
    # = 3 infectious + 5 non-infectious, overridden dynamically by WorldFLClient
    num_labels: int = 24

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

    # Suppress expected warnings: UNEXPECTED keys = MLM head not used for
    # classification; MISSING keys = new classification head (expected).
    hf_logging.set_verbosity_error()

    base = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=config.num_labels,
    )

    hf_logging.set_verbosity_warning()  # restore for other HF calls
    peft_cfg = PeftLoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    return get_peft_model(base, peft_cfg)


# ── Weight extraction / loading ───────────────────────────────────────────────

def get_lora_weights(model) -> list[np.ndarray]:
    """Return all trainable LoRA adapter tensors as numpy arrays (ordered)."""
    return [
        p.detach().cpu().numpy()
        for name, p in model.named_parameters()
        if "lora_" in name and p.requires_grad
    ]


def set_lora_weights(model, weights: list[np.ndarray]) -> None:
    """Write averaged LoRA weights back into the model in-place."""
    import torch

    idx = 0
    for name, p in model.named_parameters():
        if "lora_" in name and p.requires_grad:
            # Respect the parameter's current device — avoids CPU↔GPU mismatch
            # when the model has been moved to GPU before set_weights() is called.
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
