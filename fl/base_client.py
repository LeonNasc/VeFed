"""
WorldFLClient — framework-agnostic FL client for one WorldEngine instance.

Workflow per FL round:
  1. receive_weights()  — load globally aggregated LoRA weights
  2. run_simulation_round() — advance the world by sim_days, collect events
  3. train_on_events()  — LoRA fine-tune on (symptom_text → triage_label) pairs
  4. get_weights()      — return updated LoRA weights for aggregation

Training data:
  Each DiagnosticEvent contributes one (text, label) pair:
    text  = patient's opening symptom statement
    label = oracle triage label (mild / moderate / severe / critical)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from simulation.world import WorldEngine
from fl.lora import LoRAConfig, build_model, get_lora_weights, set_lora_weights

# Maps ground_truth strings (from Agent.build_diagnostic_event) to class indices
_LABEL_MAP: dict[str, int] = {
    "mild viral infection":         0,
    "moderate influenza":           1,
    "severe influenza":             2,
    "critical respiratory infection": 3,
}
ID2LABEL = {v: k for k, v in _LABEL_MAP.items()}


class WorldFLClient:
    """
    One FL participant = one WorldEngine silo.

    Parameters
    ----------
    world : WorldEngine
        The simulated epidemic world.
    lora_config : LoRAConfig
        Model + LoRA hyperparameters.
    sim_days : int
        Simulation days to run before each local training step.
    local_epochs : int
        Gradient steps per FL round.
    batch_size : int
    lr : float
    """

    def __init__(
        self,
        world: WorldEngine,
        lora_config: Optional[LoRAConfig] = None,
        sim_days: int = 7,
        local_epochs: int = 3,
        batch_size: int = 8,
        lr: float = 1e-4,
    ):
        self.world = world
        self.lora_config = lora_config or LoRAConfig()
        self.sim_days = sim_days
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model = None  # lazy-built on first use

    @property
    def model(self):
        if self._model is None:
            self._model = build_model(self.lora_config)
        return self._model

    # ── Simulation ─────────────────────────────────────────────────────────────

    def run_simulation_round(self) -> list:
        """
        Advance the world by sim_days and return newly processed DiagnosticEvents.
        Uses the world's default (rule-based) diagnostic stub so no LLM is needed
        during FL training — the LLM role is data generation, not FL itself.
        """
        start = len(self.world.clinic_queue.processed)
        for _ in range(self.sim_days * self.world.TICKS_PER_DAY):
            self.world.step_tick()
        return self.world.clinic_queue.processed[start:]

    # ── Local LoRA training ────────────────────────────────────────────────────

    def _build_dataset(self, events: list) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        for ev in events:
            label = _LABEL_MAP.get(ev.ground_truth or "")
            if label is None:
                continue
            patient_turns = [
                t["text"] for t in ev.conversation if t["role"] == "patient"
            ]
            if not patient_turns:
                continue
            texts.append(patient_turns[-1])
            labels.append(label)
        return texts, labels

    def train_on_events(self, events: list) -> int:
        """
        Fine-tune LoRA adapter on collected events.
        Returns number of training examples used.
        """
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer

        texts, labels = self._build_dataset(events)
        if not texts:
            return 0

        tokenizer = AutoTokenizer.from_pretrained(self.lora_config.model_name_or_path)
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        label_t = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], label_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = self.model
        model.train()
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
        )

        for _ in range(self.local_epochs):
            for input_ids, attn_mask, batch_labels in loader:
                optimizer.zero_grad()
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=batch_labels,
                ).loss
                loss.backward()
                optimizer.step()

        return len(texts)

    # ── Weight I/O ─────────────────────────────────────────────────────────────

    def get_weights(self) -> list[np.ndarray]:
        return get_lora_weights(self.model)

    def set_weights(self, weights: list[np.ndarray]) -> None:
        set_lora_weights(self.model, weights)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, events: Optional[list] = None) -> dict:
        """
        Evaluate LoRA model on events (defaults to all processed events).
        Returns {"loss": float, "accuracy": float, "num_examples": int}.
        """
        import torch
        from transformers import AutoTokenizer

        events = events or self.world.clinic_queue.processed
        texts, labels = self._build_dataset(events)
        if not texts:
            return {"loss": 0.0, "accuracy": 0.0, "num_examples": 0}

        tokenizer = AutoTokenizer.from_pretrained(self.lora_config.model_name_or_path)
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        label_t = torch.tensor(labels, dtype=torch.long)

        model = self.model
        model.eval()
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=label_t,
            )
        acc = (out.logits.argmax(dim=-1) == label_t).float().mean().item()
        return {
            "loss": float(out.loss),
            "accuracy": acc,
            "num_examples": len(texts),
        }

    # ── Convenience ────────────────────────────────────────────────────────────

    def run_round(self) -> dict:
        """
        One complete FL round: simulate → train → return metrics.
        Convenience wrapper used in standalone / testing mode.
        """
        events = self.run_simulation_round()
        n = self.train_on_events(events)
        metrics = self.evaluate(events)
        metrics["trained_on"] = n
        return metrics
