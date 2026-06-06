"""
Centralized training baseline for federated learning comparison.

Runs N independent WorldEngine silos (identical seeds to the federated run)
but instead of FedAvg, pools all events from all silos each round and trains
one shared model on the union.  This is the data-pooling oracle upper bound.

Comparison framing
------------------
  Local-only  : each silo trains on its own data, no sharing
  Federated   : FedAvg over adapter weights (no raw data sharing)
  Centralized : one model trained on pooled data from all silos (oracle)

If federated ≈ centralized → FedAvg effectively recovers pooled performance
  without sharing raw data.
If federated < centralized → gap quantifies the FedAvg communication cost.
If federated > centralized on a specific silo → non-IID advantage: the global
  model generalises better than a pooled model because pooling dilutes
  rare-disease signal from small silos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fl.learner import build_label_map, _split_label
from simulation.world import WorldEngine
from fl.lora import LoRAConfig, build_model, get_lora_weights, set_lora_weights


@dataclass
class CentralizedRoundMetrics:
    round_num:   int
    loss:        float
    triage_acc:  float
    diag_acc:    float
    danger_rate: float
    num_events:  int
    trained_on:  int
    epoch_losses: list[float]
    elapsed:     float
    all_done:    bool = False


class CentralizedTrainer:
    """
    Centralized oracle: N worlds, one shared model trained on pooled events.

    Parameters
    ----------
    silos : list[WorldEngine]
        Pre-built silo worlds. Events are pooled across all silos each round.
    lora_config : LoRAConfig
    local_epochs : int
    batch_size : int
    lr : float
    min_events_to_train : int
    """

    def __init__(
        self,
        silos: list[WorldEngine],
        lora_config: Optional[LoRAConfig] = None,
        local_epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-4,
        min_events_to_train: int = 10,
        sim_days: int = 7,
        train_sample_cap: int = 512,
        device: Optional[str] = None,
    ):
        from transformers import AutoTokenizer
        self.silos               = silos
        self._sim_days           = sim_days
        self.local_epochs        = local_epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.min_events_to_train = min_events_to_train
        self.train_sample_cap    = train_sample_cap
        # Default CPU: same rationale as FLLearner (Ollama holds the GPU).
        self.device = device if device is not None else "cpu"

        self._label2id, self._id2label = build_label_map()

        cfg = lora_config or LoRAConfig()
        if cfg.num_labels != len(self._label2id):
            from dataclasses import replace
            cfg = replace(cfg, num_labels=len(self._label2id))
        self.lora_config = cfg
        self._tokenizer  = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

        self._model        = None   # lazy-built
        self._pool_buffer: list = []   # accumulated events from all silos
        self._pool_rng = __import__("random").Random(42)

    @property
    def model(self):
        if self._model is None:
            self._model = build_model(self.lora_config).to(self.device)
        return self._model

    def get_weights(self) -> list[np.ndarray]:
        return get_lora_weights(self.model)

    def release_model(self) -> None:
        import gc
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def run_round(self) -> CentralizedRoundMetrics:
        """
        One centralized round:
          1. Advance every silo's world by sim_days
          2. Pool all new events into the shared buffer
          3. Train one model on the full buffer
          4. Evaluate on this round's new events

        NOTE: does NOT release the model — the model must persist across rounds
        so that training accumulates. The caller (run_centralized_training) is
        responsible for calling release_model() once after all rounds finish.
        """
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset

        t0 = time.time()

        # ── Step 1: advance all worlds, collect events ────────────────────────
        round_events: list = []
        for silo in self.silos:
            if not silo.is_done:
                start = len(silo.clinic_queue.processed)
                for _ in range(self._sim_days * silo.TICKS_PER_DAY):
                    silo.step_tick()
                    if silo.is_done:
                        break
                round_events.extend(silo.clinic_queue.processed[start:])
                # Trim processed archive to prevent unbounded memory growth;
                # round_events already holds references to the new events.
                del silo.clinic_queue.processed[:]

        all_done = all(s.is_done for s in self.silos)

        # ── Step 2: extend shared pool, capped at 4× train_sample_cap ────────
        self._pool_buffer.extend(round_events)
        max_pool = self.train_sample_cap * 4
        if len(self._pool_buffer) > max_pool:
            self._pool_buffer = self._pool_buffer[-max_pool:]

        # ── Step 3: evaluate BEFORE training (on new events only) ────────────
        eval_metrics = self._evaluate(round_events) if round_events else {}

        # ── Step 4: train on a stratified sample from the pool ───────────────
        trained_on   = 0
        epoch_losses: list[float] = []

        if len(self._pool_buffer) >= self.min_events_to_train:
            sample = (
                self._pool_rng.sample(self._pool_buffer, self.train_sample_cap)
                if len(self._pool_buffer) > self.train_sample_cap
                else self._pool_buffer
            )
            texts, labels = self._build_dataset(sample)
            if texts:
                enc = self._tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=128, return_tensors="pt",
                )
                label_t = torch.tensor(labels, dtype=torch.long)
                dataset = TensorDataset(
                    enc["input_ids"], enc["attention_mask"], label_t
                )
                loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                self.model.train()
                opt = AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.lr,
                )
                for _ in range(self.local_epochs):
                    batch_losses = []
                    for iids, amask, blabels in loader:
                        iids    = iids.to(self.device)
                        amask   = amask.to(self.device)
                        blabels = blabels.to(self.device)
                        opt.zero_grad()
                        out = self.model(
                            input_ids=iids, attention_mask=amask, labels=blabels
                        )
                        out.loss.backward()
                        opt.step()
                        batch_losses.append(out.loss.detach().item())
                    epoch_losses.append(sum(batch_losses) / len(batch_losses))

                trained_on = len(texts)

        return CentralizedRoundMetrics(
            round_num    = -1,   # caller sets this
            loss         = epoch_losses[-1] if epoch_losses else float("nan"),
            triage_acc   = eval_metrics.get("triage_acc",  float("nan")),
            diag_acc     = eval_metrics.get("diag_acc",    float("nan")),
            danger_rate  = eval_metrics.get("danger_rate", float("nan")),
            num_events   = len(round_events),
            trained_on   = trained_on,
            epoch_losses = epoch_losses,
            elapsed      = time.time() - t0,
            all_done     = all_done,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_dataset(self, events: list) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        for ev in events:
            label = self._label2id.get(ev.ground_truth or "")
            if label is None:
                continue
            turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
            if not turns:
                continue
            texts.append(turns[-1])
            labels.append(label)
        return texts, labels

    def _evaluate(self, events: list) -> dict:
        import torch

        texts, labels = self._build_dataset(events)
        if not texts:
            return {}

        enc = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        label_t = torch.tensor(labels, dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
                labels=label_t,
            )

        preds = out.logits.argmax(dim=-1).cpu().tolist()
        n = len(preds)

        sev_ok = diag_ok = danger = n_severe_true = 0

        for pred_idx, true_idx in zip(preds, labels):
            pred_lbl = self._id2label[pred_idx]
            true_lbl = self._id2label[true_idx]
            pred_disease, pred_sev = _split_label(pred_lbl)
            true_disease, true_sev = _split_label(true_lbl)

            sev_ok  += pred_sev  == true_sev
            diag_ok += pred_disease == true_disease

            if true_sev == "severe":
                n_severe_true += 1
                if pred_sev not in ("severe",) or pred_lbl == "non-infectious":
                    danger += 1

        return {
            "triage_acc": sev_ok  / n,
            "diag_acc":   diag_ok / n,
            "danger_rate": danger / n_severe_true if n_severe_true > 0 else float("nan"),
        }
