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

from fl.base_client import WorldFLClient, CANONICAL_ICD_CODES, build_label_map
from fl.lora import LoRAConfig, build_model, get_lora_weights, set_lora_weights


@dataclass
class CentralizedRoundMetrics:
    round_num:       int
    loss:            float
    triage_acc:      float
    diag_acc:        float
    combined_acc:    float
    triage_macro_f1: float
    danger_rate:     float
    num_events:      int
    trained_on:      int
    epoch_losses:    list[float]
    elapsed:         float
    all_done:        bool = False


class CentralizedTrainer:
    """
    Centralized oracle: N worlds, one shared model trained on pooled events.

    Parameters
    ----------
    silos : list[WorldFLClient]
        Pre-built silo clients.  Only their worlds and replay buffers are used;
        their individual federated models are ignored.
    lora_config : LoRAConfig
    local_epochs : int
    batch_size : int
    lr : float
    min_events_to_train : int
    """

    def __init__(
        self,
        silos: list[WorldFLClient],
        lora_config: Optional[LoRAConfig] = None,
        local_epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-4,
        min_events_to_train: int = 10,
    ):
        import torch
        self.silos               = silos
        self.local_epochs        = local_epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.min_events_to_train = min_events_to_train
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._label2id, self._id2label = build_label_map(CANONICAL_ICD_CODES)

        cfg = lora_config or LoRAConfig()
        if cfg.num_labels != len(self._label2id):
            from dataclasses import replace
            cfg = replace(cfg, num_labels=len(self._label2id))
        self.lora_config = cfg

        self._model        = None   # lazy-built
        self._pool_buffer: list = []   # accumulated events from all silos

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
        """
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer

        t0 = time.time()

        # ── Step 1: advance all worlds, collect events ────────────────────────
        round_events: list = []
        for silo in self.silos:
            if not silo.world.is_done:
                events = silo.run_simulation_round()
                round_events.extend(events)

        all_done = all(s.world.is_done for s in self.silos)

        # ── Step 2: extend shared pool ────────────────────────────────────────
        self._pool_buffer.extend(round_events)

        # ── Step 3: evaluate BEFORE training (on new events only) ────────────
        eval_metrics = self._evaluate(round_events) if round_events else {}

        # ── Step 4: train on full pool ────────────────────────────────────────
        trained_on   = 0
        epoch_losses: list[float] = []

        if len(self._pool_buffer) >= self.min_events_to_train:
            texts, labels = self._build_dataset(self._pool_buffer)
            if texts:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.lora_config.model_name_or_path
                )
                enc = tokenizer(
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
        self.release_model()

        return CentralizedRoundMetrics(
            round_num       = -1,   # caller sets this
            loss            = epoch_losses[-1] if epoch_losses else float("nan"),
            triage_acc      = eval_metrics.get("triage_acc",      float("nan")),
            diag_acc        = eval_metrics.get("diag_acc",        float("nan")),
            combined_acc    = eval_metrics.get("combined_acc",    float("nan")),
            triage_macro_f1 = eval_metrics.get("triage_macro_f1", float("nan")),
            danger_rate     = eval_metrics.get("danger_rate",     float("nan")),
            num_events      = len(round_events),
            trained_on      = trained_on,
            epoch_losses    = epoch_losses,
            elapsed         = time.time() - t0,
            all_done        = all_done,
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
        from collections import defaultdict
        from transformers import AutoTokenizer

        texts, labels = self._build_dataset(events)
        if not texts:
            return {}

        tokenizer = AutoTokenizer.from_pretrained(self.lora_config.model_name_or_path)
        enc = tokenizer(
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

        icd_cat = mgmt_ok = combined = danger = 0
        triage_tp: dict[str, int] = defaultdict(int)
        triage_fp: dict[str, int] = defaultdict(int)
        triage_fn: dict[str, int] = defaultdict(int)

        for pred_idx, true_idx in zip(preds, labels):
            pred_lbl = self._id2label[pred_idx]
            true_lbl = self._id2label[true_idx]
            pred_icd, pred_mgmt = pred_lbl.rsplit(" / ", 1)
            true_icd, true_mgmt = true_lbl.rsplit(" / ", 1)

            cat  = pred_icd[:3] == true_icd[:3]
            mgmt = pred_mgmt == true_mgmt
            icd_cat  += cat
            mgmt_ok  += mgmt
            combined += (cat and mgmt)

            if mgmt:
                triage_tp[true_mgmt] += 1
            else:
                triage_fp[pred_mgmt] += 1
                triage_fn[true_mgmt] += 1
                if true_mgmt == "hospitalise" and pred_mgmt == "home rest":
                    danger += 1

        tiers = ("home rest", "treat", "hospitalise")
        f1s = []
        for tier in tiers:
            tp = triage_tp[tier]; fp = triage_fp[tier]; fn = triage_fn[tier]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        n_hosp = triage_tp["hospitalise"] + triage_fn["hospitalise"]

        return {
            "triage_acc":      mgmt_ok  / n,
            "diag_acc":        icd_cat  / n,
            "combined_acc":    combined / n,
            "triage_macro_f1": sum(f1s) / len(f1s),
            "danger_rate":     danger / n_hosp if n_hosp > 0 else float("nan"),
        }
