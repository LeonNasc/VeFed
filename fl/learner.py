"""
FLLearner — LoRA training, evaluation, and weight management for one FL silo.

Separated from simulation logic so WorldEngine can delegate to it without
importing PyTorch at module level.  Created lazily when WorldEngine.run_round()
is first called on an FL-enabled world.

Training strategy
-----------------
Each round, new clinic events are:
  1. Evaluated against the current model (prequential — model hasn't seen them).
  2. Serialised to the silo's on-disk SiloDataset (train / holdout split).
  3. The model is then fine-tuned on a stratified sample from the train split.

This decouples dataset size from per-round compute: early rounds train on all
available data (dataset < sample_cap); later rounds use a representative fixed-
size sample so training time stays bounded as the dataset grows.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from fl.lora import LoRAConfig, build_model, get_lora_weights, set_lora_weights

# ── Label spaces ─────────────────────────────────────────────────────────────

# Nurse: severity-only classifier (5 classes)
SEVERITY_LABELS: list[str] = ["discharge", "mild", "moderate", "severe", "critical"]

# Doctor: disease-only classifier. "non-infectious" = genuine worried-well visits;
# common_cold/gastroenteritis/uti/migraine = real but mild non-epidemic background
# illnesses (see simulation/complaints.py:OTHER_DISEASE_COMPLAINTS); "unknown" is
# the disease-discovery hook.
DISEASE_LABELS: list[str] = [
    "non-infectious", "influenza", "pneumonia",
    "common_cold", "gastroenteritis", "uti", "migraine",
    "unknown",
]

# Legacy combined space kept for backward compatibility with existing datasets / checkpoints
DIAGNOSTIC_LABELS: list[str] = [
    "influenza/mild",
    "influenza/moderate",
    "influenza/severe",
    "pneumonia/mild",
    "pneumonia/moderate",
    "pneumonia/severe",
    "non-infectious",
]

# Fictional disease label spaces — used when disease_glossary is set in FLTrainConfig
FICTIONAL_DISEASE_LABELS: list[str] = [
    "non-infectious", "velarex", "sornathis", "unknown",
]

# 5-class variant adds "morven" once it has been positively identified
FICTIONAL_DISEASE_5_LABELS: list[str] = [
    "non-infectious", "velarex", "sornathis", "unknown", "morven",
]

FICTIONAL_DIAGNOSTIC_LABELS: list[str] = [
    "velarex/mild",
    "velarex/moderate",
    "velarex/severe",
    "sornathis/mild",
    "sornathis/moderate",
    "sornathis/severe",
    "non-infectious",
]


def build_severity_map() -> tuple[dict[str, int], dict[int, str]]:
    label2id = {lbl: i for i, lbl in enumerate(SEVERITY_LABELS)}
    id2label  = {i: lbl for i, lbl in enumerate(SEVERITY_LABELS)}
    return label2id, id2label


def build_disease_map() -> tuple[dict[str, int], dict[int, str]]:
    label2id = {lbl: i for i, lbl in enumerate(DISEASE_LABELS)}
    id2label  = {i: lbl for i, lbl in enumerate(DISEASE_LABELS)}
    return label2id, id2label


def build_fictional_disease_map() -> tuple[dict[str, int], dict[int, str]]:
    """Disease-only map for fictional disease FL experiments."""
    label2id = {lbl: i for i, lbl in enumerate(FICTIONAL_DISEASE_LABELS)}
    id2label  = {i: lbl for i, lbl in enumerate(FICTIONAL_DISEASE_LABELS)}
    return label2id, id2label


def build_fictional_disease_5_map() -> tuple[dict[str, int], dict[int, str]]:
    """5-class map: adds 'morven' slot for the attribution phase experiment."""
    label2id = {lbl: i for i, lbl in enumerate(FICTIONAL_DISEASE_5_LABELS)}
    id2label  = {i: lbl for i, lbl in enumerate(FICTIONAL_DISEASE_5_LABELS)}
    return label2id, id2label


def build_fictional_label_map() -> tuple[dict[str, int], dict[int, str]]:
    """Combined (disease/severity) map for fictional disease static training."""
    label2id = {lbl: i for i, lbl in enumerate(FICTIONAL_DIAGNOSTIC_LABELS)}
    id2label  = {i: lbl for i, lbl in enumerate(FICTIONAL_DIAGNOSTIC_LABELS)}
    return label2id, id2label


def build_label_map() -> tuple[dict[str, int], dict[int, str]]:
    """Legacy combined map — kept for backward compat."""
    label2id = {lbl: i for i, lbl in enumerate(DIAGNOSTIC_LABELS)}
    id2label  = {i: lbl for i, lbl in enumerate(DIAGNOSTIC_LABELS)}
    return label2id, id2label


def _split_label(lbl: str) -> tuple[str, str]:
    """Returns (disease, severity). 'non-infectious' → ('non-infectious', 'none')."""
    if "/" in lbl:
        d, s = lbl.split("/", 1)
        return d, s
    return lbl, "none"


def disease_match_score(pred_label: str, true_label: str) -> float:
    if pred_label == true_label:
        return 1.0
    if _split_label(pred_label)[0] == _split_label(true_label)[0]:
        return 0.5
    return 0.0


def _normalize_item(item) -> dict:
    """
    Return a uniform dict representation whether item is a DiagnosticEvent
    or a dict record loaded from SiloDataset.

    Text field: concatenate all patient turns (not just the last one).
    Multi-turn conversations produce richer embeddings; single-turn events
    (old behaviour or background visits) are unchanged because they only
    have one patient turn.
    """
    if isinstance(item, dict):
        return item
    # Use structured case summary when available; fall back to concatenated patient turns
    case_summary = getattr(item, "case_summary", None)
    if case_summary:
        text = case_summary
    else:
        patient_turns = [t["text"] for t in item.conversation if t["role"] == "patient"]
        text = " ".join(patient_turns) if patient_turns else ""
    return {
        "text":          text,
        "label":         item.ground_truth or "",
        "gt_disease":    getattr(item, "gt_disease",  None) or "",
        "gt_severity":   getattr(item, "gt_severity", None) or "",
        "is_background": bool(getattr(item, "is_background", False)),
        "action":        item.action.name if item.action is not None else None,
        "num_turns":     len(patient_turns),
        "severity":      item.severity or 0.0,
        "days_infected": int(getattr(item, "days_infected", 0) or 0),
    }


class FLLearner:
    """
    Handles LoRA model lifecycle for one FL silo.

    Owns two models:
    - federated model  : receives global weights each round via set_weights()
    - local shadow model: never receives global weights — local-only baseline

    label_space controls which task this learner trains on:
      "severity" — triage nurse: 5-class (discharge/mild/moderate/severe/critical)
      "disease"  — diagnostic doctor: 4-class (non-infectious/influenza/pneumonia/unknown)
      "combined" — legacy 7-class combined space (backward compat)
    """

    def __init__(
        self,
        lora_config: Optional[LoRAConfig] = None,
        label_space: str = "disease",   # "severity" | "disease" | "combined"
        min_events_to_train: int = 10,
        local_epochs: int = 3,
        batch_size: int = 8,
        lr: float = 1e-4,
        # On-disk dataset (preferred). When provided, replay_buffer_size is unused.
        dataset=None,             # SiloDataset | None
        train_sample_cap: int = 512,
        # Fallback in-memory replay buffer (used when dataset=None).
        replay_buffer_size: int = 2048,
        device: Optional[str] = None,
    ):
        import torch
        from dataclasses import replace
        from transformers import AutoTokenizer
        base_cfg = lora_config or LoRAConfig()

        self.label_space = label_space
        if label_space == "severity":
            label2id, id2label = build_severity_map()
        elif label_space == "disease":
            label2id, id2label = build_disease_map()
        elif label_space == "fictional_disease":
            label2id, id2label = build_fictional_disease_map()
        elif label_space == "fictional_disease_5":
            label2id, id2label = build_fictional_disease_5_map()
        else:
            label2id, id2label = build_label_map()

        base_cfg = replace(base_cfg, num_labels=len(label2id))

        self.lora_config         = base_cfg
        self.min_events_to_train = min_events_to_train
        self.local_epochs        = local_epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.dataset             = dataset          # SiloDataset | None
        self.train_sample_cap    = train_sample_cap
        self.replay_buffer_size  = replay_buffer_size
        # Default to CPU: Ollama keeps the GPU pre-allocated (~3.7 GB on RTX 3050),
        # leaving insufficient headroom for DistilBERT training + optimizer states.
        # Pass device="cuda" explicitly only when the GPU is free.
        self.device = device if device is not None else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(base_cfg.model_name_or_path)

        self._label2id = label2id
        self._id2label = id2label

        self._model:       object = None
        self._local_model: object = None

        # In-memory fallback buffers (only used when dataset=None)
        self._replay_buffer:       list = []
        self._local_replay_buffer: list = []

        # Cached sample so fed model and local shadow train on identical data
        self._last_train_sample: Optional[list] = None

        # Frozen-silo state
        self._frozen_weights:          Optional[list] = None
        self._frozen_eval_batch:       list           = []
        self._frozen_triage_acc:       float          = 0.0
        self._frozen_local_triage_acc: float          = 0.0
        self._frozen_n:                int            = 0

        # Saved local model weights — persisted across release() / rebuild cycles
        # so the local shadow model accumulates learning over rounds, giving a
        # valid local-only baseline to compare against the federated model.
        self._saved_local_weights:     Optional[list] = None

    # ── Model access (lazy build) ─────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            self._model = build_model(self.lora_config).to(self.device)
        return self._model

    @property
    def local_model(self):
        if self._local_model is None:
            self._local_model = build_model(self.lora_config).cpu()
            if self._saved_local_weights is not None:
                set_lora_weights(self._local_model, self._saved_local_weights)
                self._saved_local_weights = None
        return self._local_model

    @property
    def frozen_n(self) -> int:
        return self._frozen_n

    # ── Weight I/O ────────────────────────────────────────────────────────────

    def get_weights(self) -> list[np.ndarray]:
        return get_lora_weights(self.model)

    def set_weights(self, weights: list[np.ndarray]) -> None:
        set_lora_weights(self.model, weights)

    def release(self) -> None:
        import gc
        self._model = None
        # Save local weights before freeing so the shadow model can resume
        # from its last trained state next round rather than from random init.
        if self._local_model is not None:
            self._saved_local_weights = get_lora_weights(self._local_model)
        self._local_model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Training ──────────────────────────────────────────────────────────────

    def _build_dataset(self, items: list) -> tuple[list[str], list[int]]:
        """
        Build (texts, label_ids) from either DiagnosticEvent objects or
        dict records loaded from SiloDataset.

        Label field selected by label_space:
          "severity"  → rec["gt_severity"]  (nurse)
          "disease"   → rec["gt_disease"]   (doctor)
          "combined"  → rec["label"]        (legacy ground_truth)
        """
        texts, labels = [], []
        field = {"severity": "gt_severity", "disease": "gt_disease"}.get(
            self.label_space, "label"
        )
        for item in items:
            rec   = _normalize_item(item)
            text  = rec.get("text") or ""
            label = self._label2id.get(rec.get(field) or "")
            if label is None or not text:
                continue
            texts.append(text)
            labels.append(label)
        return texts, labels

    @staticmethod
    def _model_device(model):
        return next(model.parameters()).device

    def _train_model(self, model, items: list) -> tuple[int, list[float]]:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset

        texts, labels = self._build_dataset(items)
        if not texts:
            return 0, []

        dev = self._model_device(model)
        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
        label_t = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], label_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=self.lr)
        epoch_losses: list[float] = []
        for _ in range(self.local_epochs):
            batch_losses = []
            for input_ids, attn_mask, batch_labels in loader:
                input_ids    = input_ids.to(dev)
                attn_mask    = attn_mask.to(dev)
                batch_labels = batch_labels.to(dev)
                optimizer.zero_grad()
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            labels=batch_labels)
                out.loss.backward()
                optimizer.step()
                batch_losses.append(out.loss.detach().item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        return len(texts), epoch_losses

    def _extend_buffer(self, buf: list, new_events: list) -> None:
        buf.extend(new_events)
        excess = len(buf) - self.replay_buffer_size
        if excess > 0:
            del buf[:excess]

    def train(self, events: list, round_num: int = 0) -> tuple[int, list[float]]:
        """
        Fine-tune the federated model.

        With a SiloDataset:
          1. Append new events to the on-disk dataset.
          2. Draw a stratified sample (up to train_sample_cap) from the training
             split — grows naturally with the dataset until the cap is hit.
          3. Cache the sample so train_local() reuses the same data.

        Without a dataset (fallback):
          Extends in-memory replay buffer and trains on the full buffer.
        """
        if self.dataset is not None:
            self.dataset.add(events, round_num)
            self._last_train_sample = self.dataset.sample_train(self.train_sample_cap)
            return self._train_model(self.model, self._last_train_sample)
        else:
            self._extend_buffer(self._replay_buffer, events)
            self._last_train_sample = list(self._replay_buffer)
            return self._train_model(self.model, self._replay_buffer)

    def init_attribution_class(self, source_idx: int, target_idx: int) -> None:
        """
        Copy source class head weights to target slot (warm-start for attribution).

        Call before train_head_only() at the attribution round so the new class
        starts from the position of the existing catch-all class rather than from
        random init. This eliminates the cold-start disadvantage for the new class.
        """
        import torch
        model = self.model
        with torch.no_grad():
            model.classifier.weight.data[target_idx] = (
                model.classifier.weight.data[source_idx].clone()
            )
            if model.classifier.bias is not None:
                model.classifier.bias.data[target_idx] = (
                    model.classifier.bias.data[source_idx].clone()
                )

    def train_head_only(self, events: list) -> tuple[int, list[float]]:
        """
        Train only the classification head; freeze backbone + LoRA adapters.

        Used in Phase 2 (attribution): the embedding is already correct
        (the backbone learned the disease representation during Phase 1).
        Only the linear head needs updating to split the "unknown" catch-all
        into a named class.
        """
        model = self.model
        # Save original requires_grad so we can restore exactly — don't blindly
        # unfreeze everything (PEFT keeps base model weights frozen via requires_grad=False
        # and get_lora_weights filters by requires_grad, so accidentally thawing base
        # weights causes FedAvg shape mismatches across silos).
        saved_grad = {name: p.requires_grad for name, p in model.named_parameters()}
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name

        try:
            result = self._train_model(model, events)
        finally:
            for name, param in model.named_parameters():
                param.requires_grad = saved_grad[name]

        return result

    def extract_embeddings(self, events: list) -> tuple[np.ndarray, list[str]]:
        """
        Extract [CLS] embeddings for events using the current backbone.
        Returns (embeddings [N, H], label_strings [N]).
        Only events with a known label and non-empty text are included.
        """
        import torch

        texts, label_ids = self._build_dataset(events)
        if not texts:
            return np.zeros((0, 768)), []

        model = self.model
        dev   = self._model_device(model)
        enc   = self._tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        enc = {k: v.to(dev) for k, v in enc.items()}

        model.eval()
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        cls_emb = out.hidden_states[-1][:, 0, :].cpu().numpy()   # [N, H]
        labels  = [self._id2label[i] for i in label_ids]
        return cls_emb, labels

    def train_local(self, events: list, round_num: int = 0) -> None:
        """
        Fine-tune the local shadow model on the same sample used by train().
        Must be called after train() so _last_train_sample is populated.
        """
        if self._last_train_sample is not None:
            self._train_model(self.local_model, self._last_train_sample)
        elif self.dataset is not None:
            sample = self.dataset.sample_train(self.train_sample_cap)
            self._train_model(self.local_model, sample)
        else:
            self._extend_buffer(self._local_replay_buffer, events)
            self._train_model(self.local_model, self._local_replay_buffer)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_model(self, model, items: list) -> dict:
        """
        Evaluate model on items (DiagnosticEvent objects or dict records).

        The prequential eval in run_round() passes raw events — model sees them
        before they are added to the dataset, giving an unbiased performance
        estimate on unseen cases.

        evaluate_holdout() passes dict records loaded from disk.
        """
        import torch
        from collections import defaultdict

        texts, labels = self._build_dataset(items)
        if not texts:
            return {
                "loss": float("nan"), "num_examples": 0,
                "triage_acc": float("nan"), "diag_acc": float("nan"),
                "danger_rate": float("nan"), "per_class": {},
                "avg_turns": float("nan"), "first_turn_rate": float("nan"),
                "action_accuracy": float("nan"), "fp_escalation_rate": float("nan"),
            }

        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
        dev     = self._model_device(model)
        label_t = torch.tensor(labels, dtype=torch.long).to(dev)

        model.eval()
        with torch.no_grad():
            out = model(
                input_ids      = enc["input_ids"].to(dev),
                attention_mask = enc["attention_mask"].to(dev),
                labels         = label_t,
            )

        preds     = out.logits.argmax(dim=-1).cpu().tolist()
        true_idxs = label_t.cpu().tolist()
        n = len(preds)

        sev_ok = diag_ok = 0
        per_class_correct: dict = defaultdict(int)
        per_class_total:   dict = defaultdict(int)
        danger = 0
        n_severe_true = 0

        for pred_idx, true_idx in zip(preds, true_idxs):
            pred_lbl = self._id2label[pred_idx]
            true_lbl = self._id2label[true_idx]

            if self.label_space == "severity":
                # Nurse: direct severity comparison
                pred_sev     = pred_lbl
                true_sev     = true_lbl
                pred_disease = "n/a"
                true_disease = "n/a"
                sev_ok  += pred_sev == true_sev
                diag_ok  = sev_ok   # mirror — no separate disease metric for nurse
            elif self.label_space == "disease":
                # Doctor: direct disease comparison
                pred_disease = pred_lbl
                true_disease = true_lbl
                pred_sev     = "n/a"
                true_sev     = "n/a"
                diag_ok += pred_disease == true_disease
                sev_ok   = diag_ok      # mirror — no separate severity metric for doctor
            else:
                # Combined (legacy): split "disease/severity"
                pred_disease, pred_sev = _split_label(pred_lbl)
                true_disease, true_sev = _split_label(true_lbl)
                sev_ok  += pred_sev     == true_sev
                diag_ok += pred_disease == true_disease

            per_class_total[true_lbl]   += 1
            per_class_correct[true_lbl] += int(pred_lbl == true_lbl)

            if true_sev in ("severe", "critical"):
                n_severe_true += 1
                if pred_sev not in ("severe", "critical") or pred_lbl in ("non-infectious", "discharge"):
                    danger += 1

        sev_to_action = {
            "mild": "RECOVER", "moderate": "RESOLVE",
            "severe": "HOSPITALISE", "critical": "HOSPITALISE",
            "discharge": "RECOVER", "none": "RECOVER",
        }

        # Rich metrics — normalise items to dicts for uniform field access
        norm_items = [_normalize_item(it) for it in items]

        turns_list = [it["num_turns"] for it in norm_items if it.get("num_turns", 0) > 0]
        action_correct = action_total = 0
        for it in norm_items:
            if not it.get("num_turns") or it.get("action") is None:
                continue
            # Pick severity from the appropriate label field
            if self.label_space == "severity":
                sev = it.get("gt_severity") or it.get("label") or ""
            elif self.label_space == "disease":
                # Doctor knows disease; action from stored gt_severity
                sev = it.get("gt_severity") or _split_label(it.get("label") or "")[1]
            else:
                sev = _split_label(it.get("label") or "")[1]
            expected = sev_to_action.get(sev, "RECOVER")
            if not expected:
                continue
            action_total   += 1
            action_correct += int(it["action"] == expected)

        avg_turns       = sum(turns_list) / len(turns_list) if turns_list else float("nan")
        first_turn_rate = sum(1 for t in turns_list if t <= 3) / len(turns_list) if turns_list else float("nan")
        action_accuracy = action_correct / action_total if action_total > 0 else float("nan")

        noise_total = noise_escalated = 0
        for it in norm_items:
            is_noise = it.get("is_background") or (
                it.get("severity", 1.0) == 0.0 and it.get("days_infected", 0) > 0
            )
            if not is_noise or it.get("action") is None or not it.get("num_turns"):
                continue
            noise_total += 1
            if it["action"] != "RECOVER":
                noise_escalated += 1
        fp_escalation_rate = noise_escalated / noise_total if noise_total > 0 else float("nan")

        return {
            "loss":               float(out.loss),
            "num_examples":       n,
            "triage_acc":         sev_ok  / n,
            "diag_acc":           diag_ok / n,
            "danger_rate":        danger / n_severe_true if n_severe_true > 0 else float("nan"),
            "per_class":          {k: per_class_correct[k] / per_class_total[k]
                                   for k in per_class_total},
            "avg_turns":          avg_turns,
            "first_turn_rate":    first_turn_rate,
            "action_accuracy":    action_accuracy,
            "fp_escalation_rate": fp_escalation_rate,
        }

    def evaluate(self, events: list) -> dict:
        return self._evaluate_model(self.model, events)

    def evaluate_local(self, events: list) -> dict:
        return self._evaluate_model(self.local_model, events)

    def evaluate_holdout(self) -> dict:
        """Evaluate federated model on the held-out test split."""
        if self.dataset is None:
            return {}
        records = self.dataset.load_holdout()
        if not records:
            return {}
        return self._evaluate_model(self.model, records)

    def evaluate_local_holdout(self) -> dict:
        """Evaluate local shadow model on the held-out test split."""
        if self.dataset is None:
            return {}
        records = self.dataset.load_holdout()
        if not records:
            return {}
        return self._evaluate_model(self.local_model, records)

    # ── Frozen-silo helpers ───────────────────────────────────────────────────

    def snapshot_for_freeze(self, events: list) -> None:
        """Called once when the world's epidemic ends. Locks in the evaluation baseline."""
        self._frozen_weights    = self.get_weights()
        self._frozen_eval_batch = list(events)
        if self._frozen_eval_batch:
            m  = self.evaluate(self._frozen_eval_batch)
            lm = self.evaluate_local(self._frozen_eval_batch)
            self._frozen_triage_acc       = m.get("triage_acc", 0.0)
            self._frozen_local_triage_acc = lm.get("triage_acc", 0.0)
        self._frozen_n = max(len(self._frozen_eval_batch), self.min_events_to_train)

    def try_accept_global(self, global_weights: list) -> bool:
        """
        Passive-observer ratchet: adopt global weights only on strict improvement.
        Done silos never contribute to FedAvg (fedavg_n=0 in the orchestrator),
        so stale weights never pollute aggregation.
        """
        if not self._frozen_eval_batch:
            set_lora_weights(self.model, global_weights)
            return True

        set_lora_weights(self.model, global_weights)
        acc = self.evaluate(self._frozen_eval_batch).get("triage_acc", float("nan"))

        if acc != acc or acc <= self._frozen_triage_acc:
            set_lora_weights(self.model, self._frozen_weights)
            return False

        self._frozen_weights    = self.get_weights()
        self._frozen_triage_acc = acc
        return True

    def frozen_metrics(self, sir) -> dict:
        nan = float("nan")
        fed   = self._frozen_triage_acc
        local = self._frozen_local_triage_acc
        gain  = fed - local if (fed == fed and local == local) else nan
        return {
            "loss": nan, "num_examples": 0,
            "triage_acc": fed, "diag_acc": nan,
            "danger_rate": nan, "per_class": {}, "avg_turns": nan,
            "first_turn_rate": nan, "action_accuracy": nan, "fp_escalation_rate": nan,
            "trained_on": 0, "trained": 0, "num_events": 0, "epoch_losses": [],
            "sir_s": sir.S, "sir_i": sir.I, "sir_r": sir.R,
            "local_triage_acc": local, "fl_gain": gain,
        }
