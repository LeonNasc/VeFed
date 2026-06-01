"""
FLLearner — LoRA training, evaluation, and weight management for one FL silo.

Separated from simulation logic so WorldEngine can delegate to it without
importing PyTorch at module level.  Created lazily when WorldEngine.run_round()
is first called on an FL-enabled world.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from fl.lora import LoRAConfig, build_model, get_lora_weights, set_lora_weights

MANAGEMENT_TIERS = ("home rest", "treat", "hospitalise")

CANONICAL_ICD_CODES: list[str] = [
    # Infectious (circulate via SIR)
    "J11.1",    # Standard Flu
    "U07.2",    # Mild Corona
    "A41.9",    # Slow Burn / Sepsis
    # Non-infectious (background clinic visits)
    "M54.5",    # Low back pain
    "R51",      # Headache
    "F41.1",    # Generalised anxiety disorder
    "Z87.39",   # Hypertension follow-up
    "R53.83",   # Fatigue, non-specific
]


def build_label_map(icd_codes: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(
        f"{icd} / {mgmt}"
        for icd in sorted(set(icd_codes))
        for mgmt in MANAGEMENT_TIERS
    )
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label  = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


def icd_match_score(pred_label: str, true_label: str) -> float:
    pred_icd = pred_label.split(" / ")[0]
    true_icd = true_label.split(" / ")[0]
    if pred_icd == true_icd:
        return 1.0
    if pred_icd[:3] == true_icd[:3]:
        return 0.5
    return 0.0


class FLLearner:
    """
    Handles LoRA model lifecycle for one FL silo.

    Owns two models:
    - federated model  : receives global weights each round via set_weights()
    - local shadow model: never receives global weights — local-only baseline

    Both train on the same accumulated replay buffer.
    """

    def __init__(
        self,
        lora_config: Optional[LoRAConfig] = None,
        min_events_to_train: int = 10,
        local_epochs: int = 3,
        batch_size: int = 8,
        lr: float = 1e-4,
        replay_buffer_size: int = 2048,
        device: Optional[str] = None,
    ):
        import torch
        base_cfg = lora_config or LoRAConfig()
        label2id, id2label = build_label_map(CANONICAL_ICD_CODES)
        if base_cfg.num_labels != len(label2id):
            from dataclasses import replace
            base_cfg = replace(base_cfg, num_labels=len(label2id))

        self.lora_config         = base_cfg
        self.min_events_to_train = min_events_to_train
        self.local_epochs        = local_epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.replay_buffer_size  = replay_buffer_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._label2id = label2id
        self._id2label = id2label

        self._model:       object = None
        self._local_model: object = None

        self._replay_buffer:       list = []
        self._local_replay_buffer: list = []

        # Frozen-silo state
        self._frozen_weights:          Optional[list] = None
        self._frozen_eval_batch:       list           = []
        self._frozen_triage_acc:       float          = 0.0
        self._frozen_local_triage_acc: float          = 0.0
        self._frozen_n:                int            = 0

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
        self._model       = None
        self._local_model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Training ──────────────────────────────────────────────────────────────

    def _build_dataset(self, events: list) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        for ev in events:
            label = self._label2id.get(ev.ground_truth or "")
            if label is None:
                continue
            patient_turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
            if not patient_turns:
                continue
            texts.append(patient_turns[-1])
            labels.append(label)
        return texts, labels

    @staticmethod
    def _model_device(model):
        return next(model.parameters()).device

    def _train_model(self, model, events: list) -> tuple[int, list[float]]:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer

        texts, labels = self._build_dataset(events)
        if not texts:
            return 0, []

        dev = self._model_device(model)
        tokenizer = AutoTokenizer.from_pretrained(self.lora_config.model_name_or_path)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
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
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=batch_labels)
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

    def train(self, events: list) -> tuple[int, list[float]]:
        """Fine-tune federated model on accumulated replay buffer."""
        self._extend_buffer(self._replay_buffer, events)
        return self._train_model(self.model, self._replay_buffer)

    def train_local(self, events: list) -> None:
        """Fine-tune local shadow model on its own replay buffer."""
        self._extend_buffer(self._local_replay_buffer, events)
        self._train_model(self.local_model, self._local_replay_buffer)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_model(self, model, events: list) -> dict:
        import torch
        from collections import defaultdict
        from transformers import AutoTokenizer

        texts, labels = self._build_dataset(events)
        if not texts:
            return {
                "loss": float("nan"), "num_examples": 0,
                "triage_acc": float("nan"), "diag_acc": float("nan"),
                "icd_exact_acc": float("nan"), "combined_acc": float("nan"),
                "triage_macro_f1": float("nan"), "triage_f1": {},
                "danger_rate": float("nan"), "per_class": {},
                "avg_turns": float("nan"), "first_turn_rate": float("nan"),
                "action_accuracy": float("nan"), "fp_escalation_rate": float("nan"),
            }

        tokenizer = AutoTokenizer.from_pretrained(self.lora_config.model_name_or_path)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        dev = self._model_device(model)
        label_t = torch.tensor(labels, dtype=torch.long).to(dev)

        model.eval()
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(dev),
                attention_mask=enc["attention_mask"].to(dev),
                labels=label_t,
            )

        preds     = out.logits.argmax(dim=-1).cpu().tolist()
        true_idxs = label_t.cpu().tolist()
        n = len(preds)

        icd_exact = icd_cat = mgmt_ok = combined = 0
        per_class_correct: dict = defaultdict(int)
        per_class_total:   dict = defaultdict(int)
        triage_tp: dict = defaultdict(int)
        triage_fp: dict = defaultdict(int)
        triage_fn: dict = defaultdict(int)
        danger = 0

        for pred_idx, true_idx in zip(preds, true_idxs):
            pred_lbl = self._id2label[pred_idx]
            true_lbl = self._id2label[true_idx]
            pred_icd, pred_mgmt = pred_lbl.rsplit(" / ", 1)
            true_icd, true_mgmt = true_lbl.rsplit(" / ", 1)

            exact = pred_icd == true_icd
            cat   = pred_icd[:3] == true_icd[:3]
            mgmt  = pred_mgmt == true_mgmt

            icd_exact += exact
            icd_cat   += cat
            mgmt_ok   += mgmt
            combined  += (cat and mgmt)

            per_class_total[true_lbl]   += 1
            per_class_correct[true_lbl] += int(cat and mgmt)

            if mgmt:
                triage_tp[true_mgmt] += 1
            else:
                triage_fp[pred_mgmt] += 1
                triage_fn[true_mgmt] += 1
                if true_mgmt == "hospitalise" and pred_mgmt == "home rest":
                    danger += 1

        tiers = ("home rest", "treat", "hospitalise")
        triage_f1: dict = {}
        for tier in tiers:
            tp = triage_tp[tier]; fp = triage_fp[tier]; fn = triage_fn[tier]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            triage_f1[tier] = {"precision": prec, "recall": rec, "f1": f1}
        macro_f1 = sum(v["f1"] for v in triage_f1.values()) / len(tiers)
        n_hosp_true = triage_tp["hospitalise"] + triage_fn["hospitalise"]

        gt_to_action = {"home rest": "RECOVER", "treat": "RESOLVE", "hospitalise": "HOSPITALISE"}
        turns_list = [ev.num_turns for ev in events if ev.num_turns > 0]
        action_correct = action_total = 0
        for ev in events:
            if ev.num_turns == 0 or not ev.ground_truth or ev.action is None:
                continue
            expected = gt_to_action.get(ev.ground_truth.rsplit(" / ", 1)[-1])
            if expected is not None:
                action_total   += 1
                action_correct += int(ev.action.name == expected)

        avg_turns       = sum(turns_list) / len(turns_list) if turns_list else float("nan")
        first_turn_rate = sum(1 for t in turns_list if t <= 3) / len(turns_list) if turns_list else float("nan")
        action_accuracy = action_correct / action_total if action_total > 0 else float("nan")

        noise_total = noise_escalated = 0
        for ev in events:
            is_noise = ev.is_background or (ev.severity == 0.0 and ev.days_infected > 0)
            if not is_noise or ev.action is None or ev.num_turns == 0:
                continue
            noise_total += 1
            if ev.action.name != "RECOVER":
                noise_escalated += 1
        fp_escalation_rate = noise_escalated / noise_total if noise_total > 0 else float("nan")

        return {
            "loss":               float(out.loss),
            "num_examples":       n,
            "triage_acc":         mgmt_ok   / n,
            "diag_acc":           icd_cat   / n,
            "icd_exact_acc":      icd_exact / n,
            "combined_acc":       combined  / n,
            "triage_macro_f1":    macro_f1,
            "triage_f1":          triage_f1,
            "danger_rate":        danger / n_hosp_true if n_hosp_true > 0 else float("nan"),
            "per_class":          {k: per_class_correct[k] / per_class_total[k] for k in per_class_total},
            "avg_turns":          avg_turns,
            "first_turn_rate":    first_turn_rate,
            "action_accuracy":    action_accuracy,
            "fp_escalation_rate": fp_escalation_rate,
        }

    def evaluate(self, events: list) -> dict:
        return self._evaluate_model(self.model, events)

    def evaluate_local(self, events: list) -> dict:
        return self._evaluate_model(self.local_model, events)

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
            "triage_acc": fed, "diag_acc": nan, "icd_exact_acc": nan,
            "combined_acc": nan, "triage_macro_f1": nan, "triage_f1": {},
            "danger_rate": nan, "per_class": {}, "avg_turns": nan,
            "first_turn_rate": nan, "action_accuracy": nan, "fp_escalation_rate": nan,
            "trained_on": 0, "trained": 0, "num_events": 0, "epoch_losses": [],
            "sir_s": sir.S, "sir_i": sir.I, "sir_r": sir.R,
            "local_triage_acc": local, "fl_gain": gain,
        }
