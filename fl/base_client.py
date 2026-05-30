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

MANAGEMENT_TIERS = ("home rest", "treat", "hospitalise")

# Fixed label vocabulary spanning all progression strategies.
# Every silo uses this same set so FedAvg weight shapes always match and
# icd_category_acc is non-trivial even in single-disease runs: the model must
# learn to map symptom text to the correct disease, not just to management tier.
CANONICAL_ICD_CODES: list[str] = [
    "J09.X1",   # Aggressive Flu  — novel influenza A with pneumonia
    "J10.89",   # Persistent Flu  — identified influenza, other manifestations
    "J11.1",    # Standard Flu    — unidentified influenza, respiratory
    "J18.9",    # Deadly          — pneumonia, unspecified
    "U07.2",    # Mild Corona     — COVID-19, virus not identified
    "A41.9",    # Slow Burn       — sepsis, unspecified organism
]


def build_label_map(icd_codes: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build a label map from a list of ICD codes × 3 management tiers.

    Labels are sorted so the map is deterministic regardless of insertion order —
    critical for FedAvg: all silos must share the same label-to-index mapping.

    Returns (label2id, id2label).
    """
    labels = sorted(
        f"{icd} / {mgmt}"
        for icd in sorted(set(icd_codes))
        for mgmt in MANAGEMENT_TIERS
    )
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label  = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


def icd_match_score(pred_label: str, true_label: str) -> float:
    """
    Hierarchical ICD match score for one prediction.

    Splits the label on ' / ' to isolate the ICD part, then:
      exact subcategory match (e.g. J11.1  == J11.1 ) → 1.0
      3-char category  match  (e.g. J11    == J11.1 ) → 0.5
      no match                                         → 0.0

    The 3-char threshold mirrors clinical coding practice: codes in the same
    category represent closely related conditions (e.g. J09.X1, J09.X2 are
    both novel influenza A with different presentations).
    """
    pred_icd = pred_label.split(" / ")[0]
    true_icd = true_label.split(" / ")[0]
    if pred_icd == true_icd:
        return 1.0
    if pred_icd[:3] == true_icd[:3]:
        return 0.5
    return 0.0


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
        min_events_to_train: int = 10,
        local_epochs: int = 3,
        batch_size: int = 8,
        lr: float = 1e-4,
        device: Optional[str] = None,
    ):
        import torch
        self.world               = world
        self.sim_days            = sim_days
        self.min_events_to_train = min_events_to_train
        self.local_epochs        = local_epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._label2id, self._id2label = build_label_map(CANONICAL_ICD_CODES)

        # LoRA config must match the actual number of output classes
        base_cfg = lora_config or LoRAConfig()
        if base_cfg.num_labels != len(self._label2id):
            from dataclasses import replace
            base_cfg = replace(base_cfg, num_labels=len(self._label2id))
        self.lora_config = base_cfg

        self._model       = None  # lazy-built on first use
        self._local_model = None  # shadow local-only model (never receives global weights)
        self.last_round_events: list = []   # events from the most recent round

        # Frozen-silo state — populated when world.is_done fires
        self._frozen_weights:          Optional[list] = None
        self._frozen_eval_batch:       list           = []
        self._frozen_triage_acc:       float          = 0.0
        self._frozen_local_triage_acc: float          = 0.0
        self._frozen_n:                int            = 0   # FedAvg weight for done silo

    @property
    def model(self):
        if self._model is None:
            self._model = build_model(self.lora_config).to(self.device)
        return self._model

    @property
    def local_model(self):
        if self._local_model is None:
            # Keep on CPU to avoid VRAM pressure — with N silos there would be
            # 2N DistilBERT instances on GPU; local shadow model trains on CPU.
            self._local_model = build_model(self.lora_config).cpu()
        return self._local_model

    # ── Simulation ─────────────────────────────────────────────────────────────

    def run_simulation_round(self) -> list:
        """
        Advance the world by up to sim_days and return newly processed DiagnosticEvents.

        Exits early if world.is_done fires (end condition met). Subsequent calls
        return an empty list immediately so the training loop receives no new
        events from a finished silo without raising an error.
        """
        if self.world.is_done:
            return []
        start = len(self.world.clinic_queue.processed)
        for _ in range(self.sim_days * self.world.TICKS_PER_DAY):
            self.world.step_tick()
            if self.world.is_done:
                break
        return self.world.clinic_queue.processed[start:]

    # ── Local LoRA training ────────────────────────────────────────────────────

    def _build_dataset(self, events: list) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        for ev in events:
            label = self._label2id.get(ev.ground_truth or "")
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

    @staticmethod
    def _model_device(model):
        """Return the device of the first model parameter."""
        return next(model.parameters()).device

    def _train_model(self, model, events: list) -> tuple[int, list[float]]:
        """Fine-tune a LoRA model on events. Uses the model's own device."""
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer

        texts, labels = self._build_dataset(events)
        if not texts:
            return 0, []

        dev = self._model_device(model)
        tokenizer = AutoTokenizer.from_pretrained(self.lora_config.model_name_or_path)
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        label_t = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], label_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=self.lr
        )
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

    def train_on_events(self, events: list) -> tuple[int, list[float]]:
        """Fine-tune the federated LoRA model. Returns (num_examples, epoch_losses)."""
        return self._train_model(self.model, events)

    # ── Weight I/O ─────────────────────────────────────────────────────────────

    def get_weights(self) -> list[np.ndarray]:
        return get_lora_weights(self.model)

    def set_weights(self, weights: list[np.ndarray]) -> None:
        set_lora_weights(self.model, weights)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def _evaluate_model(self, model, events: list) -> dict:
        """
        Core evaluation logic for any LoRA model on a list of DiagnosticEvents.

        Returns
        -------
        triage_acc, diag_acc, icd_exact_acc, combined_acc,
        triage_macro_f1, triage_f1, danger_rate, per_class,
        avg_turns, first_turn_rate, action_accuracy,
        fp_escalation_rate, loss, num_examples
        """
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
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
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
        per_class_correct: dict[str, int] = defaultdict(int)
        per_class_total:   dict[str, int] = defaultdict(int)
        triage_tp: dict[str, int] = defaultdict(int)
        triage_fp: dict[str, int] = defaultdict(int)
        triage_fn: dict[str, int] = defaultdict(int)
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
        triage_f1: dict[str, dict] = {}
        for tier in tiers:
            tp = triage_tp[tier]; fp = triage_fp[tier]; fn = triage_fn[tier]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            triage_f1[tier] = {"precision": prec, "recall": rec, "f1": f1}
        macro_f1 = sum(v["f1"] for v in triage_f1.values()) / len(tiers)
        n_hosp_true = triage_tp["hospitalise"] + triage_fn["hospitalise"]

        # Doctor LLM metrics (zero when Ollama not active)
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

        # Q24: false-positive escalation rate — noise cases that doctor escalated
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

    def evaluate(self, events: Optional[list] = None) -> dict:
        """Evaluate the federated (global) LoRA model on events."""
        events = events or self.world.clinic_queue.processed
        return self._evaluate_model(self.model, events)

    # ── Memory management ──────────────────────────────────────────────────────

    def release_model(self) -> None:
        """Free both LoRA models from memory. Rebuilt automatically on next use."""
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

    # ── Frozen-silo helpers ────────────────────────────────────────────────────

    def try_accept_global(
        self, global_weights: list, threshold: float = 0.10
    ) -> bool:
        """
        For done silos: evaluate the global model on the frozen evaluation batch.
        Accept if triage_acc degradation ≤ threshold; otherwise revert to frozen weights.

        The frozen batch is the last active round's events — the best available proxy
        for whether the global model still serves this silo's patient population.
        Returns True if the global model was accepted.
        """
        if not self._frozen_eval_batch:
            return True  # nothing to protect; accept unconditionally

        set_lora_weights(self.model, global_weights)
        global_metrics = self.evaluate(self._frozen_eval_batch)
        global_acc = global_metrics.get("triage_acc", float("nan"))

        if global_acc != global_acc:  # NaN — evaluation failed
            set_lora_weights(self.model, self._frozen_weights)
            return False

        if self._frozen_triage_acc - global_acc > threshold:
            # Regression too large — revert
            set_lora_weights(self.model, self._frozen_weights)
            return False

        # Accept — update frozen reference to the new accepted weights
        self._frozen_weights    = self.get_weights()
        self._frozen_triage_acc = global_acc
        return True

    def _run_frozen_round(self) -> dict:
        """
        Called when world.is_done. No simulation, no training.
        On first call: snapshots weights + eval batch for both models.
        Subsequent calls: returns frozen metrics immediately.
        """
        if self._frozen_weights is None:
            self._frozen_weights    = self.get_weights()
            self._frozen_eval_batch = list(self.last_round_events)
            if self._frozen_eval_batch:
                m = self.evaluate(self._frozen_eval_batch)
                self._frozen_triage_acc = m.get("triage_acc", 0.0)
                lm = self._evaluate_model(self.local_model, self._frozen_eval_batch)
                self._frozen_local_triage_acc = lm.get("triage_acc", 0.0)
            self._frozen_n = max(len(self._frozen_eval_batch), self.min_events_to_train)

        nan = float("nan")
        sir = self.world.sir_model
        local_acc = self._frozen_local_triage_acc
        fed_acc   = self._frozen_triage_acc
        fl_gain   = fed_acc - local_acc if (fed_acc == fed_acc and local_acc == local_acc) else nan
        return {
            "loss":               nan,
            "num_examples":       0,
            "triage_acc":         fed_acc,
            "diag_acc":           nan,
            "icd_exact_acc":      nan,
            "combined_acc":       nan,
            "triage_macro_f1":    nan,
            "triage_f1":          {},
            "danger_rate":        nan,
            "per_class":          {},
            "avg_turns":          nan,
            "first_turn_rate":    nan,
            "action_accuracy":    nan,
            "fp_escalation_rate": nan,
            "trained_on":         0,
            "trained":            0,
            "num_events":         0,
            "epoch_losses":       [],
            "sir_s":              sir.S,
            "sir_i":              sir.I,
            "sir_r":              sir.R,
            "local_triage_acc":   local_acc,
            "fl_gain":            fl_gain,
        }

    # ── Convenience ────────────────────────────────────────────────────────────

    def run_round(self) -> dict:
        """
        One complete FL round: simulate → evaluate (global + local) → train both → metrics.

        Evaluate-before-train gives generalisation accuracy on unseen cases.
        The local shadow model trains on the same events but never receives global
        weights — its triage_acc is the local-only baseline for measuring FL gain.
        """
        if self.world.is_done:
            return self._run_frozen_round()

        events     = self.run_simulation_round()
        self.last_round_events = events
        num_events = len(events)
        sir        = self.world.sir_model

        # Evaluate global (federated) model BEFORE training
        metrics = self.evaluate(events)

        # Evaluate local shadow model BEFORE its training (fair comparison)
        local_metrics = self._evaluate_model(self.local_model, events)
        local_triage_acc = local_metrics.get("triage_acc", float("nan"))

        if num_events >= self.min_events_to_train:
            n, epoch_losses = self.train_on_events(events)
            self._train_model(self.local_model, events)
            trained = 1
        else:
            n, epoch_losses = 0, []
            trained = 0

        nan = float("nan")
        fed_acc = metrics.get("triage_acc", nan)
        fl_gain = fed_acc - local_triage_acc if (fed_acc == fed_acc and local_triage_acc == local_triage_acc) else nan

        metrics.update(
            trained_on       = n,
            trained          = trained,
            num_events       = num_events,
            epoch_losses     = epoch_losses,
            sir_s            = sir.S,
            sir_i            = sir.I,
            sir_r            = sir.R,
            local_triage_acc = local_triage_acc,
            fl_gain          = fl_gain,
        )
        return metrics
