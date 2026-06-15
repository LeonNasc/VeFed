"""
FLSilo — one federated learning participant.

Wraps a WorldEngine (data source) and two FLLearner instances (doctor +
nurse), decoupling simulation from training.  The orchestrator in
fl/train.py sees only FLSilo, never WorldEngine directly.

Interface contract
──────────────────
  silo.run_round(round_num) -> dict     # advance sim, eval, train, return metrics
  silo.get_weights() / set_weights()    # doctor model (FedAvg)
  silo.get_nurse_weights() / set_nurse_weights()
  silo.release_model()                  # free VRAM between rounds
  silo.is_done / silo.stop_reason       # terminal state — owned here, not by world
  silo.last_round_events                # for embedding tracker, Ollama few-shot
  silo.world / silo.learner             # read-only access for external probes
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.world import WorldEngine
    from simulation.end_conditions import EndCondition
    from fl.learner import FLLearner


class FLSilo:
    """One FL participant: world + doctor learner + nurse learner + end condition."""

    def __init__(
        self,
        world:               "WorldEngine",
        learner:             "FLLearner",
        nurse_learner:       "FLLearner",
        sim_days:            int                     = 2,
        min_events_to_train: int                     = 10,
        end_condition:       Optional["EndCondition"] = None,
    ):
        self.world          = world
        self.learner        = learner
        self.nurse_learner  = nurse_learner
        self.sim_days       = sim_days
        self.min_events_to_train = min_events_to_train
        self._end_condition = end_condition

        self.is_done:           bool         = False
        self.stop_reason:       Optional[str] = None
        self.last_round_events: list          = []

    # ── Per-round training ────────────────────────────────────────────────────

    def run_round(self, round_num: int = 0) -> dict:
        """
        One FL round: advance sim → prequential eval → train → holdout eval.

        Returns a metrics dict consumed by the orchestrator for logging and
        FedAvg weight selection.  When is_done, returns a zero-event stub so
        the orchestrator can skip FedAvg contribution without branching.
        """
        if self.is_done:
            return {"num_events": 0, "trained": 0, "num_examples": 0}

        # ── 1. Advance simulation ─────────────────────────────────────────────
        events = self.world.run_sim_days(self.sim_days)
        self.last_round_events = events
        num_events = len(events)
        sir = self.world.sir_model

        # ── 2. Prequential eval (before training — unseen data) ───────────────
        fed_metrics   = self.learner.evaluate(events)
        local_metrics = self.learner.evaluate_local(events)
        nurse_metrics = self.nurse_learner.evaluate(events)

        fed_diag    = fed_metrics.get("diag_acc",   float("nan"))
        local_diag  = local_metrics.get("diag_acc", float("nan"))
        fed_triage  = nurse_metrics.get("triage_acc", float("nan"))

        local_nurse_metrics = self.nurse_learner.evaluate_local(events)
        local_triage_fed    = local_nurse_metrics.get("triage_acc", float("nan"))

        nan = float("nan")
        fl_gain      = _diff(fed_triage, local_triage_fed)
        fl_diag_gain = _diff(fed_diag,   local_diag)

        # ── 3. Train if sufficient events ─────────────────────────────────────
        if num_events >= self.min_events_to_train:
            n_trained, epoch_losses = self.learner.train(events, round_num=round_num)
            self.learner.train_local(events, round_num=round_num)
            self.nurse_learner.train(events, round_num=round_num)
            self.nurse_learner.train_local(events, round_num=round_num)
            trained = 1
        else:
            n_trained, epoch_losses = 0, []
            trained = 0

        # ── 4. Holdout eval ───────────────────────────────────────────────────
        holdout_doc   = self.learner.evaluate_holdout()
        holdout_nurse = self.nurse_learner.evaluate_holdout()
        if self.learner.dataset is not None:
            ds_train, ds_holdout = self.learner.dataset.size()
        else:
            ds_train = ds_holdout = 0

        # ── 5. Check end condition ────────────────────────────────────────────
        if self._end_condition and not self.is_done:
            if self._end_condition.check(self.world):
                self.is_done    = True
                self.stop_reason = self._end_condition.reason
                self.world.event_log.append(
                    f"Day {self.world.current_day}: Silo ended — {self.stop_reason}"
                )

        # ── 6. Build metrics dict ─────────────────────────────────────────────
        metrics = {**fed_metrics}
        metrics["triage_acc"]          = fed_triage
        metrics["trained"]             = trained
        metrics["trained_on"]          = n_trained
        metrics["num_events"]          = num_events
        metrics["num_examples"]        = n_trained
        metrics["epoch_losses"]        = epoch_losses
        metrics["sir_s"]               = sir.S
        metrics["sir_i"]               = sir.I
        metrics["sir_r"]               = sir.R
        metrics["local_triage_acc"]    = local_triage_fed
        metrics["local_diag_acc"]      = local_diag
        metrics["fl_gain"]             = fl_gain
        metrics["fl_diag_gain"]        = fl_diag_gain
        metrics["holdout_diag_acc"]    = holdout_doc.get("diag_acc",    nan)
        metrics["holdout_triage_acc"]  = holdout_nurse.get("triage_acc", nan)
        metrics["dataset_train_n"]     = ds_train
        metrics["dataset_holdout_n"]   = ds_holdout
        return metrics

    # ── Weight protocol ───────────────────────────────────────────────────────

    def get_weights(self) -> list:
        return self.learner.get_weights()

    def set_weights(self, weights) -> None:
        self.learner.set_weights(weights)

    def get_nurse_weights(self) -> list:
        return self.nurse_learner.get_weights()

    def set_nurse_weights(self, weights) -> None:
        self.nurse_learner.set_weights(weights)

    def release_model(self) -> None:
        """Free GPU/CPU memory held by both learners between rounds."""
        self.learner.release()
        if self.nurse_learner is not None:
            self.nurse_learner.release()

    def try_accept_global(self, global_weights) -> bool:
        return self.learner.try_accept_global(global_weights)

    # ── Passthrough helpers (keep train.py surface small) ─────────────────────

    @property
    def clinic_queue(self):
        return self.world.clinic_queue

    def register_diagnostic_fn(self, fn) -> None:
        self.world.register_diagnostic_fn(fn)

    def attach_interaction_logger(self, logger) -> None:
        self.world.attach_interaction_logger(logger)

    @property
    def sir_model(self):
        return self.world.sir_model

    def set_patient_llm(self, client) -> None:
        """Replace the world's data source with OllamaDataSource(client)."""
        self.world.set_patient_llm(client)

    def evaluate(self, parameters=None, config=None):
        """Flower evaluate callback shim — delegates to learner holdout eval."""
        if parameters is not None:
            self.set_weights(parameters)
        return self.learner.evaluate_holdout()


# ── Factory ───────────────────────────────────────────────────────────────────

def make_silo(
    world_config,           # simulation.world_config.WorldConfig
    fl_cfg,                 # fl.train.FLTrainConfig
    silo_idx:    int,
    dataset,                # fl.dataset.SiloDataset
    end_condition,          # EndCondition
    seed:        int,
) -> "FLSilo":
    """
    Public factory for external FL engines (Flower, FedN, custom orchestrators).

    Separates world construction from the FL orchestration loop so that
    external engines can instantiate a silo from clean config objects without
    importing fl.train internals.

    Note: fl.train._build_fl_silo() is the internal analog that handles the
    flat SiloPresetConfig → SimWorldConfig conversion.  Once that migration
    is complete, _build_fl_silo should delegate here.
    """
    from simulation.world import WorldEngine
    from fl.learner import FLLearner

    world = WorldEngine(world_config, seed=seed)

    lora_cfg = fl_cfg.lora_config()
    learner_kwargs = dict(
        min_events_to_train = fl_cfg.min_events_to_train,
        local_epochs        = fl_cfg.local_epochs,
        batch_size          = fl_cfg.batch_size,
        lr                  = fl_cfg.lr,
        dataset             = dataset,
        train_sample_cap    = fl_cfg.train_sample_cap,
        replay_buffer_size  = getattr(fl_cfg, "replay_buffer_size", 2048),
        device              = fl_cfg.training_device,
    )
    doctor = FLLearner(
        lora_config = lora_cfg,
        label_space = fl_cfg.doctor_label_space,
        **learner_kwargs,
    )
    nurse = FLLearner(
        lora_config = lora_cfg,
        label_space = "severity",
        **learner_kwargs,
    )
    return FLSilo(
        world               = world,
        learner             = doctor,
        nurse_learner       = nurse,
        sim_days            = fl_cfg.sim_days,
        min_events_to_train = fl_cfg.min_events_to_train,
        end_condition       = end_condition,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _diff(a: float, b: float) -> float:
    """Return a - b, or nan if either operand is nan."""
    return a - b if (a == a and b == b) else float("nan")
