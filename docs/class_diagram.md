# Class Diagram — Federated Simulated World

**Last updated: 2026-05-30**

Render with any Mermaid-compatible viewer (GitHub, VS Code Mermaid Preview, mermaid.live).

Three diagrams are provided:
- **A** — Simulation core (world engine, agents, disease model, complaint types, LLM layer)
- **B** — Federated learning layer (FL clients, centralized baseline, aggregation, end conditions)
- **C** — Visualisation & reporting layer

---

## Diagram A — Simulation Core

```mermaid
classDiagram
    direction TB

    %% ── World Engine ─────────────────────────────────────────────────────────
    class WorldEngine {
        +agents : list[Agent]
        +locations : dict[str, Location]
        +sir_model : SIRModel
        +clinic_queue : ClinicQueue
        +strategy : DiseaseStrategy
        +progressions : list[DiseaseProgressionStrategy]
        +end_condition : EndCondition
        +background_visit_rate : float
        +disease_weights : list[float]
        Note: Dirichlet-sampled per silo
        +current_day : int
        +is_done : bool
        +stop_reason : str
        +step_tick()
        +inject_disease_cloud(location_id, intensity)
        +collect_end_of_day_cases() list
        -_collect_background_cases() list
        Note: returns NonInfectiousComplaint visits
        -_generate_opening(agent, event) str
        Note: routes to PatientLLMClient or templates
        +run_fl_round()
    }

    %% ── Agent ────────────────────────────────────────────────────────────────
    class Agent {
        +id : str
        +name : str
        +health_state : HealthState
        +personality : Personality
        +severity_threshold : float
        +hospitalised : bool
        +inner_state : InnerState
        +should_seek_care() bool
        +opening_statement() str
        +followup_answer(question) str
        +build_diagnostic_event() DiagnosticEvent
        +apply_daily_infection(progression, disease_weights) bool
        Note: uses rng.choices() with Dirichlet weights
    }

    class HealthState {
        +status : HealthStatus
        +severity : float
        +symptoms : float
        +days_infected : int
        +infect(trajectory)
        +step()
        +apply_treatment(decay_boost)
        +sir_compartment : str
    }

    class InnerState {
        +severity : float
        +symptoms : float
        +days_infected : int
        +trend : str
        +fatigue : float
        +pain : float
        +mood : float
        +top_vital : tuple
    }

    %% ── Disease Trajectory ───────────────────────────────────────────────────
    class DiseaseTrajectory {
        +peak_severity : float
        +rise_days : float
        +decay_rate : float
        +symptom_sensitivity : float
        +severity_floor_weight : float
        +disease_name : str
        +icd_code : str
        +incubation_days : int
        +vital_slopes : dict
        +trend : str
        +cleared : bool
        +step() tuple[float,float]
        +apply_treatment(decay_boost)
    }

    %% ── Non-Infectious Complaints ────────────────────────────────────────────
    class NonInfectiousComplaint {
        <<dataclass, frozen>>
        +name : str
        +icd_code : str
        +management : str
        Note: home rest or treat
        +prompt_context : str
        Note: injected into patient LLM system prompt
    }

    class NON_INFECTIOUS_COMPLAINTS {
        <<registry>>
        M54.5 — Low back pain
        R51   — Headache
        F41.1 — Generalised anxiety
        Z87.39 — Hypertension follow-up
        R53.83 — Fatigue
    }

    %% ── Case Table ───────────────────────────────────────────────────────────
    class CaseTable {
        +variables : list[str]
        +get(variable, day) float
        +band(variable, day) str
    }
    class MimicCaseTable {
        +_load_from_record(record) dict
    }

    %% ── LLM Layer ────────────────────────────────────────────────────────────
    class PatientLLMClient {
        +model : str
        Note: tinyllama (frozen, never fine-tuned)
        +opening_statement(severity, days, personality) str
        +complaint_opening(prompt_context, personality) str
        Note: for non-infectious complaint visits
        +followup_answer(question, severity, personality, case_table, day) str
        +health_check() bool
    }

    class OllamaDiagnosticClient {
        +model : str
        Note: phi3:mini
        +diagnose(event, _queue) DiagnosticEvent
        +update_examples(round_events)
        Note: federated few-shot bank
        +health_check() bool
    }

    %% ── Symptom Narration ────────────────────────────────────────────────────
    class SymptomNarrator {
        +full_opening_statement(inner_state, days, personality) str
        +followup_answer(symptoms, personality, question, case_table, day) str
    }

    class Personality {
        <<enumeration>>
        STOIC
        NEUTRAL
        ANXIOUS
    }

    %% ── Diagnostic Pipeline ──────────────────────────────────────────────────
    class DiagnosticEvent {
        +agent_id : str
        +severity : float
        +symptoms : float
        +days_infected : int
        +ground_truth : str
        Note: ICD-code / management-tier
        Note: e.g. A41.9 / treat
        +conversation : list
        +case_table : CaseTable
        +action : DiagnosticAction
        +is_background : bool
        Note: True for non-infectious visits
        +complaint_context : str
        Note: prompt hint for patient LLM
        +num_turns : int
    }

    class ClinicQueue {
        +patients : deque
        +processed : list
        +enqueue(event)
        +process_all(diagnostic_fn) list
        +apply_outcomes(results, agents, hospital_id) list
    }

    class SIRModel {
        +S : int
        +I : int
        +R : int
        +beta : float
        +gamma : float
        +history : list
        +step(agents)
    }

    %% ── Disease Strategies ───────────────────────────────────────────────────
    class DiseaseStrategy {
        <<abstract>>
        +name : str
        +transmission_rate(location_type) float
    }
    class StandardFluStrategy
    class MildCoronaStrategy
    class SlowBurnStrategy

    class DiseaseProgressionStrategy {
        <<abstract>>
        +name : str
        +icd_code : str
        +sample_trajectory(rng) DiseaseTrajectory
    }
    class StandardFluProgression {
        icd_code = J11.1
        incubation = 2d
    }
    class MildCoronaProgression {
        icd_code = U07.2
        incubation = 4d
        silent hypoxia pattern
    }
    class SlowBurnProgression {
        icd_code = A41.9
        incubation = 4d
        floor_weight = 0.15
    }
    class MimicProgression {
        icd_code per diagnosis_group
    }
    note for MimicProgression "AggressiveFlu (J09.X1), PersistentFlu (J10.89)\nand Deadly (J18.9) exist as classes but are\nexcluded from PROGRESSION_STRATEGIES registry.\nThe active registry contains: StandardFlu,\nMildCorona, SlowBurn, MIMIC."

    %% ── Relationships ────────────────────────────────────────────────────────
    WorldEngine "1" *-- "N" Agent
    WorldEngine "1" *-- "1" SIRModel
    WorldEngine "1" *-- "1" ClinicQueue
    WorldEngine --> DiseaseStrategy
    WorldEngine "1" --> "1..*" DiseaseProgressionStrategy
    WorldEngine ..> NonInfectiousComplaint : samples randomly

    Agent "1" *-- "1" HealthState
    Agent "1" ..> "1" InnerState : computes
    Agent "1" *-- "1" CaseTable
    Agent "1" *-- "1" SymptomNarrator

    HealthState --> DiseaseTrajectory

    ClinicQueue "1" o-- "N" DiagnosticEvent
    DiagnosticEvent --> CaseTable

    CaseTable <|-- MimicCaseTable

    DiseaseStrategy <|-- StandardFluStrategy
    DiseaseStrategy <|-- MildCoronaStrategy
    DiseaseStrategy <|-- SlowBurnStrategy

    DiseaseProgressionStrategy <|-- StandardFluProgression
    DiseaseProgressionStrategy <|-- MildCoronaProgression
    DiseaseProgressionStrategy <|-- SlowBurnProgression
    DiseaseProgressionStrategy <|-- MimicProgression

    NON_INFECTIOUS_COMPLAINTS "1" o-- "5" NonInfectiousComplaint

    SymptomNarrator ..> Personality
    Agent ..> Personality

    WorldEngine ..> PatientLLMClient : optional, wired at startup
    WorldEngine ..> OllamaDiagnosticClient : via register_diagnostic_fn()
    PatientLLMClient ..> NonInfectiousComplaint : uses prompt_context
```

---

## Diagram B — Federated Learning Layer

```mermaid
classDiagram
    direction TB

    %% ── End Conditions ───────────────────────────────────────────────────────
    class EndCondition {
        <<abstract>>
        +reason : str
        +check(world) bool
        +reset()
    }
    class HorizonCondition { +max_days : int }
    class ExtinctionCondition { +consecutive_days : int }
    class NoSusceptiblesCondition
    class TrainingBudgetCondition { +target_events : int }
    class AnyCondition { +conditions : list[EndCondition] }

    EndCondition <|-- HorizonCondition
    EndCondition <|-- ExtinctionCondition
    EndCondition <|-- NoSusceptiblesCondition
    EndCondition <|-- TrainingBudgetCondition
    EndCondition <|-- AnyCondition
    AnyCondition o-- EndCondition

    %% ── Config ───────────────────────────────────────────────────────────────
    class WorldConfig {
        +num_agents : int
        +progressions : list[str]
        +disease_strategy : str
        +beta_scale : float
        +background_visit_rate : float
        +end_condition : str
        +seed_offset : int
        +initial_seeds : int
    }

    class FLTrainConfig {
        +num_silos : int
        +max_rounds : int
        Note: simulation-guided; safety cap
        +num_agents : int
        +sim_days : int
        +local_epochs : int
        Note: default 10
        +lr : float
        +lora_rank : int
        +dirichlet_alpha : float
        Note: default 0.3; controls non-IID degree
        +replay_buffer_size : int
        Note: default 2048
        +progressions : list[str]
        +end_condition : str
        +world_configs : list[WorldConfig]
        Note: per-silo non-IID override
        +use_ollama : bool
        +wandb_project : str
        +lora_config() LoRAConfig
    }

    %% ── Label Space ──────────────────────────────────────────────────────────
    class LabelMap {
        <<utility>>
        CANONICAL_ICD_CODES : 8 codes
        Note: 3 infectious (J11.1 / U07.2 / A41.9)
        Note: 5 non-infectious (M54.5 / R51 / F41.1 / Z87.39 / R53.83)
        24 classes = 8 ICD × 3 management tiers
        build_label_map(icd_codes)
        icd_match_score(pred, true) float
        MANAGEMENT_TIERS = home rest / treat / hospitalise
    }

    class LoRAConfig {
        +model_name_or_path : str
        Note: distilbert-base-uncased
        +rank : int
        +lora_alpha : float
        +lora_dropout : float
        +target_modules : list[str]
        Note: q_lin, v_lin
        +num_labels : int
        Note: 24 (overridden dynamically)
    }

    %% ── FL Client ────────────────────────────────────────────────────────────
    class WorldFLClient {
        +world : WorldEngine
        +lora_config : LoRAConfig
        +sim_days : int
        +local_epochs : int
        +replay_buffer_size : int
        +_replay_buffer : list
        Note: accumulates events across rounds
        +_local_replay_buffer : list
        Note: shadow model's own buffer
        +_label2id : dict
        +_id2label : dict
        +run_simulation_round() list
        +train_on_events(events) tuple
        Note: extends buffer then trains on full buffer
        +train_local_on_events(events)
        +evaluate(events) dict
        Note: triage_acc, diag_acc, danger_rate, fl_gain
        +run_round() dict
        +get_weights() list[ndarray]
        +set_weights(weights)
        +release_model()
        Note: called after each round to free VRAM
        Note: peak VRAM = 1 model regardless of N silos
        +try_accept_global(global_weights, threshold) bool
    }

    %% ── Centralized Baseline ─────────────────────────────────────────────────
    class CentralizedTrainer {
        +silos : list[WorldFLClient]
        Note: worlds used for data only
        +local_epochs : int
        +_pool_buffer : list
        Note: all events from all silos pooled
        +run_round() CentralizedRoundMetrics
        Note: advance all worlds → pool → train one model
        +get_weights() list[ndarray]
        +release_model()
    }

    class CentralizedRoundMetrics {
        +round_num : int
        +loss : float
        +triage_acc : float
        +diag_acc : float
        +danger_rate : float
        +num_events : int
        +trained_on : int
        +all_done : bool
    }

    %% ── Aggregation ──────────────────────────────────────────────────────────
    class FedAvg {
        <<utility>>
        _fedavg(weights_list, n_examples) list[ndarray]
        Note: weighted mean of LoRA adapter arrays
    }

    %% ── MIMIC Layer ──────────────────────────────────────────────────────────
    class MimicDatabase {
        <<abstract>>
        +cohort_ids(diagnosis_group) list[int]
        +get_record(subject_id) MimicRecord
        +random_subject(diagnosis_group, rng) MimicRecord
    }
    class MockMimicDatabase {
        150 synthetic ICU patients
        groups: sepsis / pneumonia / heart_failure
    }

    MimicDatabase <|-- MockMimicDatabase

    %% ── Flower / W&B ─────────────────────────────────────────────────────────
    class WandBFedAvg {
        +wandb_run : Run
        +aggregate_fit(round, results, failures)
    }

    %% ── Relationships ────────────────────────────────────────────────────────
    WorldFLClient --> WorldEngine
    WorldFLClient --> LoRAConfig
    CentralizedTrainer "1" o-- "N" WorldFLClient
    CentralizedTrainer --> CentralizedRoundMetrics
    FLTrainConfig --> LoRAConfig
    FLTrainConfig "1" o-- "N" WorldConfig
    FLTrainConfig ..> EndCondition : builds via from_config()
    WorldEngine --> EndCondition
    WorldFLClient ..> LabelMap
    CentralizedTrainer ..> LabelMap
```

---

## Key Design Patterns

| Pattern | Where used |
|---|---|
| **Strategy** | `DiseaseStrategy`, `DiseaseProgressionStrategy` — swap disease behaviour at runtime |
| **Template Method** | `DiseaseProgressionStrategy.sample_trajectory()` → `_sample()` helper |
| **Observer** | `SIRModel.step(agents)` — recomputes S/I/R from agent states each day |
| **Factory** | `end_conditions.from_config(name, param)` — builds EndCondition from string |
| **Adapter** | `MimicDiseaseTrajectory` — adapts `MimicRecord` to `DiseaseTrajectory` interface |
| **Decorator** | `WandBFedAvg` — extends `FedAvg` with W&B callbacks |
| **Null Object** | `step_tick()` no-op when `is_done=True` |
| **Object Pool** | `release_model()` + lazy rebuild — VRAM scheduling; peak = 1 model at a time |
| **Registry** | `PROGRESSION_STRATEGIES`, `NON_INFECTIOUS_COMPLAINTS` — lookup by name |
