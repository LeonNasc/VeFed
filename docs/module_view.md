# Module View — Federated Simulated World

**Last updated: 2026-05-29**

Render with any Mermaid-compatible viewer (GitHub, VS Code Mermaid Preview, mermaid.live).

Two views are provided:
- **A** — Package dependency graph (what imports what)
- **B** — Data flow (how information moves from simulation to W&B)

---

## View A — Package Structure & Dependencies

```mermaid
graph TD
    subgraph simulation["simulation/"]
        WE["world.py\nWorldEngine · ClinicQueue\nDiseaseCloud · Location"]
        MOD["models.py\nAgent · HealthState · InnerState\nDiagnosticEvent · SIRModel"]
        PROG["progression.py\nDiseaseProgressionStrategy\nDiseaseTrajectory\n+ 5 concrete strategies"]
        STRAT["strategies.py\nDiseaseStrategy\n+ 4 concrete strategies"]
        SYMP["symptom_language.py\nSymptomNarrator · Personality\nTREND_ADDONS · VITAL_ADDONS"]
        CT["case_table.py\nCaseTable · MimicCaseTable\nNORMAL_RANGES · SEVERITY_SLOPES"]
        EC["end_conditions.py\nEndCondition ABC\nHorizon · Extinction\nNoSusceptibles · Budget\nAnyCondition"]
        MIMI["mimic_db.py\nMimicDatabase ABC\nMockMimicDatabase\nMimicRecord\nMimicDiseaseTrajectory"]
        OLL["ollama_client.py\nOllamaDiagnosticClient"]
        ILOG["interaction_log.py\nInteractionLogger"]
    end

    subgraph fl["fl/"]
        BASE["base_client.py\nWorldFLClient"]
        LORA["lora.py\nLoRAConfig · build_model\nget/set_lora_weights"]
        TRAIN["train.py\nFLTrainConfig\nrun_federated_training\n_fedavg"]
        SERVER["server.py\nWandBFedAvg\nrun_flower_server"]
        FLWR["flwr_client.py\nmake_flower_client\nstart_flower_client"]
        FEDN["fedn_client.py\nmake_fedn_callbacks\nrun_fedn_client"]
    end

    subgraph viz["viz/"]
        DVIZ["disease_viz.py\nplot_progression_curves\ngenerate_disease_atlas\nsave_embedding_plot"]
    end

    subgraph ui["ui/"]
        TUI["tui.py\nTUI"]
        RTUI["rogue_tui.py\nRogueTUI"]
        PYG["pygame_ui.py\nPygameUI"]
    end

    MAIN["main.py\nlauncher"]

    %% simulation internal deps
    WE --> MOD
    WE --> PROG
    WE --> STRAT
    WE --> EC
    MOD --> SYMP
    MOD --> CT
    PROG --> MIMI
    CT --> MIMI

    %% fl deps
    BASE --> WE
    BASE --> LORA
    TRAIN --> BASE
    TRAIN --> WE
    TRAIN --> PROG
    TRAIN --> EC
    SERVER --> LORA
    FLWR --> BASE
    FEDN --> BASE
    FEDN --> LORA

    %% viz deps
    DVIZ --> PROG
    DVIZ --> WE

    %% ui deps
    TUI --> WE
    RTUI --> WE
    PYG --> WE

    %% main deps
    MAIN --> WE
    MAIN --> OLL
    MAIN --> ILOG
    MAIN --> TUI

    %% external
    OLL["ollama_client.py"] -->|"HTTP/JSON"| OLLAMA[("Ollama\nphi3:mini")]
    SERVER -->|"Flower protocol"| FLOWER[("Flower\nserver")]
    TRAIN -->|"wandb.log()"| WANDB[("Weights &\nBiases")]
    SERVER -->|"wandb.log()"| WANDB

    style simulation fill:#dbeafe,stroke:#3b82f6
    style fl fill:#dcfce7,stroke:#22c55e
    style viz fill:#fef9c3,stroke:#eab308
    style ui fill:#fce7f3,stroke:#ec4899
    style OLLAMA fill:#f3f4f6,stroke:#9ca3af
    style FLOWER fill:#f3f4f6,stroke:#9ca3af
    style WANDB fill:#f3f4f6,stroke:#9ca3af
```

---

## View B — Data Flow: Simulation → Training → Aggregation

```mermaid
flowchart LR
    subgraph SIM["Simulation Layer"]
        direction TB
        WORLD["WorldEngine\n(1 per silo)"]
        AGENTS["N Agents\n(SIR dynamics)"]
        PROG2["Disease\nProgression\nStrategy"]
        EC2["End\nCondition"]
        WORLD --> AGENTS
        WORLD --> PROG2
        WORLD --> EC2
    end

    subgraph DIAG["Diagnosis Layer"]
        direction TB
        CQ["ClinicQueue"]
        IS["InnerState\n(severity · trend\nvitals · mood)"]
        OS["opening_statement()\n(personality-filtered\nnatural language)"]
        LLM["LLM Doctor\n(phi3:mini / Ollama)\nor rule-based oracle"]
        DE["DiagnosticEvent\n(symptom_text\nground_truth\nconversation)"]
        ILOG2["InteractionLogger\n(.jsonl)"]
        CQ --> IS --> OS --> LLM --> DE
        DE --> ILOG2
    end

    subgraph FL["Federated Learning Layer"]
        direction TB
        CLIENT["WorldFLClient\n(per silo)"]
        DATASET["Local dataset\n(symptom_text → label)\nfrom DiagnosticEvents"]
        FINETUNE["LoRA fine-tune\nDistilBERT\n(local_epochs)"]
        WEIGHTS["LoRA adapter\nweights\n(q_lin, v_lin only)"]
        CLIENT --> DATASET --> FINETUNE --> WEIGHTS
    end

    subgraph AGG["Aggregation"]
        direction TB
        FEDAVG["FedAvg\n_fedavg() / WandBFedAvg\nweighted mean of\nadapter weights"]
        GLOBAL["Global model\n(updated after\neach round)"]
        FEDAVG --> GLOBAL
    end

    subgraph LOG["Experiment Tracking"]
        direction TB
        WB["Weights & Biases\nsilo_N/loss · accuracy\nsilo_N/sir_s/i/r\naggregated/loss\naggregated/accuracy"]
    end

    %% Data flow between layers
    WORLD -->|"step_tick() × sim_days"| CQ
    AGENTS -->|"should_seek_care()"| CQ
    DE -->|"events list"| CLIENT
    WEIGHTS -->|"N silos"| FEDAVG
    GLOBAL -->|"set_weights() before\nnext round"| CLIENT
    CLIENT -->|"metrics dict\nloss · acc · SIR"| LOG
    FEDAVG -->|"aggregated metrics"| LOG

    style SIM fill:#dbeafe,stroke:#3b82f6
    style DIAG fill:#fce7f3,stroke:#ec4899
    style FL fill:#dcfce7,stroke:#22c55e
    style AGG fill:#fef9c3,stroke:#eab308
    style LOG fill:#f3e8ff,stroke:#a855f7
```

---

## View C — FL Round Sequence

```mermaid
sequenceDiagram
    participant Loop as run_federated_training()
    participant S0 as Silo 0 (WorldFLClient)
    participant S1 as Silo 1 (WorldFLClient)
    participant SN as Silo N-1
    participant WB as Weights & Biases

    Note over Loop: Round r = 1..num_rounds

    Loop->>S0: set_weights(global_weights)
    Loop->>S1: set_weights(global_weights)
    Loop->>SN: set_weights(global_weights)

    par simulate + train (per silo)
        S0->>S0: run_simulation_round() — sim_days ticks
        S0->>S0: train_on_events() — local LoRA fine-tune
        S0->>S0: evaluate() — post-train loss + accuracy
    and
        S1->>S1: run_simulation_round()
        S1->>S1: train_on_events()
        S1->>S1: evaluate()
    and
        SN->>SN: run_simulation_round()
        SN->>SN: train_on_events()
        SN->>SN: evaluate()
    end

    S0-->>Loop: weights_0, metrics_0 (loss, acc, SIR, num_events)
    S1-->>Loop: weights_1, metrics_1
    SN-->>Loop: weights_N, metrics_N

    Loop->>Loop: global_weights = _fedavg([w0..wN], [n0..nN])

    Loop->>WB: wandb.log({silo_i/*, aggregated/*}, step=r)

    alt all silos is_done
        Loop->>WB: wandb.log({aggregated/all_silos_done: 1})
        Note over Loop: Early exit
    end
```

---

## Key Data Structures

| Structure | From | To | Contents |
|---|---|---|---|
| `DiagnosticEvent.ground_truth` | `Agent.build_diagnostic_event()` | `WorldFLClient._build_dataset()` | `"{ICD-10 code} / {management tier}"` e.g. `"J10.89 / treat"` |
| `DiseaseTrajectory.icd_code` | `DiseaseProgressionStrategy.sample_trajectory()` | `Agent.build_diagnostic_event()` | ICD-10 code stamped at infection time |
| `DiagnosticEvent` | `ClinicQueue` | `WorldFLClient` | symptom text, ground_truth (ICD/mgmt), severity, conversation, CaseTable |
| `InnerState` | `Agent.inner_state` | `SymptomNarrator` | severity, σ, trend, fatigue, pain, mood, top_vital |
| LoRA weights | `WorldFLClient.get_weights()` | `_fedavg()` | list of np.ndarray (q_lin + v_lin adapters only) |
| `run_round()` dict | `WorldFLClient` | `run_federated_training()` | loss, accuracy (ICD-cat+mgmt), icd_exact_acc, icd_category_acc, mgmt_acc, SIR, num_events |
| W&B log dict | `run_federated_training()` | `wandb.log()` | silo_N/\* + aggregated/\* including icd_category_acc, mgmt_acc |

## ICD-10 Accuracy Scoring

| Prediction vs Ground Truth | Score | Rationale |
|---|:---:|---|
| Exact subcategory match (J10.89 == J10.89) | 1.0 | Fully correct disease identification |
| 3-char category match (J10.xx == J10.89) | 0.5 | Same disease family, wrong specification |
| No match (J11.1 vs J10.89) | 0.0 | Wrong disease |

**Primary metric `accuracy`** = `(icd_category_correct AND mgmt_correct) / N` — clinically acceptable: right disease family, right treatment intensity.
