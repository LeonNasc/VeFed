# Module View — Federated Simulated World

**Last updated: 2026-05-29 (session 6)**

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
        BASE["base_client.py\nWorldFLClient\ntry_accept_global()"]
        LORA["lora.py\nLoRAConfig · build_model\nget/set_lora_weights"]
        TRAIN["train.py\nFLTrainConfig · WorldConfig\nrun_federated_training\n_fedavg"]
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

    RUN["run.py\nlauncher · presets\nRunConfig"]

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

    %% run.py deps
    RUN --> WE
    RUN --> OLL
    RUN --> ILOG
    RUN --> TUI
    RUN --> TRAIN

    %% fl also uses ollama (wired at run_federated_training startup)
    TRAIN --> OLL

    %% external
    OLL["ollama_client.py\n+ DiagnosticExampleStore\n(few-shot bank)"] -->|"HTTP/JSON"| OLLAMA[("Ollama\nphi3:mini\n(single server)")]
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
        LLM["LLM Doctor\n(phi3:mini / Ollama)\nsingle server\n+ few-shot bank"]
        DE["DiagnosticEvent\n(symptom_text\nground_truth\nconversation\noracle_label\ndiagnosis)"]
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
    FEDAVG -->|"update_examples(round_events)\nbest cross-silo conversations\ninjected into Ollama prompt"| LLM

    style SIM fill:#dbeafe,stroke:#3b82f6
    style DIAG fill:#fce7f3,stroke:#ec4899
    style FL fill:#dcfce7,stroke:#22c55e
    style AGG fill:#fef9c3,stroke:#eab308
    style LOG fill:#f3e8ff,stroke:#a855f7
```

---

## View C — FL Round Sequence

Round count is **simulation-guided** — the loop runs until all silos hit their end condition
(default: `ExtinctionCondition`, I=0 for 3 consecutive days) or `max_rounds` is reached.
Silos that finish early enter **frozen mode**: they re-upload their last model each round
and conditionally accept the global model only if it doesn't regress on their frozen eval batch.

```mermaid
sequenceDiagram
    participant Loop as run_federated_training()
    participant SA as Active Silo (WorldFLClient)
    participant SD as Done Silo (WorldFLClient)
    participant OLL as OllamaDiagnosticClient
    participant WB as Weights & Biases

    Note over Loop: Round r = 1..max_rounds (exits early when all done)

    Loop->>SA: set_weights(global_weights)
    Loop->>SD: try_accept_global(global_weights, threshold=0.10)
    Note over SD: Eval global on frozen batch.<br/>Accept if triage_acc drop ≤ 10 %,<br/>else revert to frozen weights.

    par active silo
        SA->>SA: run_simulation_round() — sim_days × TICKS_PER_DAY
        SA->>SA: evaluate() — pre-train metrics on new events
        SA->>SA: train_on_events() — local LoRA fine-tune
    and done silo (frozen)
        SD->>SD: _run_frozen_round() — no simulation, returns cached metrics
    end

    SA-->>Loop: weights (updated), metrics, n_examples
    SD-->>Loop: weights (frozen/accepted), frozen_n for FedAvg weight

    Loop->>Loop: global_weights = _fedavg([all weights], [n_examples or frozen_n])

    Note over Loop: Federated few-shot update (active silos only)
    Loop->>OLL: update_examples(round_events)
    Loop->>WB: wandb.log({silo_i/*, aggregated/*}, step=r)

    alt all silos is_done
        Loop->>WB: wandb.log({aggregated/all_silos_done: 1})
        Note over Loop: Early exit — epidemic extinct
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

## Run Presets (`python run.py --preset <name>`)

| Preset | Silos | Agents | Diseases | Notes |
|---|:---:|:---:|---|---|
| `smoke` | 2 | 15 | Standard Flu | Fastest; offline W&B, no Ollama. Use for CI/debugging. |
| `standard` | 3 | 60 | Flu + Mild Corona | Default research run. |
| `multi-disease` | 5 | 100 | Flu, Corona, Slow Burn, Aggressive Flu | IID multi-disease; tests disease identity classification. |
| `non-iid` | 5 | varies | per silo (WorldConfig) | Asymmetric: different disease, beta_scale, population per silo. Key FL research configuration. |
| `hard-triage` | 3 | 80 | Slow Burn + Deadly | Maximum triage difficulty: plateau symptoms + near-zero decay. |

Wizard: select a preset at startup, then optionally customize individual fields.
CLI: `--preset <name>` loads the preset; any additional flags override specific fields.

## Multi-Disease Dynamics

When `progression_strategies` contains multiple diseases:
- Each susceptible agent that crosses the infection threshold is assigned one disease drawn **uniformly at random** from the list.
- The **single-disease-slot rule** (`HealthState.infect()` guards against double-infection) ensures agents can only carry one disease at a time.
- A **14-day refractory period** (`REFRACTORY_DAYS`) runs after recovery before the agent re-enters `SUSCEPTIBLE`. This prevents weekly cycling through multiple diseases.
- Non-IID silos use `WorldConfig.progressions` to give each silo its own disease mix; FedAvg shares learned weights across disease boundaries.

## ICD-10 Accuracy Scoring

| Prediction vs Ground Truth | Score | Rationale |
|---|:---:|---|
| Exact subcategory match (J10.89 == J10.89) | 1.0 | Fully correct disease identification |
| 3-char category match (J10.xx == J10.89) | 0.5 | Same disease family, wrong specification |
| No match (J11.1 vs J10.89) | 0.0 | Wrong disease |

**Primary metric `accuracy`** = `(icd_category_correct AND mgmt_correct) / N` — clinically acceptable: right disease family, right treatment intensity.
