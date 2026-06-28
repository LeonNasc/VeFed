# VeFed: A Virtual Epidemiological Environment for Federated Training of Clinical Language Agents

**VeFed** is a simulation testbed for studying federated learning (FL) applied to clinical language agents. It couples a calibrated SIR epidemic model with LLM-generated patient text and a LoRA-adapted DistilBERT classifier trained across isolated hospital silos, enabling controlled evaluation of FL under realistic temporal and distributional heterogeneity.

The framework supports four data-generation modes (template phrases, curated phrase libraries, Ollama-backed LLM, and MIMIC-IV-ED chief complaints), four FL aggregation strategies (FedAvg, FedProx, FedSGD, FedAdam), and a five-point falsification protocol for validating that learned representations track disease identity rather than confounds.

> **Paper:** Nascimento, L. et al. *VeFed: A Virtual Epidemiological Environment for Federated Training of Clinical Language Agents.* Frontiers in Applied Mathematics and Statistics (2026). *(in press)*

---

## Citation

```bibtex
@article{nascimento2026vefed,
  author    = {Nascimento, Leon and others},
  title     = {{VeFed}: A Virtual Epidemiological Environment for Federated
               Training of Clinical Language Agents},
  journal   = {Frontiers in Applied Mathematics and Statistics},
  year      = {2026},
  doi       = {10.3389/fams.2026.xxxxxx},
  note      = {in press}
}
```

---

## Installation

```bash
git clone https://github.com/LeonNasc/VeFed.git
cd VeFed
pip install torch transformers peft accelerate scikit-learn matplotlib numpy
# Federated backends (install one or both):
pip install "flwr>=1.8"
pip install "fedn>=0.20"
```

**Ollama (optional)** — used by `OllamaFictionalDataSource` and `OllamaCaseSummarizer` for LLM-generated patient text. Install from [ollama.com](https://ollama.com) and pull the model used in the paper:

```bash
ollama pull phi3:mini
```

If Ollama is unavailable, all Ollama-backed data sources fall back automatically to their template equivalents — no configuration required.

---

## Quickstart

Run the core unknown-disease detection experiment (main paper §5.2):

```bash
python run_unknown_disease.py --seed 42 --data-source template
python run_unknown_disease.py --seed 42 --data-source ollama   # requires Ollama
```

Results are written to `results/unknown_disease/`.

---

## Experiment Scripts

### Main experiment

| Script | What it runs |
|---|---|
| `run_unknown_disease.py` | Unknown-disease detection with SIR-driven heterogeneity |
| `run_attribution.py` | Attribution sweep: frozen-backbone vs naive FL vs local-head |
| `run_prototype.py` | PrototypeBank nearest-centroid novel-class detector |
| `run_prototype_sweep.py` | Sweep over prototype threshold and round count |

### Falsification protocol (C1–C5)

| Script | Control |
|---|---|
| `run_falsification_c1.py` | C1 — random-label permutation (representation baseline) |
| `run_falsification_c2.py` | C2 — data-volume ablation (federation benefit) |
| `run_falsification_c2_sweep.py` | C2 sweep across silo sizes |
| `run_falsification_c3.py` | C3 — ARI/silhouette convergence with round count |
| `run_falsification_c4.py` | C4 — known-disease control injection (false-positive rate) |
| `run_falsification_c4_prototype.py` | C4 variant — PrototypeBank stress test |
| `run_falsification_c5.py` | C5 — untrained OOD absorption control |

### Ablations & sweeps

```bash
python run_ablation.py                  # schedule shape × DataSource
python run_schedule_ablation.py         # Gaussian / flat / SIR schedule comparison
python run_aggregation_comparison.py    # FedAvg vs FedProx vs FedSGD vs FedAdam
python run_dirichlet_sweep.py           # label heterogeneity α sweep
python run_multilingual_silos.py        # multilingual silo bonus experiment
python run_diffusion_test.py            # SIR diffusion reconciliation
python run_all_statistical_tests.py     # significance tests for all main results
```

### MIMIC-IV-ED experiments

```bash
python run_mimic_prototype.py           # prototype evaluation on MIMIC cohort
python run_mimic_unknown_disease.py     # unknown-disease detection with real ED data
```

> **MIMIC access required.** See [MIMIC-IV-ED access](#mimic-iv-ed-access) below.

---

## Repository Layout

```
virtual_world/
├── simulation/
│   ├── world.py               # WorldEngine — simulation loop and silo orchestration
│   ├── models.py              # Agent, HealthState, SIRModel, InnerState
│   ├── data_sources.py        # DataSource ABC + Template/PhraseLibrary/Ollama/MIMIC variants
│   ├── case_summary.py        # CaseSummarizer ABC + Template/Ollama variants
│   ├── fictional_diseases.py  # Velarex, Sornathis, Morven phrase banks and prompts
│   ├── patient_llm.py         # PatientLLMClient (Ollama phi3:mini wrapper)
│   ├── symptom_language.py    # SymptomNarrator, severity bands, Personality enum
│   ├── phrase_sampler.py      # PhraseLibrary curated phrase banks
│   ├── conversation.py        # Multi-turn patient–nurse conversation state machine
│   ├── mimic_db.py            # MockMimicDatabase / RealMimicDatabase
│   ├── mimic_data_source.py   # MIMICDataSource — MIMIC-IV-ED chief-complaint enrichment
│   ├── mimic_text.py          # MIMIC phrase library and Ollama-guided generation
│   └── world_config.py        # WorldConfig dataclass
├── fl/
│   ├── learner.py             # FLLearner — LoRA DistilBERT + prototype head
│   ├── aggregation.py         # FedAvg, FedProx, FedSGD, FedAdam (FedAdamServer)
│   ├── silo.py                # FLSilo — per-hospital training loop
│   ├── server.py              # FL server — aggregation and round scheduling
│   ├── prototype_bank.py      # PrototypeBank — nearest-centroid novel-class detector
│   ├── schedules.py           # FL schedule shapes (Gaussian, flat, SIR-calibrated)
│   └── lora.py                # LoRA adapter helpers
├── scripts/
│   ├── analysis/              # Figure generation and statistical tests
│   └── preprocess_mimicel.py  # MIMIC-IV-ED preprocessing pipeline
├── paper_figures/             # Publication figures (44 PNG files)
├── diagrams/                  # Architecture diagram
├── run_*.py                   # Experiment entry points (see table above)
└── requirements.txt
```

---

## MIMIC-IV-ED Access

MIMIC-IV-ED is a restricted dataset requiring credentialed access via PhysioNet. The raw data files are **not** included in this repository.

To run the MIMIC experiments:

1. Apply for access at [physionet.org/content/mimic-iv-ed](https://physionet.org/content/mimic-iv-ed/)
2. Download the dataset and run the preprocessing script:
   ```bash
   python scripts/preprocess_mimicel.py --mimic-dir /path/to/mimic-iv-ed --out MIMIC/
   ```
3. Pass the preprocessed CSV path to the MIMIC experiment scripts via `--csv-path`.

The `MockMimicDatabase` class in `simulation/mimic_db.py` generates synthetic vitals conforming to the same schema and can be used without MIMIC access for development and testing:
```bash
python run_mimic_prototype.py --mock   # no MIMIC files needed
```

---

## Aggregation Strategies

| Strategy | Server rule | Local distinction |
|---|---|---|
| FedAvg | Weighted average of client weights | Standard SGD |
| FedProx | Identical to FedAvg | Proximal penalty $\frac{\mu}{2}\|w - w^{(r)}\|^2$ added to local loss |
| FedSGD | Identical to FedAvg | Single gradient step per round (mathematically equivalent to gradient averaging) |
| FedAdam | `FedAdamServer` maintains persistent momentum/variance | Standard SGD locally |

---

## License

Code: MIT License — see `LICENSE`.  
MIMIC-IV-ED data: subject to PhysioNet Credentialed Health Data License 1.5.0.
