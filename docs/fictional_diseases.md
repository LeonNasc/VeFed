# Fictional Disease Experiment — Design & Mapping

**Module:** `simulation/fictional_diseases.py`  
**Preset configs:** `fictional-noniid`, `fictional-iid`, `fictional-noniid-explicit`, `fictional-iid-explicit`

---

## Real-world analog mapping

| Fictional name | Real-world analog    | Why this pairing |
|----------------|----------------------|------------------|
| **Velarex**    | Influenza            | Fast-spreading (high β); predominantly mild-to-moderate; similar SIR arc |
| **Sornathis**  | Bacterial Pneumonia  | Slow-spreading (lower β); higher severity ceiling; significant respiratory compromise |

This mapping is **epidemic-dynamics only**.  The symptom profiles are completely
invented — joint mottling, metallic taste, photophobia (Velarex) and peripheral
paraesthesia, blurred vision, earache (Sornathis) have no standard clinical analog
that phi3:mini would recognise.

The programmatic anchor for all reporting and result-interpretation code is:

```python
from simulation.fictional_diseases import REAL_DISEASE_ANALOG, FICTIONAL_NAME_FOR
# REAL_DISEASE_ANALOG  = {"velarex": "influenza", "sornathis": "bacterial_pneumonia"}
# FICTIONAL_NAME_FOR   = {"influenza": "velarex", "bacterial_pneumonia": "sornathis"}
```

---

## Motivation

phi3:mini has memorised influenza and pneumonia from pretraining.  Even when a
silo sees only one disease, the LLM can generate plausible cross-disease symptom
text from memory — not from the SIR signal.  This inflates local `diag_acc` and
undermines the non-IID FL claim: if the doctor can already guess "this sounds like
pneumonia" from pattern memory, it never needed FL to learn from the other silo.

Fictional names break this shortcut.  A pure-Velarex silo genuinely cannot produce
Sornathis symptom descriptions without seeing Sornathis cases; the only source of
truth is the symptom definition injected in the system prompt.

---

## Three prompt variants

All three use the same fictional disease names (`velarex`, `sornathis`).  They
differ only in what the LLM is told about real-world disease terminology.

### 1. Standard fictional (`explicit_exclusion=False`)

**Presets:** `fictional-noniid`, `fictional-iid`

The prompts define the two diseases from scratch and instruct the model to reason
only from those definitions.  Real disease names are simply never mentioned.

```
Do NOT use any outside medical knowledge — base your diagnosis only on the
definitions below.
```

### 2. Explicit exclusion (`explicit_exclusion=True`)

**Presets:** `fictional-noniid-explicit`, `fictional-iid-explicit`

Same novel names, but every prompt also carries a nomenclature-restriction
paragraph that names "influenza" and "pneumonia" explicitly as unknown terminology:

```
IMPORTANT — NOMENCLATURE RESTRICTION:
In this world the disease names "influenza", "pneumonia", and "the flu" are NOT
known terminology.  The conditions themselves may still exist under different
names — use the condition definitions above to identify them.  If a symptom
pattern reminds you of influenza or pneumonia from your training, that clinical
reasoning is valid; you must simply label the condition using the names listed
above (e.g. Velarex or Sornathis), never using the words "influenza",
"pneumonia", or "flu" in your response.
```

**Key design decision:** the disclaimer explicitly says *"the clinical reasoning
is valid"* — the LLM is not told the syndrome doesn't exist, only that the name
is unknown.  This preserves the doctor's ability to recognise the syndrome and
map it correctly to the fictional label, while suppressing raw label leakage from
pretraining.

### 3. Real-disease baseline (`disease_glossary=None`)

**Presets:** `iid`, `non-iid`, `extreme-noniid`, …

Standard influenza/pneumonia label space; LLM uses pretraining knowledge freely.
This is the **contaminated baseline**: if local `diag_acc` is high even in the
non-IID arm, it is partly or wholly explained by memorised disease patterns rather
than FL weight sharing.

---

## Ablation interpretation

Running `fictional-noniid` vs `fictional-noniid-explicit` with `--mode local_only`
gives a direct test:

- If `diag_acc` is similar in both, **naming novelty alone** is sufficient to
  suppress prior contamination.
- If `explicit` gives lower local accuracy but stronger FL gain, the model was
  still leaking real-world label associations even with novel names.
- The real-disease `non-iid` baseline sets the contaminated ceiling; the
  fictional runs should show a larger FL gain (lower local, higher federated).

---

## Where the mapping is used

- **`simulation/fictional_diseases.py`** — `REAL_DISEASE_ANALOG`, `FICTIONAL_NAME_FOR`
- **`simulation/progression.py`** — `VelarexProgression` / `SornathisProgression`
  docstrings reference the analog; SIR β values are calibrated to match the
  real-disease equivalents.
- **Result reporting** — any script that logs per-disease metrics should import
  `REAL_DISEASE_ANALOG` to annotate fictional labels with their real-world
  counterparts in tables and figures.
