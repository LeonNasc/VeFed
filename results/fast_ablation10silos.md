# Federated Learning Silo Scalability — 10-Silo Results

**Generated:** 2026-06-15  
**Experiment:** Silo scalability study — 3 silos vs 10 silos under SIR-Gaussian dynamics  
**Model:** LoRA DistilBERT (rank=8, α=16), FedAvg, 20 FL rounds  
**Seed:** 42 (single seed; scalability is structural, not statistical)  
**Disease setup:** Influenza + Bacterial Pneumonia (SIR runs); Velarex + Sornathis + Morven (unknown disease)  
**Baseline:** 3-silo results from `fast_ablation/report.md` §7-8 (sir-cal-2x, 150 agents/silo)

---

## Contents

1. [Motivation and Experiment Design](#1-motivation-and-experiment-design)
2. [SIR-Gaussian IID — 3 vs 10 Silos](#2-sir-gaussian-iid--3-vs-10-silos)
3. [SIR-Gaussian Non-IID — 3 vs 10 Silos](#3-sir-gaussian-non-iid--3-vs-10-silos)
4. [Zombie Silo Analysis](#4-zombie-silo-analysis)
5. [Unknown Disease Detection — 3 vs 10 Silos](#5-unknown-disease-detection--3-vs-10-silos)
6. [Summary and Scalability Conclusions](#6-summary-and-scalability-conclusions)

---

## 1. Motivation and Experiment Design

The results in `fast_ablation/report.md` were obtained with **3 silos**. This document extends those findings to **10 silos**, motivated by a realistic hospital-network scenario where regional health authorities operate 10 or more independent clinic systems under a federated learning agreement.

The key research questions are:

1. **Convergence scaling**: Does FedAvg maintain convergence quality when N grows from 3 to 10?
2. **Non-IID amplification**: With a 10-step disease gradient (instead of 3), does the heterogeneity
   advantage (§C3 in the main report) still hold?
3. **Zombie silo amplification**: With 10 independent SIR worlds, the epidemic tail affects more silos.
   Does the zombie-silo problem scale with N?
4. **Novel disease detection**: Does the "unknown" embedding cluster remain detectable when 9/10 silos
   have no Morven exposure (vs 2/3 in the original experiment)?

### Setup

| Parameter | 3-silo (reference) | 10-silo (scaled) |
|---|---|---|
| Silos | 3 | 10 |
| Agents/silo | 150 | 150 |
| Total agents | 450 | 1500 |
| FL rounds | 20 | 20 |
| Sim-days/round | 2 | 2 |
| β (transmission) | 2.0 | 2.0 |
| End condition | horizon=40 | horizon=40 |
| Ollama | off (templates) | off (templates) |
| Seed | 42 | 42 |

**Non-IID gradient:** 10 silos span the spectrum from 95% flu / 5% pneumo (silo 0) to 5% flu / 95% pneumo (silo 9), with equal steps between. The 5% floor prevents pathologically pure-class silos that would never observe the opposing disease.

**Unknown disease:** Gaussian-controlled schedule (160 events/silo), Morven injected into silo_0 only from round 10. Silos 1–9 (7 additional silos vs the original) see only Velarex+Sornathis throughout.

![SIR comparison](../scalability_10silo/sir_comparison.png)

---

## 2. SIR-Gaussian IID — 3 vs 10 Silos

**Setup:** All silos see 50/50 Influenza+Pneumonia. Tests pure scaling behaviour with no heterogeneity.

![IID silo heatmap 3-silo](../scalability_10silo/iid_3silo_silo_heatmap.png)
*3-silo per-silo accuracy heatmap (rounds × silos)*

![IID silo heatmap 10-silo](../scalability_10silo/iid_10silo_silo_heatmap.png)
*10-silo per-silo accuracy heatmap (rounds × silos)*

### IID accuracy table

| Condition | Peak acc | Final acc (R20) | Convergence |
|---|---|---|---|
| 3 silos IID | — | — | — |
| 10 silos IID | — | — | — |

*(Fill in from `results/scalability_10silo/summary.json` after run completes)*

**Key observations:**

- [TBD after run]

---

## 3. SIR-Gaussian Non-IID — 3 vs 10 Silos

**Setup:** Disease gradient across silos. 3-silo version: [100% flu, 50/50, 100% pneumo].
10-silo version: 10 equidistant steps from [95% flu] to [95% pneumo].

The 10-silo Non-IID case is more extreme than the 3-silo case: silos 0 and 9 see almost none of the
opposing disease and rely entirely on FedAvg to learn it.

![NonIID silo heatmap 3-silo](../scalability_10silo/noniid_3silo_silo_heatmap.png)
*3-silo Non-IID per-silo accuracy (silos 0 and 2 are disease-specialist silos)*

![NonIID silo heatmap 10-silo](../scalability_10silo/noniid_10silo_silo_heatmap.png)
*10-silo Non-IID per-silo accuracy (silos 0 and 9 are near-specialist silos)*

### Non-IID accuracy table

| Condition | Peak acc | Final acc (R20) | Specialist silo gap (max−min) |
|---|---|---|---|
| 3 silos Non-IID | — | — | — |
| 10 silos Non-IID | — | — | — |

*(Fill in from `results/scalability_10silo/summary.json`)*

---

## 4. Zombie Silo Analysis

With 10 independent SIR worlds (β=2.0, 150 agents/silo, horizon=40 sim-days), the epidemic arc
plays out at slightly different speeds across silos due to stochastic seed contact variation
(contact_rate_sigma=0.5). The zombie silo problem — silos whose epidemic has ended and contribute
zero training examples — should scale with N.

![IID 3-silo zombie](../scalability_10silo/iid_3silo_zombie.png)
*Training examples per silo per round — IID 3-silo*

![IID 10-silo zombie](../scalability_10silo/iid_10silo_zombie.png)
*Training examples per silo per round — IID 10-silo*

![NonIID 10-silo zombie](../scalability_10silo/noniid_10silo_zombie.png)
*Training examples per silo per round — Non-IID 10-silo*

**Expected finding:** With 10 silos, the late-round FedAvg aggregate receives contributions from
more zombie silos, but the non-zombie silos remain in the majority for longer (the epidemic
extinctions are staggered). The net effect on accuracy depends on whether the zombie fraction
exceeds ~50% simultaneously.

---

## 5. Unknown Disease Detection — 3 vs 10 Silos, Federated vs Local-Only

**Setup:** 10 silos train on Velarex+Sornathis (Gaussian schedule, 160 events/silo).
Morven is injected into **silo_0 only** from round 10. Silos 1–9 never see Morven.
The global model is the FedAvg of all 10 silos, which dilutes the "unknown" signal 10-fold
compared to the original 3-silo experiment (where silo_0's Morven signal had 2:1 dilution).

This section directly tests **paper claim (iii)**: *cross-institutional diffusion enables early
detection in silos that have never encountered the novel disease.*

Each run has a paired **local-only baseline** where FedAvg is disabled. In local-only mode,
silo_0 (Morven-exposed) learns from its own data; silos 1–9 never see Morven and cannot detect it.
The federated run lets silo_0's Morven signal propagate to all silos through FedAvg.

![Diffusion claim](../scalability_10silo/diffusion_claim.png)

*The key figure: federated global silhouette (teal) vs local-only silo_0 (orange, exposed) vs
local-only silo_1 (red, never sees Morven). Silo_1's local silhouette should stay ≈ 0; the
federated model achieves high silhouette because silo_0's signal diffuses via FedAvg.*

![Unknown disease comparison](../scalability_10silo/unknown_comparison.png)

### Detection metrics

| Condition | Mode | Detection round (sil>0) | Silhouette @R15 | Silhouette @R20 | Known-disease acc @R20 |
|---|---|---|---|---|---|
| 3 silos (§9 ref) | federated | R12 | 0.870 | — | 1.00 |
| 10 silos | federated | — | — | — | — |
| 3 silos | local-only silo_0 | — | — | — | — |
| 3 silos | local-only silo_1 | — | — | — | — |
| 10 silos | local-only silo_0 | — | — | — | — |
| 10 silos | local-only silo_1 | — | — | — | — |

*(Fill in after run completes)*

**Expected finding:** Silo_1's local-only silhouette stays near 0 throughout all 20 rounds —
it never encounters Morven, so its local model has no signal. The federated model's silhouette
rises toward 0.87 because silo_0's "unknown"-labelled Morven examples shift the backbone's
embedding geometry, and FedAvg propagates this shift globally. This is claim (iii)
demonstrated experimentally.

---

## 6. Summary and Scalability Conclusions

### Summary table

| Metric | 3-silo | 10-silo | Δ |
|---|---|---|---|
| IID final acc | — | — | — |
| IID peak acc | — | — | — |
| Non-IID final acc | — | — | — |
| Non-IID peak acc | — | — | — |
| Morven detection round | R12 | — | — |
| Morven silhouette @R15 | 0.87 | — | — |

### Scalability conclusions

**CS1: [TBD — convergence scaling]**

**CS2: [TBD — Non-IID gradient amplification]**

**CS3: [TBD — zombie silo scaling behaviour]**

**CS4: [TBD — novel disease detection under 9:1 dilution]**

---

*Results generated by `run_10silo.py`. Raw per-round data in `results/scalability_10silo/*/round_metrics.json`.
Summary in `results/scalability_10silo/summary.json`.*
