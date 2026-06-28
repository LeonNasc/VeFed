#!/usr/bin/env python3
"""
Falsification C1 — round-0 baseline (falsification.md, Claim 1).

Checks whether the pretrained DistilBERT + freshly-initialized LoRA adapter
already separates Morven from Velarex/Sornathis in embedding space, BEFORE
any FL training happens. If round-0 silhouette is already high, the
pretrained representations are doing the work and FL contributes nothing —
Claim 1 ("FL training produces disease-structured embeddings") would be
falsified per the decision matrix in falsification.md.

No FL rounds are run — this is a single forward pass on a fresh model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_unknown_disease import (
    generate_fictional_probe_events, _extract_embeddings, _project_umap, _silhouette_morven,
)

OUT_DIR = Path("results/falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from fl.lora import LoRAConfig
    from fl.learner import FLLearner

    probe_events = generate_fictional_probe_events(n_per_band=12, seed=999)
    probe_labels = [ev.ground_truth for ev in probe_events]

    results = []
    for seed in [42, 43, 44]:
        lora_cfg = LoRAConfig(model_name_or_path="distilbert-base-uncased", num_labels=None,
                              rank=8, lora_alpha=16.0, lora_dropout=0.05)
        learner = FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                            device="cuda")
        # No .train() call -- model is freshly initialized (pretrained backbone +
        # randomly-initialized LoRA adapters + randomly-initialized classifier head).
        cls, logits = _extract_embeddings(learner, probe_events)
        coords = _project_umap(logits, seed=seed)
        sil = _silhouette_morven(coords, probe_labels)
        results.append({"seed": seed, "round0_silhouette": sil})
        print(f"seed={seed}  round-0 silhouette (Morven vs known) = {sil:.4f}")
        learner.release()

    sils = [r["round0_silhouette"] for r in results]
    mean_sil = float(np.mean(sils))
    print(f"\nMean round-0 silhouette across {len(sils)} seeds: {mean_sil:.4f}")

    # Compare against the federated/trained results already on record.
    trained_peak_sils = {
        "gauss_inject_×8 (phrase-bank)":  0.815,
        "gauss_inject_×32 (phrase-bank)": 0.936,
        "SIR inject (cal-2x)":            0.787,
    }
    print("\nFor comparison, previously-reported TRAINED peak silhouettes:")
    for name, v in trained_peak_sils.items():
        print(f"  {name}: {v}")

    verdict = (
        "C1 PASSES (round-0 baseline is near zero/negative; trained silhouette is "
        "substantially higher) -- FL training, not pre-training, is producing the "
        "disease-structured embedding."
        if mean_sil < 0.15 else
        "C1 FAILS -- round-0 silhouette is already substantial; pretrained "
        "representations may be doing the separating work, not FL training. "
        "Re-examine Claim 1."
    )
    print(f"\nVerdict: {verdict}")

    summary = {
        "control": "falsification.md C1 -- round-0 baseline",
        "per_seed": results,
        "mean_round0_silhouette": mean_sil,
        "trained_peak_silhouettes_for_comparison": trained_peak_sils,
        "verdict": verdict,
    }
    with (OUT_DIR / "c1_round0_baseline.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'c1_round0_baseline.json'}")


if __name__ == "__main__":
    main()
