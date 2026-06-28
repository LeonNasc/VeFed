#!/usr/bin/env python3
"""
Recompute MI(text; disease) for phrase_library vs Ollama generators across
seeds 42/43/44, replacing the single-run estimate in research_notes/text_entropy_analysis.md.

Ollama text: reused from existing 3-seed pools (results/ollama_ablation/pools/seed{N}),
filtered to influenza/pneumonia (excludes non-infectious, matching the original
2-class comparison).

Phrase-library text: regenerated per seed via simulation.phrase_sampler.PhraseLibrary
(deterministic, no LLM call), matched 50/50 influenza/pneumonia, severity drawn
uniformly over mild/moderate/severe, n=1080 per seed to match the original corpus size.

MI = H_marginal - H_within, where H_within is the per-class token entropy
averaged across classes weighted by class document count, and H_marginal is the
entropy of the pooled token distribution. Both in bits (log2).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from simulation.phrase_sampler import PhraseLibrary

SEEDS = [42, 43, 44]
OLLAMA_POOL_ROOT = Path("results/ollama_ablation/pools")
OUT_PATH = Path("results/falsification/mi_per_seed.json")

_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def mi_for_texts(texts_by_class: dict[str, list[str]]) -> float:
    class_counters = {}
    pooled = Counter()
    n_per_class = {}
    for cls, texts in texts_by_class.items():
        ctr = Counter()
        for t in texts:
            ctr.update(tokenize(t))
        class_counters[cls] = ctr
        pooled.update(ctr)
        n_per_class[cls] = len(texts)

    h_marginal = entropy_bits(pooled)
    total_docs = sum(n_per_class.values())
    h_within = sum(
        (n_per_class[cls] / total_docs) * entropy_bits(class_counters[cls])
        for cls in texts_by_class
    )
    return h_marginal - h_within


def load_ollama_texts(seed: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"influenza": [], "pneumonia": []}
    for split in ("iid", "noniid"):
        for silo_dir in sorted((OLLAMA_POOL_ROOT / f"seed{seed}" / split).glob("silo_*")):
            f = silo_dir / "train.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                d = rec.get("gt_disease")
                if d in out:
                    out[d].append(rec["text"])
    return out


def generate_phrase_library_texts(seed: int, n_per_class: int = 540) -> dict[str, list[str]]:
    lib = PhraseLibrary(seed=seed)
    rng = lib._rng  # reuse the same seeded RNG for severity draws
    severities = ["mild", "moderate", "severe"]
    out: dict[str, list[str]] = {"influenza": [], "pneumonia": []}
    for disease in ("influenza", "pneumonia"):
        for _ in range(n_per_class):
            sev = rng.choice(severities)
            rec = lib.sample(disease, sev)
            out[disease].append(rec["text"])
    return out


def main():
    per_seed = {"phrase_library": {}, "ollama": {}}
    for seed in SEEDS:
        phrase_texts = generate_phrase_library_texts(seed)
        ollama_texts = load_ollama_texts(seed)
        per_seed["phrase_library"][seed] = mi_for_texts(phrase_texts)
        per_seed["ollama"][seed] = mi_for_texts(ollama_texts)
        print(f"seed={seed}  phrase_library n={sum(len(v) for v in phrase_texts.values())} "
              f"MI={per_seed['phrase_library'][seed]:.3f}   "
              f"ollama n={sum(len(v) for v in ollama_texts.values())} "
              f"MI={per_seed['ollama'][seed]:.3f}")

    summary = {}
    for gen in ("phrase_library", "ollama"):
        vals = list(per_seed[gen].values())
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        summary[gen] = {"mean": mean, "sd": sd, "per_seed": per_seed[gen]}
        print(f"{gen}: MI = {mean:.3f} ± {sd:.3f}  (n={len(vals)} seeds)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
