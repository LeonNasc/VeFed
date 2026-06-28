#!/usr/bin/env python3
"""
Bonus experiment: novel-disease detection (Morven) under multilingual silos.

Extends run_multilingual_silos.py's setup (one language per silo, Velarex +
Sornathis known, distilbert-base-multilingual-cased backbone) with the
original run_unknown_disease.py protocol: from `injection_round` onward, one
silo (--inject-language) additionally receives Morven cases in its own
language, labelled "unknown". A fixed probe set -- Velarex, Sornathis, AND
Morven, generated in EVERY active language, not just the injected one -- is
passed through the global model each round.

This adds a question the single-language original couldn't ask: does novel-
disease detection diffuse across a LANGUAGE boundary the same way Section
3.5's diffusion_claim.png showed it diffusing across a SILO boundary? Morven
text only ever enters training in --inject-language; the per-language probe
breakdown checks whether Morven separates in the *other* languages' probes
too, via FedAvg weight-sharing alone.

Not a falsification.md control -- exploratory, single-seed, bonus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_samples

from run_multilingual_silos import (
    DISEASES, _extract_cls, _generate_events, _project_umap,
)

OUT_DIR = Path("results/multilingual_silos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOVEL_DISEASE = "morven"


def _silhouette_morven(coords: np.ndarray, labels: list[str]) -> float:
    """Mean silhouette of Morven points vs. the known-disease backdrop in `labels`."""
    group = np.array([1 if lbl == NOVEL_DISEASE else 0 for lbl in labels], dtype=int)
    if group.sum() < 2 or (group == 0).sum() < 2:
        return float("nan")
    try:
        scores = silhouette_samples(coords, group)
        return float(np.mean(scores[group == 1]))
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="en,es,fr",
                        help="Comma-separated subset of en,es,fr,et -- one silo per language.")
    parser.add_argument("--inject-language", default=None,
                        help="Which silo's language receives Morven text. Default: last in --languages.")
    parser.add_argument("--events-per-silo-per-disease", type=int, default=30)
    parser.add_argument("--probe-per-disease-per-language", type=int, default=6)
    parser.add_argument("--morven-events-per-round", type=int, default=4)
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--injection-round", type=int, default=6)
    parser.add_argument("--total-rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="distilbert-base-multilingual-cased")
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    inject_language = args.inject_language or languages[-1]
    assert inject_language in languages

    from fl.aggregation import fedavg
    from fl.learner import FLLearner
    from fl.lora import LoRAConfig
    from simulation.patient_llm import PatientLLMClient

    client = PatientLLMClient()
    if not client.health_check():
        raise SystemExit("Ollama unreachable at localhost:11434 -- required for live generation.")

    print(f"Languages: {languages}; Morven injected into: {inject_language} "
          f"from round {args.injection_round}")

    print("Generating known-disease training pools...")
    known_pools = {
        lang: _generate_events(lang, args.events_per_silo_per_disease, args.seed, client)
        for lang in languages
    }

    n_morven_rounds = args.total_rounds - args.injection_round + 1
    n_morven_total = max(1, n_morven_rounds * args.morven_events_per_round)
    print(f"Generating {n_morven_total} Morven events in {inject_language}...")
    morven_pool = _generate_events(
        inject_language, n_morven_total, args.seed + 555, client, diseases=[NOVEL_DISEASE]
    )
    morven_cursor = 0

    print("Generating probe set (known diseases + Morven, every language)...")
    probe_texts, probe_disease, probe_language = [], [], []
    for lang in languages:
        known_probes = _generate_events(lang, args.probe_per_disease_per_language,
                                        args.seed + 999, client)
        morven_probes = _generate_events(lang, args.probe_per_disease_per_language,
                                         args.seed + 999, client, diseases=[NOVEL_DISEASE])
        for ev in known_probes + morven_probes:
            probe_texts.append(ev["text"])
            probe_disease.append(ev["gt_disease"])
            probe_language.append(lang)

    lora_cfg = LoRAConfig(model_name_or_path=args.backbone, num_labels=None)
    learners = {
        lang: FLLearner(lora_config=lora_cfg, label_space="fictional_disease",
                        min_events_to_train=4, device=args.device)
        for lang in languages
    }

    def evaluate_global(weights) -> dict:
        learners[languages[0]].set_weights(weights)
        embs = _extract_cls(learners[languages[0]], probe_texts)
        learners[languages[0]].release()
        coords = _project_umap(embs, seed=args.seed)

        result = {"morven_silhouette_pooled": _silhouette_morven(coords, probe_disease)}
        for lang in languages:
            idx = [i for i, l in enumerate(probe_language) if l == lang]
            sub_coords = coords[idx]
            sub_labels = [probe_disease[i] for i in idx]
            result[f"morven_silhouette_{lang}"] = _silhouette_morven(sub_coords, sub_labels)
        return result

    init_weights = learners[languages[0]].get_weights()
    learners[languages[0]].release()
    round0_metrics = evaluate_global(init_weights)
    print(f"Round 0 (pre-training): {round0_metrics}")
    curve = [{"round": 0, **round0_metrics}]

    global_weights = init_weights
    for r in range(args.total_rounds):
        rnd = r + 1
        round_weights, train_sizes = [], []
        for lang in languages:
            learner = learners[lang]
            if global_weights is not None:
                learner.set_weights(global_weights)

            events_this_round = list(known_pools[lang])
            if lang == inject_language and rnd >= args.injection_round:
                take = morven_pool[morven_cursor: morven_cursor + args.morven_events_per_round]
                events_this_round += take
                morven_cursor += args.morven_events_per_round

            n_trained, _ = learner.train(events_this_round, round_num=rnd)
            train_sizes.append(n_trained)
            round_weights.append(learner.get_weights())
            learner.release()

        global_weights = fedavg(round_weights, [max(1, n) for n in train_sizes])
        metrics = evaluate_global(global_weights)
        tag = " [INJECTED]" if rnd >= args.injection_round else ""
        print(f"Round {rnd}{tag}: {metrics}")
        curve.append({"round": rnd, **metrics})

    lang_tag = "-".join(languages)

    # Final-round global-model embeddings (known diseases + Morven, every
    # language), for a 2D visualization of the global clusters.
    learners[languages[0]].set_weights(global_weights)
    final_embs = _extract_cls(learners[languages[0]], probe_texts)
    learners[languages[0]].release()
    coords = _project_umap(final_embs, seed=args.seed)
    embedding_dump = {
        "coords": coords.tolist(),
        "disease": probe_disease,
        "language": probe_language,
    }
    emb_path = OUT_DIR / f"multilingual_unknown_disease_{lang_tag}_inject-{inject_language}_seed{args.seed}_final_embeddings.json"
    with emb_path.open("w") as f:
        json.dump(embedding_dump, f, indent=2)
    print(f"Saved: {emb_path}")

    summary = {
        "experiment": "bonus_multilingual_unknown_disease",
        "languages": languages,
        "inject_language": inject_language,
        "injection_round": args.injection_round,
        "known_diseases": DISEASES,
        "novel_disease": NOVEL_DISEASE,
        "backbone": args.backbone,
        "seed": args.seed,
        "curve": curve,
    }
    out_path = OUT_DIR / f"multilingual_unknown_disease_{lang_tag}_inject-{inject_language}_seed{args.seed}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
