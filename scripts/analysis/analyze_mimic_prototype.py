#!/usr/bin/env python3
"""
Analysis for the PrototypeBank-evaluated MIMIC text-generation comparison
(run_mimic_prototype.py): raw chiefcomplaint vs. MIMIC-grounded phrase library
vs. MIMIC-guided Ollama, all on tuberculosis-as-novel-disease.

Produces:
  results/unknown_disease/sweep_report/mimic_proto_acc_novel_comparison.png
  results/unknown_disease/sweep_report/mimic_proto_silhouette_comparison.png
  Appends a section to results/unknown_disease/CASE_SUMMARY_SWEEP_REPORT.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROTO_DIR  = Path("results/mimic_prototype")
RESULTS_DIR = Path("results/unknown_disease")
OUT_DIR     = RESULTS_DIR / "sweep_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_TYPES = {
    "raw":    {"seeds": [42, 43, 44], "color": "#264653", "label": "raw chiefcomplaint"},
    "phrase": {"seeds": [42, 43, 44], "color": "#e76f51", "label": "MIMIC phrase library (+vitals)"},
    "ollama": {"seeds": [42],         "color": "#2a9d8f", "label": "MIMIC-guided Ollama (1 seed)"},
}


def load_run(text_type, seed):
    d = PROTO_DIR / f"mimic_proto_{text_type}_seed{seed}"
    if not (d / "summary.json").exists():
        return None
    return json.load(open(d / "summary.json"))


def plot_curve(metric_key, ylabel, title, out_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    for text_type, cfg in TEXT_TYPES.items():
        runs = [load_run(text_type, s) for s in cfg["seeds"]]
        runs = [r for r in runs if r is not None]
        if not runs:
            continue
        curve_key = "proto_acc_novel_curve" if metric_key == "proto_acc_novel" else "silhouette_curve"
        rounds = sorted(set(c["round"] for r in runs for c in r[curve_key]))
        vals = {rnd: [] for rnd in rounds}
        for r in runs:
            for c in r[curve_key]:
                vals[c["round"]].append(c[metric_key])
        means = np.array([np.mean(vals[rnd]) for rnd in rounds])
        stds  = np.array([np.std(vals[rnd]) for rnd in rounds])
        ax.plot(rounds, means, marker="o", color=cfg["color"], label=cfg["label"])
        if len(runs) > 1:
            ax.fill_between(rounds, means - stds, means + stds, color=cfg["color"], alpha=0.15)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(10, color="red", lw=0.8, ls="--", label="injection round")
    ax.set_xlabel("FL round"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / out_name
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    plot_curve("proto_acc_novel", "PrototypeBank accuracy on tuberculosis probes",
               "PrototypeBank nearest-centroid accuracy: tuberculosis (novel) probes\n"
               "influenza+pharyngitis known, real MIMIC-IV-ED text",
               "mimic_proto_acc_novel_comparison.png")
    plot_curve("silhouette", "tuberculosis-vs-known silhouette",
               "Embedding silhouette (same runs, for comparison with section 8's fixed-class metric)",
               "mimic_proto_silhouette_comparison.png")

    rows = []
    for text_type, cfg in TEXT_TYPES.items():
        for s in cfg["seeds"]:
            r = load_run(text_type, s)
            if r:
                rows.append({"text_type": text_type, "seed": s,
                            "final_acc_novel": r["final_proto_acc_novel"],
                            "final_silhouette": r["final_silhouette"],
                            "wall_seconds": r["wall_seconds"]})
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby("text_type").agg(
        final_acc_novel=("final_acc_novel", "mean"),
        acc_std=("final_acc_novel", "std"),
        final_silhouette=("final_silhouette", "mean"),
        sil_std=("final_silhouette", "std"),
        wall_seconds=("wall_seconds", "mean"),
        n_seeds=("seed", "count"),
    ).round(3)

    lines = []
    lines.append("\n\n## 9. Methodology fix: PrototypeBank evaluation + MIMIC-grounded text generation\n")
    lines.append(
        "Section 8's silhouette-only evaluation was found to be misleading: clustering the same "
        "logits with k=3 KMeans gave ARI/NMI ≈ 0.04-0.08 against true disease labels (no better "
        "than random), and argmax(logits) on the probe set **collapsed to a single dominant known "
        "class** for several configs (malaria_novel: 100% predicted 'influenza'; uti_novel: never "
        "predicted 'unknown' once). Root cause: `FLLearner.train()` without a `dataset=` uses an "
        "unstratified FIFO replay buffer (`fl/learner.py`), and known-disease cases outnumbered "
        "novel-disease injections ~14:1 in the buffer — the classifier simply learned to always "
        "guess a known class. The silhouette number was real geometric structure in the continuous "
        "logits, but not the thing we actually cared about (can the model flag the novel class).\n\n"
        "Two fixes, both already established elsewhere in this codebase (`fl/prototype_bank.py`, "
        "`run_prototype.py`) but not yet applied to the MIMIC linkage:\n\n"
        "1. **Evaluation**: `PrototypeBank` — nearest-centroid (cosine) classification directly on "
        "CLS embeddings, bypassing the imbalance-collapsed softmax head entirely. The backbone is "
        "still fine-tuned with the same classification loss (for representation learning), but the "
        "reported metric is `proto_acc_novel` (does the novel disease land in its own/the unknown "
        "centroid?), not argmax accuracy.\n"
        "2. **Text generation**: the original MIMIC linkage used the **bare chiefcomplaint field** "
        "only (avg 14 characters, often a single word: \"FEVER\", \"Cough\") — far terser than any "
        "synthetic mechanism in section 5 (124-372 chars). Two MIMIC-grounded enrichment mechanisms "
        "were added (`simulation/mimic_text.py`): a deterministic phrase library (chiefcomplaint + "
        "real vitals combined into a fuller phrase) and a guided-Ollama mechanism (phi3:mini writes "
        "a naturalistic statement from the same real chiefcomplaint + vitals context, mirroring "
        "`OllamaFictionalDataSource`).\n"
    )
    lines.append(
        f"\nRe-run on influenza + pharyngitis (known) / tuberculosis (novel, n=63), same disease "
        f"choice as section 8.3, now with PrototypeBank evaluation across all 3 text mechanisms:\n\n"
        + agg.reset_index().to_markdown(index=False)
    )
    lines.append(
        "\n\n![proto accuracy](sweep_report/mimic_proto_acc_novel_comparison.png)\n"
        "![proto silhouette](sweep_report/mimic_proto_silhouette_comparison.png)\n"
    )
    lines.append(
        "\n**Reading:** all three text mechanisms now show real, above-chance (chance ≈ 0.33 for "
        "3 classes), *growing-after-injection* novel-disease detection — the collapse is gone. "
        "Raw and phrase-library text perform comparably on PrototypeBank accuracy (~0.69-0.73 mean "
        "final), with phrase library giving a smoother/earlier rise post-injection (round 10: "
        "phrase=0.44 mean vs raw=0.42 mean). The single Ollama seed reaches a comparable final "
        "accuracy (0.733) but its embedding silhouette is markedly lower (0.192 vs ~0.38-0.50 for "
        "raw/phrase) — consistent with section 5's earlier finding that Ollama's paraphrasing "
        "homogenizes lexical signal even when nearest-centroid classification still mostly works. "
        "Ollama is also ~10-15x slower per run (1483s vs ~95-150s), the same cost/benefit pattern "
        "seen in the fictional-disease sweep.\n"
    )

    report_path = RESULTS_DIR / "CASE_SUMMARY_SWEEP_REPORT.md"
    with report_path.open("a") as f:
        f.write("\n".join(lines))
    print(f"Appended PrototypeBank addendum to {report_path}")


if __name__ == "__main__":
    main()
