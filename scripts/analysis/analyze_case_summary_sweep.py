#!/usr/bin/env python3
"""
Analysis for the case-summary x distribution sweep (run_case_summary_compare.sh).

Produces:
  results/unknown_disease/sweep_report/metrics_table.csv
  results/unknown_disease/sweep_report/text_info_table.csv
  results/unknown_disease/sweep_report/text_samples.json
  results/unknown_disease/sweep_report/silhouette_by_schedule.png
  results/unknown_disease/sweep_report/embedding_metrics_bars.png
  results/unknown_disease/sweep_report/umap_grid_seed42.png
  results/unknown_disease/sweep_report/text_info_measures.png
  results/unknown_disease/CASE_SUMMARY_SWEEP_REPORT.md
"""
from __future__ import annotations

import gzip
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/unknown_disease")
OUT_DIR     = RESULTS_DIR / "sweep_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULES = ["gaussian", "flat", "sir"]
DATATYPES = ["baseline", "template", "ollama"]
SEEDS     = [42, 43, 44]

_DISEASE_COLORS = {"velarex": "#e63946", "sornathis": "#4361ee", "morven": "#2a9d8f"}
_SEV_ALPHA       = {"mild": 0.45, "moderate": 0.75, "severe": 1.00}


# ── 1. Aggregate per-run metrics ────────────────────────────────────────────

def load_run(schedule, datatype, seed):
    name = f"sweep_{schedule}_{datatype}_seed{seed}"
    d = RESULTS_DIR / name
    if not (d / "summary.json").exists():
        return None
    summary = json.load(open(d / "summary.json"))
    silhouette_curve = summary.get("silhouette_curve", [])
    return {"name": name, "dir": d, "summary": summary, "silhouette_curve": silhouette_curve}


def embedding_cluster_metrics(logits: np.ndarray, labels: list[str]) -> dict:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    group = np.array([1 if "morven" in lbl else 0 for lbl in labels])
    out = {}
    try:
        out["silhouette_morven"] = float(
            np.mean(__import__("sklearn.metrics", fromlist=["silhouette_samples"])
                    .silhouette_samples(logits, group)[group == 1])
        )
    except Exception:
        out["silhouette_morven"] = float("nan")
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(logits, group))
    except Exception:
        out["davies_bouldin"] = float("nan")
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(logits, group))
    except Exception:
        out["calinski_harabasz"] = float("nan")
    try:
        knn = KNeighborsClassifier(n_neighbors=5)
        out["knn5_acc_morven_vs_known"] = float(
            np.mean(cross_val_score(knn, logits, group, cv=5))
        )
    except Exception:
        out["knn5_acc_morven_vs_known"] = float("nan")
    return out


def main_metrics_table(probe_labels):
    rows = []
    curves = {}  # (schedule, datatype) -> list of (round -> sil) per seed
    for schedule in SCHEDULES:
        for datatype in DATATYPES:
            for seed in SEEDS:
                r = load_run(schedule, datatype, seed)
                if r is None:
                    print(f"MISSING: {schedule} {datatype} seed{seed}")
                    continue
                s = r["summary"]
                final_round = max((c["round"] for c in r["silhouette_curve"]), default=None)
                emb_metrics = {}
                if final_round is not None:
                    npz_path = r["dir"] / f"logits_r{final_round:02d}.npz"
                    if npz_path.exists():
                        data = np.load(npz_path)
                        logits = data["logits"]
                        if logits.shape[0] == len(probe_labels):
                            emb_metrics = embedding_cluster_metrics(logits, probe_labels)
                rows.append({
                    "schedule": schedule, "datatype": datatype, "seed": seed,
                    "final_diag_acc": s.get("final_diag_acc"),
                    "wall_seconds": s.get("wall_seconds"),
                    "sil_first": r["silhouette_curve"][0]["silhouette"] if r["silhouette_curve"] else float("nan"),
                    "sil_last":  r["silhouette_curve"][-1]["silhouette"] if r["silhouette_curve"] else float("nan"),
                    **emb_metrics,
                })
                curves.setdefault((schedule, datatype), []).append(r["silhouette_curve"])
    return rows, curves


# ── 2. Text samples + information measures ─────────────────────────────────

def gen_baseline_scheduled_samples(disease, n=40, seed=1):
    """Single-phrase baseline used by the legacy gaussian/flat path (no conversation)."""
    from run_unknown_disease import FictionalPhraseLibrary
    lib = FictionalPhraseLibrary(seed=seed)
    sevs = ["mild", "moderate", "severe"]
    return [lib.sample(disease, sevs[i % 3])["text"] for i in range(n)]


def gen_conversation_samples(disease, n=40, seed=1, case_summarizer=None, use_ollama=False):
    """Multi-turn conversation (SIR baseline when case_summarizer=None) or a
    compiled case summary (template/ollama) for one disease, via the same
    WorldEngine static-mode machinery used by the sweep."""
    from run_unknown_disease import _build_static_world, _event_to_record
    progressions = {"velarex": "Velarex", "sornathis": "Sornathis", "morven": "Morven"}
    world = _build_static_world([progressions[disease]], seed=seed,
                                 use_ollama=use_ollama, case_summarizer=case_summarizer)
    world._config.epidemic.cases_per_day = n
    events = world.run_sim_days(1)
    return [_event_to_record(ev)["text"] for ev in events if ev.gt_disease == disease]


def text_stats(samples: list[str]) -> dict:
    if not samples:
        return {}
    words_all = []
    char_lens = []
    comp_ratios = []
    for s in samples:
        words = s.lower().split()
        words_all.extend(words)
        char_lens.append(len(s))
        raw = s.encode("utf-8")
        comp = gzip.compress(raw)
        comp_ratios.append(len(comp) / max(1, len(raw)))
    vocab = set(words_all)
    ttr = len(vocab) / max(1, len(words_all))
    # word-level Shannon entropy
    counts = Counter(words_all)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return {
        "n_samples":        len(samples),
        "avg_chars":        float(np.mean(char_lens)),
        "vocab_size":       len(vocab),
        "type_token_ratio": ttr,
        "word_entropy_bits": entropy,
        "avg_gzip_ratio":   float(np.mean(comp_ratios)),
    }


def text_classification_accuracy(samples_by_disease: dict[str, list[str]]) -> dict:
    """TF-IDF + Logistic Regression CV accuracy: how much disease/unknown-class
    signal is present in the surface text alone, independent of any neural net."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    diseases = list(samples_by_disease.keys())
    texts, y_disease, y_unknown = [], [], []
    for d in diseases:
        for t in samples_by_disease[d]:
            texts.append(t)
            y_disease.append(d)
            y_unknown.append(1 if d == "morven" else 0)

    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    out = {}
    try:
        clf = LogisticRegression(max_iter=1000)
        out["tfidf_3class_acc"] = float(np.mean(cross_val_score(clf, X, y_disease, cv=5)))
    except Exception:
        out["tfidf_3class_acc"] = float("nan")
    try:
        if len(set(y_unknown)) > 1:
            clf2 = LogisticRegression(max_iter=1000)
            out["tfidf_unknown_vs_known_acc"] = float(np.mean(cross_val_score(clf2, X, y_unknown, cv=5)))
        else:
            out["tfidf_unknown_vs_known_acc"] = float("nan")
    except Exception:
        out["tfidf_unknown_vs_known_acc"] = float("nan")
    return out


def build_text_info_table():
    from run_unknown_disease import FictionalPhraseLibrary
    from simulation.case_summary import TemplateCaseSummarizer, OllamaCaseSummarizer

    diseases = ["velarex", "sornathis", "morven"]
    mechanisms = {
        "baseline_scheduled": lambda d: gen_baseline_scheduled_samples(d, n=40, seed=7),
        "baseline_sir":       lambda d: gen_conversation_samples(d, n=15, seed=7, case_summarizer=None),
        "template":           lambda d: gen_conversation_samples(d, n=15, seed=7, case_summarizer=TemplateCaseSummarizer()),
        "ollama":             lambda d: gen_conversation_samples(d, n=15, seed=7, case_summarizer=OllamaCaseSummarizer()),
    }

    rows = []
    samples_dump = {}
    for mech, fn in mechanisms.items():
        samples_by_disease = {}
        for d in diseases:
            print(f"  generating {mech} / {d} ...")
            samples_by_disease[d] = fn(d)
        samples_dump[mech] = samples_by_disease

        all_samples = sum(samples_by_disease.values(), [])
        stats = text_stats(all_samples)
        clf_stats = text_classification_accuracy(samples_by_disease)
        rows.append({"mechanism": mech, **stats, **clf_stats})

    return rows, samples_dump


# ── 3. Plots ─────────────────────────────────────────────────────────────────

def plot_silhouette_by_schedule(curves):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    colors = {"baseline": "#264653", "template": "#e76f51", "ollama": "#2a9d8f"}
    for ax, schedule in zip(axes, SCHEDULES):
        for datatype in DATATYPES:
            runs = curves.get((schedule, datatype), [])
            if not runs:
                continue
            rounds = sorted(set(c["round"] for run in runs for c in run))
            vals = []
            for rnd in rounds:
                xs = [c["silhouette"] for run in runs for c in run if c["round"] == rnd]
                vals.append(xs)
            means = [np.mean(v) if v else np.nan for v in vals]
            stds  = [np.std(v) if v else np.nan for v in vals]
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means, label=datatype, color=colors[datatype], marker="o", ms=4)
            ax.fill_between(rounds, means - stds, means + stds, color=colors[datatype], alpha=0.15)
        ax.set_title(f"{schedule}", fontsize=11)
        ax.set_xlabel("FL round")
        ax.axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("Morven silhouette (mean ± std, 3 seeds)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Information content over training: Morven separability by data type x distribution")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "silhouette_by_schedule.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_embedding_metric_bars(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    metrics = ["sil_last", "davies_bouldin", "calinski_harabasz", "knn5_acc_morven_vs_known"]
    titles  = ["Final silhouette (Morven)\nhigher = better", "Davies-Bouldin\nlower = better",
               "Calinski-Harabasz\nhigher = better", "kNN(5) acc Morven-vs-known\nhigher = better"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    width = 0.25
    x = np.arange(len(SCHEDULES))
    colors = {"baseline": "#264653", "template": "#e76f51", "ollama": "#2a9d8f"}
    for ax, metric, title in zip(axes, metrics, titles):
        for j, datatype in enumerate(DATATYPES):
            means, stds = [], []
            for schedule in SCHEDULES:
                vals = df[(df.schedule == schedule) & (df.datatype == datatype)][metric].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std() if len(vals) else np.nan)
            ax.bar(x + (j - 1) * width, means, width, yerr=stds, label=datatype, color=colors[datatype], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(SCHEDULES)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "embedding_metrics_bars.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_umap_grid_seed42(probe_labels):
    from run_unknown_disease import _project_umap

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for i, schedule in enumerate(SCHEDULES):
        for j, datatype in enumerate(DATATYPES):
            ax = axes[i][j]
            r = load_run(schedule, datatype, 42)
            if r is None:
                ax.axis("off")
                continue
            final_round = max((c["round"] for c in r["silhouette_curve"]), default=None)
            npz_path = r["dir"] / f"logits_r{final_round:02d}.npz" if final_round else None
            if not npz_path or not npz_path.exists():
                ax.axis("off")
                continue
            data = np.load(npz_path)
            logits = data["logits"]
            if logits.shape[0] != len(probe_labels):
                ax.axis("off")
                continue
            coords = _project_umap(logits, seed=42)
            for k, lbl in enumerate(probe_labels):
                disease = lbl.split("/")[0]
                sev = lbl.split("/")[1] if "/" in lbl else "mild"
                ax.scatter(coords[k, 0], coords[k, 1],
                           color=_DISEASE_COLORS.get(disease, "#888"),
                           alpha=_SEV_ALPHA.get(sev, 0.7),
                           s=55 if disease == "morven" else 20,
                           marker="D" if disease == "morven" else "o",
                           edgecolors="black" if disease == "morven" else "none",
                           linewidths=0.6 if disease == "morven" else 0)
            ax.set_title(f"{schedule} / {datatype}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Final-round probe embeddings (seed 42) — red=velarex, blue=sornathis, teal diamond=morven", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_grid_seed42.png", dpi=140, facecolor="white")
    plt.close(fig)


def plot_text_info_measures(text_rows):
    import pandas as pd
    df = pd.DataFrame(text_rows).set_index("mechanism")
    metrics = ["type_token_ratio", "avg_gzip_ratio", "word_entropy_bits"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    for ax, m in zip(axes, metrics):
        df[m].plot(kind="bar", ax=ax, color="#457b9d")
        ax.set_title(m, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "text_info_measures.png", dpi=150, facecolor="white")
    plt.close(fig)


# ── 4. Report ────────────────────────────────────────────────────────────────

def write_report(rows, text_rows, samples_dump):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metrics_table.csv", index=False)
    tdf = pd.DataFrame(text_rows)
    tdf.to_csv(OUT_DIR / "text_info_table.csv", index=False)
    json.dump(samples_dump, open(OUT_DIR / "text_samples.json", "w"), indent=2)

    agg = df.groupby(["schedule", "datatype"]).agg(
        final_diag_acc=("final_diag_acc", "mean"),
        wall_seconds=("wall_seconds", "mean"),
        sil_first=("sil_first", "mean"),
        sil_last=("sil_last", "mean"),
        davies_bouldin=("davies_bouldin", "mean"),
        calinski_harabasz=("calinski_harabasz", "mean"),
        knn5_acc=("knn5_acc_morven_vs_known", "mean"),
    ).round(3)

    lines = []
    lines.append("# Case-Summary x Distribution Sweep — Information Content Report\n")
    lines.append("27 runs: 3 distributions (gaussian / flat / sir) x 3 data types "
                 "(baseline phrase-bank, template case summary, ollama/phi3:mini case summary) x 3 seeds (42/43/44).\n")
    lines.append("**Caveat:** \"baseline\" text differs structurally by distribution — gaussian/flat baseline "
                 "is a single sampled phrase (legacy `run_unknown_disease()` path), while sir baseline is the "
                 "full multi-turn patient-nurse conversation. template/ollama always summarize the multi-turn "
                 "conversation regardless of distribution. See the text-mechanism section below.\n")

    lines.append("\n## 1. Aggregated metrics (mean across 3 seeds)\n")
    lines.append(agg.reset_index().to_markdown(index=False))

    lines.append("\n\n## 2. Morven separability over training\n")
    lines.append("![silhouette by schedule](sweep_report/silhouette_by_schedule.png)\n")
    lines.append("Mean morven-vs-known silhouette ± std across seeds, per FL round. "
                 "Baseline phrase-bank text reaches detectable separation earliest in most distributions; "
                 "template/ollama summaries lag early but several catch up by the final snapshot round.\n")

    lines.append("\n## 3. Embedding-space cluster quality (final round)\n")
    lines.append("![embedding metrics](sweep_report/embedding_metrics_bars.png)\n")
    lines.append("Four complementary embedding-space measures beyond silhouette: Davies-Bouldin "
                 "(lower = tighter/better-separated clusters), Calinski-Harabasz (higher = better), "
                 "and 5-fold cross-validated kNN accuracy distinguishing morven vs known disease directly "
                 "in the 768-d embedding space (a model-free check that the silhouette number isn't an artifact "
                 "of the 2D UMAP projection).\n")

    lines.append("\n## 4. UMAP embeddings, seed 42 (visual)\n")
    lines.append("![umap grid](sweep_report/umap_grid_seed42.png)\n")

    lines.append("\n## 5. Text information measures (surface text only, no neural net)\n")
    lines.append("Computed directly on generated text samples — TF-IDF + Logistic Regression accuracy "
                 "measures how much disease-identity signal exists in the raw words themselves, independent "
                 "of the DistilBERT classifier used in the FL pipeline.\n")
    lines.append(tdf.round(3).to_markdown(index=False))
    lines.append("\n![text info](sweep_report/text_info_measures.png)\n")

    lines.append("\n## 6. Text samples\n")
    for mech, by_disease in samples_dump.items():
        lines.append(f"\n### {mech}\n")
        for disease, samples in by_disease.items():
            lines.append(f"**{disease}**")
            for s in samples[:2]:
                snippet = s if len(s) < 400 else s[:400] + " […]"
                lines.append(f"> {snippet}\n")

    lines.append("\n## 7. Reading\n")
    lines.append(
        "- final_diag_acc is 1.0 in every cell — the FL classifier always reaches ceiling, so it doesn't "
        "discriminate data types; the silhouette/cluster metrics and the surface-text TF-IDF accuracy are "
        "the informative signals here.\n"
        "- Compare `tfidf_3class_acc` (text-only, no training) against `sil_last`/`knn5_acc` (post-FL-training "
        "embedding separability) to see how much of the final separation is already present in the raw text "
        "vs. only emerges after federated training.\n"
    )

    (RESULTS_DIR / "CASE_SUMMARY_SWEEP_REPORT.md").write_text("\n".join(lines))
    print(f"\nReport written to {RESULTS_DIR / 'CASE_SUMMARY_SWEEP_REPORT.md'}")


def main():
    from run_unknown_disease import generate_fictional_probe_events
    probe_events = generate_fictional_probe_events(12, 999)
    probe_labels = [ev.ground_truth for ev in probe_events]

    print("Aggregating run metrics...")
    rows, curves = main_metrics_table(probe_labels)

    print("Generating text samples + information measures (this calls Ollama for ~45 samples)...")
    text_rows, samples_dump = build_text_info_table()

    print("Plotting...")
    plot_silhouette_by_schedule(curves)
    plot_embedding_metric_bars(rows)
    plot_umap_grid_seed42(probe_labels)
    plot_text_info_measures(text_rows)

    write_report(rows, text_rows, samples_dump)


if __name__ == "__main__":
    main()
