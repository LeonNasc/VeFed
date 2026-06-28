#!/usr/bin/env python3
"""
Sentiment-analysis (IMDB) analog of the C2 data-volume sweep.

Tests whether the ~60-events/silo threshold found on the disease-text task is
tied to clinical/disease vocabulary specifically, or holds for DistilBERT +
LoRA text classification more broadly. Same architecture and modality as the
disease task (unlike the MNIST check, which varied both) -- isolates the
"is it the content domain" question specifically.

Bypasses the WorldEngine/simulation framework entirely (no clinical agents
needed) and feeds IMDB review text + binary label directly into
fl.lora.build_model()'s LoRA-adapted DistilBERT, mirroring FLLearner's actual
training hyperparameters (lr=1e-4, batch_size=8, AdamW, local_epochs=3) for a
fair comparison against the disease-text result.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from fl.aggregation import fedavg
from fl.lora import LoRAConfig, build_model

OUT_DIR = Path("results/sentiment_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_TOKENIZER = None
_IMDB_CACHE = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return _TOKENIZER


def load_imdb():
    global _IMDB_CACHE
    if _IMDB_CACHE is None:
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/imdb")
        _IMDB_CACHE = ds
    return _IMDB_CACHE


def get_weights(model) -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters() if p.requires_grad]


def set_weights(model, weights: list[np.ndarray]) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p, w in zip(trainable, weights):
        p.data = torch.as_tensor(w, dtype=p.dtype).to(p.device)


def tokenize(texts: list[str], device):
    tok = get_tokenizer()
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def extract_cls(model, texts: list[str], device) -> np.ndarray:
    enc = tokenize(texts, device)
    model.eval()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[-1][:, 0, :].cpu().numpy()


def supervised_accuracy(model, texts: list[str], labels: list[int], device) -> float:
    """Direct argmax accuracy on the model's own classification head -- unlike
    unsupervised KMeans ARI, this doesn't require sentiment to be the dominant
    axis of CLS-embedding variance, which natural-language sentiment text
    (unlike engineered disease text or simple digit images) does not satisfy."""
    enc = tokenize(texts, device)
    model.eval()
    with torch.no_grad():
        out = model(**enc)
    preds = out.logits.argmax(dim=-1).cpu().numpy()
    return float((preds == np.array(labels)).mean())


def train_step(model, texts: list[str], labels: list[int], device, local_epochs: int, lr: float, batch_size: int):
    enc = tokenize(texts, device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr)

    n = len(labels)
    idx_all = list(range(n))
    model.train()
    for _ in range(local_epochs):
        random.shuffle(idx_all)
        for start in range(0, n, batch_size):
            batch_idx = idx_all[start:start + batch_size]
            batch_enc = {k: v[batch_idx] for k, v in enc.items()}
            batch_y = y[batch_idx]
            optimizer.zero_grad()
            out = model(**batch_enc, labels=batch_y)
            out.loss.backward()
            optimizer.step()


def run_condition(federated: bool, seed: int, n_rounds: int, n_silos: int,
                  reviews_per_silo: int, local_epochs: int, device: str) -> dict:
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = torch.device(device)

    imdb = load_imdb()
    train_texts = list(imdb["train"]["text"])
    train_labels = list(imdb["train"]["label"])
    test_texts = list(imdb["test"]["text"])
    test_labels = list(imdb["test"]["label"])

    by_label_train = {0: [], 1: []}
    for t, l in zip(train_texts, train_labels):
        by_label_train[l].append(t)
    for l in (0, 1):
        rng.shuffle(by_label_train[l])

    silo_pools = {
        l: [by_label_train[l][i::n_silos][:reviews_per_silo // 2] for i in range(n_silos)]
        for l in (0, 1)
    }

    by_label_test = {0: [], 1: []}
    for t, l in zip(test_texts, test_labels):
        by_label_test[l].append(t)
    n_probe = 30
    probe_texts, probe_labels = [], []
    for l in (0, 1):
        rng.shuffle(by_label_test[l])
        probe_texts.extend(by_label_test[l][:n_probe])
        probe_labels.extend([l] * n_probe)
    probe_y = np.array(probe_labels)

    reviews_per_round = max(1, (reviews_per_silo // 2) // n_rounds + 1)
    cursors = {l: [0] * n_silos for l in (0, 1)}

    lora_cfg = LoRAConfig(num_labels=2)
    models = [build_model(lora_cfg).to(dev) for _ in range(n_silos)]
    silo_weights = [None] * n_silos
    snap_rounds = [2, 5, 8, 10, 12, 15, 18, 20]
    curve = []

    tag = "FEDERATED" if federated else "ISOLATED"
    print(f"\n  Sentiment-C2 [{tag}] reviews_per_silo={reviews_per_silo} seed={seed}")

    for r in range(n_rounds):
        rnd = r + 1
        round_weights, train_sizes = [], []
        for i, model in enumerate(models):
            if silo_weights[i] is not None:
                set_weights(model, silo_weights[i])

            texts, labels = [], []
            for l in (0, 1):
                pool = silo_pools[l][i]
                c = cursors[l][i]
                batch = pool[c: c + reviews_per_round]
                cursors[l][i] = c + len(batch)
                texts.extend(batch)
                labels.extend([l] * len(batch))

            if not texts:
                train_sizes.append(0)
                round_weights.append(get_weights(model))
                continue

            train_step(model, texts, labels, dev, local_epochs, lr=1e-4, batch_size=8)
            train_sizes.append(len(texts))
            round_weights.append(get_weights(model))

        if federated:
            active_idx = [i for i, s in enumerate(train_sizes) if s > 0]
            if active_idx:
                global_w = fedavg([round_weights[i] for i in active_idx], [train_sizes[i] for i in active_idx])
                silo_weights = [global_w] * n_silos
        else:
            silo_weights = round_weights

        if rnd in snap_rounds and any(w is not None for w in silo_weights):
            if federated:
                set_weights(models[-1], silo_weights[-1])
                emb_np = extract_cls(models[-1], probe_texts, dev)
                coords = PCA(n_components=2, random_state=seed).fit_transform(emb_np)
                sil = float(silhouette_score(coords, probe_y)) if len(set(probe_y.tolist())) > 1 else float("nan")
                km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(emb_np)
                ari = float(adjusted_rand_score(probe_y, km.labels_))
                acc = supervised_accuracy(models[-1], probe_texts, probe_labels, dev)
                per_silo_sil, per_silo_ari, per_silo_acc = [sil] * n_silos, [ari] * n_silos, [acc] * n_silos
            else:
                per_silo_sil, per_silo_ari, per_silo_acc = [], [], []
                for i, model in enumerate(models):
                    if silo_weights[i] is None:
                        continue
                    set_weights(model, silo_weights[i])
                    emb_np = extract_cls(model, probe_texts, dev)
                    coords = PCA(n_components=2, random_state=seed).fit_transform(emb_np)
                    sil = float(silhouette_score(coords, probe_y)) if len(set(probe_y.tolist())) > 1 else float("nan")
                    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(emb_np)
                    ari = float(adjusted_rand_score(probe_y, km.labels_))
                    acc = supervised_accuracy(model, probe_texts, probe_labels, dev)
                    per_silo_sil.append(sil)
                    per_silo_ari.append(ari)
                    per_silo_acc.append(acc)

            mean_sil = float(np.nanmean(per_silo_sil))
            mean_ari = float(np.nanmean(per_silo_ari))
            mean_acc = float(np.nanmean(per_silo_acc))
            curve.append({"round": rnd, "silhouette_mean": mean_sil, "kmeans_ari_mean": mean_ari, "acc_mean": mean_acc})
            print(f"    R{rnd:02d}  mean_ari={mean_ari:.3f}  mean_acc={mean_acc:.3f}")

    return {"federated": federated, "curve": curve,
           "final_silhouette": curve[-1]["silhouette_mean"] if curve else float("nan"),
           "final_kmeans_ari": curve[-1]["kmeans_ari_mean"] if curve else float("nan"),
           "final_acc": curve[-1]["acc_mean"] if curve else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews-per-silo-grid", default="160,80,40,20,10")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--n-silos", type=int, default=3)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    grid = [int(x) for x in args.reviews_per_silo_grid.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    results = {}
    for rps in grid:
        fed_aris, iso_aris, fed_accs, iso_accs = [], [], [], []
        for seed in seeds:
            fed = run_condition(True, seed, args.n_rounds, args.n_silos, rps, args.local_epochs, args.device)
            iso = run_condition(False, seed, args.n_rounds, args.n_silos, rps, args.local_epochs, args.device)
            fed_aris.append(fed["final_kmeans_ari"])
            iso_aris.append(iso["final_kmeans_ari"])
            fed_accs.append(fed["final_acc"])
            iso_accs.append(iso["final_acc"])
            print(f"  >> reviews_per_silo={rps} seed={seed}: fed_acc={fed['final_acc']:.3f}  iso_acc={iso['final_acc']:.3f}  "
                 f"(fed_ari={fed['final_kmeans_ari']:.3f}  iso_ari={iso['final_kmeans_ari']:.3f})")

        results[rps] = {
            "fed_ari_mean": float(np.mean(fed_aris)), "fed_ari_std": float(np.std(fed_aris)),
            "iso_ari_mean": float(np.mean(iso_aris)), "iso_ari_std": float(np.std(iso_aris)),
            "fed_acc_mean": float(np.mean(fed_accs)), "fed_acc_std": float(np.std(fed_accs)),
            "iso_acc_mean": float(np.mean(iso_accs)), "iso_acc_std": float(np.std(iso_accs)),
            "fed_aris": fed_aris, "iso_aris": iso_aris, "fed_accs": fed_accs, "iso_accs": iso_accs,
        }
        gap_acc = results[rps]["fed_acc_mean"] - results[rps]["iso_acc_mean"]
        print(f"\n=== reviews_per_silo={rps}: fed_acc={results[rps]['fed_acc_mean']:.3f}±{results[rps]['fed_acc_std']:.3f}  "
             f"iso_acc={results[rps]['iso_acc_mean']:.3f}±{results[rps]['iso_acc_std']:.3f}  gap_acc={gap_acc:+.3f} ===\n")

    out_path = OUT_DIR / "sentiment_c2_reviews_per_silo_sweep.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n=== SWEEP SUMMARY (supervised accuracy, the meaningful metric for this domain) ===")
    print(f"{'reviews/silo':>12}{'fed_acc':>16}{'iso_acc':>16}{'gap':>10}")
    for rps in grid:
        r = results[rps]
        gap = r["fed_acc_mean"] - r["iso_acc_mean"]
        print(f"{rps:>12}{r['fed_acc_mean']:>10.3f}±{r['fed_acc_std']:<5.3f}{r['iso_acc_mean']:>10.3f}±{r['iso_acc_std']:<5.3f}{gap:>+10.3f}")
    print("\n(Unsupervised KMeans ARI is also saved but stays near 0 throughout -- sentiment is not the "
         "dominant axis of CLS-embedding variance for natural movie-review text, unlike engineered disease "
         "text or simple digit images. See write-up.)")


if __name__ == "__main__":
    main()
