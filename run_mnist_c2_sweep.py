#!/usr/bin/env python3
"""
MNIST analog of the C2 data-volume sweep (run_falsification_c2_sweep.py).

Tests whether the "federation only helps in a narrow data-volume band"
finding from the clinical-text pipeline (DistilBERT + LoRA) generalizes to a
completely different architecture and domain (small CNN + MNIST digits) --
a stronger check than swapping to a different transformer variant, since it
varies architecture AND modality at once, reusing infrastructure already
built for the MNIST cross-domain replication (run_mnist_falsification.py).

Two known digits (0, 1), no novel/injection class at all -- pure federated
vs. isolated general two-class clustering quality, swept over total images
per silo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fl.aggregation import fedavg
from run_mnist_falsification import SmallCNN, get_weights, set_weights, load_mnist

OUT_DIR = Path("results/mnist_falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_DIGITS = [0, 1]


def run_condition(federated: bool, seed: int, n_rounds: int, n_silos: int,
                  images_per_silo: int, local_epochs: int, device: str) -> dict:
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    x_tr, y_tr, x_te, y_te = load_mnist()

    def pool_for(digit):
        idx = (y_tr == digit).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(idx)
        return idx

    pools = {d: pool_for(d) for d in KNOWN_DIGITS}
    silo_pools = {d: [pools[d][i::n_silos][:images_per_silo // 2] for i in range(n_silos)] for d in KNOWN_DIGITS}

    n_probe = 40
    probe_idx, probe_labels = [], []
    for d in KNOWN_DIGITS:
        idx = (y_te == d).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(idx)
        probe_idx.extend(idx[:n_probe].tolist())
        probe_labels.extend([d] * min(n_probe, len(idx)))
    probe_x = x_te[probe_idx].to(dev)
    probe_y = np.array(probe_labels)

    images_per_round = max(1, (images_per_silo // 2) // n_rounds + 1)
    cursors = {d: [0] * n_silos for d in KNOWN_DIGITS}
    models = [SmallCNN(num_classes=2).to(dev) for _ in range(n_silos)]
    silo_weights = [None] * n_silos
    snap_rounds = [2, 5, 8, 10, 12, 15, 18, 20]
    curve = []

    tag = "FEDERATED" if federated else "ISOLATED"
    print(f"\n  MNIST-C2 [{tag}] images_per_silo={images_per_silo} seed={seed}")

    for r in range(n_rounds):
        rnd = r + 1
        round_weights, train_sizes = [], []
        for i, model in enumerate(models):
            if silo_weights[i] is not None:
                set_weights(model, silo_weights[i])

            xs, ys = [], []
            for d in KNOWN_DIGITS:
                pool = silo_pools[d][i]
                c = cursors[d][i]
                batch_idx = pool[c: c + images_per_round]
                cursors[d][i] = c + len(batch_idx)
                if len(batch_idx) > 0:
                    xs.append(x_tr[batch_idx])
                    ys.extend([d] * len(batch_idx))

            if not xs:
                train_sizes.append(0)
                round_weights.append(get_weights(model))
                continue

            batch_x = torch.cat(xs, dim=0).to(dev)
            batch_y = torch.tensor(ys, dtype=torch.long, device=dev)

            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loader = DataLoader(TensorDataset(batch_x, batch_y), batch_size=16, shuffle=True)
            for _ in range(local_epochs):
                for bx, by in loader:
                    opt.zero_grad()
                    logits = model(bx)
                    loss = F.cross_entropy(logits, by)
                    loss.backward()
                    opt.step()

            train_sizes.append(batch_x.shape[0])
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
                models[-1].eval()
                with torch.no_grad():
                    _, emb = models[-1](probe_x, return_embedding=True)
                emb_np = emb.cpu().numpy()
                coords = PCA(n_components=2, random_state=seed).fit_transform(emb_np)
                sil = float(silhouette_score(coords, probe_y)) if len(set(probe_y.tolist())) > 1 else float("nan")
                km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(emb_np)
                ari = float(adjusted_rand_score(probe_y, km.labels_))
                per_silo_sil, per_silo_ari = [sil] * n_silos, [ari] * n_silos
            else:
                per_silo_sil, per_silo_ari = [], []
                for i, model in enumerate(models):
                    if silo_weights[i] is None:
                        continue
                    set_weights(model, silo_weights[i])
                    model.eval()
                    with torch.no_grad():
                        _, emb = model(probe_x, return_embedding=True)
                    emb_np = emb.cpu().numpy()
                    coords = PCA(n_components=2, random_state=seed).fit_transform(emb_np)
                    sil = float(silhouette_score(coords, probe_y)) if len(set(probe_y.tolist())) > 1 else float("nan")
                    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(emb_np)
                    ari = float(adjusted_rand_score(probe_y, km.labels_))
                    per_silo_sil.append(sil)
                    per_silo_ari.append(ari)

            mean_sil = float(np.nanmean(per_silo_sil))
            mean_ari = float(np.nanmean(per_silo_ari))
            curve.append({"round": rnd, "silhouette_mean": mean_sil, "kmeans_ari_mean": mean_ari})
            print(f"    R{rnd:02d}  mean_ari={mean_ari:.3f}")

    return {"federated": federated, "curve": curve,
           "final_silhouette": curve[-1]["silhouette_mean"] if curve else float("nan"),
           "final_kmeans_ari": curve[-1]["kmeans_ari_mean"] if curve else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-per-silo-grid", default="320,160,80,40,20")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--n-silos", type=int, default=3)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    grid = [int(x) for x in args.images_per_silo_grid.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    results = {}
    for ips in grid:
        fed_aris, iso_aris = [], []
        for seed in seeds:
            fed = run_condition(True, seed, args.n_rounds, args.n_silos, ips, args.local_epochs, args.device)
            iso = run_condition(False, seed, args.n_rounds, args.n_silos, ips, args.local_epochs, args.device)
            fed_aris.append(fed["final_kmeans_ari"])
            iso_aris.append(iso["final_kmeans_ari"])
            print(f"  >> images_per_silo={ips} seed={seed}: fed_ari={fed['final_kmeans_ari']:.3f}  iso_ari={iso['final_kmeans_ari']:.3f}")

        results[ips] = {
            "fed_ari_mean": float(np.mean(fed_aris)), "fed_ari_std": float(np.std(fed_aris)),
            "iso_ari_mean": float(np.mean(iso_aris)), "iso_ari_std": float(np.std(iso_aris)),
            "fed_aris": fed_aris, "iso_aris": iso_aris,
        }
        gap = results[ips]["fed_ari_mean"] - results[ips]["iso_ari_mean"]
        print(f"\n=== images_per_silo={ips}: fed={results[ips]['fed_ari_mean']:.3f}±{results[ips]['fed_ari_std']:.3f}  "
             f"iso={results[ips]['iso_ari_mean']:.3f}±{results[ips]['iso_ari_std']:.3f}  gap={gap:+.3f} ===\n")

    out_path = OUT_DIR / "mnist_c2_images_per_silo_sweep.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n=== SWEEP SUMMARY ===")
    print(f"{'images/silo':>12}{'fed_ari':>16}{'iso_ari':>16}{'gap':>10}")
    for ips in grid:
        r = results[ips]
        gap = r["fed_ari_mean"] - r["iso_ari_mean"]
        print(f"{ips:>12}{r['fed_ari_mean']:>10.3f}±{r['fed_ari_std']:<5.3f}{r['iso_ari_mean']:>10.3f}±{r['iso_ari_std']:<5.3f}{gap:>+10.3f}")


if __name__ == "__main__":
    main()
