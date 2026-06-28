#!/usr/bin/env python3
"""
MNIST generalization check for the C7 false-positive finding.

Tests whether the central methodological result -- silhouette/softmax
evaluation is fooled by relabeling a KNOWN class "unknown" mid-federation,
while PrototypeBank nearest-centroid is not -- reproduces on a standard FL
benchmark (MNIST digits), not just the synthetic clinical-text simulator.
Directly mirrors run_falsification_c7.py / run_aggregation_comparison.py's
design:

  known classes   : digits {0, 1}      (analog: Velarex, Sornathis)
  novel class     : digit 2, injected into silo_0 from round 10,
                    labelled "unknown"  (analog: Morven)
  C7 stress test  : digit 0 (a KNOWN class) injected into silo_0 from
                    round 10, ALSO labelled "unknown" -- same images,
                    contradictory label (analog: Velarex-relabeled-unknown)

Reuses fl.aggregation.fedavg and fl.prototype_bank.PrototypeBank directly --
both are generic over embedding vectors, nothing text/LoRA-specific.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fl.aggregation import fedavg
from fl.prototype_bank import PrototypeBank

OUT_DIR = Path("results/mnist_falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_KNOWN_DIGITS = [0, 1]   # "Velarex", "Sornathis"
DEFAULT_NOVEL_DIGIT  = 2        # "Morven" -- real novel class
ROLE_NAMES = ["velarex", "sornathis"]   # role labels for the two known digits, in order


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        emb = F.relu(self.fc1(x))
        logits = self.fc2(emb)
        if return_embedding:
            return logits, emb
        return logits


def get_weights(model) -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_weights(model, weights: list[np.ndarray]) -> None:
    for p, w in zip(model.parameters(), weights):
        p.data = torch.as_tensor(w, dtype=p.dtype).to(p.device)


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(root="/tmp/mnist_data", train=True, download=True)
    te = torchvision.datasets.MNIST(root="/tmp/mnist_data", train=False, download=True)
    x_tr = tr.data.float().unsqueeze(1) / 255.0
    y_tr = tr.targets
    x_te = te.data.float().unsqueeze(1) / 255.0
    y_te = te.targets
    return x_tr, y_tr, x_te, y_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["novel", "c7"])
    ap.add_argument("--n-silos", type=int, default=3)
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--injection-round", type=int, default=10)
    ap.add_argument("--injection-per-round", type=int, default=8)
    ap.add_argument("--events-per-round", type=int, default=20, help="known-digit images revealed per silo per round")
    ap.add_argument("--n-probe", type=int, default=30, help="held-out probes per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--known-digits", default=",".join(str(d) for d in DEFAULT_KNOWN_DIGITS),
                    help="comma-separated, exactly 2 digits")
    ap.add_argument("--novel-digit", type=int, default=DEFAULT_NOVEL_DIGIT,
                    help="digit used as the real novel class ('novel' scenario) or as the "
                         "probe-only held-out class ('c7' scenario)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    KNOWN_DIGITS = [int(d) for d in args.known_digits.split(",")]
    assert len(KNOWN_DIGITS) == 2, "exactly 2 known digits required"
    NOVEL_DIGIT = args.novel_digit
    DIGIT_NAME = {KNOWN_DIGITS[0]: ROLE_NAMES[0], KNOWN_DIGITS[1]: ROLE_NAMES[1], NOVEL_DIGIT: "morven"}
    C7_DIGIT = KNOWN_DIGITS[0]   # known digit relabeled "unknown" in the stress test

    inject_digit = NOVEL_DIGIT if args.scenario == "novel" else C7_DIGIT
    run_name = args.run_name or f"mnist_{args.scenario}_seed{args.seed}"
    out_dir = OUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr, x_te, y_te = load_mnist()

    def pool_for(digit):
        idx = (y_tr == digit).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(idx)
        return idx

    known_pools = {d: pool_for(d) for d in KNOWN_DIGITS}
    inject_pool = pool_for(inject_digit)

    # Held-out probes (test split, never used in training) for velarex/sornathis/(real)morven.
    probe_idx, probe_labels = [], []
    for d in KNOWN_DIGITS + [NOVEL_DIGIT]:
        idx = (y_te == d).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(idx)
        probe_idx.extend(idx[:args.n_probe].tolist())
        probe_labels.extend([DIGIT_NAME[d]] * min(args.n_probe, len(idx)))
    probe_x = x_te[probe_idx].to(dev)

    # Split known-digit pools round-robin across silos.
    silo_known_pools = {
        d: [known_pools[d][i::args.n_silos] for i in range(args.n_silos)]
        for d in KNOWN_DIGITS
    }
    cursors = {d: [0] * args.n_silos for d in KNOWN_DIGITS}
    inject_cursor = 0

    models = [SmallCNN(num_classes=3).to(dev) for _ in range(args.n_silos)]
    silo_banks  = [PrototypeBank(pca_components=20, dbscan_eps=0.30, dbscan_min_samples=5)
                   for _ in range(args.n_silos)]
    global_bank = PrototypeBank(pca_components=20, dbscan_eps=0.30, dbscan_min_samples=5)
    global_w = None

    velarex_idx_p = [i for i, l in enumerate(probe_labels) if l == "velarex"]
    morven_idx_p  = [i for i, l in enumerate(probe_labels) if l == "morven"]
    snap_rounds = [5, 8, 10, 12, 15, 18, 20]
    round_metrics = []

    print(f"\n{'='*60}\n  MNIST falsification check -- scenario={args.scenario} "
         f"(inject digit {inject_digit} as 'unknown')\n{'='*60}\n")

    for r in range(args.n_rounds):
        rnd = r + 1
        round_weights, train_sizes = [], []
        for i, model in enumerate(models):
            if global_w is not None:
                set_weights(model, global_w)

            xs, ys = [], []
            for d in KNOWN_DIGITS:
                pool = silo_known_pools[d][i]
                c = cursors[d][i]
                batch_idx = pool[c: c + args.events_per_round]
                cursors[d][i] = c + len(batch_idx)
                xs.append(x_tr[batch_idx])
                ys.extend([DIGIT_NAME[d]] * len(batch_idx))

            if rnd >= args.injection_round and i == 0 and inject_cursor < len(inject_pool):
                end = min(inject_cursor + args.injection_per_round, len(inject_pool))
                batch_idx = inject_pool[inject_cursor:end]
                inject_cursor = end
                xs.append(x_tr[batch_idx])
                ys.extend(["unknown"] * len(batch_idx))

            if not xs or sum(x.shape[0] for x in xs) == 0:
                train_sizes.append(0)
                round_weights.append(get_weights(model))
                continue

            batch_x = torch.cat(xs, dim=0).to(dev)
            label2id = {"velarex": 0, "sornathis": 1, "unknown": 2}
            batch_y = torch.tensor([label2id[l] for l in ys], dtype=torch.long, device=dev)

            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loader = DataLoader(TensorDataset(batch_x, batch_y), batch_size=16, shuffle=True)
            for _ in range(args.local_epochs):
                for bx, by in loader:
                    opt.zero_grad()
                    logits = model(bx)
                    loss = F.cross_entropy(logits, by)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                _, emb = model(batch_x, return_embedding=True)
            emb_np = emb.cpu().numpy()
            for cls_name in set(ys):
                mask = np.array([l == cls_name for l in ys])
                silo_banks[i].update(cls_name, emb_np[mask])

            train_sizes.append(batch_x.shape[0])
            round_weights.append(get_weights(model))

        active_idx = [i for i, s in enumerate(train_sizes) if s > 0]
        if active_idx:
            global_w = fedavg([round_weights[i] for i in active_idx], [train_sizes[i] for i in active_idx])
            global_bank = PrototypeBank.fedavg(
                [silo_banks[i] for i in active_idx], [train_sizes[i] for i in active_idx],
                pca_components=20, dbscan_eps=0.30, dbscan_min_samples=5)

        probe_metrics = {}
        if global_w is not None and rnd in snap_rounds:
            set_weights(models[-1], global_w)
            models[-1].eval()
            with torch.no_grad():
                logits, emb = models[-1](probe_x, return_embedding=True)
            emb_np = emb.cpu().numpy()
            logits_np = logits.cpu().numpy()

            # Softmax/argmax false-positive check.
            pred_ids = logits_np.argmax(axis=1)
            id2label = {0: "velarex", 1: "sornathis", 2: "unknown"}
            softmax_preds = [id2label[i] for i in pred_ids]
            velarex_as_unk_softmax = sum(1 for i in velarex_idx_p if softmax_preds[i] == "unknown") / max(len(velarex_idx_p), 1)
            morven_as_unk_softmax  = sum(1 for i in morven_idx_p  if softmax_preds[i] == "unknown") / max(len(morven_idx_p), 1)

            # Silhouette (binary: "unknown"-injected digit vs rest) on raw embeddings.
            from sklearn.metrics import silhouette_samples
            inject_name = DIGIT_NAME[inject_digit] if args.scenario == "novel" else "velarex"
            group = np.array([1 if l == inject_name else 0 for l in probe_labels])
            sil = float("nan")
            if group.sum() >= 2 and (group == 0).sum() >= 2:
                sil = float(np.mean(silhouette_samples(emb_np, group)[group == 1]))

            # PrototypeBank nearest-centroid false-positive check.
            proto_preds = global_bank.classify(emb_np)
            velarex_as_unk_proto = sum(1 for i in velarex_idx_p if "unknown" in proto_preds[i]) / max(len(velarex_idx_p), 1)
            morven_as_unk_proto  = sum(1 for i in morven_idx_p  if "unknown" in proto_preds[i]) / max(len(morven_idx_p), 1)

            probe_metrics = {
                "softmax_velarex_as_unk": velarex_as_unk_softmax, "softmax_morven_as_unk": morven_as_unk_softmax,
                "proto_velarex_as_unk": velarex_as_unk_proto, "proto_morven_as_unk": morven_as_unk_proto,
                "silhouette_inject_vs_rest": sil, "proto_names": list(global_bank.names()),
            }
            print(f"  R{rnd:02d}: softmax[velarex_as_unk={velarex_as_unk_softmax:.3f} morven_as_unk={morven_as_unk_softmax:.3f}]  "
                 f"proto[velarex_as_unk={velarex_as_unk_proto:.3f} morven_as_unk={morven_as_unk_proto:.3f}]  sil={sil:.3f}")

        round_metrics.append({"round": rnd, "train_sizes": train_sizes, **probe_metrics})

    snap_metrics = [m for m in round_metrics if "proto_names" in m]
    final = snap_metrics[-1] if snap_metrics else {}
    summary = {"scenario": args.scenario, "inject_digit": inject_digit, "seed": args.seed,
              "round_metrics": round_metrics, "final": final}
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
