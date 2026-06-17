"""
PrototypeBank — embedding-space classifier with dynamic class discovery.

Each named class has a centroid (mean [CLS] embedding). Classification is
nearest-centroid (cosine distance). New classes are discovered by running
DBSCAN on all probe embeddings and comparing cluster count against the number
of named prototypes. Attribution is a rename call — no retraining required.

FedAvg for prototypes: weighted average of centroid positions (weight = n_examples),
one operation per class name. Classes absent from a silo are excluded from that
silo's contribution so novel sub-clusters from one silo propagate to the global bank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _Proto:
    name:     str
    centroid: np.ndarray   # mean [CLS] embedding (hidden_size-dim)
    n:        int           # examples used


class PrototypeBank:
    def __init__(
        self,
        pca_components:      int   = 50,
        dbscan_eps:          float = 0.30,
        dbscan_min_samples:  int   = 5,
    ):
        self._protos:         dict[str, _Proto] = {}
        self._pca_components: int               = pca_components
        self._pca                               = None   # sklearn PCA, fitted lazily
        self.dbscan_eps         = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    # ── Centroid management ───────────────────────────────────────────────────

    def update(self, name: str, embeddings: np.ndarray) -> None:
        """Recompute centroid from a full batch of embeddings."""
        if len(embeddings) == 0:
            return
        self._protos[name] = _Proto(
            name=name, centroid=embeddings.mean(axis=0), n=len(embeddings)
        )

    def names(self) -> list[str]:
        return list(self._protos.keys())

    def has(self, name: str) -> bool:
        return name in self._protos

    def rename(self, old_name: str, new_name: str) -> None:
        """Attribution: give a sub-cluster its confirmed disease name (no retraining)."""
        if old_name not in self._protos:
            raise KeyError(old_name)
        p = self._protos.pop(old_name)
        self._protos[new_name] = _Proto(name=new_name, centroid=p.centroid, n=p.n)

    # ── PCA projection (for DBSCAN) ───────────────────────────────────────────

    def _ensure_pca(self, embeddings: np.ndarray) -> None:
        if self._pca is not None:
            return
        from sklearn.decomposition import PCA
        n_comp = min(self._pca_components, embeddings.shape[0] - 1, embeddings.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=0)
        self._pca.fit(embeddings)

    def _project(self, embeddings: np.ndarray) -> np.ndarray:
        if self._pca is None:
            return embeddings
        return self._pca.transform(embeddings)

    # ── Nearest-centroid classification ───────────────────────────────────────

    def classify(self, embeddings: np.ndarray) -> list[str]:
        """Return nearest-centroid class name for each row in embeddings."""
        if not self._protos:
            return ["unknown"] * len(embeddings)
        names = list(self._protos.keys())
        C = np.stack([self._protos[n].centroid for n in names])   # [K, H]
        # Cosine similarity: normalise both sides
        E_n = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        C_n = C          / (np.linalg.norm(C,          axis=1, keepdims=True) + 1e-12)
        sims = E_n @ C_n.T   # [N, K]
        return [names[int(i)] for i in sims.argmax(axis=1)]

    def accuracy(self, embeddings: np.ndarray, true_labels: list[str]) -> float:
        if not true_labels:
            return float("nan")
        preds = self.classify(embeddings)
        return sum(p == t for p, t in zip(preds, true_labels)) / len(true_labels)

    # ── DBSCAN cluster discovery ──────────────────────────────────────────────

    def dbscan_cluster_count(self, embeddings: np.ndarray) -> int:
        """
        Run DBSCAN on all embeddings (projected to PCA space, cosine distance).
        Returns the number of non-noise clusters found.
        """
        from sklearn.cluster import DBSCAN

        if len(embeddings) < 2 * self.dbscan_min_samples:
            return 1

        self._ensure_pca(embeddings)
        proj = self._project(embeddings)
        proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)

        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="cosine",
        ).fit_predict(proj)

        return int(len(set(labels) - {-1}))

    def check_split(
        self, name: str, embeddings: np.ndarray
    ) -> Optional[list[tuple[np.ndarray, int]]]:
        """
        Run DBSCAN on embeddings classified as `name`.
        Returns list of (centroid, n) per sub-cluster if >1 found, else None.
        """
        from sklearn.cluster import DBSCAN

        if len(embeddings) < 2 * self.dbscan_min_samples:
            return None

        self._ensure_pca(embeddings)
        proj = self._project(embeddings)
        proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)

        cluster_labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="cosine",
        ).fit_predict(proj)

        ids = sorted(c for c in set(cluster_labels) if c != -1)
        if len(ids) <= 1:
            return None

        return [
            (embeddings[cluster_labels == cid].mean(axis=0), int((cluster_labels == cid).sum()))
            for cid in ids
        ]

    def split(self, name: str, sub_clusters: list[tuple[np.ndarray, int]]) -> list[str]:
        """
        Replace `name` with `name_0`, `name_1`, … using the given sub-cluster centroids.
        Returns the new names in order (largest cluster first).
        """
        sub_clusters = sorted(sub_clusters, key=lambda x: -x[1])   # largest first
        del self._protos[name]
        new_names = []
        for i, (centroid, n) in enumerate(sub_clusters):
            new_name = f"{name}_{i}"
            self._protos[new_name] = _Proto(name=new_name, centroid=centroid, n=n)
            new_names.append(new_name)
        return new_names

    # ── FedAvg ────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, dict]:
        return {
            name: {"centroid": p.centroid.copy(), "n": p.n}
            for name, p in self._protos.items()
        }

    @classmethod
    def from_dict(cls, d: dict[str, dict], **kwargs) -> "PrototypeBank":
        bank = cls(**kwargs)
        for name, v in d.items():
            bank._protos[name] = _Proto(name=name, centroid=np.asarray(v["centroid"]), n=v["n"])
        return bank

    @classmethod
    def fedavg(
        cls,
        banks: "list[PrototypeBank]",
        weights: list[int],
        **kwargs,
    ) -> "PrototypeBank":
        """
        Weighted average of centroid positions across silos.
        Classes absent from a silo are skipped (not zeroed) so that a novel
        sub-cluster discovered by one silo propagates into the global bank.
        """
        all_names: set[str] = set()
        for b in banks:
            all_names.update(b.names())

        merged: dict[str, dict] = {}
        for name in all_names:
            contributors = [
                (b._protos[name].centroid, w, b._protos[name].n)
                for b, w in zip(banks, weights)
                if b.has(name)
            ]
            if not contributors:
                continue
            contrib_w = sum(w for _, w, _ in contributors)
            centroid = np.zeros_like(contributors[0][0], dtype=float)
            for c, w, _ in contributors:
                centroid = centroid + c * (w / contrib_w)
            n = sum(n_ for _, _, n_ in contributors)
            merged[name] = {"centroid": centroid, "n": n}

        result = cls.from_dict(merged, **kwargs)
        # Propagate fitted PCA from first bank that has one
        for b in banks:
            if b._pca is not None:
                result._pca = b._pca
                break
        return result

    def __repr__(self) -> str:
        return f"PrototypeBank({list(self._protos)})"
