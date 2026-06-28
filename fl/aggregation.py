"""
Server-side aggregation strategies for FL backbone weights.

FedAvg   : weighted average of local weights (fl.train._fedavg). The baseline.
FedProx  : SAME server-side rule as FedAvg. The entire FedProx mechanism is a
           proximal term added to each silo's LOCAL objective (see
           FLLearner.prox_mu / set_weights() anchor), which constrains how far
           local training can drift from the global weights before submission.
           Provided here as an explicit alias so experiment configs can name
           it directly.
FedSGD   : SAME server-side rule as FedAvg. The distinguishing mechanism is
           local: a single gradient step per round on plain SGD (no momentum,
           no Adam), vs. FedAvg's multiple local AdamW epochs (see
           FLLearner.local_epochs=1, optimizer="sgd", max_local_batches=1).
           With a shared starting point each round, averaging post-step
           weights is mathematically equivalent to averaging gradients.
FedAdam  : genuinely different server-side rule (Reddi et al. 2020,
           "Adaptive Federated Optimization"). Treats the FedAvg-aggregated
           weight delta as a pseudo-gradient and runs server-side Adam with
           its own persistent momentum/variance state -- separate from any
           silo's local optimizer state.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def fedavg(weights_list: list[list[np.ndarray]], n_examples: list[int]) -> list[np.ndarray]:
    """Weighted average of LoRA weight arrays (FedAvg eq. 1)."""
    total = sum(n_examples)
    if total == 0:
        return weights_list[0]
    return [
        sum(w[i] * (n / total) for w, n in zip(weights_list, n_examples))
        for i in range(len(weights_list[0]))
    ]


# FedProx and FedSGD use the identical server-side rule as FedAvg -- see module
# docstring. Named separately so an experiment config can select them by name
# without implying a different aggregation formula exists.
fedprox_aggregate = fedavg
fedsgd_aggregate  = fedavg


class FedAdamServer:
    """
    Server-side Adam (Reddi et al. 2020). Stateful across rounds -- one
    instance per training run (not per silo).

    pseudo-gradient g_t = global_w_t - fedavg(local_weights)
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    w_{t+1} = w_t - server_lr * m_t / (sqrt(v_t) + eps)

    v is initialised to tau^2 (small positive constant, per Reddi et al.) for
    numerical stability rather than zero.
    """

    def __init__(
        self,
        server_lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-3,
        tau: float = 1e-3,
    ):
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.tau   = tau
        self._m: Optional[list[np.ndarray]] = None
        self._v: Optional[list[np.ndarray]] = None
        self.t  = 0

    def step(
        self,
        global_w: list[np.ndarray],
        weights_list: list[list[np.ndarray]],
        n_examples: list[int],
    ) -> list[np.ndarray]:
        avg_w = fedavg(weights_list, n_examples)
        g = [gw - aw for gw, aw in zip(global_w, avg_w)]   # pseudo-gradient

        if self._m is None:
            self._m = [np.zeros_like(gi) for gi in g]
            self._v = [np.full_like(gi, self.tau ** 2) for gi in g]

        self.t += 1
        new_w = []
        for i, gi in enumerate(g):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * gi
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (gi ** 2)
            update = self.server_lr * self._m[i] / (np.sqrt(self._v[i]) + self.eps)
            new_w.append(global_w[i] - update)
        return new_w

    def reset(self) -> None:
        self._m = None
        self._v = None
        self.t = 0
