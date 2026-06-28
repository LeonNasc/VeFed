#!/usr/bin/env python3
"""Exact permutation tests for the diffusion-latency null result (n=3 per arm).

With only 3 seeds per condition, a t-test's normality assumption is not
defensible. An exact permutation test (enumerate all C(6,3)=20 ways to split
the 6 pooled values into two groups of 3) is appropriate at this sample size
and makes no distributional assumption -- the minimum achievable two-sided
p-value at this n is 1/20 = 0.05, which the writeup states explicitly so the
power ceiling isn't mistaken for a precise estimate.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path

import numpy as np

SEEDS = [42, 43, 44]


def load(cond: str, seed: int, thin: bool) -> float:
    suffix = "_epoch1_thin" if thin else ""
    d = json.load(open(f"results/diffusion_test/{cond}{suffix}_seed{seed}/summary.json"))
    return d["final_morven_recall"]


def exact_permutation_test(a: list[float], b: list[float]) -> dict:
    pooled = a + b
    n_a = len(a)
    observed = float(np.mean(a) - np.mean(b))
    extreme = 0
    total = 0
    for idx in itertools.combinations(range(len(pooled)), n_a):
        group_a = [pooled[i] for i in idx]
        group_b = [pooled[i] for i in range(len(pooled)) if i not in idx]
        diff = np.mean(group_a) - np.mean(group_b)
        if abs(diff) >= abs(observed) - 1e-9:
            extreme += 1
        total += 1
    return {"observed_diff": observed, "p_two_sided": extreme / total, "n_permutations": total}


def main():
    results = {}
    for regime, thin in [("generous", False), ("thin", True)]:
        vals = {cond: [load(cond, s, thin) for s in SEEDS] for cond in ["isolated", "naive", "pre_exposed"]}
        comparisons = {
            "naive_vs_pre_exposed": exact_permutation_test(vals["naive"], vals["pre_exposed"]),
            "isolated_vs_naive": exact_permutation_test(vals["isolated"], vals["naive"]),
            "isolated_vs_pre_exposed": exact_permutation_test(vals["isolated"], vals["pre_exposed"]),
        }
        results[regime] = {"values": vals, "tests": comparisons}
        print(f"\n=== {regime} regime ===")
        for cond, v in vals.items():
            print(f"  {cond}: {[round(x, 3) for x in v]}")
        for name, t in comparisons.items():
            print(f"  {name}: diff={t['observed_diff']:+.3f}  p={t['p_two_sided']:.3f}  (n_perm={t['n_permutations']})")

    out = Path("results/diffusion_test/significance_tests.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    print("\nNote: for a clean (zero-overlap) separation at n=3 per arm, the minimum achievable "
         "two-sided p-value is 2/20=0.10, not 0.05 -- splits come in complementary pairs of equal "
         "|difference|, so an odd count (and hence p=0.05) is structurally unreachable. "
         "treat all p-values here as bounded-power estimates, not precise significance levels.")


if __name__ == "__main__":
    main()
