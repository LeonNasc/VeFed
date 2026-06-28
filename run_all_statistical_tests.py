#!/usr/bin/env python3
"""
Consolidated statistical-testing pass across every n=3 comparison in the
session that previously had only descriptive mean+-std, no formal test.

Two test types, both honest about n=3's limits:
  - exact permutation test (two independent groups, n=3 each): enumerates
    all C(6,3)=20 splits of the pooled 6 values, the only assumption-free
    test that fits this sample size. Minimum achievable two-sided p is 0.05.
  - one-sample case (a single n=3 group vs a fixed null, e.g. "is this rate
    different from 0"): permutation tests don't apply (nothing to shuffle
    between groups), so this script reports the n=3 values directly plus a
    bootstrap CI, flagging explicitly how coarse a 3-point bootstrap is
    rather than dressing it up as a real p-value.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path

import numpy as np

RESULTS = {}


def exact_permutation_test(a: list[float], b: list[float]) -> dict:
    pooled = a + b
    n_a = len(a)
    observed = float(np.mean(a) - np.mean(b))
    extreme, total = 0, 0
    for idx in itertools.combinations(range(len(pooled)), n_a):
        group_a = [pooled[i] for i in idx]
        group_b = [pooled[i] for i in range(len(pooled)) if i not in idx]
        diff = np.mean(group_a) - np.mean(group_b)
        if abs(diff) >= abs(observed) - 1e-9:
            extreme += 1
        total += 1
    return {"observed_diff": observed, "p_two_sided": extreme / total, "n_permutations": total}


def bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"values": vals, "mean": float(np.mean(arr)), "bootstrap_ci_95": [float(lo), float(hi)],
           "note": f"bootstrap from only n={len(vals)} distinct values -- coarse, not a substitute for more seeds"}


def section_aggregation_comparison():
    print("\n" + "=" * 70 + "\n  AGGREGATION COMPARISON -- pairwise C7-FP tests\n" + "=" * 70)
    aggregators = ["fedavg", "fedprox", "fedsgd", "fedadam"]
    vals = {}
    for agg in aggregators:
        v = [json.load(open(f"results/aggregation_comparison/{agg}_c7_seed{s}/summary.json"))["final_velarex_as_unknown_rate"]
            for s in [42, 43, 44]]
        vals[agg] = v
        print(f"  {agg}: {[round(x,3) for x in v]}")

    pairwise = {}
    for a, b in itertools.combinations(aggregators, 2):
        t = exact_permutation_test(vals[a], vals[b])
        pairwise[f"{a}_vs_{b}"] = t
        print(f"  {a} vs {b}: diff={t['observed_diff']:+.3f}  p={t['p_two_sided']:.3f}")
    RESULTS["aggregation_c7_pairwise"] = {"values": vals, "tests": pairwise}

    print("\n  AGGREGATION COMPARISON -- pairwise real-Morven-TP tests")
    vals_tp = {}
    for agg in aggregators:
        v = [json.load(open(f"results/aggregation_comparison/{agg}_morven_seed{s}/summary.json"))["final_morven_as_unknown_rate"]
            for s in [42, 43, 44]]
        vals_tp[agg] = v
        print(f"  {agg}: {[round(x,3) for x in v]}")
    pairwise_tp = {}
    for a, b in itertools.combinations(aggregators, 2):
        t = exact_permutation_test(vals_tp[a], vals_tp[b])
        pairwise_tp[f"{a}_vs_{b}"] = t
        print(f"  {a} vs {b}: diff={t['observed_diff']:+.3f}  p={t['p_two_sided']:.3f}")
    RESULTS["aggregation_tp_pairwise"] = {"values": vals_tp, "tests": pairwise_tp}


def section_c3():
    print("\n" + "=" * 70 + "\n  C3 -- normal vs shuffled ARI\n" + "=" * 70)

    def load(seed):
        path = f"results/falsification/c3_shuffled_label_control_seed{seed}.json" if seed != 42 else "results/falsification/c3_shuffled_label_control.json"
        return json.load(open(path))

    normal = [load(s)["normal"]["final_kmeans_ari"] for s in [42, 43, 44]]
    shuffled = [load(s)["shuffled"]["final_kmeans_ari"] for s in [42, 43, 44]]
    print(f"  normal: {normal}")
    print(f"  shuffled: {shuffled}")
    t = exact_permutation_test(normal, shuffled)
    print(f"  diff={t['observed_diff']:+.3f}  p={t['p_two_sided']:.3f}")
    RESULTS["c3_normal_vs_shuffled"] = {"normal": normal, "shuffled": shuffled, "test": t}


def section_c2_main_and_c2b():
    print("\n" + "=" * 70 + "\n  C2 (eps=160) and C2b (3-disease) -- federated vs isolated\n" + "=" * 70)
    fed = [json.load(open(f"results/falsification/c2_isolated_training_control_seed{s}.json"))["federated"]["final_kmeans_ari"] for s in [42, 43, 44]]
    iso = [json.load(open(f"results/falsification/c2_isolated_training_control_seed{s}.json"))["isolated"]["final_kmeans_ari"] for s in [42, 43, 44]]
    t = exact_permutation_test(fed, iso)
    print(f"  C2 (eps=160): fed={fed}  iso={iso}  diff={t['observed_diff']:+.3f}  p={t['p_two_sided']:.3f}")
    RESULTS["c2_eps160_fed_vs_iso"] = {"federated": fed, "isolated": iso, "test": t}

    fed2 = [json.load(open(f"results/falsification/c2b_3disease_isolated_training_control_seed{s}.json"))["federated"]["final_kmeans_ari"] for s in [42, 43, 44]]
    iso2 = [json.load(open(f"results/falsification/c2b_3disease_isolated_training_control_seed{s}.json"))["isolated"]["final_kmeans_ari"] for s in [42, 43, 44]]
    t2 = exact_permutation_test(fed2, iso2)
    print(f"  C2b (3-disease): fed={fed2}  iso={iso2}  diff={t2['observed_diff']:+.3f}  p={t2['p_two_sided']:.3f}")
    RESULTS["c2b_fed_vs_iso"] = {"federated": fed2, "isolated": iso2, "test": t2}


def section_one_sample():
    print("\n" + "=" * 70 + "\n  ONE-SAMPLE CASES (n=3 vs a fixed null) -- bootstrap CI, not a p-value\n" + "=" * 70)

    c7_proto = [0.0, 0.0, 0.194]  # seeds 42,43,44, from c7_prototype_known_disease_control_seed*.json
    ci = bootstrap_ci(c7_proto)
    print(f"  C7 PrototypeBank false-positive rate: {ci}")
    RESULTS["c7_prototype_fp_one_sample"] = ci

    c8 = [json.load(open(f"results/falsification/c8_ood_absorption_seed{s}.json"))["final_ood_absorption"] for s in [42, 43, 44]]
    ci8 = bootstrap_ci(c8)
    print(f"  C8 OOD absorption rate: {ci8}")
    RESULTS["c8_ood_absorption_one_sample"] = ci8


def main():
    section_aggregation_comparison()
    section_c3()
    section_c2_main_and_c2b()
    section_one_sample()

    out = Path("results/all_statistical_tests.json")
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
