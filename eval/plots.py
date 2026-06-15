"""
Canonical plot functions for FL experiment evaluation.

Each function accepts a list of run dicts (from eval.loader.load_results) and
returns a matplotlib Figure.  Callers savefig() or show() as needed.

Functions
─────────
accuracy_curves(runs, ...)      per-round accuracy, one line per group
loss_curves(runs, ...)          per-round training loss, one line per group
bar_final_accuracy(runs, ...)   grouped bar of final accuracy (e.g. IID vs non-IID)
confusion_heatmap(preds, trues, class_names, ...)  annotated confusion matrix

All curve functions support a ``shade_std`` flag that draws a ±1 std band when
multiple runs share the same group key (i.e. different seeds).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

# ── Palette ───────────────────────────────────────────────────────────────────

_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _group_runs(runs: list[dict], by: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in runs:
        k = str(r.get(by, "?"))
        groups.setdefault(k, []).append(r)
    return dict(sorted(groups.items()))


def _round_series(
    runs: list[dict],
    key:  str,
) -> tuple[list[int], list[float], list[float]]:
    """
    Average a per-round metric across runs.

    Returns (rounds, means, stds).  Rounds with no valid values are skipped.
    """
    bucket: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        for row in run.get("rounds", []):
            v = row.get(key)
            if v is not None and not math.isnan(float(v)):
                bucket[int(row["round"])].append(float(v))

    if not bucket:
        return [], [], []

    xs     = sorted(bucket)
    means  = [sum(bucket[x]) / len(bucket[x]) for x in xs]
    stds   = [
        math.sqrt(sum((v - means[i]) ** 2 for v in bucket[xs[i]]) / len(bucket[xs[i]]))
        if len(bucket[xs[i]]) > 1 else 0.0
        for i in range(len(xs))
    ]
    return xs, means, stds


# ── Public plot functions ─────────────────────────────────────────────────────

def accuracy_curves(
    runs:      list[dict],
    group_by:  str           = "schedule",
    metric:    str           = "agg_diag_acc",
    title:     str           = "",
    shade_std: bool          = True,
    ax=None,
) -> "matplotlib.figure.Figure":
    """
    Per-round accuracy curves, one line per group.

    group_by:  metadata key to split runs into series
               (e.g. "schedule", "distribution", "experiment")
    metric:    round-level key to plot  (default: "agg_diag_acc")
    shade_std: draw ±1 std band when multiple seeds share a group
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for ci, (label, group) in enumerate(_group_runs(runs, group_by).items()):
        xs, ys, errs = _round_series(group, metric)
        if not xs:
            continue
        col = _color(ci)
        ax.plot(xs, ys, label=label, color=col, linewidth=1.8)
        if shade_std and any(e > 0 for e in errs):
            lo = [y - e for y, e in zip(ys, errs)]
            hi = [y + e for y, e in zip(ys, errs)]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15)

    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace("_", " ").removeprefix("agg "))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)

    return fig or ax.get_figure()


def loss_curves(
    runs:      list[dict],
    group_by:  str   = "schedule",
    metric:    str   = "mean_loss",
    title:     str   = "",
    shade_std: bool  = True,
    ax=None,
) -> "matplotlib.figure.Figure":
    """Per-round training loss curves, one line per group."""
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for ci, (label, group) in enumerate(_group_runs(runs, group_by).items()):
        xs, ys, errs = _round_series(group, metric)
        # Drop leading None/NaN (no training in early rounds)
        valid = [(x, y, e) for x, y, e in zip(xs, ys, errs)
                 if not math.isnan(y)]
        if not valid:
            continue
        xs, ys, errs = zip(*valid)
        col = _color(ci)
        ax.plot(xs, ys, label=label, color=col, linewidth=1.8)
        if shade_std and any(e > 0 for e in errs):
            lo = [y - e for y, e in zip(ys, errs)]
            hi = [y + e for y, e in zip(ys, errs)]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15)

    ax.set_xlabel("Round")
    ax.set_ylabel("Training loss")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)

    return fig or ax.get_figure()


def bar_final_accuracy(
    runs:     list[dict],
    group_by: str   = "schedule",
    split_by: str   = "distribution",
    metric:   str   = "final_diag_acc",
    title:    str   = "",
    ax=None,
) -> "matplotlib.figure.Figure":
    """
    Grouped bar chart of final accuracy.

    group_by:  x-axis tick labels  (e.g. "schedule")
    split_by:  bar colour within each group  (e.g. "distribution" → IID vs non-IID)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    split_vals = sorted(set(str(r.get(split_by, "?")) for r in runs))
    group_vals = sorted(set(str(r.get(group_by, "?")) for r in runs))

    # table[split_val][group_val] = list of final accuracy values (over seeds)
    table: dict[str, dict[str, list[float]]] = {
        sv: {gv: [] for gv in group_vals} for sv in split_vals
    }
    for r in runs:
        sv = str(r.get(split_by, "?"))
        gv = str(r.get(group_by,  "?"))
        v  = r.get(metric)
        if v is not None:
            table[sv][gv].append(float(v))

    n_groups = len(group_vals)
    n_splits = len(split_vals)
    width    = 0.8 / max(n_splits, 1)
    xs       = np.arange(n_groups)

    for si, sv in enumerate(split_vals):
        vals   = [table[sv][gv] for gv in group_vals]
        means  = [sum(v) / len(v) if v else 0.0 for v in vals]
        errors = [
            (max(v) - min(v)) / 2 if len(v) > 1 else 0.0
            for v in vals
        ]
        offset = (si - n_splits / 2 + 0.5) * width
        ax.bar(xs + offset, means, width, label=sv, color=_color(si),
               yerr=errors, capsize=3, alpha=0.85, error_kw={"linewidth": 1.2})

    ax.set_xticks(xs)
    ax.set_xticklabels(group_vals, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1.05)
    ax.legend(title=split_by.replace("_", " "), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)

    return fig or ax.get_figure()


def confusion_heatmap(
    preds:       list[int],
    trues:       list[int],
    class_names: list[str],
    normalize:   bool = True,
    title:       str  = "Confusion matrix",
    ax=None,
) -> "matplotlib.figure.Figure":
    """
    Annotated confusion matrix heatmap.

    preds / trues: integer class indices aligned with class_names
    normalize:     if True, show row-normalised recall rates; else raw counts
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n   = len(class_names)
    mat = np.zeros((n, n), dtype=int)
    for p, t in zip(preds, trues):
        if 0 <= t < n and 0 <= p < n:
            mat[t, p] += 1

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        display  = np.where(row_sums > 0, mat / row_sums.astype(float), 0.0)
        fmt      = ".2f"
        vmax     = 1.0
    else:
        display = mat.astype(float)
        fmt     = "d"
        vmax    = None

    fig = None
    if ax is None:
        size = max(4.0, n * 0.7)
        fig, ax = plt.subplots(figsize=(size + 1, size))

    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10)

    thresh = 0.5 if normalize else (display.max() / 2 if display.max() > 0 else 1)
    for i in range(n):
        for j in range(n):
            v = display[i, j]
            s = f"{v:{fmt}}" if fmt == "d" else f"{v:.2f}"
            ax.text(j, i, s, ha="center", va="center", fontsize=7,
                    color="white" if v > thresh else "black")

    return fig or ax.get_figure()


# ── Convenience: multi-panel summary figure ───────────────────────────────────

def summary_panel(
    runs:      list[dict],
    group_by:  str  = "schedule",
    split_by:  str  = "distribution",
    out_path:  Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    3-panel summary: accuracy curves | loss curves | final-accuracy bar.

    Saves to out_path if given, otherwise returns the figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.subplots_adjust(wspace=0.35)

    accuracy_curves(runs, group_by=group_by, ax=axes[0],
                    title=f"Diagnostic accuracy by {group_by}")
    loss_curves(runs, group_by=group_by, ax=axes[1],
                title="Training loss")
    bar_final_accuracy(runs, group_by=group_by, split_by=split_by,
                       ax=axes[2], title=f"Final accuracy: {group_by} × {split_by}")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  [eval] Saved {out_path}")

    return fig
