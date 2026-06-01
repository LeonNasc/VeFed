"""
Publication-quality metric curve plots for FedWorld runs.

Generates a single multi-panel figure suitable for paper figures and
supervisor presentations.  Works for three scenarios:

  federated-only  : fed + local-only traces
  centralized-only: single centralized trace
  comparison      : all three traces on the same axes (fed / local / centralized)

Panels
------
  1. Triage accuracy      (management tier, 3-class)
  2. Diagnosis accuracy   (ICD category match)
  3. Training loss
  4. FL gain              (fed triage_acc − local triage_acc)
  5. Danger rate          (hospitalise cases sent home)
  6. SIR bell curve       (active infectious I per silo over rounds)

Usage
-----
    from viz.metrics_plot import MetricsPlotter
    plotter = MetricsPlotter()
    # call once per round during training:
    plotter.add_federated_round(round_num, log_dict)
    plotter.add_centralized_round(round_num, cen_log_dict)   # optional
    # at end of run:
    fig_path = plotter.save("reports/figures/run_id_metrics.png")
    html_tag = plotter.as_html_img()   # base64 data-URI <img> tag
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional


# Colour palette — light-mode, consistent across all plots in the paper
FED_COLOR    = "#1f77b4"   # blue   — federated global model
LOCAL_COLOR  = "#adb5bd"   # grey   — local-only shadow model
CEN_COLOR    = "#ff7f0e"   # orange — centralized oracle
DANGER_COLOR = "#d62728"   # red    — danger / hospitalise-miss rate

# Light-mode chrome
BG_FIGURE = "white"
BG_AXES   = "#f8f9fa"
GRID_COL  = "#dee2e6"
SPINE_COL = "#ced4da"
TEXT_COL  = "#212529"
TICK_COL  = "#495057"
LABEL_COL = "#6c757d"


class MetricsPlotter:
    """
    Accumulates per-round metrics and renders a 6-panel comparison figure.

    Parameters
    ----------
    title : str
        Figure super-title (e.g. "non-iid · 3 silos · Standard Flu + Slow Burn")
    n_silos : int
        Number of silos (determines SIR panel layout).
    """

    def __init__(self, title: str = "FedWorld Training Metrics", n_silos: int = 3):
        self.title   = title
        self.n_silos = n_silos

        # Federated traces
        self._fed_rounds:   list[int]   = []
        self._fed_triage:   list[float] = []
        self._fed_diag:     list[float] = []
        self._fed_loss:     list[float] = []
        self._fed_gain:     list[float] = []
        self._fed_danger:   list[float] = []
        self._local_triage: list[float] = []

        # Centralized traces (optional)
        self._cen_rounds:   list[int]   = []
        self._cen_triage:   list[float] = []
        self._cen_diag:     list[float] = []
        self._cen_loss:     list[float] = []
        self._cen_danger:   list[float] = []

        # SIR: dict[silo_idx] → list[I_count]
        self._sir_rounds:   list[int]            = []
        self._sir_i:        dict[int, list[int]] = {}

    # ── Data ingestion ────────────────────────────────────────────────────────

    def add_federated_round(self, round_num: int, log: dict) -> None:
        nan = float("nan")
        self._fed_rounds.append(round_num)
        self._fed_triage.append(log.get("aggregated/triage_acc",      nan))
        self._fed_diag.append(  log.get("aggregated/diag_acc",        nan))
        self._fed_loss.append(  log.get("aggregated/loss",            nan))
        self._fed_gain.append(  log.get("aggregated/fl_gain",         nan))
        self._fed_danger.append(log.get("aggregated/danger_rate",     nan))
        self._local_triage.append(log.get("aggregated/local_triage_acc", nan))

        self._sir_rounds.append(round_num)
        for i in range(self.n_silos):
            val = int(log.get(f"silo_{i}/sir_i", 0))
            self._sir_i.setdefault(i, []).append(val)

    def add_centralized_round(self, round_num: int, log: dict) -> None:
        nan = float("nan")
        self._cen_rounds.append(round_num)
        self._cen_triage.append(log.get("centralized/triage_acc",  nan))
        self._cen_diag.append(  log.get("centralized/diag_acc",    nan))
        self._cen_loss.append(  log.get("centralized/loss",        nan))
        self._cen_danger.append(log.get("centralized/danger_rate", nan))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _build_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.suptitle(self.title, fontsize=12, fontweight="bold", y=1.01, color=TEXT_COL)
        fig.patch.set_facecolor(BG_FIGURE)

        for ax in axes.flat:
            ax.set_facecolor(BG_AXES)
            ax.tick_params(colors=TICK_COL, labelsize=8)
            ax.spines[:].set_color(SPINE_COL)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            ax.grid(axis="y", color=GRID_COL, linewidth=0.5, linestyle="--")
            ax.set_xlabel("Round", fontsize=8, color=LABEL_COL)

        def _plot(ax, xs, ys, label, color, linestyle="-", linewidth=1.6, alpha=1.0):
            pts = [(x, y) for x, y in zip(xs, ys) if y == y]
            if not pts:
                return
            x_arr, y_arr = zip(*pts)
            ax.plot(x_arr, y_arr, color=color, label=label,
                    linestyle=linestyle, linewidth=linewidth, alpha=alpha)

        def _legend(ax):
            ax.legend(fontsize=7, framealpha=0.9, labelcolor=TEXT_COL,
                      facecolor="white", edgecolor=SPINE_COL)

        # Panel 1: Triage accuracy
        ax = axes[0, 0]
        _plot(ax, self._fed_rounds,   self._fed_triage,   "Federated",   FED_COLOR)
        _plot(ax, self._fed_rounds,   self._local_triage, "Local-only",  LOCAL_COLOR, "--", 1.2, 0.8)
        _plot(ax, self._cen_rounds,   self._cen_triage,   "Centralized", CEN_COLOR,   "-.", 1.4)
        ax.set_title("Triage accuracy (management tier)", fontsize=9, color=TEXT_COL)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        _legend(ax)

        # Panel 2: Diagnosis accuracy
        ax = axes[0, 1]
        _plot(ax, self._fed_rounds, self._fed_diag, "Federated",   FED_COLOR)
        _plot(ax, self._cen_rounds, self._cen_diag, "Centralized", CEN_COLOR, "-.", 1.4)
        ax.set_title("Diagnosis accuracy (ICD category)", fontsize=9, color=TEXT_COL)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        _legend(ax)

        # Panel 3: Training loss
        ax = axes[0, 2]
        _plot(ax, self._fed_rounds, self._fed_loss, "Federated",   FED_COLOR)
        _plot(ax, self._cen_rounds, self._cen_loss, "Centralized", CEN_COLOR, "-.", 1.4)
        ax.set_title("Training loss (cross-entropy)", fontsize=9, color=TEXT_COL)
        _legend(ax)

        # Panel 4: FL gain
        ax = axes[1, 0]
        _plot(ax, self._fed_rounds, self._fed_gain, "FL gain (fed − local)", FED_COLOR)
        ax.axhline(0, color=SPINE_COL, linewidth=0.8, linestyle="--")
        ax.set_title("FL gain (fed triage_acc − local)", fontsize=9, color=TEXT_COL)
        gains = [(x, y) for x, y in zip(self._fed_rounds, self._fed_gain) if y == y]
        if gains:
            xs, ys = zip(*gains)
            ax.fill_between(xs, 0, ys, where=[y > 0 for y in ys],
                            alpha=0.15, color=FED_COLOR, interpolate=True)
            ax.fill_between(xs, 0, ys, where=[y < 0 for y in ys],
                            alpha=0.12, color=LOCAL_COLOR, interpolate=True)
        _legend(ax)

        # Panel 5: Danger rate
        ax = axes[1, 1]
        _plot(ax, self._fed_rounds, self._fed_danger, "Federated",   FED_COLOR)
        _plot(ax, self._cen_rounds, self._cen_danger, "Centralized", CEN_COLOR, "-.", 1.4)
        ax.set_title("Danger rate (hospitalise → home rest)", fontsize=9, color=DANGER_COLOR)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", colors=DANGER_COLOR)
        _legend(ax)

        # Panel 6: SIR bell curve
        ax = axes[1, 2]
        silo_colors = plt.cm.get_cmap("tab10", max(self.n_silos, 1))
        for i in range(self.n_silos):
            ys = self._sir_i.get(i, [])
            if ys:
                ax.plot(self._sir_rounds[:len(ys)], ys,
                        color=silo_colors(i), linewidth=1.2, alpha=0.85, label=f"Silo {i}")
        ax.set_title("Active infections I per silo", fontsize=9, color=TEXT_COL)
        ax.set_ylabel("I", fontsize=8, color=LABEL_COL)
        if self.n_silos <= 5:
            _legend(ax)

        for ax in axes.flat:
            ax.tick_params(colors=TICK_COL)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color(TICK_COL)

        fig.tight_layout()
        return fig

    def save(self, path: str | Path, dpi: int = 150) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = self._build_figure()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path

    def as_png_bytes(self, dpi: int = 130) -> bytes:
        import matplotlib.pyplot as plt
        fig = self._build_figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()

    def as_html_img(self, dpi: int = 130) -> str:
        data = base64.b64encode(self.as_png_bytes(dpi)).decode()
        return f'<img src="data:image/png;base64,{data}" style="max-width:100%;margin:1em 0">'
