"""
HTML run report — written at the end of every FL training run.

Captures: run config, per-round SIR + FL metrics, infection bell curve,
and sample conversations with GT vs LLM comparison.
"""
from __future__ import annotations

import html as _html
from datetime import datetime
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _h(s: object) -> str:
    return _html.escape(str(s))

def _pct(v: float) -> str:
    return f"{v:.0%}" if v == v else "—"

def _f(v: float, dec: int = 3) -> str:
    return f"{v:.{dec}f}" if v == v else "—"

BARS = "▁▂▃▄▅▆▇█"

def _spark(values: list[int]) -> str:
    mx = max(values) if values else 1
    return "".join(BARS[min(7, int(v / max(mx, 1) * 7))] for v in values)


# ── Main class ────────────────────────────────────────────────────────────────

class RunReport:
    """
    Accumulates per-round data during a run and renders a self-contained
    HTML report.

    Usage:
        report = RunReport(cfg, run_id="20260529_120000")
        # inside round loop:
        report.add_round(round_num, log, silo_events_list)
        # after loop:
        path = report.write("reports/run_20260529_120000.html")
    """

    def __init__(self, cfg, run_id: str):
        self.cfg     = cfg
        self.run_id  = run_id
        self._rounds: list[dict] = []
        self._convs:  list[dict] = []

        from viz.metrics_plot import MetricsPlotter
        progs = getattr(cfg, "progressions", [])
        prog_str = " + ".join(progs) if progs else getattr(cfg, "disease_strategy", "")
        self.plotter = MetricsPlotter(
            title   = f"{getattr(cfg,'num_silos','?')} silos · {prog_str}",
            n_silos = getattr(cfg, "num_silos", 3),
        )

    # ── Data ingestion ────────────────────────────────────────────────────────

    def add_round(self, round_num: int, log: dict,
                  silo_events: list[list]) -> None:
        """
        Call at the end of each round with the W&B log dict and the list of
        event lists (one per silo from silo.last_round_events).
        """
        n_silos = self.cfg.num_silos
        nan = float("nan")
        silos = []
        for i in range(n_silos):
            silos.append({
                "sir_s":      int(log.get(f"silo_{i}/sir_s",      0)),
                "sir_i":      int(log.get(f"silo_{i}/sir_i",      0)),
                "sir_r":      int(log.get(f"silo_{i}/sir_r",      0)),
                "events":     int(log.get(f"silo_{i}/num_events",  0)),
                "triage_acc": log.get(f"silo_{i}/triage_acc",    nan),
                "diag_acc":   log.get(f"silo_{i}/diag_acc",      nan),
                "trained":    bool(log.get(f"silo_{i}/trained",   0)),
                "done":       bool(log.get(f"silo_{i}/done",      0)),
            })

        self._rounds.append({
            "round":    round_num,
            "silos":    silos,
            "agg_loss":    log.get("aggregated/loss",              float("nan")),
            "agg_triage":  log.get("aggregated/triage_acc",       float("nan")),
            "agg_diag":    log.get("aggregated/diag_acc",         float("nan")),
            "agg_f1":      log.get("aggregated/danger_rate",       float("nan")),
            "agg_danger":  log.get("aggregated/danger_rate",      float("nan")),
            "n_trained":   int(log.get("aggregated/num_trained",  0)),
        })

        # Feed plotter — handles both federated and centralized keys gracefully
        if "aggregated/triage_acc" in log:
            self.plotter.add_federated_round(round_num, log)
        if "centralized/triage_acc" in log:
            self.plotter.add_centralized_round(round_num, log)

        # SIR per silo (present in federated rounds)
        silo_i = [int(log.get(f"silo_{i}/sir_i", 0))
                  for i in range(self.cfg.num_silos)]
        if any(v > 0 for v in silo_i):
            self.plotter._sir_rounds.append(round_num)
            for i, v in enumerate(silo_i):
                self.plotter._sir_i.setdefault(i, []).append(v)

        # Harvest up to 2 rich conversations per silo per round
        for i, events in enumerate(silo_events):
            rich = [ev for ev in events
                    if ev.conversation and len(ev.conversation) >= 4]
            for ev in rich[:2]:
                self._convs.append({
                    "round":  round_num,
                    "silo":   i,
                    "agent":  ev.agent_id,
                    "gt":     ev.ground_truth or "—",
                    "action": ev.action.value if ev.action else "—",
                    "label":  ev.oracle_label or "—",
                    "days":   ev.days_infected,
                    "turns":  ev.conversation,
                    "match":  _gt_match(ev),
                })

    # ── Rendering ─────────────────────────────────────────────────────────────

    def write(self, out_path: str | Path, tracker=None) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save standalone PNG alongside the HTML
        png_path = out_path.with_suffix(".png")
        try:
            self.plotter.save(png_path)
        except Exception:
            png_path = None

        # Collect embedding images from tracker if provided
        embed_imgs: dict = {}
        if tracker is not None:
            try:
                embed_imgs = tracker.as_html_imgs()
            except Exception:
                pass

        out_path.write_text(self._render(png_path, embed_imgs), encoding="utf-8")
        return out_path

    def _render(self, png_path=None, embed_imgs: dict | None = None) -> str:
        cfg    = self.cfg
        now    = datetime.now().strftime("%Y-%m-%d %H:%M")
        n_s    = cfg.num_silos

        # ── Sparklines ───────────────────────────────────────────────────────
        sparks = ""
        for i in range(n_s):
            i_vals = [r["silos"][i]["sir_i"] if i < len(r["silos"]) else 0
                      for r in self._rounds]
            sparks += f'<div class="spark">Silo {i} · infected I: <code>{_spark(i_vals)}</code></div>'
        ev_vals = [sum(r["silos"][i]["events"] for i in range(n_s) if i < len(r["silos"]))
                   for r in self._rounds]
        sparks += f'<div class="spark">Total clinic events: <code>{_spark(ev_vals)}</code></div>'

        # ── Round table ───────────────────────────────────────────────────────
        silo_th = "".join(
            f"<th>S{i} S/I/R</th><th>S{i} ev</th><th>S{i} triage</th><th>S{i} diag</th>"
            for i in range(n_s)
        )
        rows = ""
        for r in self._rounds:
            cells = f'<td class="rn">{r["round"]}</td>'
            for i, s in enumerate(r["silos"]):
                sir  = f'{s["sir_s"]}/{s["sir_i"]}/{s["sir_r"]}'
                tr   = _pct(s["triage_acc"]) if s["trained"] else "—"
                dg   = _pct(s["diag_acc"])   if s["trained"] else "—"
                done = " ✓" if s["done"] else ""
                cells += (
                    f'<td class="sir">{sir}{done}</td>'
                    f'<td>{s["events"]}</td>'
                    f'<td class="acc">{tr}</td>'
                    f'<td class="acc">{dg}</td>'
                )
            cells += (
                f'<td class="agg">{_pct(r["agg_triage"])}</td>'
                f'<td class="agg">{_pct(r["agg_diag"])}</td>'
                f'<td class="agg">{_f(r["agg_f1"])}</td>'
                f'<td class="agg" style="color:#f85149">{_pct(r["agg_danger"])}</td>'
                f'<td class="agg">{_f(r["agg_loss"])}</td>'
            )
            rows += f"<tr>{cells}</tr>\n"

        # ── Conversations ─────────────────────────────────────────────────────
        # Show at most 10, prefer rounds where training happened
        shown = sorted(self._convs, key=lambda c: (not c["match"], c["round"]))[:10]
        convs_html = ""
        for c in shown:
            match_cls  = "match" if c["match"] else "mismatch"
            match_text = "✓ correct" if c["match"] else "✗ mismatch"
            convs_html += f"""
<div class="conv">
  <div class="conv-hdr">
    Round {c['round']} · Silo {c['silo']} · {_h(c['agent'])} · day {c['days']}
    &ensp;|&ensp; GT: <b>{_h(c['gt'])}</b>
    &ensp;|&ensp; LLM: <b>{_h(c['action'])} ({_h(c['label'])})</b>
    &ensp;<span class="{match_cls}">{match_text}</span>
  </div>
  <div class="conv-body">"""
            for turn in c["turns"]:
                role = turn["role"].upper()
                cls  = {"patient": "tp", "doctor": "td", "vitals": "tv"}.get(turn["role"], "tx")
                convs_html += (
                    f'<div class="turn {cls}">'
                    f'<span class="role">{role}</span>'
                    f'<span class="txt">{_h(turn["text"])}</span></div>\n'
                )
            convs_html += "  </div>\n</div>\n"

        if not convs_html:
            convs_html = "<p class='dim'>No conversations collected — min_events_to_train threshold not met.</p>"

        # ── Metric curves figure ─────────────────────────────────────────────
        try:
            metrics_img = self.plotter.as_html_img()
        except Exception:
            metrics_img = "<p class='dim'>Metric curves unavailable (matplotlib error).</p>"

        # ── Summary stats ─────────────────────────────────────────────────────
        total_ev  = sum(ev_vals)
        n_correct = sum(1 for c in self._convs if c["match"])
        n_conv    = len(self._convs)
        conv_acc  = _pct(n_correct / n_conv) if n_conv else "—"
        trained_rounds = [r for r in self._rounds if r["n_trained"] > 0]
        best_acc  = max((r["agg_triage"] for r in trained_rounds if r["agg_triage"] == r["agg_triage"]), default=float("nan"))
        best_diag = max((r["agg_diag"]   for r in trained_rounds if r["agg_diag"]   == r["agg_diag"]),   default=float("nan"))
        danger_vals = [r["agg_danger"] for r in trained_rounds if r["agg_danger"] == r["agg_danger"]]
        avg_danger  = sum(danger_vals) / len(danger_vals) if danger_vals else float("nan")

        embed_imgs = embed_imgs or {}
        embed_tab_html = ""
        if embed_imgs:
            plot_labels = {
                "evolution_cls":    "Global evolution — CLS space",
                "evolution_logits": "Global evolution — logit space",
                "final_all_models": "Final round — all models",
                "fl_gain":          "FL gain (fed vs local)",
            }
            for key, img_tag in embed_imgs.items():
                label = plot_labels.get(key, key.replace("_", " ").title())
                embed_tab_html += f'<h3 style="font-size:.95em;color:#495057;margin:1em 0 .4em">{label}</h3>\n{img_tag}\n'
        else:
            embed_tab_html = "<p class='dim'>No embedding snapshots available for this run.</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FedWorld Report — {self.run_id}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fff;color:#212529;padding:2em;line-height:1.5}}
h1{{color:#0d6efd;font-size:1.5em;margin-bottom:.3em}}
h2{{color:#0550ae;font-size:1.1em;border-bottom:1px solid #dee2e6;padding-bottom:.3em;margin:1.5em 0 .8em}}
.meta{{color:#6c757d;font-size:.85em;margin-bottom:1.5em}}
.cfg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.5em;margin-bottom:1.5em}}
.cfg-item{{background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;padding:.5em .8em}}
.cfg-label{{color:#6c757d;font-size:.75em;text-transform:uppercase;letter-spacing:.05em}}
.cfg-val{{color:#212529;font-size:1em;margin-top:.15em}}
.spark{{color:#6c757d;font-size:.9em;margin:.2em 0}} code{{color:#d63384}}
/* ── Tabs ── */
.tabs{{display:flex;gap:.25em;border-bottom:2px solid #dee2e6;margin-bottom:1.2em}}
.tab-btn{{background:none;border:none;padding:.5em 1.1em;cursor:pointer;font-size:.9em;color:#6c757d;border-bottom:2px solid transparent;margin-bottom:-2px;border-radius:4px 4px 0 0;transition:color .15s}}
.tab-btn:hover{{color:#0d6efd;background:#f8f9fa}}
.tab-btn.active{{color:#0d6efd;border-bottom-color:#0d6efd;font-weight:600}}
.tab-content{{display:none}} .tab-content.active{{display:block}}
/* ── Table ── */
table{{width:100%;border-collapse:collapse;font-size:.82em;margin-bottom:1.5em;font-family:'Courier New',monospace}}
th{{background:#f8f9fa;color:#6c757d;padding:.4em .7em;text-align:left;white-space:nowrap;border-bottom:2px solid #dee2e6}}
td{{padding:.3em .7em;border-bottom:1px solid #eaeef2}}
tr:hover td{{background:#f8f9fa}}
.rn{{color:#adb5bd}} .sir{{color:#6c757d}} .acc{{color:#198754;font-weight:bold}}
.agg{{color:#6f42c1;font-weight:bold}}
/* ── Conversations ── */
.conv{{background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;margin-bottom:.8em}}
.conv-hdr{{padding:.4em .8em;background:#e9ecef;border-radius:6px 6px 0 0;font-size:.8em;color:#6c757d}}
.conv-body{{padding:.6em .8em}}
.turn{{display:flex;gap:.6em;padding:.2em 0;font-size:.82em;border-bottom:1px solid #eaeef2;font-family:'Courier New',monospace}}
.turn:last-child{{border-bottom:none}}
.role{{min-width:5.5em;font-weight:bold;flex-shrink:0}}
.txt{{color:#212529}}
.tp .role{{color:#0d6efd}} .tp .txt{{color:#0d6efd}}
.td .role{{color:#198754}} .td .txt{{color:#495057}}
.tv .role{{color:#fd7e14}} .tv .txt{{color:#fd7e14}}
.match{{color:#198754;font-weight:bold}} .mismatch{{color:#dc3545;font-weight:bold}}
/* ── Summary cards ── */
.summary{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.5em;margin-top:.5em}}
.stat{{background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;padding:.5em .8em}}
.stat-val{{font-size:1.4em;font-weight:bold;color:#0d6efd}}
.stat-label{{color:#6c757d;font-size:.8em}}
.dim{{color:#6c757d;font-size:.9em}}
</style>
<script>
function showTab(name,btn){{
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}}
</script>
</head>
<body>

<h1>FedWorld Simulation Report</h1>
<div class="meta">Generated {now} &nbsp;·&nbsp; Run ID: <code>{_h(self.run_id)}</code></div>

<h2>Configuration</h2>
<div class="cfg">
  <div class="cfg-item"><div class="cfg-label">Silos</div><div class="cfg-val">{getattr(cfg,'num_silos','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">Agents / silo</div><div class="cfg-val">{getattr(cfg,'num_agents','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">Sim-days / round</div><div class="cfg-val">{getattr(cfg,'sim_days','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">End condition</div><div class="cfg-val">{_h(getattr(cfg,'end_condition','—'))}</div></div>
  <div class="cfg-item"><div class="cfg-label">Local epochs</div><div class="cfg-val">{getattr(cfg,'local_epochs','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">LR</div><div class="cfg-val">{getattr(cfg,'lr','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">Seed</div><div class="cfg-val">{getattr(cfg,'seed','—')}</div></div>
</div>

<h2>Summary</h2>
<div class="summary">
  <div class="stat"><div class="stat-val">{total_ev}</div><div class="stat-label">Total clinic events</div></div>
  <div class="stat"><div class="stat-val">{n_conv}</div><div class="stat-label">Conversations logged</div></div>
  <div class="stat"><div class="stat-val">{conv_acc}</div><div class="stat-label">LLM triage accuracy</div></div>
  <div class="stat"><div class="stat-val">{_pct(best_acc)}</div><div class="stat-label">Best LoRA triage acc</div></div>
  <div class="stat"><div class="stat-val">{_pct(best_diag)}</div><div class="stat-label">Best LoRA diag acc</div></div>
  <div class="stat" style="border-color:#dc3545"><div class="stat-val" style="color:#dc3545">{_pct(avg_danger)}</div><div class="stat-label">Avg danger rate</div></div>
  <div class="stat"><div class="stat-val">{len(self._rounds)}</div><div class="stat-label">Rounds run</div></div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('metrics',this)">Metrics</button>
  <button class="tab-btn" onclick="showTab('embeddings',this)">Embeddings</button>
  <button class="tab-btn" onclick="showTab('rounds',this)">Round table</button>
  <button class="tab-btn" onclick="showTab('conversations',this)">Conversations</button>
</div>

<div id="tab-metrics" class="tab-content active">
  <h2>Metric Curves</h2>
  {metrics_img}
  <h2>Epidemic Bell Curve</h2>
  {sparks}
</div>

<div id="tab-embeddings" class="tab-content">
  <h2>Embedding Visualisations</h2>
  <p class="dim" style="margin-bottom:1em">
    Scatter plots show CLS / logit-space UMAP projections of the probe event set.
    Warm colours (●) = infectious diseases &nbsp;·&nbsp; Cool colours (▲) = non-infectious.
  </p>
  {embed_tab_html}
</div>

<div id="tab-rounds" class="tab-content">
  <h2>Round-by-Round Metrics</h2>
  <table>
  <tr><th>Round</th>{silo_th}<th>Agg triage%</th><th>Agg diag%</th><th>Macro-F1</th><th>Danger rate</th><th>Agg loss</th></tr>
  {rows}
  </table>
</div>

<div id="tab-conversations" class="tab-content">
  <h2>Sample Conversations</h2>
  {convs_html}
</div>

</body>
</html>"""


def _gt_match(ev) -> bool:
    """True if the LLM action matches expected action for the ground truth severity."""
    if not ev.ground_truth or not ev.action:
        return False
    # New format: "disease/severity" or "non-infectious"
    sev_to_action = {"mild": "home_recovery", "moderate": "resolve", "severe": "hospitalise", "none": "home_recovery"}
    gt = ev.ground_truth
    if "/" in gt:
        sev = gt.split("/", 1)[1]
    else:
        sev = "none"  # "non-infectious"
    expected_action = sev_to_action.get(sev)
    return ev.action.value == expected_action
