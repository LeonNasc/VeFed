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
        self.cfg    = cfg
        self.run_id = run_id
        self._rounds: list[dict] = []
        self._convs:  list[dict] = []

    # ── Data ingestion ────────────────────────────────────────────────────────

    def add_round(self, round_num: int, log: dict,
                  silo_events: list[list]) -> None:
        """
        Call at the end of each round with the W&B log dict and the list of
        event lists (one per silo from silo.last_round_events).
        """
        n_silos = self.cfg.num_silos
        silos = []
        for i in range(n_silos):
            silos.append({
                "sir_s":    int(log.get(f"silo_{i}/sir_s",    0)),
                "sir_i":    int(log.get(f"silo_{i}/sir_i",    0)),
                "sir_r":    int(log.get(f"silo_{i}/sir_r",    0)),
                "events":   int(log.get(f"silo_{i}/num_events", 0)),
                "mgmt_acc": log.get(f"silo_{i}/mgmt_acc",    float("nan")),
                "trained":  bool(log.get(f"silo_{i}/trained", 0)),
                "done":     bool(log.get(f"silo_{i}/done",    0)),
            })

        self._rounds.append({
            "round":    round_num,
            "silos":    silos,
            "agg_loss": log.get("aggregated/loss",     float("nan")),
            "agg_acc":  log.get("aggregated/mgmt_acc", float("nan")),
            "n_trained": int(log.get("aggregated/num_trained", 0)),
        })

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

    def write(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self._render(), encoding="utf-8")
        return out_path

    def _render(self) -> str:
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
            f"<th>S{i} S/I/R</th><th>S{i} ev</th><th>S{i} mgmt%</th>"
            for i in range(n_s)
        )
        rows = ""
        for r in self._rounds:
            cells = f'<td class="rn">{r["round"]}</td>'
            for i, s in enumerate(r["silos"]):
                sir = f'{s["sir_s"]}/{s["sir_i"]}/{s["sir_r"]}'
                acc = _pct(s["mgmt_acc"]) if s["trained"] else "—"
                done_mark = " ✓" if s["done"] else ""
                cells += (
                    f'<td class="sir">{sir}{done_mark}</td>'
                    f'<td>{s["events"]}</td>'
                    f'<td class="acc">{acc}</td>'
                )
            ma = _pct(r["agg_acc"])
            lo = _f(r["agg_loss"])
            cells += f'<td class="agg">{ma}</td><td class="agg">{lo}</td>'
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

        # ── Summary stats ─────────────────────────────────────────────────────
        total_ev  = sum(ev_vals)
        n_correct = sum(1 for c in self._convs if c["match"])
        n_conv    = len(self._convs)
        conv_acc  = _pct(n_correct / n_conv) if n_conv else "—"
        trained_rounds = [r for r in self._rounds if r["n_trained"] > 0]
        best_acc  = max((r["agg_acc"] for r in trained_rounds), default=float("nan"))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FedWorld Report — {self.run_id}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Courier New',monospace;background:#0d1117;color:#c9d1d9;padding:2em;line-height:1.5}}
h1{{color:#58a6ff;font-size:1.5em;margin-bottom:.3em}}
h2{{color:#79c0ff;font-size:1.1em;border-bottom:1px solid #30363d;padding-bottom:.3em;margin:1.5em 0 .8em}}
.meta{{color:#8b949e;font-size:.85em;margin-bottom:1.5em}}
.cfg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.5em;margin-bottom:1.5em}}
.cfg-item{{background:#161b22;border:1px solid #30363d;border-radius:4px;padding:.5em .8em}}
.cfg-label{{color:#8b949e;font-size:.75em;text-transform:uppercase;letter-spacing:.05em}}
.cfg-val{{color:#e6edf3;font-size:1em;margin-top:.15em}}
.spark{{color:#8b949e;font-size:.9em;margin:.2em 0}} code{{color:#ffa657}}
table{{width:100%;border-collapse:collapse;font-size:.85em;margin-bottom:1.5em}}
th{{background:#161b22;color:#8b949e;padding:.4em .7em;text-align:left;white-space:nowrap;border-bottom:2px solid #30363d}}
td{{padding:.3em .7em;border-bottom:1px solid #21262d}}
tr:hover td{{background:#161b22}}
.rn{{color:#8b949e}} .sir{{color:#6e7681}} .acc{{color:#3fb950;font-weight:bold}}
.agg{{color:#d2a679;font-weight:bold}}
.conv{{background:#161b22;border:1px solid #30363d;border-radius:6px;margin-bottom:.8em}}
.conv-hdr{{padding:.4em .8em;background:#1c2128;border-radius:6px 6px 0 0;font-size:.8em;color:#8b949e}}
.conv-body{{padding:.6em .8em}}
.turn{{display:flex;gap:.6em;padding:.2em 0;font-size:.82em;border-bottom:1px solid #21262d}}
.turn:last-child{{border-bottom:none}}
.role{{min-width:5.5em;font-weight:bold;flex-shrink:0}}
.txt{{color:#c9d1d9}}
.tp .role{{color:#79c0ff}} .tp .txt{{color:#79c0ff}}
.td .role{{color:#3fb950}} .td .txt{{color:#adbac7}}
.tv .role{{color:#ffa657}} .tv .txt{{color:#ffa657}}
.match{{color:#3fb950;font-weight:bold}} .mismatch{{color:#f85149;font-weight:bold}}
.summary{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.5em;margin-top:.5em}}
.stat{{background:#161b22;border:1px solid #30363d;border-radius:4px;padding:.5em .8em}}
.stat-val{{font-size:1.4em;font-weight:bold;color:#58a6ff}}
.stat-label{{color:#8b949e;font-size:.8em}}
.dim{{color:#8b949e;font-size:.9em}}
</style>
</head>
<body>

<h1>🦠 FedWorld Simulation Report</h1>
<div class="meta">Generated {now} &nbsp;·&nbsp; Run ID: <code>{_h(self.run_id)}</code></div>

<h2>Configuration</h2>
<div class="cfg">
  <div class="cfg-item"><div class="cfg-label">Disease</div><div class="cfg-val">{_h(getattr(cfg,'progression','—'))}</div></div>
  <div class="cfg-item"><div class="cfg-label">Silos</div><div class="cfg-val">{getattr(cfg,'num_silos','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">Agents / silo</div><div class="cfg-val">{getattr(cfg,'num_agents','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">Sim-days / round</div><div class="cfg-val">{getattr(cfg,'sim_days','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">End condition</div><div class="cfg-val">{_h(getattr(cfg,'end_condition','—'))}</div></div>
  <div class="cfg-item"><div class="cfg-label">LR</div><div class="cfg-val">{getattr(cfg,'lr','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">Epochs</div><div class="cfg-val">{getattr(cfg,'local_epochs','—')}</div></div>
  <div class="cfg-item"><div class="cfg-label">Seed</div><div class="cfg-val">{getattr(cfg,'seed','—')}</div></div>
</div>

<h2>Epidemic Bell Curve</h2>
{sparks}

<h2>Round-by-Round Metrics</h2>
<table>
<tr><th>Round</th>{silo_th}<th>Agg mgmt%</th><th>Agg loss</th></tr>
{rows}
</table>

<h2>Summary</h2>
<div class="summary">
  <div class="stat"><div class="stat-val">{total_ev}</div><div class="stat-label">Total clinic events</div></div>
  <div class="stat"><div class="stat-val">{n_conv}</div><div class="stat-label">Conversations logged</div></div>
  <div class="stat"><div class="stat-val">{conv_acc}</div><div class="stat-label">LLM mgmt accuracy (logged convs)</div></div>
  <div class="stat"><div class="stat-val">{_pct(best_acc)}</div><div class="stat-label">Best aggregated mgmt acc</div></div>
  <div class="stat"><div class="stat-val">{len(self._rounds)}</div><div class="stat-label">Rounds run</div></div>
</div>

<h2>Sample Conversations</h2>
{convs_html}

</body>
</html>"""


def _gt_match(ev) -> bool:
    """True if the LLM management tier matches ground truth."""
    if not ev.ground_truth or not ev.action:
        return False
    tier_map = {"home rest": "home_recovery", "treat": "resolve", "hospitalise": "hospitalise"}
    gt_parts = ev.ground_truth.rsplit(" / ", 1)
    if len(gt_parts) != 2:
        return False
    expected_action = tier_map.get(gt_parts[1])
    return ev.action.value == expected_action
