"""
Retrograde report regenerator.

Rebuilds existing dark-mode HTML reports into the new light-mode tabbed format,
and regenerates embedding plots with the infectious/non-infectious palette.

Usage:
    python retrograde_reports.py                  # process all reports
    python retrograde_reports.py reports/run_X.html  # specific report(s)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# ── HTML section extractor ────────────────────────────────────────────────────

def _between(html: str, start_h2: str, end_h2: str | None = None) -> str:
    """Extract HTML between <h2>start_h2</h2> and the next <h2> (or </body>)."""
    start_pat = rf'<h2>{re.escape(start_h2)}</h2>'
    end_pat   = r'(?=<h2>|</body>)' if end_h2 is None else rf'(?=<h2>{re.escape(end_h2)}</h2>|</body>)'
    m = re.search(start_pat + r'(.*?)' + end_pat, html, re.DOTALL)
    return m.group(1).strip() if m else ""


def _meta_line(html: str) -> str:
    m = re.search(r'<div class="meta">(.*?)</div>', html, re.DOTALL)
    return m.group(1).strip() if m else ""


def _cfg_block(html: str) -> str:
    return _between(html, "Configuration")


def _metrics_img(html: str) -> str:
    """Extract the base64 <img> tag from Metric Curves section."""
    section = _between(html, "Metric Curves")
    m = re.search(r'<img[^>]+>', section)
    return m.group(0) if m else "<p class='dim'>No metrics figure.</p>"


def _sparks(html: str) -> str:
    return _between(html, "Epidemic Bell Curve")


def _round_table(html: str) -> str:
    return _between(html, "Round-by-Round Metrics")


def _summary_block(html: str) -> str:
    return _between(html, "Summary")


def _conversations(html: str) -> str:
    return _between(html, "Sample Conversations")


def _title(html: str) -> str:
    m = re.search(r'<title>(.*?)</title>', html)
    return m.group(1) if m else "FedWorld Report"


# ── Embedding plot regeneration ───────────────────────────────────────────────

def _load_tracker(embed_dir: Path, run_id: str) -> object | None:
    """
    Reconstruct an EmbeddingTracker _snapshots dict from saved NPZ files
    and return the tracker, ready to call as_html_imgs().
    """
    run_dir = embed_dir / run_id
    if not run_dir.exists():
        return None

    round_dirs = sorted(run_dir.glob("round_*"))
    if not round_dirs:
        return None

    import numpy as np
    from fl.lora import LoRAConfig
    from viz.embedding_tracker import EmbeddingTracker

    # Build a minimal tracker — we only need _snapshots and _probe_labels
    tracker = EmbeddingTracker.__new__(EmbeddingTracker)
    tracker.output_dir   = run_dir
    tracker.lora_config  = LoRAConfig()
    tracker._probe_model = None
    tracker._tokenizer   = None
    tracker._snapshots   = {}
    tracker._probe_labels = None

    for rd in round_dirs:
        rnum = int(rd.name.split("_")[1])
        snap: dict[str, dict] = {}
        for npz_path in sorted(rd.glob("*.npz")):
            data = np.load(npz_path)
            name = npz_path.stem
            # Support both old schema (raw) and new schema (cls + logits)
            cls    = data["cls"]    if "cls"    in data else data["raw"]
            logits = data["logits"] if "logits" in data else data["raw"]
            snap[name] = {"cls": cls, "logits": logits}
            if tracker._probe_labels is None and "labels" in data:
                tracker._probe_labels = data["labels"].tolist()
        tracker._snapshots[rnum] = snap

    if tracker._probe_labels is None:
        tracker._probe_labels = []

    return tracker


# ── New CSS + JS ──────────────────────────────────────────────────────────────

NEW_STYLE = """<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fff;color:#212529;padding:2em;line-height:1.5}
h1{color:#0d6efd;font-size:1.5em;margin-bottom:.3em}
h2{color:#0550ae;font-size:1.1em;border-bottom:1px solid #dee2e6;padding-bottom:.3em;margin:1.5em 0 .8em}
.meta{color:#6c757d;font-size:.85em;margin-bottom:1.5em}
.cfg{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.5em;margin-bottom:1.5em}
.cfg-item{background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;padding:.5em .8em}
.cfg-label{color:#6c757d;font-size:.75em;text-transform:uppercase;letter-spacing:.05em}
.cfg-val{color:#212529;font-size:1em;margin-top:.15em}
.spark{color:#6c757d;font-size:.9em;margin:.2em 0} code{color:#d63384}
.tabs{display:flex;gap:.25em;border-bottom:2px solid #dee2e6;margin-bottom:1.2em}
.tab-btn{background:none;border:none;padding:.5em 1.1em;cursor:pointer;font-size:.9em;color:#6c757d;border-bottom:2px solid transparent;margin-bottom:-2px;border-radius:4px 4px 0 0;transition:color .15s}
.tab-btn:hover{color:#0d6efd;background:#f8f9fa}
.tab-btn.active{color:#0d6efd;border-bottom-color:#0d6efd;font-weight:600}
.tab-content{display:none} .tab-content.active{display:block}
table{width:100%;border-collapse:collapse;font-size:.82em;margin-bottom:1.5em;font-family:'Courier New',monospace}
th{background:#f8f9fa;color:#6c757d;padding:.4em .7em;text-align:left;white-space:nowrap;border-bottom:2px solid #dee2e6}
td{padding:.3em .7em;border-bottom:1px solid #eaeef2}
tr:hover td{background:#f8f9fa}
.rn{color:#adb5bd} .sir{color:#6c757d} .acc{color:#198754;font-weight:bold}
.agg{color:#6f42c1;font-weight:bold}
.conv{background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;margin-bottom:.8em}
.conv-hdr{padding:.4em .8em;background:#e9ecef;border-radius:6px 6px 0 0;font-size:.8em;color:#6c757d}
.conv-body{padding:.6em .8em}
.turn{display:flex;gap:.6em;padding:.2em 0;font-size:.82em;border-bottom:1px solid #eaeef2;font-family:'Courier New',monospace}
.turn:last-child{border-bottom:none}
.role{min-width:5.5em;font-weight:bold;flex-shrink:0}
.txt{color:#212529}
.tp .role{color:#0d6efd} .tp .txt{color:#0d6efd}
.td .role{color:#198754} .td .txt{color:#495057}
.tv .role{color:#fd7e14} .tv .txt{color:#fd7e14}
.match{color:#198754;font-weight:bold} .mismatch{color:#dc3545;font-weight:bold}
.summary{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.5em;margin-top:.5em}
.stat{background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;padding:.5em .8em}
.stat-val{font-size:1.4em;font-weight:bold;color:#0d6efd}
.stat-label{color:#6c757d;font-size:.8em}
.dim{color:#6c757d;font-size:.9em}
</style>
<script>
function showTab(name,btn){
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}
</script>"""


# ── Report rebuilder ──────────────────────────────────────────────────────────

def rebuild(html_path: Path, embed_dir: Path) -> None:
    html     = html_path.read_text(encoding="utf-8")
    run_id   = html_path.stem.replace("run_", "")
    title    = _title(html)
    meta     = _meta_line(html)
    cfg      = _cfg_block(html)
    img      = _metrics_img(html)
    sparks   = _sparks(html)
    tbl      = _round_table(html)
    summary  = _summary_block(html)
    convs    = _conversations(html)

    # Embedding tab
    tracker = _load_tracker(embed_dir, f"run_{run_id}")
    embed_tab_html = ""
    if tracker and tracker._snapshots:
        print(f"  [embed] regenerating plots for {run_id}…")
        plot_labels = {
            "evolution_cls":    "Global evolution — CLS space",
            "evolution_logits": "Global evolution — logit space",
            "final_all_models": "Final round — all models",
            "fl_gain":          "FL gain (fed vs local)",
        }
        try:
            imgs = tracker.as_html_imgs()
            for key, img_tag in imgs.items():
                label = plot_labels.get(key, key.replace("_", " ").title())
                embed_tab_html += (
                    f'<h3 style="font-size:.95em;color:#495057;margin:1em 0 .4em">'
                    f'{label}</h3>\n{img_tag}\n'
                )
        except Exception as e:
            embed_tab_html = f"<p class='dim'>Could not regenerate plots: {e}</p>"
    else:
        embed_tab_html = "<p class='dim'>No embedding snapshots found for this run.</p>"

    new_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
{NEW_STYLE}
</head>
<body>

<h1>FedWorld Simulation Report</h1>
<div class="meta">{meta}</div>

<h2>Configuration</h2>
{cfg}

<h2>Summary</h2>
{summary}

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('metrics',this)">Metrics</button>
  <button class="tab-btn" onclick="showTab('embeddings',this)">Embeddings</button>
  <button class="tab-btn" onclick="showTab('rounds',this)">Round table</button>
  <button class="tab-btn" onclick="showTab('conversations',this)">Conversations</button>
</div>

<div id="tab-metrics" class="tab-content active">
  <h2>Metric Curves</h2>
  {img}
  <h2>Epidemic Bell Curve</h2>
  {sparks}
</div>

<div id="tab-embeddings" class="tab-content">
  <h2>Embedding Visualisations</h2>
  <p class="dim" style="margin-bottom:1em">
    Warm colours (●) = infectious &nbsp;·&nbsp; Cool colours (▲) = non-infectious.
  </p>
  {embed_tab_html}
</div>

<div id="tab-rounds" class="tab-content">
  <h2>Round-by-Round Metrics</h2>
  {tbl}
</div>

<div id="tab-conversations" class="tab-content">
  <h2>Sample Conversations</h2>
  {convs}
</div>

</body>
</html>"""

    html_path.write_text(new_html, encoding="utf-8")
    print(f"  ✓ {html_path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    reports_dir = Path("reports")
    embed_dir   = Path("viz_output/embeddings")

    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = sorted(reports_dir.glob("run_*.html"))

    if not targets:
        print("No reports found.")
        sys.exit(0)

    print(f"Rebuilding {len(targets)} report(s)…\n")
    for path in targets:
        print(f"→ {path.name}")
        try:
            rebuild(path, embed_dir)
        except Exception as e:
            print(f"  ✗ failed: {e}")

    print(f"\nDone.")
