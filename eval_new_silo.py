"""
Evaluate a saved LoRA model on a fresh unseen silo.

The silo runs a new SIR epidemic (never seen during training), generates
patient conversations via Ollama, and the model does inference-only —
no further training. This cleanly tests out-of-distribution generalization.

Usage examples:
  # Evaluate FL non-IID final model on a new mixed-disease silo
  python eval_new_silo.py --weights datasets/20260611_143904/weights/fl_final.npz \\
                          --disease mixed --n-events 80

  # Evaluate centralized non-IID model on a pneumonia-heavy silo
  python eval_new_silo.py --weights datasets/merged_noniid/weights/centralized.npz \\
                          --disease pneu --n-events 80

  # Compare FL vs local-only by running both against the same silo
  python eval_new_silo.py --weights datasets/20260611_143904/weights/fl_final.npz \\
                          datasets/20260611_143904/weights/silo_0.npz \\
                          --labels "FL non-IID" "Local silo-0" \\
                          --disease mixed --n-events 80 --seed 99
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from fl.lora import LoRAConfig, build_model, set_lora_weights
from fl.learner import DIAGNOSTIC_LABELS, build_label_map, _split_label
from simulation.end_conditions import from_config as end_condition_from_config

# ── Calibrated SIR params (from sweep results) ───────────────────────────────
DEFAULT_NUM_AGENTS      = 75
DEFAULT_BETA_SCALE      = 2.0
DEFAULT_SIGMA           = 0.5
DEFAULT_INITIAL_SEEDS   = 8
DEFAULT_SIM_DAYS        = 2
DEFAULT_BG_RATE         = 0.025

DISEASE_PRESETS = {
    "flu":   (["Influenza"],                          None),         # pure influenza
    "pneu":  (["Bacterial Pneumonia"],                None),         # pure pneumonia
    "mixed": (["Influenza", "Bacterial Pneumonia"],   [0.5, 0.5]),   # 50/50
    "mixed-pneu": (["Influenza", "Bacterial Pneumonia"], [0.3, 0.7]),# pneumonia-heavy
    "mixed-flu":  (["Influenza", "Bacterial Pneumonia"], [0.7, 0.3]),# flu-heavy
}

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

label2id, id2label = build_label_map()

LORA_CFG = LoRAConfig(
    model_name_or_path = MODEL_NAME,
    num_labels         = len(DIAGNOSTIC_LABELS),
    rank               = 8,
    lora_alpha         = 16.0,
    lora_dropout       = 0.05,
)


def load_weights(npz_path: str) -> list[np.ndarray]:
    data = np.load(npz_path)
    return [data[k] for k in sorted(data.files)]


def build_model_from_weights(weights: list[np.ndarray]):
    model = build_model(LORA_CFG).to(DEVICE)
    set_lora_weights(model, weights)
    model.eval()
    return model


def collect_events(disease: str, n_events: int, seed: int, ollama_url: str):
    """Run a fresh WorldEngine until n_events clinic visits are collected.

    Uses step_tick() directly (no FL learner attached) — pure data collection.
    The Ollama diagnostic function is registered so conversations are generated.
    """
    from simulation.world import WorldEngine
    from simulation.ollama_client import OllamaDiagnosticClient
    from simulation.world_config import WorldConfig as SimWorldConfig, AgentConfig, EpidemicConfig
    from simulation.data_sources import TemplateDataSource

    prog_names, d_weights = DISEASE_PRESETS[disease]
    end_cond = end_condition_from_config("horizon", 40)

    sim_wc = SimWorldConfig(
        agents=AgentConfig(
            num_agents=DEFAULT_NUM_AGENTS,
            data_source=TemplateDataSource(seed=seed),
            background_visit_rate=DEFAULT_BG_RATE,
        ),
        epidemic=EpidemicConfig(
            progressions=prog_names,
            disease_strategy=prog_names[0] if prog_names else "Influenza",
            disease_weights=d_weights,
            beta_scale=DEFAULT_BETA_SCALE,
            contact_rate_sigma=DEFAULT_SIGMA,
            initial_seeds=DEFAULT_INITIAL_SEEDS,
        ),
    )
    world = WorldEngine(sim_wc, seed=seed, end_condition=end_cond)

    client = OllamaDiagnosticClient()
    if not client.health_check():
        sys.exit(f"Ollama not reachable at {ollama_url}. Start it first.")

    q = world.clinic_queue
    world.register_diagnostic_fn(lambda ev: client.diagnose(ev, _queue=q))

    ticks_per_round = DEFAULT_SIM_DAYS * WorldEngine.TICKS_PER_DAY
    events = []
    round_num = 0
    print(f"\n  New-silo sim  disease={disease}  target={n_events} events  seed={seed}")
    while len(events) < n_events and not world.is_done:
        round_num += 1
        start = len(q.processed)
        for _ in range(ticks_per_round):
            world.step_tick()
            if world.is_done:
                break
        new_evs = q.processed[start:]
        events.extend(new_evs)
        sir = world.sir_model
        print(f"  round {round_num:>2}  S={sir.S:>3} I={sir.I:>3} R={sir.R:>3}  "
              f"new={len(new_evs):>3}  total={len(events):>4}", flush=True)

    print(f"  Collected {len(events)} events over {round_num} rounds\n")
    return events


def event_to_text_label(ev) -> tuple[str, str] | None:
    """Extract (text, label) from a DiagnosticEvent. Returns None if label unknown."""
    patient_turns = [t["text"] for t in ev.conversation if t["role"] == "patient"]
    text = patient_turns[-1] if patient_turns else ""
    label = ev.ground_truth or ""
    if label not in label2id or not text:
        return None
    return text, label


def run_inference(model, texts: list[str]) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE))
    return out.logits.argmax(dim=-1).cpu().tolist()


def confusion_table(preds, trues, label_fn, title):
    classes = sorted(set(label_fn(id2label[t]) for t in trues)
                     | set(label_fn(id2label[p]) for p in preds))
    idx = {c: i for i, c in enumerate(classes)}
    n   = len(classes)
    mat = [[0]*n for _ in range(n)]
    for p, t in zip(preds, trues):
        mat[idx[label_fn(id2label[t])]][idx[label_fn(id2label[p])]] += 1

    col_w = max(len(c) for c in classes) + 1
    row_w = max(len(c) for c in classes) + 1
    header = f"{'true \\ pred':<{row_w}}" + "".join(f"{c:>{col_w}}" for c in classes) + f"{'|total':>7}"
    sep    = "─" * len(header)
    print(f"\n  {title}")
    print(f"  {header}")
    print(f"  {sep}")
    for ri, rc in enumerate(classes):
        row   = mat[ri]
        total = sum(row)
        cells = "".join(f"{v:>{col_w}}" for v in row)
        print(f"  {rc:<{row_w}}{cells}{total:>7}")
    print(f"  {sep}")
    col_totals = [sum(mat[r][c] for r in range(n)) for c in range(n)]
    cells = "".join(f"{v:>{col_w}}" for v in col_totals)
    print(f"  {'total':<{row_w}}{cells}")


def evaluate(model, events, label: str):
    pairs = [event_to_text_label(ev) for ev in events]
    pairs = [(t, l) for p in pairs if p is not None for t, l in [p]]
    if not pairs:
        print(f"  {label}: no evaluable events (all labels outside DIAGNOSTIC_LABELS?)")
        return

    texts  = [t for t, _ in pairs]
    labels = [label2id[l] for _, l in pairs]
    preds  = run_inference(model, texts)
    n      = len(preds)

    sev_ok = diag_ok = danger = n_severe = 0
    for p, t in zip(preds, labels):
        pd, ps = _split_label(id2label[p])
        td, ts = _split_label(id2label[t])
        sev_ok  += ps == ts
        diag_ok += pd == td
        if ts == "severe":
            n_severe += 1
            if ps != "severe" or id2label[p] == "non-infectious":
                danger += 1

    _f = lambda v: f"{v:.3f}" if v == v else "—"
    dr = danger / n_severe if n_severe > 0 else float("nan")
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"  n={n}  triage={_f(sev_ok/n)}  diag={_f(diag_ok/n)}  danger={_f(dr)}")
    print(f"{'═'*60}")

    confusion_table(preds, labels, lambda l: _split_label(l)[0], f"{label} — Disease")
    confusion_table(preds, labels, lambda l: _split_label(l)[1] or "none", f"{label} — Severity")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True,
                    help="One or more .npz weight files to evaluate")
    ap.add_argument("--labels", nargs="*",
                    help="Display labels for each weights file (default: filename)")
    ap.add_argument("--disease", choices=list(DISEASE_PRESETS), default="mixed",
                    help="Disease mix for the new eval silo")
    ap.add_argument("--n-events", type=int, default=80,
                    help="Target number of clinic events to collect")
    ap.add_argument("--seed", type=int, default=999,
                    help="RNG seed for the new silo (use a seed not in training)")
    ap.add_argument("--ollama-url", default="http://localhost:11434",
                    help="Ollama server URL")
    args = ap.parse_args()

    labels = args.labels or [Path(w).stem for w in args.weights]
    if len(labels) != len(args.weights):
        ap.error("--labels count must match --weights count")

    t0 = time.time()
    events = collect_events(args.disease, args.n_events, args.seed, args.ollama_url)

    for npz_path, label in zip(args.weights, labels):
        print(f"\n  Loading {npz_path} …")
        weights = load_weights(npz_path)
        model   = build_model_from_weights(weights)
        evaluate(model, events, label)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  Total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
