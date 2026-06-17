"""
Print confusion matrices for all 4 static ablation stages (rep 0).
Re-trains each stage with the same hypers used in run_ablation.py,
then captures all (pred, true) pairs on the holdout set.

Stages evaluated:
  3 — centralized non-IID   (datasets/merged_noniid)
  4 — centralized IID       (datasets/20260611_111743)
  5 — local-only IID        (datasets/20260611_121029)
  6 — local-only non-IID   (datasets/20260611_143904)
"""
import sys, os
os.environ["WANDB_MODE"] = "disabled"

import json, random, time
from pathlib import Path
from collections import Counter

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from fl.lora import LoRAConfig, build_model
from fl.learner import DIAGNOSTIC_LABELS, build_label_map, _split_label

# ── Hypers (match run_ablation.py) ───────────────────────────────────────────
EPOCHS     = 15
LR         = 1e-4
BATCH      = 8
MAX_LEN    = 128
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"

label2id, id2label = build_label_map()

LORA_CFG = LoRAConfig(
    model_name_or_path = MODEL_NAME,
    num_labels         = len(DIAGNOSTIC_LABELS),
    rank               = 8,
    lora_alpha         = 16,
    lora_dropout       = 0.1,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def extract(records: list[dict]):
    texts, labels = [], []
    for r in records:
        lid = label2id.get(r.get("label", ""))
        if lid is None:
            continue
        texts.append(r["text"])
        labels.append(lid)
    return texts, labels


def train_and_eval(train_recs, eval_recs):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_texts, train_labels = extract(train_recs)
    eval_texts,  eval_labels  = extract(eval_recs)

    model = build_model(LORA_CFG).to(DEVICE)

    if train_texts:
        enc = tokenizer(train_texts, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        label_t = torch.tensor(train_labels, dtype=torch.long)
        loader  = DataLoader(
            TensorDataset(enc["input_ids"], enc["attention_mask"], label_t),
            batch_size=BATCH, shuffle=True,
        )
        model.train()
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        for ep in range(EPOCHS):
            for iids, amask, blabels in loader:
                opt.zero_grad()
                out = model(input_ids=iids.to(DEVICE),
                            attention_mask=amask.to(DEVICE),
                            labels=blabels.to(DEVICE))
                out.loss.backward()
                opt.step()

    preds, trues = [], []
    if eval_texts:
        enc = tokenizer(eval_texts, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(DEVICE),
                        attention_mask=enc["attention_mask"].to(DEVICE))
        preds = out.logits.argmax(dim=-1).cpu().tolist()
        trues = eval_labels
    return preds, trues


def load_stage(dataset_dir: Path, mode: str):
    """Returns list of (tag, train_recs, eval_recs) tuples."""
    n_silos = sum(1 for d in dataset_dir.iterdir()
                  if d.is_dir() and d.name.startswith("silo_"))
    silo_data = []
    for i in range(n_silos):
        train = load_jsonl(dataset_dir / f"silo_{i}" / "train.jsonl")
        hold  = load_jsonl(dataset_dir / f"silo_{i}" / "holdout.jsonl")
        silo_data.append((train, hold))

    if mode == "centralized":
        rng = random.Random(SEED)
        train_all = [r for t, _ in silo_data for r in t]
        eval_all  = [r for _, h in silo_data for r in h]
        rng.shuffle(train_all)
        return [("pooled", train_all, eval_all)]
    else:
        return [(f"silo_{i}", t, h) for i, (t, h) in enumerate(silo_data)]


# ── Confusion matrix printer ─────────────────────────────────────────────────

def confusion_table(preds, trues, label_fn, title):
    classes = sorted(set(label_fn(id2label[t]) for t in trues))
    # also include any predicted classes not in trues
    classes = sorted(set(classes) | set(label_fn(id2label[p]) for p in preds))
    idx = {c: i for i, c in enumerate(classes)}
    n   = len(classes)
    mat = [[0]*n for _ in range(n)]
    for p, t in zip(preds, trues):
        pc = label_fn(id2label[p])
        tc = label_fn(id2label[t])
        mat[idx[tc]][idx[pc]] += 1

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


def get_disease(label):  return _split_label(label)[0]
def get_severity(label): return _split_label(label)[1] or "none"


# ── Stages ───────────────────────────────────────────────────────────────────

STAGES = [
    ("Stage 3 — Centralized non-IID", "datasets/merged_noniid",      "centralized"),
    ("Stage 4 — Centralized IID",     "datasets/20260611_111743",     "centralized"),
    ("Stage 5 — Local-only IID",      "datasets/20260611_121029",     "local"),
    ("Stage 6 — Local-only non-IID",  "datasets/20260611_143904",     "local"),
]

BASE = Path(__file__).parent
random.seed(SEED)
torch.manual_seed(SEED)

for stage_name, ds_path, mode in STAGES:
    dataset_dir = BASE / ds_path
    print(f"\n{'═'*60}")
    print(f"  {stage_name}")
    print(f"  source={ds_path}  mode={mode}")
    print(f"{'═'*60}")

    t0 = time.time()
    groups = load_stage(dataset_dir, mode)
    all_preds, all_trues = [], []

    for tag, train_recs, eval_recs in groups:
        print(f"\n  Training {tag}: {len(train_recs)} train / {len(eval_recs)} holdout ...", flush=True)
        preds, trues = train_and_eval(train_recs, eval_recs)
        all_preds.extend(preds)
        all_trues.extend(trues)

        n = len(preds)
        if n == 0:
            print(f"  {tag}: no eval samples")
            continue

        sev_ok = diag_ok = 0
        for p, t in zip(preds, trues):
            pd, ps = _split_label(id2label[p])
            td, ts = _split_label(id2label[t])
            sev_ok  += ps == ts
            diag_ok += pd == td
        print(f"  {tag}: n={n}  triage={sev_ok/n:.3f}  diag={diag_ok/n:.3f}")

        confusion_table(preds, trues, get_disease,  f"{tag} — Disease confusion")
        confusion_table(preds, trues, get_severity, f"{tag} — Severity confusion")

    if mode == "centralized" and all_preds:
        pass  # already printed above
    elif mode == "local" and all_preds:
        print(f"\n  ── Macro (all silos pooled for display) ──")
        confusion_table(all_preds, all_trues, get_disease,  "All silos — Disease confusion")
        confusion_table(all_preds, all_trues, get_severity, "All silos — Severity confusion")

    print(f"\n  [{stage_name} done in {time.time()-t0:.0f}s]")
