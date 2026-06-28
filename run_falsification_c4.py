#!/usr/bin/env python3
"""
Falsification C4 — known-disease control injection (falsification.md, Claim 2).

Injects a "new disease" that is actually just Velarex text relabeled "unknown"
at training time (same injection mechanism, timing, and volume as the real
Morven experiment). It should NOT form a new embedding cluster -- it should
be absorbed into the existing Velarex cluster, since the text content is
identical to what the model already knows as Velarex.

If Velarex probes start separating out as a distinct "unknown-like" group
just because some training examples were labeled "unknown" (even though the
text is unchanged Velarex content), that is failure mode F5 (false cluster):
the detection mechanism would be keying off the label assignment itself,
not genuine novel content.

Implementation: monkeypatch _build_morven_pool to sample Velarex text instead
of Morven text. Everything else (round loop, injection timing, label
overwrite to "unknown") is unchanged -- this isolates exactly one variable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_unknown_disease as rud

OUT_DIR = Path("results/falsification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_FAKE_NOVEL_DIST = [("velarex", "mild", 2.0), ("velarex", "moderate", 3.0), ("velarex", "severe", 1.0)]


def _build_fake_novel_pool(n: int, seed: int) -> list[dict]:
    """Drop-in replacement for _build_morven_pool: samples Velarex text, not Morven."""
    lib = rud.FictionalPhraseLibrary(seed=seed + 77777)
    return lib.sample_pool(_FAKE_NOVEL_DIST, n, seed_offset=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default="falsification_c4_fake_velarex_unknown")
    args = ap.parse_args()

    rud._build_morven_pool = _build_fake_novel_pool  # monkeypatch before calling

    cfg = rud.UnknownDiseaseConfig(
        schedule="gaussian", n_silos=3, n_rounds=20,
        injection_round=10, injection_per_round=8, do_inject=True,
        run_name=args.run_name, results_dir=str(OUT_DIR),
        seed=args.seed, training_device="cuda", local_epochs=args.local_epochs,
    )
    result = rud.run_unknown_disease(cfg)
    print("\nRun summary:", json.dumps(result["summary"], indent=2)[:500])

    # ── Post-hoc check: do VELAREX probes separate out as if they were novel? ──
    out_dir = OUT_DIR / cfg.run_name
    probe_events = rud.generate_fictional_probe_events(cfg.probe_per_band, cfg.probe_seed)
    probe_labels = [ev.ground_truth for ev in probe_events]

    snap_rounds = sorted(int(p.stem.split("_r")[1]) for p in out_dir.glob("logits_r*.npz"))
    print(f"\nSnapshot rounds available: {snap_rounds}")

    velarex_sil_curve = []
    for rnd in snap_rounds:
        logits = np.load(out_dir / f"logits_r{rnd:02d}.npz")["logits"]
        coords = rud._project_umap(logits, seed=cfg.seed)
        # Relabel: treat "velarex" as the group-of-interest, mirroring _silhouette_morven's
        # "morven" substring check, to ask "does velarex separate out like a novel cluster
        # would?" Real morven probes must be renamed too (not just left alone) so they don't
        # also match the "morven" substring check and contaminate the group-of-interest.
        fake_labels = [
            l.replace("velarex", "morven") if l.startswith("velarex")
            else l.replace("morven", "background") if l.startswith("morven")
            else l
            for l in probe_labels
        ]
        sil = rud._silhouette_morven(coords, fake_labels)
        velarex_sil_curve.append({"round": rnd, "velarex_as_novel_silhouette": sil})
        print(f"  R{rnd:02d}: velarex-as-if-novel silhouette = {sil:.4f}")

    real_morven_sil = result["summary"]["silhouette_curve"]
    print("\nFor reference, REAL morven-vs-known silhouette in this same run "
         "(should be near baseline/control level since no real Morven text was injected):")
    for c in real_morven_sil:
        print(f"  R{c['round']:02d}: {c['silhouette']:.4f}")

    final_velarex_sil = velarex_sil_curve[-1]["velarex_as_novel_silhouette"] if velarex_sil_curve else float("nan")
    control_level = 0.718  # documented control (no-Morven) silhouette from results_2026-06-16.md
    verdict = (
        f"C4 PASSES (final velarex-as-novel silhouette {final_velarex_sil:.3f} is not "
        f"higher than the documented natural-drift control level ~{control_level}) -- "
        "injecting known-disease text labeled 'unknown' does not create a spurious "
        "detected cluster; detection tracks genuine novel content, not the label alone."
        if final_velarex_sil < control_level + 0.05 else
        f"C4 FAILS (final velarex-as-novel silhouette {final_velarex_sil:.3f} exceeds the "
        f"control level ~{control_level}) -- the detection mechanism may be keying off the "
        "'unknown' label assignment itself rather than genuine novel content. False-positive risk."
    )
    print(f"\nVerdict: {verdict}")

    summary = {
        "control": "falsification.md C4 -- known-disease control injection",
        "injected_content": "Velarex text, relabeled 'unknown' at injection (same mechanism as real Morven injection)",
        "velarex_as_novel_silhouette_curve": velarex_sil_curve,
        "real_morven_silhouette_curve_same_run": real_morven_sil,
        "final_velarex_as_novel_silhouette": final_velarex_sil,
        "control_reference_level": control_level,
        "verdict": verdict,
    }
    out_path = OUT_DIR / f"c4_known_disease_control_seed{args.seed}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
