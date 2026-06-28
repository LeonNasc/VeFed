#!/usr/bin/env python3
"""Render a sample patient-nurse conversation as a messaging-app style chat figure.

Generates one real conversation via simulation.conversation.simulate_conversation
(anxious personality, Morven disease) and draws it as alternating chat bubbles
(nurse left/gray, patient right/blue), matplotlib only, saved to paper_figures/.
"""
import random
import textwrap
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.lines import Line2D

from simulation.conversation import simulate_conversation
from simulation.symptom_language import Personality
from simulation.fictional_diseases import MORVEN

OUT_PATH = Path("paper_figures/sec1_fig2_conversation_example.png")

NURSE_COLOR = "#e9e9eb"
NURSE_TEXT = "#1c1c1e"
PATIENT_COLOR = "#0a84ff"
PATIENT_TEXT = "#ffffff"
WRAP_WIDTH = 52
FONT_SIZE = 11.5
PAD = 0.35
LINE_H = 0.30
GAP = 0.35
BUBBLE_W = 8.6
AVATAR_SIZE = 0.85
AVATAR_MARGIN = 0.25
FIG_W = 10.5 + 2 * (AVATAR_SIZE + AVATAR_MARGIN)


def draw_robot_avatar(ax, cx, cy, role, size=AVATAR_SIZE):
    """Draw a small vector robot icon (no emoji/fonts needed)."""
    is_patient = role == "patient"
    body_color = "#0a84ff" if is_patient else "#9a9ea3"
    eye_color = "#bfe4ff" if is_patient else "#bfffd0"
    badge_color = "#ff3b30" if not is_patient else "#ff6961"

    # outer circle (face plate)
    ax.add_patch(Circle((cx, cy), size / 2, facecolor=body_color,
                         edgecolor="white", linewidth=1.2, zorder=4))
    # antenna
    ax.add_line(Line2D([cx, cx], [cy + size / 2, cy + size / 2 + size * 0.18],
                        color=body_color, linewidth=1.6, zorder=4))
    ax.add_patch(Circle((cx, cy + size / 2 + size * 0.18), size * 0.07,
                         facecolor=badge_color, edgecolor="none", zorder=4))
    # eyes
    eye_dx = size * 0.17
    eye_y = cy + size * 0.06
    for dx in (-eye_dx, eye_dx):
        ax.add_patch(Circle((cx + dx, eye_y), size * 0.09,
                             facecolor=eye_color, edgecolor="none", zorder=5))
    # mouth
    mouth_w = size * 0.32
    ax.add_patch(Rectangle((cx - mouth_w / 2, cy - size * 0.22), mouth_w, size * 0.06,
                            facecolor=eye_color, edgecolor="none", zorder=5))
    # role badge: cross for the doctor/nurse robot, heart-pulse dot for patient robot
    bx, by = cx + size * 0.32, cy - size * 0.32
    if is_patient:
        ax.add_patch(Circle((bx, by), size * 0.13, facecolor="white",
                             edgecolor=badge_color, linewidth=1.0, zorder=6))
        ax.add_patch(Circle((bx, by), size * 0.05, facecolor=badge_color,
                             edgecolor="none", zorder=6))
    else:
        ax.add_patch(Circle((bx, by), size * 0.14, facecolor="white",
                             edgecolor=badge_color, linewidth=1.0, zorder=6))
        cw, ch = size * 0.14, size * 0.035
        ax.add_patch(Rectangle((bx - cw / 2, by - ch / 2), cw, ch,
                                facecolor=badge_color, edgecolor="none", zorder=6))
        ax.add_patch(Rectangle((bx - ch / 2, by - cw / 2), ch, cw,
                                facecolor=badge_color, edgecolor="none", zorder=6))


def build_conversation():
    rng = random.Random(7)
    inner_state = SimpleNamespace(severity=0.5, disease_name="morven", trend="worsening")
    opener = ("I have been feeling really off the past few days. I can't seem to stay warm "
               "no matter what I do, and my stomach has been cramping on and off.")
    record = simulate_conversation(
        opener=opener,
        inner_state=inner_state,
        days=6,
        personality=Personality.NEUTRAL,
        rng=rng,
        probe_responses=MORVEN["probe_responses"],
    )
    return [(t.role, t.text) for t in record.turns]


def wrap(text, width=WRAP_WIDTH):
    return "\n".join(textwrap.wrap(text, width=width))


def main():
    turns = build_conversation()

    wrapped = [(role, wrap(text)) for role, text in turns]
    n_lines = [w.count("\n") + 1 for _, w in wrapped]
    bubble_heights = [PAD * 2 + nl * LINE_H for nl in n_lines]
    total_h = sum(bubble_heights) + GAP * (len(wrapped) + 1)

    fig, ax = plt.subplots(figsize=(FIG_W, max(4.0, total_h)))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, total_h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y = total_h - GAP
    for (role, text), h in zip(wrapped, bubble_heights):
        is_patient = role == "patient"
        n_lines_here = text.count("\n") + 1
        char_w = max(len(line) for line in text.split("\n"))
        bw = min(BUBBLE_W, max(2.2, char_w * 0.115 + PAD * 2))

        avatar_zone = AVATAR_SIZE + AVATAR_MARGIN
        x_right = FIG_W - avatar_zone - 0.15 if is_patient else avatar_zone + 0.15 + bw
        x_left = x_right - bw

        y_top = y
        y_bottom = y - h

        avatar_cx = FIG_W - avatar_zone / 2 if is_patient else avatar_zone / 2
        avatar_cy = (y_top + y_bottom) / 2
        draw_robot_avatar(ax, avatar_cx, avatar_cy, role)

        color = PATIENT_COLOR if is_patient else NURSE_COLOR
        text_color = PATIENT_TEXT if is_patient else NURSE_TEXT

        box = FancyBboxPatch(
            (x_left, y_bottom), bw, h,
            boxstyle="round,pad=0,rounding_size=0.22",
            linewidth=0,
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)

        ax.text(
            (x_left + x_right) / 2, (y_top + y_bottom) / 2, text,
            ha="center", va="center", fontsize=FONT_SIZE, color=text_color,
            family="DejaVu Sans", linespacing=1.5, zorder=3,
        )

        label = "Patient" if is_patient else "Doctor"
        label_x = x_right if is_patient else x_left
        ha = "right" if is_patient else "left"
        ax.text(
            label_x, y_top + 0.12, label, ha=ha, va="bottom",
            fontsize=9, color="#8e8e93", style="italic", zorder=3,
        )

        y = y_bottom - GAP

    fig.tight_layout(pad=0.6)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
