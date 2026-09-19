#!/usr/bin/env python3
"""Two orientation figures for the report, redesigned to be method-specific rather than
generic box diagrams:

  teaser.png    graphical abstract: the task, the two shape signals drawn as what they
                actually are (a whole-object vector vs a 4-fingertip contact diagram with
                signed distances and surface normals), and the headline zero-shot result.
  pipeline.png  method / architecture overview: the shape signal fills a z-slot (learned
                encoder or frozen raw concat) alongside the 102-d proprioceptive history,
                feeds the policy -> PD targets -> LEAP Hand in Isaac Gym -> angvel reward,
                with the per-control-step contact loop and the PPO loop drawn explicitly.

Run from the repo root (writes to docs/figures/):
  python3 src/LEAP_Hand_Sim/leapsim/tools/plot_overview.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Rectangle, Circle, Ellipse, FancyArrowPatch,
                                Arc, Polygon)

TEAL, AMBER, BRICK, GREY = "#0c8b98", "#b9761b", "#a94a3f", "#9aa7ad"
INK, FAINT, SKIN = "#243138", "#d7dee1", "#c9b39c"
OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- primitives
def fbox(ax, cx, cy, w, h, text, fc="white", ec=INK, tc=INK, fs=10, lw=1.4, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight, zorder=6)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.9, ls="-", rad=0.0, ms=15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def glyph(ax, kind, cx, cy, s, color, ec=INK):
    if kind == "cuboid":
        ax.add_patch(FancyBboxPatch((cx - s, cy - s), 2 * s, 2 * s,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=color, edgecolor=ec, lw=1.2, alpha=.92))
    elif kind == "cylinder":
        ax.add_patch(Rectangle((cx - s, cy - 1.1 * s), 2 * s, 2.2 * s, facecolor=color, edgecolor="none", alpha=.92))
        ax.add_patch(Ellipse((cx, cy + 1.1 * s), 2 * s, 0.66 * s, facecolor=color, edgecolor=ec, lw=1.1, alpha=.95))
        ax.add_patch(Ellipse((cx, cy - 1.1 * s), 2 * s, 0.66 * s, facecolor=color, edgecolor=ec, lw=1.1, alpha=.7))
        ax.plot([cx - s, cx - s], [cy - 1.1 * s, cy + 1.1 * s], color=ec, lw=1.1)
        ax.plot([cx + s, cx + s], [cy - 1.1 * s, cy + 1.1 * s], color=ec, lw=1.1)
    elif kind == "sphere":
        ax.add_patch(Circle((cx, cy), s, facecolor=color, edgecolor=ec, lw=1.2, alpha=.92))


def contacts(ax, cx, cy, R, angles, color=TEAL, normal_len=0.62, tipr=0.15, gap=0.0,
             ray=False, lw=1.6):
    """Draw fingertip contacts around an object: fingertip dot, optional signed-distance
    ray, and an outward surface normal arrow. Returns the surface points."""
    pts = []
    for a in angles:
        th = np.radians(a)
        sx, sy = cx + R * np.cos(th), cy + R * np.sin(th)             # surface point
        fx, fy = cx + (R + gap) * np.cos(th), cy + (R + gap) * np.sin(th)  # fingertip
        if ray and gap > 0:
            ax.plot([fx, fy and fy], [fy, fy], alpha=0)              # noop guard
            ax.plot([fx, sx], [fy, sy], ls=(0, (1, 1)), color=GREY, lw=1.2, zorder=3)
        arrow(ax, sx, sy, sx + normal_len * np.cos(th), sy + normal_len * np.sin(th),
              color=color, lw=lw, ms=11)
        ax.add_patch(Circle((fx, fy), tipr, facecolor=color, edgecolor="white", lw=0.9, zorder=7))
        pts.append((sx, sy, th))
    return pts


def draw_grasp(ax, cx, cy, R):
    """A stylized 4-fingered grasp reaching from the left, with per-fingertip contact
    geometry (signed distance + surface normal) called out. Method-specific hero glyph."""
    # object (rounded cuboid so face-ish normals read as surface normals)
    ax.add_patch(FancyBboxPatch((cx - R, cy - R), 2 * R, 2 * R, boxstyle="round,pad=0,rounding_size=0.28",
                                facecolor="#eef4f5", edgecolor=INK, lw=1.5, zorder=2))
    # palm bar on the left
    px = cx - R - 2.5
    ax.add_patch(FancyBboxPatch((px - 0.3, cy - 1.7), 0.62, 3.4, boxstyle="round,pad=0.03,rounding_size=0.28",
                                facecolor="#efe7dd", edgecolor=INK, lw=1.3, zorder=1))
    tips = [(138, 1.15, False), (166, 0.42, False), (198, -0.42, True), (232, -1.35, False)]
    for a, dy, detailed in tips:
        th = np.radians(a)
        g = 0.28 if detailed else 0.0                      # this finger sits slightly off -> signed distance
        sx, sy = cx + R * np.cos(th), cy + R * np.sin(th)
        fx, fy = cx + (R + g) * np.cos(th), cy + (R + g) * np.sin(th)
        bx, by = px + 0.3, cy + dy                         # finger base on palm
        kx, ky = 0.5 * bx + 0.5 * fx, 0.5 * by + 0.5 * fy + 0.28   # knuckle, bowed up
        ax.plot([bx, kx, fx], [by, ky, fy], "-", color=SKIN, lw=4.2, solid_capstyle="round", zorder=3)
        ax.add_patch(Circle((bx, by), 0.11, facecolor="#b39a80", edgecolor="none", zorder=4))
        ax.add_patch(Circle((kx, ky), 0.10, facecolor="#b39a80", edgecolor="none", zorder=4))
        ax.add_patch(Circle((fx, fy), 0.17, facecolor=TEAL, edgecolor="white", lw=1.0, zorder=6))
        # surface normal
        arrow(ax, sx, sy, sx + 0.7 * np.cos(th), sy + 0.7 * np.sin(th), color=TEAL, lw=1.8, ms=12)
        if detailed:
            ax.plot([fx, sx], [fy, sy], ls=(0, (1, 1.2)), color="#6b7a80", lw=1.5, zorder=5)
            ax.annotate("d  signed distance", xy=(0.5 * (fx + sx), 0.5 * (fy + sy)),
                        xytext=(cx - 0.2, cy - 2.15), fontsize=8.4, color=INK,
                        ha="center", arrowprops=dict(arrowstyle="-", color="#6b7a80", lw=0.9))
            ax.annotate("n  surface normal", xy=(sx + 0.55 * np.cos(th), sy + 0.55 * np.sin(th)),
                        xytext=(cx + 0.2, cy + 2.2), fontsize=8.4, color=TEAL,
                        ha="center", arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.9))


# ================================================================ TEASER
fig = plt.figure(figsize=(13.6, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.22, 1.12], wspace=0.12,
                      left=0.015, right=0.985, top=0.85, bottom=0.11)

# --- Panel A: task ---
a = fig.add_subplot(gs[0]); a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
a.set_title("A.  Leave-one-out over 3 primitive classes", fontsize=10.5, loc="left", color=INK)
a.add_patch(FancyBboxPatch((0.4, 5.7), 6.5, 3.3, boxstyle="round,pad=0.1,rounding_size=0.2",
                           fc="#f2f6f7", ec=GREY, lw=1.3))
a.text(3.65, 8.5, "train (2 classes)", fontsize=9.5, ha="center", color=INK, weight="bold")
glyph(a, "cuboid", 2.0, 7.05, 0.82, TEAL)
glyph(a, "cylinder", 5.3, 7.05, 0.78, AMBER)
a.add_patch(FancyBboxPatch((7.6, 5.7), 2.0, 3.3, boxstyle="round,pad=0.1,rounding_size=0.2",
                           fc="white", ec=BRICK, lw=1.6, linestyle="--"))
a.text(8.6, 8.5, "zero-shot", fontsize=9.5, ha="center", color=BRICK, weight="bold")
glyph(a, "sphere", 8.6, 7.05, 0.8, BRICK)
arrow(a, 7.05, 7.05, 7.55, 7.05, color=INK, lw=1.6)
a.text(5.0, 4.35, "one policy, trained on two classes,\nrotates the held-out third with no fine-tuning",
       fontsize=9.2, ha="center", va="top", color=INK)
a.text(5.0, 1.75, "metric: object angular velocity\nabout world vertical (rad/s)",
       fontsize=8.6, ha="center", va="top", color="#5c6b72", style="italic")

# --- Panel B: two shape signals, drawn as what they are ---
b = fig.add_subplot(gs[1]); b.set_xlim(0, 10); b.set_ylim(0, 10); b.axis("off")
b.set_title("B.  Two forms of the shape signal", fontsize=10.5, loc="left", color=INK)
# global static code (top)
b.text(0.2, 9.15, "global, static code", fontsize=9.6, color=BRICK, weight="bold")
for i in range(6):
    b.add_patch(Rectangle((0.35 + i * 0.52, 7.95), 0.46, 0.62, facecolor=FAINT, edgecolor=INK, lw=1.0))
b.text(3.65, 8.26, "one vector for the whole object,\nfixed for the episode", fontsize=8.4, va="center", color=INK)
b.text(9.75, 8.26, "no gain  ✗", fontsize=9.6, va="center", ha="right", color=BRICK, weight="bold")
b.plot([0.2, 9.8], [7.35, 7.35], color=FAINT, lw=1.1)
# local dynamic feature (bottom) -- the hero glyph
b.text(0.2, 6.75, "local, dynamic feature", fontsize=9.6, color=TEAL, weight="bold")
draw_grasp(b, 5.15, 3.55, 1.12)
b.text(9.75, 0.75, "16-d  =  4 fingertips × (distance, normal),  recomputed every step   ✓",
       fontsize=8.7, va="center", ha="right", color=TEAL, weight="bold")

# --- Panel C: result ---
c = fig.add_subplot(gs[2])
cats = ["cuboid", "cylinder", "sphere"]
B = [0.103, 0.114, 0.150]; LL = [0.131, 0.118, 0.195]; gain = ["+27%", "+4%", "+30%"]
x = np.arange(3); w = 0.36
c.bar(x - w / 2, B, w, color=GREY, label="proprioception (baseline)")
c.bar(x + w / 2, LL, w, color=TEAL, label="+ learned local feature")
for i in range(3):
    c.text(x[i] + w / 2, LL[i] + 0.006, gain[i], ha="center", fontsize=9,
           color=(TEAL if gain[i] != "+4%" else "#8a9298"), weight="bold")
c.set_xticks(x); c.set_xticklabels([f"held-out\n{k}" for k in cats], fontsize=9)
c.set_ylabel("zero-shot angular velocity (rad/s)", fontsize=9.2)
c.set_ylim(0, 0.235)
c.set_title("C.  Zero-shot rotation, held-out class", fontsize=10.5, loc="left", color=INK)
c.legend(frameon=False, fontsize=8.6, loc="upper left")
c.grid(axis="y", alpha=.2)
for s in ("top", "right"):
    c.spines[s].set_visible(False)
c.text(0.5, -0.235, "(the global static code loses in all three directions)",
       transform=c.transAxes, ha="center", fontsize=8.2, color="#8a9298", style="italic")

fig.suptitle("A minimal shape signal for cross-category in-hand reorientation",
             fontsize=12.5, weight="bold", color=INK, x=0.015, ha="left", y=0.975)
fig.savefig(f"{OUT}/teaser.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}/teaser.png")
plt.close(fig)


# ================================================================ PIPELINE / ARCHITECTURE
# Deliberately focused on the paper's axis: the shape signal z -> {encoder (learned) |
# raw (frozen)} -> policy, with the z0/z1/z2 (static-global) vs z_local (dynamic-local)
# distinction made explicit. RL/physics detail (Isaac Gym, PD, PPO, loops) is omitted.
fig, ax = plt.subplots(figsize=(13.8, 5.3))
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")

ax.text(8.0, 9.35, "One policy, trained on two of {box, cylinder, sphere}, evaluated "
                   "zero-shot on the held-out third", ha="center", va="center",
        fontsize=9.2, color="#5c6b72")

# --- shape signal z: the two families, spelled out ---
ax.add_patch(FancyBboxPatch((0.3, 1.5), 6.05, 6.8, boxstyle="round,pad=0.03,rounding_size=0.08",
                            fc="#fbfbfb", ec=INK, lw=1.5))
ax.text(3.32, 7.95, "shape signal  z", ha="center", fontsize=10.5, color=INK, weight="bold")
# static-global ladder: z0, z1, z2
ax.add_patch(FancyBboxPatch((0.55, 4.5), 5.55, 2.95, boxstyle="round,pad=0.02,rounding_size=0.06",
                            fc="white", ec=BRICK, lw=1.5))
ax.text(3.32, 7.02, "static · global   (one code per episode)", ha="center", fontsize=8.9, color=BRICK, weight="bold")
ax.text(0.85, 6.42, "z0    physical scalars  (non-shape baseline)", fontsize=7.7, color=INK, ha="left")
ax.text(0.85, 5.82, "z1    analytic geometry: extents, aspect, inertia, SA/V  (8-d)", fontsize=7.7, color=INK, ha="left")
ax.text(0.85, 5.22, "z2    learned shape embedding  (proposed)", fontsize=7.7, color=INK, ha="left")
# dynamic-local: z_local
ax.add_patch(FancyBboxPatch((0.55, 1.85), 5.55, 2.3, boxstyle="round,pad=0.02,rounding_size=0.06",
                            fc="white", ec=TEAL, lw=1.6))
ax.text(3.32, 3.78, "dynamic · local   (recomputed every step)", ha="center", fontsize=8.9, color=TEAL, weight="bold")
ax.add_patch(Circle((1.3, 2.72), 0.34, facecolor="#eef4f5", edgecolor=INK, lw=1.0, zorder=1))
contacts(ax, 1.3, 2.72, 0.34, [45, 135, 225, 315], color=TEAL, normal_len=0.32, tipr=0.09)
ax.text(2.1, 2.72, "z_local    4 × (signed distance, normal)  =  16-d",
        fontsize=7.9, color=INK, ha="left", va="center")

# --- how z enters the policy: encoder (learned) vs raw (frozen) ---
ax.text(8.35, 6.75, "how z enters the policy", ha="center", fontsize=8.6, color="#5c6b72", style="italic")
fbox(ax, 8.35, 5.7, 2.7, 1.15, "encoder φ    (learned)", fc="white", fs=9.2)
fbox(ax, 8.35, 4.0, 2.7, 1.15, "pass raw    (frozen / static)", fc="white", fs=9.2)
arrow(ax, 6.35, 5.5, 6.95, 5.7)
arrow(ax, 6.35, 4.35, 6.95, 4.05)
ax.text(6.62, 5.05, "z", fontsize=11, color=INK, style="italic", weight="bold")

# --- combine with proprioception, then policy ---
fbox(ax, 11.85, 4.85, 2.75, 1.75, "concat ⊕\nwith proprioception\n(102-d)", fc="white", fs=8.9)
arrow(ax, 9.7, 5.7, 10.45, 5.2)
arrow(ax, 9.7, 4.0, 10.45, 4.5)
fbox(ax, 14.6, 4.85, 1.9, 1.5, "policy π", fc="#eef4f5", ec=INK, fs=11.5, weight="bold")
arrow(ax, 13.25, 4.85, 13.6, 4.85)

# minimal output (RL/physics detail deliberately omitted)
arrow(ax, 14.6, 4.08, 14.6, 3.25)
ax.text(14.6, 2.9, "16-DoF joint targets", ha="center", fontsize=8.3, color=INK)
ax.text(14.6, 2.46, "reward: angular velocity about Z", ha="center", fontsize=7.6, color="#8a9298", style="italic")

ax.set_title("Method overview", fontsize=12.8, weight="bold", color=INK, loc="left", x=0.01, y=0.99)
fig.savefig(f"{OUT}/pipeline.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}/pipeline.png")
plt.close(fig)
