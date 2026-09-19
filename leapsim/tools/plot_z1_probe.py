#!/usr/bin/env python3
"""Render the z1 mechanistic probe as a 2-panel figure (objective; interpretation
belongs in the report caption).
  A. Input-gradient saliency of the action on each input group.
  B. Eval-time drop/launch rate on the held-out cuboid, full z1 vs masked subgroups.
Source: tools/logs/probe_z1_cond_*.json (probe_z1_analysis.py).
-> docs/figures/z1_probe.png
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TEAL, AMBER, BRICK, GREY = "#0c8b98", "#b9761b", "#a94a3f", "#9aa7ad"

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

# A. saliency
labels = ["proprio\n(mean dim)", "inertia\nratio r2", "inertia\nratio r3", "norm.\nSA/V"]
grads  = [2.848, 2.330, 2.969, 5.831]
ax[0].bar(range(len(labels)), grads, color=[GREY, AMBER, AMBER, BRICK])
ax[0].axhline(2.848, ls=":", c="#888", lw=1)
ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels, fontsize=9)
ax[0].set_ylabel("mean |input-gradient| of the action")
ax[0].set_title("A. Input-gradient saliency")

# B. ablation (early-termination / drop-launch frac; lower = better)
conds = ["baseline\n(full z1)", "mask\nSA/V", "mask all\nshape", "mask z0\n(keep shape)", "mask\nratios", "shuffle z\n(wrong obj)"]
et    = [0.503, 0.284, 0.391, 0.344, 0.742, 0.688]
base  = 0.503
ax[1].bar(range(len(conds)), et, color=[GREY] + [(TEAL if v < base else BRICK) for v in et[1:]])
ax[1].axhline(base, ls="--", c="#888", lw=1.2)
ax[1].set_xticks(range(len(conds))); ax[1].set_xticklabels(conds, fontsize=8.5)
ax[1].set_ylabel("drop / launch rate on zero-shot cuboid\n(early-termination frac, lower is better)")
ax[1].set_title("B. Eval-time ablation")

fig.tight_layout()
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/z1_probe.png", dpi=140, bbox_inches="tight", facecolor="white")
print("wrote docs/figures/z1_probe.png")
