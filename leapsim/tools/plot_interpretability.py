#!/usr/bin/env python3
"""Render the Stage-4 interpretability results as two separate figures (objective;
interpretation belongs in the report captions).
  interp_ablation.png   counterfactual + ablation, cross-seed (correct/wrong-shape/noise/zero)
  interp_traversal.png  continuous latent traversal (angvel vs told box half-extent)
The all-directions generalization result is a table in the report, not a figure.
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TEAL = "#0c8b98"
os.makedirs("docs/figures", exist_ok=True)

# ---- Figure: counterfactual + ablation, cross-seed ----
conds = ["correct", "cf_sphere\n(wrong shape)", "ablate_noise", "ablate_zero\n(killed)"]
vals = {"s42": [0.125, 0.087, 0.053, 0.004],
        "s43": [0.134, 0.070, 0.048, 0.018],
        "s44": [0.126, 0.067, 0.043, 0.010]}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(len(conds)); w = 0.26
for i, (s, c) in enumerate(zip(vals, [TEAL, "#3aa8b3", "#7cc6cd"])):
    ax.bar(x + (i-1)*w, vals[s], w, label=s, color=c)
ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=9)
ax.set_ylabel("zero-shot cuboid angvel (rad/s)")
ax.set_title("Counterfactual and ablation, three seeds")
ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig("docs/figures/interp_ablation.png", dpi=140, bbox_inches="tight", facecolor="white")
print("wrote docs/figures/interp_ablation.png")

# ---- Figure: continuous latent traversal ----
hx = [0.018, 0.026, 0.034, 0.042, 0.050, 0.058]
av = [0.051, 0.080, 0.108, 0.117, 0.108, 0.085]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(hx, av, "-o", color=TEAL, lw=2.2, ms=6)
ax.axvline(hx[int(np.argmax(av))], ls=":", c="#999", lw=1)
ax.set_xlabel("told box half-extent hx (physical object fixed)")
ax.set_ylabel("zero-shot cuboid angvel (rad/s)")
ax.set_title("Interventional latent traversal")
ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig("docs/figures/interp_traversal.png", dpi=140, bbox_inches="tight", facecolor="white")
print("wrote docs/figures/interp_traversal.png")
