#!/usr/bin/env python3
"""3D extension of synergy_analysis.py's limit-cycle panel: with 5 categories
overlaid, the 2D (PC1,PC2) projection gets crowded. This projects onto the
shared (PC1,PC2,PC3) space instead — one big 3D view plus the three pairwise
2D projections for orientations where the 3D view occludes something.

Reads the same tools/logs/synergy_dumps/synergy_<cat>.npz dumps as
synergy_analysis.py. Categories via CATS env var (comma-separated), default
all 5. Output path via FIG env var (should be an ABSOLUTE path or a path
resolved from the intended cwd — a relative "docs/figures/..." default here
previously landed in leapsim/docs/figures instead of the repo-root one).
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT", "tools/logs/synergy_dumps")
FIG = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "FIG", "/root/odyssey/dexterousmanipulation/docs/figures/synergy_limit_cycle_3d.png")
CATS = os.environ.get("CATS", "cuboid,cylinder,sphere,cone,capsule").split(",")
_DEFAULT_COL = {"cuboid": "#0c8b98", "cylinder": "#b9761b", "sphere": "#a94a3f",
                "cone": "#6a3d9a", "capsule": "#33a02c"}
COL = {c: _DEFAULT_COL.get(c, "#555555") for c in CATS}

samples, env0 = {}, {}
for c in CATS:
    p = os.path.join(OUT, f"synergy_{c}.npz")
    if not os.path.exists(p):
        print(f"MISSING {p}"); sys.exit(1)
    d = np.load(p); dof, yaw = d["dof"], d["yaw"]
    T, E, D = dof.shape
    X = dof.reshape(T * E, D); Y = yaw.reshape(T * E)
    samples[c] = X[np.abs(Y) > 0.02]
    env0[c] = dof[:, 0, :]
    print(f"{c:9s} samples={samples[c].shape[0]:6d}")

def pca(X):
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = S**2; v = v / v.sum()
    return Vt, v, X.mean(0)

n = min(len(samples[c]) for c in CATS)
pooled = np.concatenate([samples[c][:n] for c in CATS])
Vt_all, var_all, mean_all = pca(pooled)
print(f"\nPC1-3 variance explained: {var_all[0]:.3f}, {var_all[1]:.3f}, {var_all[2]:.3f}  "
      f"(cum={var_all[:3].sum():.3f})")

proj = {c: (env0[c] - mean_all) @ Vt_all[:3].T for c in CATS}   # [T,3] per category

fig = plt.figure(figsize=(14, 10))
ax3d = fig.add_subplot(2, 2, (1, 2), projection="3d")
for c in CATS:
    p = proj[c]
    ax3d.plot(p[:, 0], p[:, 1], p[:, 2], "-", color=COL[c], lw=1.3, alpha=.85, label=c)
ax3d.set_xlabel("shared synergy 1"); ax3d.set_ylabel("shared synergy 2"); ax3d.set_zlabel("shared synergy 3")
ax3d.set_title("Rotation limit cycle (env 0), shared PC1-PC2-PC3")
ax3d.legend(frameon=False, fontsize=9, loc="upper left")
ax3d.view_init(elev=22, azim=-55)

pairs = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3"), (1, 2, "PC2", "PC3")]
for k, (i, j, xl, yl) in enumerate(pairs):
    ax = fig.add_subplot(2, 3, 4 + k)
    for c in CATS:
        p = proj[c]
        ax.plot(p[:, i], p[:, j], "-", color=COL[c], lw=1.0, alpha=.8, label=c)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal", "datalim")
    ax.grid(alpha=.2)
    if k == 0:
        ax.legend(frameon=False, fontsize=7.5, loc="best")

fig.suptitle("Rotation limit cycles in shared synergy space — 3D view + pairwise 2D projections",
             fontsize=13)
fig.tight_layout()
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG, dpi=140, bbox_inches="tight", facecolor="white")
print(f"\nfigure -> {FIG}")
