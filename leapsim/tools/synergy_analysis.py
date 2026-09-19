#!/usr/bin/env python3
"""Stage-4 SYNERGY PROJECTION - offline analysis of the learned-local policy's
16-DoF rotation gait, per object category. Reads the dumps from run_synergy_dump.sh.

Produces:
  1. Eigengrasp spectrum  - cumulative variance vs #postural synergies, per category
     (how low-dimensional the control is).
  2. Limit cycles          - env-0 trajectory projected onto the shared PC1-PC2
     synergy plane (the cyclic rotation gait), per category.
  3. Synergy similarity    - principal angles between each category pair's top-3
     synergy subspaces (small = shared gait; large = category-specific).
Figure -> docs/figures/synergy_analysis.png ; numbers -> stdout.
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT", ".")
FIG = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/synergy_analysis.png"
CATS = os.environ.get("CATS", "cuboid,cylinder,sphere").split(",")
_DEFAULT_COL = {"cuboid": "#0c8b98", "cylinder": "#b9761b", "sphere": "#a94a3f",
                "cone": "#6a3d9a", "capsule": "#33a02c"}
COL = {c: _DEFAULT_COL.get(c, "#555555") for c in CATS}

samples, env0 = {}, {}
for c in CATS:
    p = os.path.join(OUT, f"synergy_{c}.npz")
    if not os.path.exists(p):
        print(f"MISSING {p} - run run_synergy_dump.sh first"); sys.exit(1)
    d = np.load(p); dof, yaw = d["dof"], d["yaw"]          # [T,E,16], [T,E]
    T, E, D = dof.shape
    X = dof.reshape(T * E, D); Y = yaw.reshape(T * E)
    samples[c] = X[np.abs(Y) > 0.02]                        # steady-state rotating samples
    env0[c] = dof[:, 0, :]                                  # env-0 time-ordered trajectory
    print(f"{c:9s} samples={samples[c].shape[0]:6d}  mean|yaw|={np.abs(yaw).mean():.3f}")

def pca(X):
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = S**2; v = v / v.sum()
    return Vt, v, X.mean(0)

pcs, var = {}, {}
for c in CATS:
    pcs[c], var[c], _ = pca(samples[c])

n = min(len(samples[c]) for c in CATS)
pooled = np.concatenate([samples[c][:n] for c in CATS])
Vt_all, var_all, mean_all = pca(pooled)

print("\n=== eigengrasp spectrum (cumulative variance explained) ===")
print(f"{'k':>2} " + " ".join(f"{c:>9}" for c in CATS) + f"{'pooled':>9}")
for k in range(1, 7):
    row = " ".join(f"{var[c][:k].sum():9.3f}" for c in CATS)
    print(f"{k:>2} {row} {var_all[:k].sum():9.3f}")

def principal_angles(A, B, k=3):
    s = np.linalg.svd(A[:k] @ B[:k].T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, 0, 1)))

print("\n=== principal angles between top-3 synergy subspaces (deg; small=shared gait) ===")
for i, a in enumerate(CATS):
    for b in CATS[i+1:]:
        ang = principal_angles(pcs[a], pcs[b], 3)
        print(f"  {a:8s} vs {b:8s}: {np.round(ang,1)}")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for c in CATS:
    cum = np.cumsum(var[c])
    ax[0].plot(range(1, len(cum)+1), cum, "-o", color=COL[c], label=c, lw=2, ms=4)
ax[0].axhline(0.9, ls=":", c="#999", lw=1)
ax[0].set_xlim(0.7, 8); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel("# postural synergies (eigengrasps)"); ax[0].set_ylabel("cumulative variance")
ax[0].set_title("Cumulative variance explained"); ax[0].legend(frameon=False); ax[0].grid(alpha=.2)

for c in CATS:
    proj = (env0[c] - mean_all) @ Vt_all[:2].T                # onto shared PC1-PC2
    ax[1].plot(proj[:, 0], proj[:, 1], "-", color=COL[c], lw=1.1, alpha=.85, label=c)
ax[1].set_xlabel("shared synergy 1"); ax[1].set_ylabel("shared synergy 2")
ax[1].set_title("Rotation limit cycle (env 0)"); ax[1].legend(frameon=False); ax[1].grid(alpha=.2)
ax[1].set_aspect("equal", "datalim")

fig.suptitle("Postural synergies of the learned-local rotation gait", fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG, dpi=140, bbox_inches="tight")
print(f"\nfigure -> {FIG}")
