#!/usr/bin/env python3
"""Contrast the postural synergies of the proprio baseline B (weaker generalization)
against the learned-local policy C (generalizes), so the synergy analysis becomes a
test of the hypothesis rather than a description of the winner.
  A. Cross-shape principal angle of the DOMINANT synergy, per shape pair, B vs C
     (lower = the rotate primitive is more shared across shapes).
  B. Eigengrasp spectrum (cumulative variance), mean across categories, B vs C.
-> docs/figures/synergy_B_vs_C.png
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1]
GREY, TEAL = "#9aa7ad", "#0c8b98"
CATS = ["cuboid", "cylinder", "sphere"]

def load(prefix):
    S = {}
    for c in CATS:
        d = np.load(f"{OUT}/{prefix}_{c}.npz"); dof, yaw = d["dof"], d["yaw"]
        T, E, D = dof.shape; X = dof.reshape(T*E, D); Y = yaw.reshape(T*E)
        S[c] = X[np.abs(Y) > 0.02]
    return S
def pca(X):
    Xc = X - X.mean(0); _, s, Vt = np.linalg.svd(Xc, full_matrices=False); v = s**2; v /= v.sum(); return Vt, v
def dom_angle(A, B):
    s = np.linalg.svd(A[:3] @ B[:3].T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, 0, 1)))[0]

reps = {"proprio B": "B_synergy", "learned-local": "synergy"}
pairs = [("cuboid", "cylinder"), ("cuboid", "sphere"), ("cylinder", "sphere")]
ang, spec = {}, {}
for name, pre in reps.items():
    S = load(pre); pcs = {c: pca(S[c])[0] for c in CATS}; var = {c: pca(S[c])[1] for c in CATS}
    ang[name] = [dom_angle(pcs[a], pcs[b]) for a, b in pairs]
    spec[name] = np.mean([np.cumsum(var[c])[:8] for c in CATS], axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
x = np.arange(len(pairs)); w = 0.38
lab = ["cuboid\ncylinder", "cuboid\nsphere", "cylinder\nsphere"]
ax[0].bar(x - w/2, ang["proprio B"], w, label="proprio B", color=GREY)
ax[0].bar(x + w/2, ang["learned-local"], w, label="learned-local", color=TEAL)
for xi in x:
    ax[0].text(xi - w/2, ang["proprio B"][xi] + 0.4, f"{ang['proprio B'][xi]:.0f}", ha="center", fontsize=8, color="#555")
    ax[0].text(xi + w/2, ang["learned-local"][xi] + 0.4, f"{ang['learned-local'][xi]:.0f}", ha="center", fontsize=8, color=TEAL)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=9)
ax[0].set_ylabel("dominant-synergy angle between shapes (deg)")
ax[0].set_title("A. Dominant-synergy angle between shapes")
ax[0].legend(frameon=False, fontsize=9); ax[0].grid(axis="y", alpha=.2)

for name, c in [("proprio B", GREY), ("learned-local", TEAL)]:
    ax[1].plot(range(1, 9), spec[name], "-o", color=c, lw=2, ms=4, label=name)
ax[1].axhline(0.9, ls=":", c="#aaa", lw=1)
ax[1].set_xlim(0.7, 8); ax[1].set_ylim(0.2, 1.0)
ax[1].set_xlabel("# postural synergies"); ax[1].set_ylabel("cumulative variance (mean over shapes)")
ax[1].set_title("B. Cumulative variance explained"); ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.2)
fig.tight_layout()
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/synergy_B_vs_C.png", dpi=140, bbox_inches="tight", facecolor="white")
print("wrote docs/figures/synergy_B_vs_C.png")
print("B dominant angles:", np.round(ang["proprio B"],1), "mean", round(np.mean(ang["proprio B"]),1))
print("C dominant angles:", np.round(ang["learned-local"],1), "mean", round(np.mean(ang["learned-local"]),1))
