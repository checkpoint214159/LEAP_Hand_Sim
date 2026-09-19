#!/usr/bin/env python3
"""Pre-flight check for a hammer grasp cache: BEFORE spending hours training,
load the cache and verify the hammer is actually held in a POWER GRIP
(handle across the palm = world Y) and not a handshake (handle along the
fingers = world X). Reset restores the object's orientation FROM THE CACHE
(leap_hand_rot.py:749), so this is exactly what the policy will train on.

Usage: python tools/check_grasp_orientation.py cache/<name>_grasp_50k_s10.npy
World axes (from URDF FK): X = finger-POINTING, Y = knuckle-ROW (power grip), Z = up.
"""
import sys, numpy as np

def qrot(q, v):  # q xyzw, rotate vector v
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v
    tx = 2*(y*vz - z*vy); ty = 2*(z*vx - x*vz); tz = 2*(x*vy - y*vx)
    return np.stack([vx + w*tx + (y*tz - z*ty),
                     vy + w*ty + (z*tx - x*tz),
                     vz + w*tz + (x*ty - y*tx)], -1)

path = sys.argv[1] if len(sys.argv) > 1 else "cache/leap_hand_in_hammer_grasp_50k_s10.npy"
c = np.load(path)
print(f"cache        : {path}")
print(f"shape        : {c.shape}  ({'v4 40-col (PD targets ✓)' if c.shape[1] >= 40 else 'LEGACY 23-col (no PD targets ✗)'})")
if c.shape[1] >= 40:  pos, quat = c[:, 32:35], c[:, 35:39]      # v4: 16 dof|16 tgt|7 pose|1 id
else:                 pos, quat = c[:, 16:19], c[:, 19:23]       # legacy: 16 dof|7 pose

n = np.linalg.norm(quat, axis=1)
h = qrot(quat, (1, 0, 0)); h /= np.linalg.norm(h, axis=1, keepdims=True)   # handle = object +X
ax, ay, az = np.abs(h[:, 0]).mean(), np.abs(h[:, 1]).mean(), np.abs(h[:, 2]).mean()
fracY = (np.abs(h[:, 1]) > np.abs(h[:, 0])).mean()                          # closer to Y than X
mh = h.mean(0); mh /= np.linalg.norm(mh)

print(f"grips        : {len(c)}   obj z mean: {pos[:,2].mean():.3f} (want ~0.60; sag→loose)   quat|n|: {n.mean():.3f}")
print(f"handle |X|,|Y|,|Z| (world) : {ax:.3f}, {ay:.3f}, {az:.3f}")
print(f"  X=finger-pointing   Y=knuckle-row(POWER GRIP)   Z=vertical")
print(f"fraction handle-along-Y (power grip) : {fracY:.2f}")
ang = lambda a: round(np.degrees(np.arccos(min(abs(a), 1.0))), 1)
print(f"mean handle vs  X:{ang(mh[0])}°   Y:{ang(mh[1])}°   Z:{ang(mh[2])}°")
verdict = "POWER GRIP ✓ (handle across palm)" if ay > 0.6 and fracY > 0.6 else \
          ("HANDSHAKE ✗ (handle along fingers)" if ax > 0.6 else "MIXED / unclear ✗")
print(f"\nVERDICT      : {verdict}")
sys.exit(0 if verdict.endswith("✓ (handle across palm)") else 1)
