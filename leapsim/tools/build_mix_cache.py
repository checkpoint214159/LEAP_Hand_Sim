#!/usr/bin/env python3
"""Build a combined multi-family grasp cache for cross-category mixes.

A mix like `object.type=cuboid_train+cylinder_train+sphere_train` assigns GLOBAL
object ids 0..N-1 across families — in split('+') order, and sorted-glob order
within each family (see LeapHandRot._setup_object_info). But each family's own
grasp cache stores LOCAL ids 0..n-1. This tool concatenates the per-family caches
scale-by-scale and REMAPS the id column (the last column of a v4 40-col row) to
global ids, so LeapHandRot's object-aware restore (cache_rows_by_obj) finds rows
for every one of the 12 mix objects instead of falling back to any-row (which
would restore e.g. a cuboid grasp onto a sphere — the object-blind bug).

Offsets are derived from the actual per-family URDF counts, so this stays correct
if the taxonomy changes.

Usage (run from src/LEAP_Hand_Sim/leapsim):
  uv run python tools/build_mix_cache.py --subset train
  uv run python tools/build_mix_cache.py --subset heldout
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

FAMILIES = ['cuboid', 'cylinder', 'sphere']          # split('+') order for the mix
ASSETS = Path(__file__).resolve().parents[2] / 'assets'   # .../LEAP_Hand_Sim/assets
CACHE = Path('cache')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True, choices=['train', 'heldout'])
    ap.add_argument('--families', nargs='+', default=FAMILIES)
    ap.add_argument('--out-name', default=None)
    args = ap.parse_args()
    sub = args.subset
    fams = args.families
    out_name = args.out_name or f'leap_hand_in_mix_{sub}'

    # Global id offsets from real per-family object counts (sorted glob == the
    # order _setup_object_info assigns).
    counts = {f: len(sorted((ASSETS / f / sub).glob('*.urdf'))) for f in fams}
    offsets, running = {}, 0
    for f in fams:
        offsets[f] = running
        running += counts[f]
    print(f"[mix {sub}] families {fams}")
    print(f"[mix {sub}] per-family object counts {counts}")
    print(f"[mix {sub}] global id offsets {offsets}  (total {running} objects)")

    # Scale suffixes come from the first family's existing caches.
    pat = f'leap_hand_in_{fams[0]}_{sub}_grasp_50k_s*.npy'
    suffixes = sorted({os.path.basename(p).split('_s')[-1][:-4] for p in glob.glob(str(CACHE / pat))})
    if not suffixes:
        raise SystemExit(f"no caches matched {CACHE/pat} — generate family caches first")
    print(f"[mix {sub}] scale suffixes: {suffixes}\n")

    for suf in suffixes:
        parts = []
        for f in fams:
            fp = CACHE / f'leap_hand_in_{f}_{sub}_grasp_50k_s{suf}.npy'
            arr = np.load(fp).copy()
            if arr.shape[1] < 40:
                raise SystemExit(f"{fp.name} has {arr.shape[1]} cols (<40) — not a v4 object-aware cache")
            arr[:, -1] = arr[:, -1] + offsets[f]      # local id -> global id
            parts.append(arr)
            print(f"    {fp.name}: {arr.shape[0]} rows, ids +{offsets[f]}")
        combined = np.concatenate(parts, axis=0)
        outp = CACHE / f'{out_name}_grasp_50k_s{suf}.npy'
        np.save(outp, combined)
        ids, cnts = np.unique(combined[:, -1].astype(int), return_counts=True)
        print(f"  -> {outp.name}: shape {combined.shape}, id counts {dict(zip(ids.tolist(), cnts.tolist()))}\n")


if __name__ == '__main__':
    main()
