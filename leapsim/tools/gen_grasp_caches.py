#!/usr/bin/env python3
"""gen_grasp_caches.py — batch, resumable grasp-cache generator for procedurally
generated object families (see tools/gen_primitive_objects.py).

For every object family (`<assets>/<category>/<subset>/`) and every scale, it runs
the LeapHandGrasp settling task to produce the shared grasp cache LeapHandRot
loads at train time:

    cache/leap_hand_in_<category>_<subset>_grasp_50k_s<scale>.npy

Grasp quality follows the hand-tuned hammer recipe (scripts/gen_hammer_grasp.sh):
a strict contact criterion — num_contact_fingers=3, min_contact_force=1.0 N, a
size-aware finger_dist_threshold — and a tight DoF search radius, so the
hill-climb must discover real multi-finger grips, not palm balances. The object
init pose is computed per family × scale from the URDF geometry (global z such
that the object's lowest point spawns just above the palm support plane), which
prevents spawn interpenetration for tall objects; cylinders are laid across the
palm. All init changes are passed via task.env.override_object_init_{x,y,z,rot}.

Rerun recording is ON by default (sparse windows) so grasp quality can be
inspected visually: runs/graspgen_<family>_s<scale>_<ts>/rerun/*.rrd.
Verify the effective criterion in cache/<name>_stats.yaml → contact_criterion
(a mistyped Hydra key fails silently; the stats echo is the ground truth).

Run from src/LEAP_Hand_Sim/leapsim (nohup for background):
    uv run python tools/gen_grasp_caches.py --dry-run            # show computed jobs
    uv run python tools/gen_grasp_caches.py --only cuboid_train --scales 1.0   # pilot
    uv run python tools/gen_grasp_caches.py                      # all families/scales
    nohup uv run python tools/gen_grasp_caches.py > tools/logs/runner.out 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

LEAPSIM_DIR = Path(__file__).resolve().parents[1]          # .../src/LEAP_Hand_Sim/leapsim
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"  # .../src/LEAP_Hand_Sim/assets
CACHE_DIR = LEAPSIM_DIR / "cache"
LOG_DIR = Path(__file__).resolve().parent / "logs"
STATUS_PATH = CACHE_DIR / "_grasp_gen_status.json"

CATEGORIES = ["cuboid", "cylinder", "sphere", "cone", "capsule"]
# Must match LeapHandRot randomizeScaleList so the trained policy finds the caches.
DEFAULT_SCALES = [0.95, 0.9, 1.0, 1.05, 1.1]

# Object spawn placement (GLOBAL coords — the palm-up hand's support plane, not
# height above the palm). Calibrated from the known-good cube: init z 0.57 for
# the 0.075 m cube puts its lowest point at 0.5325. Per family we spawn so the
# object's lowest point sits CLEARANCE above that plane — within finger reach at
# spawn (contact is possible immediately) but never inside the palm.
SUPPORT_Z = 0.5325
CLEARANCE = 0.005
INIT_X = -0.03
INIT_Y = 0.04

# Per-category init orientation (URDF rpy, radians) and how the object's
# geometry maps to spawn height / criterion distance under that orientation.
#   half_height : object half-extent along gravity at spawn → sets init z
#   center_dist : max distance from object center to any surface point → sets
#                 the finger_dist_threshold budget (fingertip link origins sit
#                 a further ~0.035 m off the surface; see --dist-slack)
# Cylinders are laid on their side (axis → y, across the palm) for a power
# grip; if Rerun shows the lay direction fighting the hand, try [0,1.5708,0].
CATEGORY_INIT_ROT = {
    "cuboid":   [0.0, 0.0, 0.0],
    "cylinder": [1.5708, 0.0, 0.0],
    "sphere":   [0.0, 0.0, 0.0],
    # Laid on their side like cylinder (axis horizontal) so the hand engages
    # the full length for a power grip and the shape's distinctive rolling
    # behavior (cone: arcs; capsule: no flat end to catch on) is what the
    # policy actually has to handle, not an upright-standing pose.
    "cone":     [1.5708, 0.0, 0.0],
    "capsule":  [1.5708, 0.0, 0.0],
}


def scale_tag(s: float) -> str:
    return str(s).replace(".", "")


def cache_path(family_name: str, s: float) -> Path:
    return CACHE_DIR / f"{family_name}_grasp_50k_s{scale_tag(s)}.npy"


# ──────────────────────────────────────────────────────────────────────────────
# Geometry: parse the (auto-generated, single-link) primitive URDFs.
# ──────────────────────────────────────────────────────────────────────────────

def _parse_urdf_geom(urdf: Path):
    """Return ('box', (sx,sy,sz)) | ('cyl', (r,l)) | ('sph', (r,)) |
    ('cone', (r,h)) | ('cap', (r,L))."""
    root = ET.parse(str(urdf)).getroot()
    geom = root.find(".//collision/geometry")
    if geom is None:
        geom = root.find(".//visual/geometry")
    box = geom.find("box")
    if box is not None:
        return "box", tuple(float(v) for v in box.get("size").split())
    cyl = geom.find("cylinder")
    if cyl is not None:
        return "cyl", (float(cyl.get("radius")), float(cyl.get("length")))
    sph = geom.find("sphere")
    if sph is not None:
        return "sph", (float(sph.get("radius")),)
    mesh = geom.find("mesh")
    if mesh is not None:
        # cone/capsule aren't native URDF primitives (URDF only has
        # box/cylinder/sphere/mesh) — tools/gen_primitive_objects.py writes
        # the exact generation params into a leading XML comment
        # ("kind=cone dims=(0.045, 0.13)"), which is the ground truth here
        # (recovering them from the mesh's own bbox would need a real mesh
        # load + risk small tessellation error — see _make_mesh's docstring).
        text = urdf.read_text()
        m = re.search(r"kind=(\w+)\s+dims=\(([^)]*)\)", text)
        if m is None:
            raise ValueError(f"mesh geometry in {urdf} but no 'kind=... dims=(...)' comment found")
        kind = m.group(1)
        dims = tuple(float(x) for x in m.group(2).split(","))
        return kind, dims
    raise ValueError(f"no primitive geometry in {urdf}")


def _instance_metrics(category: str, kind: str, dims):
    """(half_height, center_dist) at unit scale, under the category init rot."""
    if kind == "box":                       # spawns axis-aligned (identity rot)
        sx, sy, sz = dims
        return sz / 2.0, 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)
    if kind == "cyl":                       # lying on its side (axis horizontal)
        r, l = dims
        return r, math.sqrt(r * r + (l / 2.0) ** 2)
    if kind == "sph":
        r = dims[0]
        return r, r
    if kind == "cone":                      # lying on its side; dims=(base radius R, height h)
        # Vertical clearance after laying on its side: bounded by the base
        # radius R (the cone's widest point) regardless of where along its
        # length it settles — same worst-case-bound reasoning as "cyl" above,
        # just using R directly since the cone's own bbox half-width is R
        # (not variable r(z), see tools/gen_primitive_objects.py). Farthest
        # point from the bbox-center origin is the base rim.
        r, h = dims
        return r, math.sqrt(r * r + (h / 2.0) ** 2)
    if kind == "cap":                       # lying on its side; dims=(radius r, cyl-segment length L)
        # A capsule's radius is constant along its whole silhouette (unlike a
        # cone), so half_height = r exactly, not just a bound. Farthest point
        # from center is the pole tip, at L/2 + r (see the local-z geometry
        # derivation in leap_hand_rot.py).
        r, ell = dims
        return r, ell / 2.0 + r
    raise ValueError(kind)


def family_geometry(category: str, subset_dir: Path):
    """Max (half_height, center_dist) over the family's instances, unit scale."""
    hh, cd = 0.0, 0.0
    for urdf in sorted(subset_dir.glob("*.urdf")):
        kind, dims = _parse_urdf_geom(urdf)
        h, c = _instance_metrics(category, kind, dims)
        hh, cd = max(hh, h), max(cd, c)
    return hh, cd


def discover_families():
    """Yield (category, subset, object_type, family_name, half_height, center_dist)."""
    fams = []
    for category in CATEGORIES:
        cat_dir = ASSETS_DIR / category
        if not cat_dir.is_dir():
            continue
        for subset_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            if not list(subset_dir.glob("*.urdf")):
                continue
            subset = subset_dir.name
            hh, cd = family_geometry(category, subset_dir)
            fams.append((category, subset, f"{category}_{subset}",
                         f"leap_hand_in_{category}_{subset}", hh, cd))
    return fams


# ──────────────────────────────────────────────────────────────────────────────
# Job construction / execution
# ──────────────────────────────────────────────────────────────────────────────

def load_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_status(status: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True))


def job_params(category, half_height, center_dist, scale, args):
    """Computed per-job values: (init_z, dist_threshold, rot)."""
    init_z = SUPPORT_Z + half_height * scale + CLEARANCE
    dist = center_dist * scale + args.dist_slack
    return init_z, dist, CATEGORY_INIT_ROT.get(category, [0.0, 0.0, 0.0])


def build_cmd(object_type, family_name, category, scale, half_height, center_dist, args):
    init_z, dist, rot = job_params(category, half_height, center_dist, scale, args)
    cmd = [
        "uv", "run", "python", "train.py",
        "task=LeapHandGrasp", "test=true", "pipeline=cpu",
        "wandb_activate=false", "log_to_sheet=false",
        f"experiment=graspgen_{object_type}_s{scale_tag(scale)}",
        f"task.env.object.type={object_type}",
        f"task.env.grasp_cache_name={family_name}",
        f"task.env.baseObjScale={scale}",
        f"task.env.numEnvs={args.num_envs}",
        f"task.env.episodeLength={args.episode_length}",
        # ── Strict contact criterion (hammer-recipe lineage, correct keys) ──
        # Effective values are echoed in <cache>_stats.yaml → contact_criterion.
        f"task.env.num_contact_fingers={args.contact_fingers}",
        f"+task.env.min_contact_force={args.min_force}",
        f"+task.env.finger_dist_threshold={dist:.4f}",
        f"+task.env.contact_grace_steps={args.grace_steps}",
        f"+task.env.max_palm_contact_force={args.max_palm_force}",
        f"task.env.grasp_dof_search_radius={args.search_radius}",
        # ── Spawn placement (global coords; see SUPPORT_Z note above) ──
        f"task.env.override_object_init_x={INIT_X}",
        f"task.env.override_object_init_y={INIT_Y}",
        f"task.env.override_object_init_z={init_z:.4f}",
        f"task.env.override_object_init_rot=[{rot[0]},{rot[1]},{rot[2]}]",
        # High enough that the per-job --timeout is the binding limit: 1-step
        # episodes count as "games", and the previous 5M cap ended jobs early.
        "train.params.config.player.games_num=500000000",
    ]
    if args.no_rerun:
        cmd.append("task.env.rerun.enabled=false")
    else:
        cmd += [
            "task.env.rerun.enabled=true",
            f"task.env.rerun.window_length_steps={args.episode_length}",
            f"task.env.rerun.record_every_n_steps={args.rerun_every}",
        ]
    return cmd


def run_one(object_type, family_name, category, scale, half_height, center_dist, args, log_fh) -> bool:
    cmd = build_cmd(object_type, family_name, category, scale, half_height, center_dist, args)
    log_fh.write(f"\n{'='*80}\n[{datetime.now().isoformat()}] {family_name} s{scale_tag(scale)}\n{' '.join(cmd)}\n")
    log_fh.flush()
    # start_new_session=True + os.killpg on timeout: `uv run python train.py` spawns
    # train.py as a GRANDCHILD of this process (uv forks/execs it as its own
    # child) — plain subprocess.run(timeout=)/Popen.kill() only ever signals the
    # DIRECT child (uv itself), silently orphaning the grandchild, which keeps
    # running forever. Found the hard way: with no process-group kill, every
    # "timed out" job left its real worker alive, and they piled up fighting each
    # other for CPU (pipeline=cpu) — 5 orphans running concurrently collapsed a
    # later job's yield to 0.01% (vs. the original families' minutes-scale
    # convergence). killpg reaches the whole tree uv spawned.
    proc = subprocess.Popen(cmd, cwd=str(LEAPSIM_DIR), stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        log_fh.write(f"[timeout after {args.timeout}s — SIGINT to flush partial cache]\n")
        log_fh.flush()
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            log_fh.write("[SIGINT didn't exit within 90s — SIGKILL]\n")
            log_fh.flush()
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        except ProcessLookupError:
            pass
    # Success == the cache file now exists (the grasp task saves on fill/exit).
    return cache_path(family_name, scale).exists()


# ──────────────────────────────────────────────────────────────────────────────
# Per-object generation (dense families — docs/HANDOFF.md §4/§5).
#
# A family cache normally comes from ONE run that samples all N objects across
# 1024 envs (~1024/N envs per object). For dense families (N=40) that is ~25
# envs/object — far too few parallel hill-climbs to converge each object in
# reasonable time (the search's first success needs ~50k episodes; with 25 envs a
# hard object may never get there). Per-object mode instead runs each object with
# ALL 1024 envs focused on it (via object_id_whitelist=[i]) so it converges in
# ~2-3 min, then fills to --rows-per-object and exits (grasp_cache_len). The N
# per-object caches are concatenated into the family cache; each already carries
# the correct object id (env_object_type_id==i) in its last column.
#
# Spawn height: every WORKING family (cuboid/cylinder/sphere) ends up with the
# object CENTER near ~0.578-0.588 (the finger-closure height) because a tall
# member lifts the family-max init_z. Dense families are sorted (smallest edge
# vertical) so their per-object init_z from the bottom-clearance formula is low
# (~0.564) and objects spawn below the fingers. --min-init-z clamps the spawn so
# small objects still present their center at finger height.
# ──────────────────────────────────────────────────────────────────────────────

def po_cache_path(family_name: str, i: int, s: float) -> Path:
    return CACHE_DIR / f"{family_name}__po{i}_grasp_50k_s{scale_tag(s)}.npy"


def build_cmd_po(object_type, family_name, i, init_z, dist, rot, scale, args):
    """Grasp-gen command for ONE object i of a family (all envs whitelisted to it)."""
    cmd = [
        "uv", "run", "python", "train.py",
        "task=LeapHandGrasp", "test=true", "pipeline=cpu",
        "wandb_activate=false", "log_to_sheet=false",
        f"experiment=graspgen_{object_type}_po{i}_s{scale_tag(scale)}",
        f"task.env.object.type={object_type}",
        f"task.env.grasp_cache_name={family_name}__po{i}",
        f"task.env.baseObjScale={scale}",
        f"task.env.numEnvs={args.num_envs}",
        f"task.env.episodeLength={args.episode_length}",
        f"+task.env.object_id_whitelist=[{i}]",       # all envs hold object i
        f"task.env.grasp_cache_len={args.rows_per_object}",   # exit once filled
        f"task.env.num_contact_fingers={args.contact_fingers}",
        f"+task.env.min_contact_force={args.min_force}",
        f"+task.env.finger_dist_threshold={dist:.4f}",
        f"+task.env.contact_grace_steps={args.grace_steps}",
        f"+task.env.max_palm_contact_force={args.max_palm_force}",
        f"task.env.grasp_dof_search_radius={args.search_radius}",
        f"task.env.override_object_init_x={INIT_X}",
        f"task.env.override_object_init_y={INIT_Y}",
        f"task.env.override_object_init_z={init_z:.4f}",
        f"task.env.override_object_init_rot=[{rot[0]},{rot[1]},{rot[2]}]",
        "train.params.config.player.games_num=500000000",
        "task.env.rerun.enabled=false" if args.no_rerun else "task.env.rerun.enabled=true",
    ]
    return cmd


def run_per_object_family(cat, sub, ot, fam, scale, args, log_fh):
    """Grasp-gen each object of one family separately, then concatenate → family cache."""
    subset_dir = ASSETS_DIR / cat / sub
    objs = sorted(subset_dir.glob("*.urdf"))
    rot = CATEGORY_INIT_ROT.get(cat, [0.0, 0.0, 0.0])
    print(f"\n[per-object] {fam} s{scale_tag(scale)}: {len(objs)} objects "
          f"(rows/object={args.rows_per_object}, min_init_z={args.min_init_z})")
    for i, urdf in enumerate(objs):
        outp = po_cache_path(fam, i, scale)
        if outp.exists():
            print(f"  [{i+1}/{len(objs)}] obj_{i} — cache exists, skip")
            continue
        kind, dims = _parse_urdf_geom(urdf)
        hh, cd = _instance_metrics(cat, kind, dims)
        init_z = max(SUPPORT_Z + hh * scale + CLEARANCE, args.min_init_z)
        dist = cd * scale + args.dist_slack
        cmd = build_cmd_po(ot, fam, i, init_z, dist, rot, scale, args)
        log_fh.write(f"\n{'='*80}\n[{datetime.now().isoformat()}] {fam} obj_{i} "
                     f"z={init_z:.4f} d={dist:.4f}\n{' '.join(cmd)}\n")
        log_fh.flush()
        print(f"  [{i+1}/{len(objs)}] obj_{i} z={init_z:.4f} d={dist:.4f} ...", end=" ", flush=True)
        t0 = time.time()
        # On timeout send SIGINT (not SIGKILL): the grasp task's atexit hook flushes
        # whatever grasps it has collected, so a slow object keeps its partial cache.
        # subprocess.run(timeout=) SIGKILLs, which loses everything.
        # start_new_session=True + killpg (not send_signal/kill, which only ever
        # reach the DIRECT child): `uv run python train.py` execs train.py as a
        # GRANDCHILD, so signaling just the Popen PID orphans the real worker
        # instead of stopping it — see run_one's comment for how badly that bites
        # (orphans pile up and CPU-starve every later job).
        proc = subprocess.Popen(cmd, cwd=str(LEAPSIM_DIR), stdout=log_fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            log_fh.write(f"[timeout after {args.timeout}s — SIGINT to flush partial cache]\n")
            log_fh.flush()
            try:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=90)     # let atexit save
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
        n = int(np.load(outp).shape[0]) if outp.exists() else 0
        print(f"{'OK' if n else 'FAILED'} ({n} rows, {round(time.time()-t0)}s)")

    # concatenate per-object caches (ids already correct) → family cache
    parts, counts = [], {}
    for i in range(len(objs)):
        p = po_cache_path(fam, i, scale)
        if not p.exists():
            print(f"  [concat] WARNING obj_{i} missing — family cache will lack it")
            continue
        arr = np.load(p)
        parts.append(arr)
        counts[i] = arr.shape[0]
    if not parts:
        print(f"  [concat] no per-object caches for {fam} — nothing written")
        return False
    combined = np.concatenate(parts, axis=0)
    famp = cache_path(fam, scale)
    np.save(famp, combined)
    print(f"  [concat] -> {famp.name}: {combined.shape} | rows/object {counts}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="Restrict to one family, e.g. cuboid_train")
    ap.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--episode-length", type=int, default=150)
    ap.add_argument("--contact-fingers", type=int, default=3,
                    help="Min fingertips with valid-window force, every non-grace step")
    ap.add_argument("--min-force", type=float, default=1.0,
                    help="Min per-fingertip contact force [N] to count as touching")
    ap.add_argument("--dist-slack", type=float, default=0.035,
                    help="finger_dist_threshold = family max center-to-surface × scale + slack "
                         "[m] (budget for fingertip link origin sitting off the surface)")
    ap.add_argument("--grace-steps", type=int, default=15,
                    help="Waive the contact condition for the first N steps (settle grace). "
                         "0 = hammer-recipe strictness; try 10-20 only if a family's yield "
                         "is pathologically low.")
    ap.add_argument("--max-palm-force", type=float, default=0.1,
                    help="Fail the episode if the object presses on the palm with more than "
                         "this force [N] after the grace period — certified grips must be "
                         "fingertip-borne, not palm rests. Negative disables (in-palm allowed).")
    ap.add_argument("--search-radius", type=float, default=0.15,
                    help="grasp_dof_search_radius around the per-env best pose")
    ap.add_argument("--rerun-every", type=int, default=1500,
                    help="Open a Rerun window every N steps (sparse visual sampling)")
    ap.add_argument("--no-rerun", action="store_true", help="Disable Rerun recording")
    ap.add_argument("--timeout", type=int, default=3600, help="Per cache subprocess timeout [s]")
    ap.add_argument("--dry-run", action="store_true")
    # ── Per-object mode (dense families) ──
    ap.add_argument("--per-object", action="store_true",
                    help="Generate each object of a family separately (all envs focused on it "
                         "via object_id_whitelist), then concatenate. Use for dense families "
                         "(N>>4) where family-sampling gives too few envs/object to converge.")
    ap.add_argument("--rows-per-object", type=int, default=256,
                    help="Per-object grasp_cache_len — exit once this many grasps are collected.")
    ap.add_argument("--min-init-z", type=float, default=0.0,
                    help="Clamp override_object_init_z up to at least this [m] so small sorted "
                         "objects present their center at finger height (~0.578). 0 = no clamp.")
    args = ap.parse_args()

    families = discover_families()
    if args.only:
        families = [f for f in families if f[2] == args.only]
    if not families:
        print("No object families found. Run tools/gen_primitive_objects.py first "
              f"(looked under {ASSETS_DIR}/<category>/<subset>/).")
        sys.exit(1)

    if args.per_object:
        jobs = [(cat, sub, ot, fam, s)
                for (cat, sub, ot, fam, hh, cd) in families for s in args.scales]
        todo = [j for j in jobs if not cache_path(j[3], j[4]).exists()]
        print(f"[per-object] families {[f[3] for f in families]} | scales {args.scales} | "
              f"family caches to build: {len(todo)}/{len(jobs)}")
        if args.dry_run:
            for (cat, sub, ot, fam, s) in jobs:
                objs = sorted((ASSETS_DIR / cat / sub).glob("*.urdf"))
                print(f"  {fam} s{scale_tag(s)}: {len(objs)} objects, "
                      f"rows/object={args.rows_per_object}, min_init_z={args.min_init_z}")
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"grasp_gen_po_{datetime.now():%Y%m%d_%H%M%S}.log"
        print(f"Logging subprocess output to {log_path}\n")
        status = load_status()
        with open(log_path, "w") as log_fh:
            for (cat, sub, ot, fam, s) in todo:
                ok = run_per_object_family(cat, sub, ot, fam, s, args, log_fh)
                status[f"{fam}_s{scale_tag(s)}_po"] = {
                    "ok": ok, "at": datetime.now().isoformat()}
                save_status(status)
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status = load_status()
    jobs = [(cat, sub, ot, fam, hh, cd, s)
            for (cat, sub, ot, fam, hh, cd) in families for s in args.scales]
    todo = [j for j in jobs if not cache_path(j[3], j[6]).exists()]

    print(f"Families: {len(families)} | scales: {args.scales} | total jobs: {len(jobs)} | "
          f"already done: {len(jobs) - len(todo)} | to run: {len(todo)}")
    palm = "disabled" if args.max_palm_force < 0 else f"<={args.max_palm_force}N"
    print(f"Criterion: fingers>={args.contact_fingers} @ >={args.min_force}N, "
          f"grace={args.grace_steps}, palm {palm}, search_radius={args.search_radius}, "
          f"rerun={'off' if args.no_rerun else f'every {args.rerun_every} steps'}")
    for (cat, sub, ot, fam, hh, cd) in families:
        have = sum(cache_path(fam, s).exists() for s in args.scales)
        rot = CATEGORY_INIT_ROT.get(cat, [0, 0, 0])
        zs = ", ".join(f"s{scale_tag(s)}:z={job_params(cat, hh, cd, s, args)[0]:.3f}"
                       f"/d={job_params(cat, hh, cd, s, args)[1]:.3f}" for s in args.scales)
        print(f"  {fam:32s} caches {have}/{len(args.scales)} rot={rot}")
        print(f"    {zs}")
    if args.dry_run or not todo:
        return

    log_path = LOG_DIR / f"grasp_gen_{datetime.now():%Y%m%d_%H%M%S}.log"
    print(f"\nLogging subprocess output to {log_path}\n")
    with open(log_path, "w") as log_fh:
        for i, (cat, sub, ot, fam, hh, cd, s) in enumerate(todo, 1):
            key = f"{fam}_s{scale_tag(s)}"
            print(f"[{i}/{len(todo)}] {fam} scale={s} ...", end=" ", flush=True)
            t0 = time.time()
            ok = run_one(ot, fam, cat, s, hh, cd, args, log_fh)
            status[key] = {"ok": ok, "seconds": round(time.time() - t0, 1),
                           "at": datetime.now().isoformat(), "cache": str(cache_path(fam, s).name)}
            save_status(status)
            print("OK" if ok else "FAILED (no cache written — check log)",
                  f"({status[key]['seconds']}s)")

    done = sum(1 for v in status.values() if v.get("ok"))
    print(f"\nDone. {done} caches present. Status: {STATUS_PATH}")


if __name__ == "__main__":
    main()
