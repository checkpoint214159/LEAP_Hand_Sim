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
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

LEAPSIM_DIR = Path(__file__).resolve().parents[1]          # .../src/LEAP_Hand_Sim/leapsim
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"  # .../src/LEAP_Hand_Sim/assets
CACHE_DIR = LEAPSIM_DIR / "cache"
LOG_DIR = Path(__file__).resolve().parent / "logs"
STATUS_PATH = CACHE_DIR / "_grasp_gen_status.json"

CATEGORIES = ["cuboid", "cylinder", "sphere"]
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
}


def scale_tag(s: float) -> str:
    return str(s).replace(".", "")


def cache_path(family_name: str, s: float) -> Path:
    return CACHE_DIR / f"{family_name}_grasp_50k_s{scale_tag(s)}.npy"


# ──────────────────────────────────────────────────────────────────────────────
# Geometry: parse the (auto-generated, single-link) primitive URDFs.
# ──────────────────────────────────────────────────────────────────────────────

def _parse_urdf_geom(urdf: Path):
    """Return ('box', (sx,sy,sz)) | ('cyl', (r,l)) | ('sph', (r,))."""
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
    try:
        subprocess.run(cmd, cwd=str(LEAPSIM_DIR), stdout=log_fh, stderr=subprocess.STDOUT,
                       timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        log_fh.write(f"[timeout after {args.timeout}s]\n")
        log_fh.flush()
    # Success == the cache file now exists (the grasp task saves on fill/exit).
    return cache_path(family_name, scale).exists()


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
    args = ap.parse_args()

    families = discover_families()
    if args.only:
        families = [f for f in families if f[2] == args.only]
    if not families:
        print("No object families found. Run tools/gen_primitive_objects.py first "
              f"(looked under {ASSETS_DIR}/<category>/<subset>/).")
        sys.exit(1)

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
