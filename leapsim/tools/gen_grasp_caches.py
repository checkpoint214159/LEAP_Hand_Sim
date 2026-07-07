#!/usr/bin/env python3
"""gen_grasp_caches.py — batch, resumable grasp-cache generator for procedurally
generated object families (see tools/gen_primitive_objects.py).

For every object family (`<assets>/<category>/<subset>/`) and every scale, it runs
the LeapHandGrasp settling task to produce the shared grasp cache LeapHandRot
loads at train time:

    cache/leap_hand_in_<category>_<subset>_grasp_50k_s<scale>.npy

Designed to run unattended in the background: it is **resumable** (skips caches
that already exist), records progress to cache/_grasp_gen_status.json, and logs
each subprocess to tools/logs/. The LeapHandGrasp task exits itself once its cache
fills, so each family/scale is one bounded subprocess.

Run from src/LEAP_Hand_Sim/leapsim (nohup for background):
    uv run python tools/gen_grasp_caches.py                      # all families/scales
    uv run python tools/gen_grasp_caches.py --only cuboid_train  # one family
    uv run python tools/gen_grasp_caches.py --dry-run
    nohup uv run python tools/gen_grasp_caches.py > tools/logs/runner.out 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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

# Per-category Hydra overrides for the grasp settle. Empty => use the config
# defaults (canonical_pose is labelled "for cube/spherical objects" and seats
# cuboids/spheres well). Add entries here if a category needs a different seed
# pose or object placement (e.g. laying a long cylinder in the palm).
POSE_OVERRIDES = {
    "cuboid": [],
    "sphere": [],
    "cylinder": [],
}


def scale_tag(s: float) -> str:
    return str(s).replace(".", "")


def cache_path(family_name: str, s: float) -> Path:
    return CACHE_DIR / f"{family_name}_grasp_50k_s{scale_tag(s)}.npy"


def discover_families():
    """Yield (category, subset, object_type, family_name, urdf_count)."""
    fams = []
    for category in CATEGORIES:
        cat_dir = ASSETS_DIR / category
        if not cat_dir.is_dir():
            continue
        for subset_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            urdfs = sorted(subset_dir.glob("*.urdf"))
            if not urdfs:
                continue
            subset = subset_dir.name
            fams.append((category, subset, f"{category}_{subset}",
                         f"leap_hand_in_{category}_{subset}", len(urdfs)))
    return fams


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


def build_cmd(object_type, family_name, category, scale, num_envs, episode_length):
    cmd = [
        "uv", "run", "python", "train.py",
        "task=LeapHandGrasp", "test=true", "pipeline=cpu", "wandb_activate=false",
        f"task.env.object.type={object_type}",
        f"task.env.grasp_cache_name={family_name}",
        f"task.env.baseObjScale={scale}",
        f"task.env.numEnvs={num_envs}",
        f"task.env.episodeLength={episode_length}",
        "task.env.rerun.enabled=false",
        "train.params.config.player.games_num=5000000",
    ]
    cmd += list(POSE_OVERRIDES.get(category, []))
    return cmd


def run_one(object_type, family_name, category, scale, args, log_fh) -> bool:
    cmd = build_cmd(object_type, family_name, category, scale, args.num_envs, args.episode_length)
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
    ap.add_argument("--timeout", type=int, default=1800, help="Per cache subprocess timeout [s]")
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
    jobs = [(cat, sub, ot, fam, s) for (cat, sub, ot, fam, _n) in families for s in args.scales]
    todo = [j for j in jobs if not cache_path(j[3], j[4]).exists()]

    print(f"Families: {len(families)} | scales: {args.scales} | total jobs: {len(jobs)} | "
          f"already done: {len(jobs) - len(todo)} | to run: {len(todo)}")
    for (cat, sub, ot, fam, _n) in families:
        have = sum(cache_path(fam, s).exists() for s in args.scales)
        print(f"  {fam:32s} object.type={ot:18s} caches {have}/{len(args.scales)}")
    if args.dry_run or not todo:
        return

    log_path = LOG_DIR / f"grasp_gen_{datetime.now():%Y%m%d_%H%M%S}.log"
    print(f"\nLogging subprocess output to {log_path}\n")
    with open(log_path, "w") as log_fh:
        for i, (cat, sub, ot, fam, s) in enumerate(todo, 1):
            key = f"{fam}_s{scale_tag(s)}"
            print(f"[{i}/{len(todo)}] {fam} scale={s} ...", end=" ", flush=True)
            t0 = time.time()
            ok = run_one(ot, fam, cat, s, args, log_fh)
            status[key] = {"ok": ok, "seconds": round(time.time() - t0, 1),
                           "at": datetime.now().isoformat(), "cache": str(cache_path(fam, s).name)}
            save_status(status)
            print("OK" if ok else "FAILED (no cache written — check log)",
                  f"({status[key]['seconds']}s)")

    done = sum(1 for v in status.values() if v.get("ok"))
    print(f"\nDone. {done} caches present. Status: {STATUS_PATH}")


if __name__ == "__main__":
    main()
