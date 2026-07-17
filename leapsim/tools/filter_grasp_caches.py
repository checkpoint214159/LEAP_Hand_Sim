#!/usr/bin/env python3
"""filter_grasp_caches.py — post-hoc stability filter for grasp caches.

The grasp generator's success criterion (num_contact_fingers=0) admits
near-miss states: rows that restore into the rotation task and drop the
object within a second under zero actions. This tool replays every cache
row in its own env (deterministic env↔row mapping via sequential_pose_idx),
holds the hand still for --hold-steps in the *rotation* task under
settle-nominal conditions (mass 0.05 kg, no COM/friction/PD randomization,
no random forces), and keeps only the rows that never drop the object.
Filtering in the rot task also absorbs any cpu→gpu settle-transfer mismatch,
since it tests exactly the reset conditions training will use.

The canonical cache path is overwritten with the surviving rows; the
unfiltered original is preserved at cache/raw/<name>.npy and per-cache
results go to <cache>_filter_stats.yaml. Resumable: caches with an existing
filter-stats file are skipped.

Run from src/LEAP_Hand_Sim/leapsim:
    uv run python tools/filter_grasp_caches.py                    # all primitive caches
    uv run python tools/filter_grasp_caches.py --only cuboid_train
    uv run python tools/filter_grasp_caches.py --dry-run
Each cache is filtered in a child process (IsaacGym supports one sim per
process); --single is the internal child mode.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LEAPSIM_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = LEAPSIM_DIR / "cache"
RAW_DIR = CACHE_DIR / "raw"

CATEGORIES = ["cuboid", "cylinder", "sphere"]
DEFAULT_SCALES = [0.95, 0.9, 1.0, 1.05, 1.1]
# Below this many surviving rows, flag the cache for regeneration/top-up.
MIN_KEPT_ROWS = 512


def scale_tag(s: float) -> str:
    return str(s).replace(".", "")


def cache_path(family: str, s: float) -> Path:
    return CACHE_DIR / f"leap_hand_in_{family}_grasp_50k_s{scale_tag(s)}.npy"


def stats_path(family: str, s: float) -> Path:
    return CACHE_DIR / f"leap_hand_in_{family}_grasp_50k_s{scale_tag(s)}_filter_stats.yaml"


def discover_families():
    assets = LEAPSIM_DIR.parent / "assets"
    fams = []
    for category in CATEGORIES:
        cat_dir = assets / category
        if not cat_dir.is_dir():
            continue
        for subset_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            if list(subset_dir.glob("*.urdf")):
                fams.append(f"{category}_{subset_dir.name}")
    return fams


# ──────────────────────────────────────────────────────────────────────────────
# Child mode: filter ONE cache in this process.
# ──────────────────────────────────────────────────────────────────────────────

def filter_single(family: str, scale: float, hold_steps: int) -> int:
    import isaacgym  # noqa: F401  must precede torch in every entry point
    import torch
    import numpy as np
    import yaml
    import leapsim  # registers OmegaConf resolvers
    from hydra import initialize, compose

    cpath = cache_path(family, scale)
    raw = np.load(cpath)
    n = raw.shape[0]

    overrides = [
        "task=LeapHandRot",
        "headless=true",
        "wandb_activate=false",
        f"task.env.numEnvs={n}",
        f"task.env.object.type={family}",
        f"task.env.grasp_cache_name=leap_hand_in_{family}",
        f"task.env.randomization.randomizeScaleList=[{scale}]",
        "task.env.randomization.randomizeMassLower=0.05",
        "task.env.randomization.randomizeMassUpper=0.051",
        "task.env.randomization.randomizeCOM=false",
        "task.env.randomization.randomizeFriction=false",
        "task.env.randomization.randomizePDGains=false",
        "task.env.forceScale=0",
        f"task.env.episodeLength={hold_steps + 50}",
        "+task.env.sequential_pose_idx=true",
        # Object-aware caches (24-col): force env i to hold cache row i's object
        # so the row↔env replay is object-correct. Ignored for 23-col caches.
        f"+task.env.object_ids_from_cache={cpath}",
        "task.env.rerun.enabled=false",
    ]
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config", overrides=overrides)

    env = leapsim.make(
        cfg.seed, "LeapHandRot", n, cfg.sim_device, cfg.rl_device,
        cfg.graphics_device_id, True, cfg=cfg,
    )

    env.reset()
    zeros = torch.zeros((n, env.num_actions), device=env.device)
    dropped = torch.zeros(n, dtype=torch.bool, device=env.device)
    for _ in range(hold_steps):
        _, _, done, _ = env.step(zeros)
        dropped |= done.to(env.device).bool()

    kept_mask = (~dropped).cpu().numpy()
    kept = raw[kept_mask]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_backup = RAW_DIR / cpath.name
    if not raw_backup.exists():
        np.save(raw_backup, raw)
    np.save(cpath, kept)

    stats = {
        "family": family,
        "scale": scale,
        "hold_steps": hold_steps,
        "total_rows": int(n),
        "kept_rows": int(kept.shape[0]),
        "keep_rate_pct": round(100.0 * kept.shape[0] / max(1, n), 2),
        "raw_backup": str(raw_backup.relative_to(LEAPSIM_DIR)),
        "filtered_at": datetime.now().isoformat(),
    }
    with open(stats_path(family, scale), "w") as f:
        yaml.safe_dump(stats, f, default_flow_style=False)
    print(f"[filter] {cpath.name}: kept {kept.shape[0]}/{n} "
          f"({stats['keep_rate_pct']}%)")
    return 0 if kept.shape[0] >= MIN_KEPT_ROWS else 3


# ──────────────────────────────────────────────────────────────────────────────
# Parent mode: one child subprocess per cache (IsaacGym = one sim per process).
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="Restrict to one family, e.g. cuboid_train")
    ap.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    ap.add_argument("--hold-steps", type=int, default=400,
                    help="Zero-action control steps a row must survive (400 = one rot episode)")
    ap.add_argument("--timeout", type=int, default=900, help="Per-cache subprocess timeout [s]")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--single", nargs=2, metavar=("FAMILY", "SCALE"), default=None,
                    help=argparse.SUPPRESS)  # internal child mode
    args = ap.parse_args()

    if args.single:
        sys.exit(filter_single(args.single[0], float(args.single[1]), args.hold_steps))

    families = discover_families()
    if args.only:
        families = [f for f in families if f == args.only]
    jobs = [(f, s) for f in families for s in args.scales
            if cache_path(f, s).exists() and not stats_path(f, s).exists()]
    skipped = [(f, s) for f in families for s in args.scales
               if stats_path(f, s).exists()]
    missing = [(f, s) for f in families for s in args.scales
               if not cache_path(f, s).exists()]

    print(f"To filter: {len(jobs)} | already filtered: {len(skipped)} | "
          f"cache missing: {len(missing)}")
    if args.dry_run or not jobs:
        return

    low_yield = []
    for i, (fam, s) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {fam} s{s} ...", flush=True)
        r = subprocess.run(
            ["uv", "run", "python", "tools/filter_grasp_caches.py",
             "--single", fam, str(s), "--hold-steps", str(args.hold_steps)],
            cwd=str(LEAPSIM_DIR), timeout=args.timeout, check=False,
        )
        if r.returncode == 3:
            low_yield.append((fam, s))
        elif r.returncode != 0:
            print(f"  child failed (rc={r.returncode}) — check output above")

    if low_yield:
        print("\nLow-yield caches (< {} rows kept) — consider regenerating with a "
              "longer run or better pose:".format(MIN_KEPT_ROWS))
        for fam, s in low_yield:
            print(f"  {fam} s{s}")


if __name__ == "__main__":
    main()
