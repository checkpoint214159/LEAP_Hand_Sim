#!/usr/bin/env python3
"""gen_primitive_objects.py — procedural object-set generator for the
cross-object-group generalization study.

Emits parametric primitive URDFs (native box / cylinder / sphere geometry, so
IsaacGym uses analytic collision — no meshes or VHACD) into the directory layout
that leap_hand_rot.py already globs:

    <assets>/<category>/<subset>/obj_<i>.urdf

where category in {cuboid, cylinder, sphere}. A `subset` is one "object family":
LeapHandRot samples an object per env from the family, and one shared grasp cache
is generated per family (see tools/gen_grasp_caches.py).

Taxonomy (2 levels):
  * category  = strategy-defining shape class (roller / box-pivot / cylinder).
  * instance  = continuous variation within a class (aspect ratio, size).
We emit a `train` subset (in-distribution) and a `heldout` subset (tests
within-category generalization) per category. Categories themselves are the
cross-category (hard) generalization axis.

Inertia is computed for a uniform-density solid so shapes are physically
distinct (a flat slab and a rod differ in their inertia tensor — part of what
makes categories behave differently). Sizes are chosen near the known-graspable
cube (0.075 m box) so the LEAP hand can form an initial grip.

Usage (from src/LEAP_Hand_Sim/leapsim):
    uv run python tools/gen_primitive_objects.py            # writes ../assets/...
    uv run python tools/gen_primitive_objects.py --dry-run  # list, write nothing
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    import trimesh
except ImportError:
    trimesh = None

# Uniform density [kg/m^3]. ~200 puts a ~0.07 m cube near 0.07 kg, in the same
# ballpark as the hand-tuned existing assets (cube.urdf mass 0.05).
DEFAULT_DENSITY = 200.0

# ---------------------------------------------------------------------------- #
# Taxonomy. Each entry is (category, subset, [instances]). Instance params:
#   cuboid  : ('box',  (sx, sy, sz))      full extents [m]
#   cylinder: ('cyl',  (radius, length))  length along z [m]
#   sphere  : ('sph',  (radius,))         [m]
# ---------------------------------------------------------------------------- #
# Size is deliberately DECORRELATED from category: grip aperture is observable
# from proprioception, so if size tracked category, the naive proprio policy
# could infer "which category" from aperture alone (deflating H1) and z0 (which
# contains size) would partially encode category (muddying the z0→z1 shape
# ablation, the core Stage-2 claim). All train volumes sit in an overlapping
# ~195-345 cm³ band; heldouts probe the band edges symmetrically. 4 train +
# 2 heldout per category so mixture difficulty is count-matched.
TAXONOMY = {
    ("cuboid", "train"): [
        ("box", (0.070, 0.070, 0.070)),   # near-cube  343 cm³
        ("box", (0.085, 0.075, 0.045)),   # slab       287
        ("box", (0.095, 0.050, 0.050)),   # bar        237
        ("box", (0.055, 0.055, 0.090)),   # tall       272
    ],
    ("cuboid", "heldout"): [
        ("box", (0.075, 0.060, 0.065)),   #            292
        ("box", (0.100, 0.045, 0.045)),   # long bar   202
    ],
    ("cylinder", "train"): [
        ("cyl", (0.032, 0.080)),          # can        257
        ("cyl", (0.050, 0.030)),          # disk/puck  236
        ("cyl", (0.026, 0.095)),          # rod        202
        ("cyl", (0.042, 0.050)),          # squat      277
    ],
    ("cylinder", "heldout"): [
        ("cyl", (0.028, 0.090)),          #            222
        ("cyl", (0.046, 0.045)),          #            299
    ],
    ("sphere", "train"): [
        ("sph", (0.036,)),                #            195
        ("sph", (0.039,)),                #            248
        ("sph", (0.041,)),                #            289
        ("sph", (0.043,)),                #            333
    ],
    ("sphere", "heldout"): [
        ("sph", (0.0375,)),               #            221
        ("sph", (0.044,)),                #            357
    ],
    # cone/capsule: NOT native URDF geometry (URDF only defines box/cylinder/
    # sphere/mesh) — these are generated as trimesh meshes, see _make_mesh.
    # Both are new STRATEGY classes, not just new instances of an existing one
    # (see docs/research-plan-object-generalization.md): a cone laid on its
    # side rolls in a circular ARC (not a straight line like a cylinder), and
    # a capsule has no flat end to pivot against (unlike a cylinder's disk
    # faces) so it can't be "caught" the way a cylinder or cuboid can.
    # dims: cone=(base radius, height, apex at bbox +z); capsule=(radius,
    # cylindrical-segment length L; total length = L+2r, axis z).
    ("cone", "train"): [
        ("cone", (0.055, 0.075)),         # squat      238 cm³
        ("cone", (0.045, 0.130)),         # tall       276
        ("cone", (0.060, 0.060)),         # wide       226
        ("cone", (0.038, 0.155)),         # spike      234
    ],
    ("cone", "heldout"): [
        ("cone", (0.050, 0.100)),         #            262
        ("cone", (0.065, 0.050)),         #            221
    ],
    ("capsule", "train"): [
        ("cap", (0.030, 0.070)),          # pill       311
        ("cap", (0.036, 0.028)),          # squat      309
        ("cap", (0.022, 0.130)),          # rod        242
        ("cap", (0.032, 0.055)),          # medium     314
    ],
    ("capsule", "heldout"): [
        ("cap", (0.028, 0.085)),          #            301
        ("cap", (0.034, 0.038)),          #            303
    ],
}

# Kinds with no native URDF <geometry> primitive tag — written as a generated
# mesh file (obj_i.stl) referenced via <mesh filename=...>.
MESH_KINDS = {"cone", "cap"}


def _inertia(kind: str, dims, mass: float):
    """(ixx, iyy, izz, com_z_offset) of a uniform-density solid about its COM.

    com_z_offset is the COM's height along the LOCAL z-axis relative to the
    shape's own bbox-center origin (0 for every symmetric kind here; a cone's
    mass concentrates toward its base, so its COM is NOT at the bbox center).
    """
    if kind == "box":
        a, b, c = dims                       # full extents
        k = mass / 12.0
        return k * (b * b + c * c), k * (a * a + c * c), k * (a * a + b * b), 0.0
    if kind == "cyl":
        r, h = dims                          # axis along z
        ixx = mass * (3.0 * r * r + h * h) / 12.0
        return ixx, ixx, 0.5 * mass * r * r, 0.0
    if kind == "sph":
        r = dims[0]
        i = 0.4 * mass * r * r
        return i, i, i, 0.0
    if kind == "cone":
        # Solid right circular cone: base radius r, height h, axis z, base at
        # bbox z=-h/2, apex at bbox z=+h/2 (see _make_mesh). Derived by direct
        # integration (thin-disk stacking + parallel axis), not copied from a
        # table — cross-checked: I_zz=(3/10)mr^2 and COM=h/4-above-base both
        # match standard references, so the perpendicular-axis result
        # (independently re-derived here) is trusted:
        #   I_zz       = (3/10) m r^2                       (about the axis)
        #   I_xx=I_yy  = (3/20) m r^2 + (3/80) m h^2         (about the COM)
        #   COM height = h/4 above the base
        r, h = dims
        izz = 0.3 * mass * r * r
        ixx = 0.15 * mass * r * r + 0.0375 * mass * h * h
        com_z = -0.25 * h            # base sits at bbox z=-h/2; COM is h/4 above it
        return ixx, ixx, izz, com_z
    if kind == "cap":
        # Capsule = cylinder (radius r, length L, axis z) + 2 hemispherical
        # end caps, standard decomposition (cap-pair treated as one sphere of
        # radius r for mass-splitting and the axial moment; the perpendicular
        # moment additionally accounts for each cap's centroid offset via the
        # parallel-axis term). Formula checked against both limits: L->0
        # collapses to a solid sphere (2/5 m r^2); r->0 collapses to a thin
        # rod (m L^2/12) — both exact.
        r, L = dims
        v_cyl = math.pi * r * r * L
        v_sph = 4.0 / 3.0 * math.pi * r ** 3
        m_cyl = mass * v_cyl / (v_cyl + v_sph)
        m_sph = mass * v_sph / (v_cyl + v_sph)
        izz = 0.5 * m_cyl * r * r + 0.4 * m_sph * r * r
        ixx = (m_cyl * (L * L / 12.0 + r * r / 4.0)
               + m_sph * (2.0 * r * r / 5.0 + L * L / 4.0 + 3.0 * L * r / 8.0))
        return ixx, ixx, izz, 0.0
    raise ValueError(kind)


def _volume(kind: str, dims) -> float:
    if kind == "box":
        return dims[0] * dims[1] * dims[2]
    if kind == "cyl":
        return math.pi * dims[0] ** 2 * dims[1]
    if kind == "sph":
        return 4.0 / 3.0 * math.pi * dims[0] ** 3
    if kind == "cone":
        return (math.pi / 3.0) * dims[0] ** 2 * dims[1]
    if kind == "cap":
        r, L = dims
        return math.pi * r * r * L + 4.0 / 3.0 * math.pi * r ** 3
    raise ValueError(kind)


def _geometry_xml(kind: str, dims, mesh_filename: str | None = None) -> str:
    if kind == "box":
        return f'<box size="{dims[0]:.6f} {dims[1]:.6f} {dims[2]:.6f}"/>'
    if kind == "cyl":
        return f'<cylinder radius="{dims[0]:.6f}" length="{dims[1]:.6f}"/>'
    if kind == "sph":
        return f'<sphere radius="{dims[0]:.6f}"/>'
    if kind in MESH_KINDS:
        assert mesh_filename is not None, f"{kind} needs a generated mesh_filename"
        return f'<mesh filename="{mesh_filename}"/>'
    raise ValueError(kind)


def _make_mesh(kind: str, dims):
    """Build a trimesh primitive, recentered to its own bbox center — every
    closed-form formula in leap_hand_rot.py's local-z feature assumes query
    points are expressed relative to the object's bbox center, matching
    box/cylinder/sphere's existing convention (their URDF <origin> is already
    the bbox center by construction).

    Resolution (cone sections=48, capsule count=[24,24]) chosen so the
    resulting mesh's bbox matches the requested analytic dims to <0.3% —
    verified empirically (low resolution measurably shrinks the radius: 8x8
    capsule undershoots requested radius by 2.5%, 24x24 by 0.23%). This
    matters because leap_hand_rot.py derives z0/z1/local-z geometry from the
    MESH's bbox at runtime, not from these exact (r,h)/(r,L) numbers — so a
    coarse mesh would silently feed the policy slightly wrong geometry.
    """
    if trimesh is None:
        raise RuntimeError("trimesh is required to generate cone/capsule assets (pip install trimesh)")
    if kind == "cone":
        r, h = dims
        mesh = trimesh.creation.cone(radius=r, height=h, sections=48)
        mesh.apply_translation([0, 0, -h / 2.0])   # trimesh: base@z=0, apex@z=h -> recenter
    elif kind == "cap":
        r, L = dims
        mesh = trimesh.creation.capsule(radius=r, height=L, count=[24, 24])
        # trimesh's capsule is already centered at its own bbox center (verified).
    else:
        raise ValueError(kind)
    return mesh


def _urdf(kind: str, dims, density: float, mesh_filename: str | None = None) -> str:
    mass = density * _volume(kind, dims)
    ixx, iyy, izz, com_z = _inertia(kind, dims, mass)
    geom = _geometry_xml(kind, dims, mesh_filename)
    inertial_origin = f'\n      <origin xyz="0 0 {com_z:.6f}"/>' if com_z else ""
    return f"""<?xml version="1.0"?>
<!-- Auto-generated by tools/gen_primitive_objects.py. kind={kind} dims={dims} -->
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        {geom}
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        {geom}
      </geometry>
    </collision>
    <inertial>{inertial_origin}
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.8f}" ixy="0.0" ixz="0.0" iyy="{iyy:.8f}" iyz="0.0" izz="{izz:.8f}"/>
    </inertial>
  </link>
</robot>
"""


# ---------------------------------------------------------------------------- #
# Dense cuboid families (the shape-density sweep, docs/HANDOFF.md §4/§5).
#
# The Stage-1 null ("oracle z0 buys nothing") is confounded: with only ~4 shapes
# per category, z0 is indistinguishable from object identity — you cannot fit a
# continuous map geometry->strategy from 4 points, so z0 *cannot* help zero-shot
# regardless of whether shape is informative. The fix is a density sweep: many
# distinct cuboids spanning the aspect-ratio space, and watch whether z0's benefit
# emerges as the sample count grows (4 -> 16 -> 40).
#
# Design choices that keep the sweep clean and cheap:
#   * FIXED VOLUME + FIXED SCALE (1.0). At fixed scale a box's z0 reduces to
#     [1, sx, sy, sz, 1] (fill==1 for boxes) — the only informative signal is the
#     three edge lengths. Holding volume fixed collapses that to a 2-D shape
#     manifold (two aspect ratios), which 40 points can fill densely; a 3-D
#     (volume-varying) region cannot. Scale is proprioceptively observable and is
#     not the variable of interest, so we pin it.
#   * SORTED EDGES sx>=sy>=sz. Removes the orientation-relabel degeneracy (a "tall
#     bar" and a "lying bar" are one shape) so each distinct shape appears once,
#     and keeps the smallest edge vertical -> objects spawn flat, graspable, and
#     the per-family-max grasp-gen init pose stays homogeneous.
#   * HALTON-NESTED train pool. A Halton (2,3) sequence is low-discrepancy at every
#     prefix, so accepted[:4] ⊂ accepted[:16] ⊂ accepted[:40]. The whole sweep
#     therefore reuses ONE grasp cache; density is a runtime knob
#     (+task.env.object_id_whitelist=[0..N-1]).
#   * SEPARATE heldout family, interior to the training region (interpolation
#     targets, disjoint from every train point) so the zero-shot claim cannot leak.
#
# Subset dir names must be single tokens (LeapHandRot derives the subset from
# object.type.split('_')[-1]): "densetrain" / "denseheldout".
# ---------------------------------------------------------------------------- #

# Envelope matched to the VALIDATED, exonerated cuboid_train set (which grasps
# well under the strict palmless criterion): geometric-mean edge ~0.069 m, aspect
# ratios up to ~1.9, thinnest edge >=0.045 m. Fixed-volume-250 with a wide aspect
# range produced flat objects with thin edges ~0.040 m that barely grasp (the LEAP
# hand palm-rests them). Volume 320 + a tighter aspect band keeps every dense
# object inside the demonstrated-graspable region while still spanning shape space.
DENSE_VOLUME_CM3 = 320.0     # in-band (195-345); cube edge ~0.0685 m (cf. validated 0.070)
DENSE_EDGE_MIN = 0.035       # hard floor (rejection); the aspect band keeps thin edges ~0.046
DENSE_EDGE_MAX = 0.100       # cf. validated "long bar" 0.100
DENSE_LOG_ASPECT = 0.58      # → AR up to ~1.8, thinnest edge ~0.046 (validated slab was 0.045)
DENSE_HELDOUT_MARGIN = 0.004 # keep heldout strictly interior to [emin,emax]
DENSE_MIN_SEP = 0.038        # min L2 sep in log-aspect space (smaller band → smaller sep)


def _halton(i: int, base: int) -> float:
    """i-th point (1-indexed) of the 1-D van der Corput / Halton sequence."""
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _dense_box_from_uv(u: float, v: float, volume_m3: float):
    """Map a Halton point in [0,1)^2 to sorted box extents (sx>=sy>=sz) at fixed volume.

    Two independent log-aspect coordinates set two raw edge ratios; the third is 1;
    the triple is renormalised so its product == volume. Returns extents [m].
    """
    a1 = (2.0 * u - 1.0) * DENSE_LOG_ASPECT
    a2 = (2.0 * v - 1.0) * DENSE_LOG_ASPECT
    raw = [math.exp(a1), math.exp(a2), 1.0]
    gm = volume_m3 ** (1.0 / 3.0)
    gm_raw = (raw[0] * raw[1] * raw[2]) ** (1.0 / 3.0)
    edges = sorted((r / gm_raw * gm for r in raw), reverse=True)
    return edges  # [sx, sy, sz], product == volume_m3


def _log_aspect_coords(edges):
    """Shape fingerprint used for separation checks: log edge / geometric-mean edge."""
    gm = (edges[0] * edges[1] * edges[2]) ** (1.0 / 3.0)
    return [math.log(e / gm) for e in edges]


def _sep(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _sample_dense_cuboids(count: int, volume_m3: float, start_idx: int,
                          interior: bool, avoid: list):
    """Accept Halton points (in sequence, so prefixes stay nested) that land in the
    graspable edge band, are separated from every already-accepted / `avoid` shape,
    and (if interior) sit strictly inside the band. Returns list of ('box', extents)."""
    emin = DENSE_EDGE_MIN + (DENSE_HELDOUT_MARGIN if interior else 0.0)
    emax = DENSE_EDGE_MAX - (DENSE_HELDOUT_MARGIN if interior else 0.0)
    accepted, coords = [], [_log_aspect_coords(e) for _, e in avoid]
    i = start_idx
    guard = start_idx + 100000
    while len(accepted) < count and i < guard:
        edges = _dense_box_from_uv(_halton(i, 2), _halton(i, 3), volume_m3)
        i += 1
        if not all(emin <= e <= emax for e in edges):
            continue
        lc = _log_aspect_coords(edges)
        if any(_sep(lc, c) < DENSE_MIN_SEP for c in coords):
            continue
        accepted.append(("box", tuple(round(e, 6) for e in edges)))
        coords.append(lc)
    if len(accepted) < count:
        raise SystemExit(f"only sampled {len(accepted)}/{count} dense cuboids — "
                         f"loosen DENSE_LOG_ASPECT / DENSE_MIN_SEP or lower count")
    return accepted


def _write_family(out_dir: Path, instances, density: float, dry_run: bool) -> int:
    for i, (kind, dims) in enumerate(instances):
        path = out_dir / f"obj_{i}.urdf"
        mass = density * _volume(kind, dims)
        vol_cm3 = _volume(kind, dims) * 1e6
        ar = max(dims) / min(dims)
        mesh_tag = ""
        mesh_filename = None
        if kind in MESH_KINDS:
            mesh_filename = f"obj_{i}.stl"
            mesh_tag = f" mesh={mesh_filename}"
        print(f"{'[dry] ' if dry_run else ''}{path.parent.name}/obj_{i}.urdf  "
              f"{tuple(round(d, 4) for d in dims)}  V={vol_cm3:.0f}cm3 AR={ar:.2f} m={mass:.3f}kg{mesh_tag}")
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            if mesh_filename is not None:
                mesh = _make_mesh(kind, dims)
                mesh.export(str(out_dir / mesh_filename))
            path.write_text(_urdf(kind, dims, density, mesh_filename))
    return len(instances)


def gen_dense(args, assets: Path) -> None:
    vol = args.dense_volume_cm3 * 1e-6
    train = _sample_dense_cuboids(args.dense_train_count, vol, start_idx=1,
                                  interior=False, avoid=[])
    # Heldout: interior interpolation targets, disjoint from every train shape.
    heldout = _sample_dense_cuboids(args.dense_heldout_count, vol, start_idx=97,
                                    interior=True, avoid=train)
    print(f"Dense cuboids: {len(train)} train ({args.dense_train_subset}), "
          f"{len(heldout)} heldout ({args.dense_heldout_subset}); "
          f"V={args.dense_volume_cm3:.0f}cm3, scale fixed 1.0, edges "
          f"[{DENSE_EDGE_MIN},{DENSE_EDGE_MAX}] m\n")
    print("---- train (Halton-nested: obj_0..3 ⊂ ..15 ⊂ ..39) ----")
    _write_family(assets / "cuboid" / args.dense_train_subset, train, args.density, args.dry_run)
    print("\n---- heldout (interior, disjoint) ----")
    _write_family(assets / "cuboid" / args.dense_heldout_subset, heldout, args.density, args.dry_run)
    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {len(train)}+{len(heldout)} dense "
          f"cuboid URDFs under {assets}/cuboid/{{{args.dense_train_subset},{args.dense_heldout_subset}}}")
    if not args.dry_run:
        print("\nNext: grasp-gen at ONE scale, then the density sweep, e.g.\n"
              "  uv run python tools/gen_grasp_caches.py --only cuboid_"
              f"{args.dense_train_subset} --scales 1.0\n"
              "  uv run python tools/gen_grasp_caches.py --only cuboid_"
              f"{args.dense_heldout_subset} --scales 1.0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-dir", default=None,
                    help="Assets root (default: <this file>/../../assets, i.e. src/LEAP_Hand_Sim/assets)")
    ap.add_argument("--density", type=float, default=DEFAULT_DENSITY, help="Uniform density [kg/m^3]")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be written, write nothing")
    # Dense cuboid sweep (docs/HANDOFF.md §4/§5).
    ap.add_argument("--dense", action="store_true",
                    help="Emit the dense cuboid families for the shape-density sweep "
                         "(instead of the fixed taxonomy).")
    ap.add_argument("--dense-train-count", type=int, default=40)
    ap.add_argument("--dense-heldout-count", type=int, default=8)
    ap.add_argument("--dense-volume-cm3", type=float, default=DENSE_VOLUME_CM3)
    ap.add_argument("--dense-train-subset", default="densetrain")
    ap.add_argument("--dense-heldout-subset", default="denseheldout")
    ap.add_argument("--only", default=None,
                    help="Restrict to one category (e.g. 'cone' or 'capsule') — "
                         "leaves other families' existing files untouched.")
    args = ap.parse_args()

    # tools/ lives in .../leapsim/tools; assets are at .../src/LEAP_Hand_Sim/assets
    assets = Path(args.assets_dir) if args.assets_dir else (Path(__file__).resolve().parents[2] / "assets")

    if args.dense:
        gen_dense(args, assets)
        return

    families = {k: v for k, v in TAXONOMY.items() if args.only is None or k[0] == args.only}
    if not families:
        raise SystemExit(f"--only {args.only!r} matched no category "
                          f"(have: {sorted({c for c, _ in TAXONOMY})})")

    n = 0
    for (category, subset), instances in families.items():
        n += _write_family(assets / category / subset, instances, args.density, args.dry_run)

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {n} URDFs across "
          f"{len(families)} families under {assets}")
    if not args.dry_run:
        print("Object types for training/grasp-gen, e.g.: "
              "task.env.object.type=cuboid_train | cylinder_train | sphere_train | "
              "cone_train | capsule_train")


if __name__ == "__main__":
    main()
