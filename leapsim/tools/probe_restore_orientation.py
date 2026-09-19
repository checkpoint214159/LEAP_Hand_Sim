#!/usr/bin/env python3
"""probe_restore_orientation.py — verify that cache restore reproduces the saved
object ORIENTATION in the sim (not just in the .npy).

Restores rows of a cylinder cache exactly like the graspcheck (sequential rows,
object-matched envs, pinned randomization), steps zero actions, and compares the
sim-side object axis tilt against the cache rows. Separates "restore bug" (sim
tilt ~0°, cache ~78°) from "visualization bug" (sim tilt matches cache).

Run from src/LEAP_Hand_Sim/leapsim:
    uv run python tools/probe_restore_orientation.py [family] [scale]
"""
import sys

import isaacgym  # noqa: F401  must precede torch
import torch
import numpy as np
import leapsim  # registers OmegaConf resolvers
from hydra import initialize, compose

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "cylinder_train"
SCALE = sys.argv[2] if len(sys.argv) > 2 else "1.0"
CACHE_NAME = f"leap_hand_in_{FAMILY}"
for _a in sys.argv[3:]:
    if _a.startswith("cache="):
        CACHE_NAME = _a.split("=", 1)[1]
CACHE = f"cache/{CACHE_NAME}_grasp_50k_s{SCALE.replace('.', '')}.npy"
N = 64


def tilt_deg(q_xyzw: np.ndarray) -> np.ndarray:
    """Angle between object z-axis and world z (0° = upright, 90° = lying)."""
    x, y = q_xyzw[:, 0], q_xyzw[:, 1]
    zz = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(np.abs(zz), 0.0, 1.0)))


def main() -> None:
    cache = np.load(CACHE)
    print(f"{CACHE}: {cache.shape}")
    qcol = 35 if cache.shape[1] >= 40 else 19
    print(f"cache rows 0..{N-1} tilt: med {np.median(tilt_deg(cache[:N, qcol:qcol+4])):.0f}°")

    overrides = [
        "task=LeapHandRot", "headless=true", "wandb_activate=false",
        f"task.env.numEnvs={N}",
        f"task.env.object.type={FAMILY}",
        f"task.env.grasp_cache_name={CACHE_NAME}",
        f"task.env.randomization.randomizeScaleList=[{SCALE}]",
        # Short episodes: force multiple reset generations within the probe.
        "task.env.episodeLength=25",
        "task.env.rerun.enabled=false",
    ]
    if "RAW" not in sys.argv:
        # graspcheck-style pins; omit (pass RAW) for a training-like config
        overrides += [
            "+task.env.scale_list_jitter=0",
            "task.env.randomization.randomizeMassLower=0.05",
            "task.env.randomization.randomizeMassUpper=0.051",
            "task.env.randomization.randomizeCOM=false",
            "task.env.randomization.randomizeFriction=false",
            "task.env.randomization.randomizePDGains=false",
            "task.env.forceScale=0",
            "+task.env.sequential_pose_idx=true",
            f"+task.env.object_ids_from_cache={CACHE}",
        ]
    overrides += [a for a in sys.argv[3:] if a not in ("RAW", "DIRECT", "FULLSET", "REISSUE") and not a.startswith("cache=")]
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config", overrides=overrides)
    env = leapsim.make(cfg.seed, "LeapHandRot", N, cfg.sim_device, cfg.rl_device,
                       cfg.graphics_device_id, True, cfg=cfg)
    if "DIRECT" in sys.argv:
        # Clean in-run apply test: warm the sim past the pre-first-simulate dead
        # zone, stage via reset_idx, optionally re-issue the root-state call LAST
        # (after reset_idx's dof-indexed calls — Preview-4 later-setter-cancels-
        # earlier suspicion), then one bare simulate and read back.
        from isaacgym import gymtorch as _gt
        env.reset()
        zeros = torch.zeros((N, env.num_actions), device=env.device)
        for _ in range(3):
            env.step(zeros)
        ids = torch.arange(N, device=env.device)
        env.reset_idx(ids)
        staged_q = env.root_state_tensor[env.object_indices, 3:7].cpu().numpy()
        print(f"STAGED rows (pre-simulate): tilt med {np.median(tilt_deg(staged_q)):.0f}° "
              f"| quat norm med {np.median(np.linalg.norm(staged_q, axis=1)):.3f}")
        tag = "DIRECT(warm)"
        if "REISSUE" in sys.argv:
            obj_idx = env.object_indices.to(torch.int32)
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim, _gt.unwrap_tensor(env.root_state_tensor),
                _gt.unwrap_tensor(obj_idx), len(obj_idx))
            tag += "+REISSUE"
        if "FULLSET" in sys.argv:
            env.gym.set_actor_root_state_tensor(env.sim, _gt.unwrap_tensor(env.root_state_tensor))
            tag += "+FULLSET"
        staged_p = env.root_state_tensor[env.object_indices, 0:3].cpu().numpy()
        env.gym.simulate(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        q = env.root_state_tensor[env.object_indices, 3:7].cpu().numpy()
        p = env.root_state_tensor[env.object_indices, 0:3].cpu().numpy()
        dp = np.linalg.norm(p[:, :2] - staged_p[:, :2], axis=1)  # xy drift vs staged
        print(f"{tag} after 1 simulate: tilt med {np.median(tilt_deg(q)):.0f}° "
              f"z med {np.median(p[:,2]):.3f} | xy drift from staged: "
              f"med {np.median(dp)*1000:.1f}mm max {dp.max()*1000:.1f}mm")
        return

    env.reset()
    zeros = torch.zeros((N, env.num_actions), device=env.device)
    for t in range(60):
        _, _, done, _ = env.step(zeros)
        tl = tilt_deg(env.object_rot.cpu().numpy())
        if t % 5 == 0 or int(done.sum()) > 8:
            print(f"step {t}: sim tilt med {np.median(tl):.0f}° "
                  f"min {tl.min():.0f}° max {tl.max():.0f}° | resets this step: {int(done.sum())}")

    # Per-instance steady-state tilt + height: discriminates "restore never
    # happened" (creation pose settles: can/disk stable on flat end ≈ 0°) from
    # "restored then flopped unsupported" (can lands lying ≈ 90°).
    ids = env.env_object_type_id.cpu().numpy()
    z = env.object_pos[:, 2].cpu().numpy()
    cache_tilt = tilt_deg(cache[:N, qcol:qcol + 4])
    for oid in np.unique(ids):
        m = ids == oid
        print(f"obj id {oid}: n={m.sum()} sim tilt med {np.median(tl[m]):.0f}° "
              f"(cache rows med {np.median(cache_tilt[m]):.0f}°) z med {np.median(z[m]):.3f}")


if __name__ == "__main__":
    main()
