#!/usr/bin/env python3
"""probe_minimal_setter.py — minimal IsaacGym GPU-pipeline root-state setter test.

No LEAP assets, no hydra, no aggregates: N free boxes, warm the sim, stage new
root poses, apply via set_actor_root_state_tensor_indexed (and full-tensor
variant), simulate once, read back. Decides stack-config vs environment bug.

Run from anywhere:  uv run python tools/probe_minimal_setter.py [cpu|gpu]
"""
import sys

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymtorch
import torch
import numpy as np

PIPELINE = sys.argv[1] if len(sys.argv) > 1 else "gpu"
N = 16

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1.0 / 120.0
sp.substeps = 1
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, -9.81)
sp.use_gpu_pipeline = (PIPELINE == "gpu")
sp.physx.solver_type = 1
sp.physx.num_position_iterations = 8
sp.physx.use_gpu = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

plane = gymapi.PlaneParams()
plane.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane)

asset_opts = gymapi.AssetOptions()
asset_opts.density = 200.0
box = gym.create_box(sim, 0.06, 0.06, 0.06, asset_opts)

AGG = "agg" in sys.argv          # wrap actors in an aggregate like leapsim
TWO = "two" in sys.argv          # add a fixed-base second actor like the hand
HAND = "hand" in sys.argv        # second actor = the real 16-DOF LEAP hand + DOF-indexed calls
fixed_opts = gymapi.AssetOptions()
fixed_opts.fix_base_link = True
fixed_box = gym.create_box(sim, 0.1, 0.1, 0.02, fixed_opts)
if HAND:
    TWO = True
    hand_opts = gymapi.AssetOptions()
    hand_opts.fix_base_link = True
    fixed_box = gym.load_asset(sim, "../assets", "leap_hand/robot.urdf", hand_opts)

envs = []
for i in range(N):
    env = gym.create_env(sim, gymapi.Vec3(-0.5, -0.5, 0), gymapi.Vec3(0.5, 0.5, 1), 4)
    if AGG:
        gym.begin_aggregate(env, 19 * 20, 40 * 20, True)
    if TWO:
        fp = gymapi.Transform()
        fp.p = gymapi.Vec3(0, 0, 0.3)
        gym.create_actor(env, fixed_box, fp, "base", i, -1, 0)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.5)
    bh = gym.create_actor(env, box, pose, "box", i, 0, 0)
    if "scaled" in sys.argv:
        gym.set_actor_scale(env, bh, 0.95)   # leapsim: set_actor_scale on every object
    if AGG:
        gym.end_aggregate(env)
    envs.append(env)

gym.prepare_sim(sim)
root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1, 13)
device = root.device
# rows of the FREE boxes (leapsim analog: object = 2nd actor per env)
box_rows = torch.arange(1, 2 * N, 2, device=device) if TWO else torch.arange(N, device=device)
print(f"pipeline={PIPELINE} agg={AGG} two={TWO} root tensor {tuple(root.shape)} on {device}")

# warm: several steps so boxes settle on the ground (z ~0.03)
for _ in range(30):
    gym.simulate(sim)
    if PIPELINE == "cpu":
        gym.fetch_results(sim, True)
gym.refresh_actor_root_state_tensor(sim)
print(f"after warm: z med {root[box_rows, 2].median().item():.3f} (settled on ground)")

# stage: teleport all boxes to z=0.8 with a 45° tilt about x
root[box_rows, 0:3] = torch.tensor([0.0, 0.0, 0.8], device=device)
root[box_rows, 3:7] = torch.tensor([0.3826834, 0.0, 0.0, 0.9238795], device=device)
root[box_rows, 7:13] = 0.0
if "massset" in sys.argv:
    # leapsim reset: per-env get/set_actor_rigid_body_properties before staging
    for i, env in enumerate(envs):
        h = gym.find_actor_handle(env, "box")
        prop = gym.get_actor_rigid_body_properties(env, h)
        for pr in prop:
            pr.mass = 0.05
        gym.set_actor_rigid_body_properties(env, h, prop)
idx = box_rows.to(torch.int32)
gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root),
                                        gymtorch.unwrap_tensor(idx), N)
if HAND:
    # leapsim reset order: root-indexed, then dof-target-indexed, then dof-state-indexed
    dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    n_dof = dof_state.shape[0] // N
    targets = torch.zeros((N, n_dof), device=device)
    hand_idx = torch.arange(0, 2 * N, 2, dtype=torch.int32, device=device)
    gym.set_dof_position_target_tensor_indexed(sim, gymtorch.unwrap_tensor(targets),
                                               gymtorch.unwrap_tensor(hand_idx), N)
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state),
                                     gymtorch.unwrap_tensor(hand_idx), N)
gym.simulate(sim)
if PIPELINE == "cpu":
    gym.fetch_results(sim, True)
gym.refresh_actor_root_state_tensor(sim)
z = root[box_rows, 2]
qx = root[box_rows, 3]
ok = (z.median().item() > 0.7) and (abs(qx.median().item()) > 0.3)
print(f"INDEXED apply: z med {z.median().item():.3f} qx med {qx.median().item():.3f} "
      f"→ {'WORKS' if ok else 'FAILED (expected z~0.8, qx~0.38)'}")

# full-tensor variant from a fresh settled state
for _ in range(60):
    gym.simulate(sim)
    if PIPELINE == "cpu":
        gym.fetch_results(sim, True)
gym.refresh_actor_root_state_tensor(sim)
root[box_rows, 0:3] = torch.tensor([0.0, 0.0, 0.8], device=device)
root[box_rows, 3:7] = torch.tensor([0.3826834, 0.0, 0.0, 0.9238795], device=device)
root[box_rows, 7:13] = 0.0
gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
gym.simulate(sim)
if PIPELINE == "cpu":
    gym.fetch_results(sim, True)
gym.refresh_actor_root_state_tensor(sim)
z = root[box_rows, 2]
qx = root[box_rows, 3]
ok = (z.median().item() > 0.7) and (abs(qx.median().item()) > 0.3)
print(f"FULL apply:    z med {z.median().item():.3f} qx med {qx.median().item():.3f} "
      f"→ {'WORKS' if ok else 'FAILED (expected z~0.8, qx~0.38)'}")
