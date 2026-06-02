"""env_setup.py — IsaacGym environment construction helpers.

Free functions called once during env creation. No task state (self) — all
inputs are explicit, all outputs are returned. This makes them testable in
isolation and easy to reuse across task variants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from isaacgym import gymapi


def create_leap_assets(gym, sim, asset_root: Path, env_cfg, body_shape_indices, object_type_list, asset_files_dict):
    """Load hand URDF and all object URDFs into IsaacGym assets.

    asset_root is the repo root (parent of 'assets/'); the hand and object
    paths from env_cfg are resolved relative to it.

    Returns (hand_asset, object_asset_list).
    """
    hand_path = asset_root / env_cfg['asset']['handAsset']

    hand_opts = gymapi.AssetOptions()
    hand_opts.flip_visual_attachments  = False
    hand_opts.fix_base_link            = True
    hand_opts.collapse_fixed_joints    = True
    hand_opts.disable_gravity          = False
    hand_opts.thickness                = 0.001
    hand_opts.angular_damping          = 0.01
    hand_opts.vhacd_enabled            = True
    hand_opts.vhacd_params.resolution  = 300000
    hand_opts.default_dof_drive_mode   = gymapi.DOF_MODE_POS

    hand_asset = gym.load_asset(sim, str(hand_path.parent), hand_path.name, hand_opts)

    if "leap_hand" in hand_path.name:
        rsp = gym.get_asset_rigid_shape_properties(hand_asset)
        for i, (_, body_group) in enumerate(env_cfg["mask_body_collision"].items()):
            filter_value = 2 ** i
            for body_idx in body_group:
                start, count = body_shape_indices[body_idx]
                for idx in range(count):
                    rsp[idx + start].filter = rsp[idx + start].filter | filter_value
        if env_cfg["disable_self_collision"]:
            for i in range(len(rsp)):
                rsp[i].filter = 1
        gym.set_asset_rigid_shape_properties(hand_asset, rsp)

    object_asset_list = []
    for object_type in object_type_list:
        obj_opts = gymapi.AssetOptions()
        if env_cfg["disable_gravity"]:
            obj_opts.disable_gravity = True
        obj_path = asset_root / asset_files_dict[object_type]
        obj_asset = gym.load_asset(sim, str(obj_path.parent), obj_path.name, obj_opts)
        object_asset_list.append(obj_asset)

    return hand_asset, object_asset_list


def init_object_pose(env_cfg, save_init_pose, grasp_cache_name):
    """Compute initial hand and object transforms from config.

    Returns (hand_pose, object_pose) as gymapi.Transform instances.
    """
    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(0, 0, env_cfg["leap_hand_start_z"])
    hand_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi)

    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3()

    pose_dx = env_cfg.get("override_object_init_x", -0.01)
    pose_dy = env_cfg.get("override_object_init_y", -0.04)

    obj_pose.p.x = hand_pose.p.x + pose_dx
    obj_pose.p.y = hand_pose.p.y + pose_dy

    object_z = 0.66 if save_init_pose else 0.65
    if 'internal' not in grasp_cache_name:
        object_z -= 0.02
    obj_pose.p.z = env_cfg.get("override_object_init_z", object_z)

    # Initial orientation: Euler RPY in radians (URDF convention, Rz @ Ry @ Rx).
    # Default identity. Override per-object to align an asset's natural frame with
    # the hand — e.g. a hammer with its handle along the object's local x-axis
    # needs ~[0, 0, pi/2] to lay the handle across the palm.
    rpy = env_cfg.get("override_object_init_rot", [0.0, 0.0, 0.0])
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    qx = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), r)
    qy = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), p)
    qz = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), y)
    obj_pose.r = qz * qy * qx

    return hand_pose, obj_pose
