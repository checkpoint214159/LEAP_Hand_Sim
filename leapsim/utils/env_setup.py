"""env_setup.py — IsaacGym environment construction helpers.

Free functions called once during env creation. No task state (self) — all
inputs are explicit, all outputs are returned. This makes them testable in
isolation and easy to reuse across task variants.
"""
from __future__ import annotations

import os

import numpy as np
from isaacgym import gymapi


def create_leap_assets(gym, sim, env_cfg, body_shape_indices, object_type_list, asset_files_dict):
    """Load hand URDF and all object URDFs into IsaacGym assets.

    Returns (hand_asset, object_asset_list).
    """
    asset_root      = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
    hand_asset_file = env_cfg['asset']['handAsset']

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

    hand_asset = gym.load_asset(sim, asset_root, hand_asset_file, hand_opts)

    if "leap_hand" in hand_asset_file:
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
        obj_asset = gym.load_asset(sim, asset_root, asset_files_dict[object_type], obj_opts)
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

    return hand_pose, obj_pose
