# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: 
# https://github.com/HaozhiQi/hora/blob/main/hora/tasks/leap_hand_hora.py
# --------------------------------------------------------

import os
import sys
from pathlib import Path
from leapsim.utils.rerun_vis import RerunVisualizer, RerunFrame, ObjectShape
from leapsim.utils.env_setup import create_leap_assets, init_object_pose
from attr import has
from importlib_metadata import itertools
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate, to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float, scale
from glob import glob
import math
import torchvision
import warnings
import matplotlib.pyplot as plt
from .base.vec_task import VecTaskRot
from collections import deque
from typing import Dict


# Map object.type → (rerun primitive, half-sizes). Half-sizes match the
# geometry in the URDFs under assets/ — keep these in sync if URDFs change.
_OBJECT_SHAPES: Dict[str, ObjectShape] = {
    'simple_tennis_ball': ('ellipsoid', (0.04,   0.04,   0.04)),
    'block':              ('box',       (0.0375, 0.0375, 0.0375)),
    'hammer':             ('box',       (0.025,  0.01,   0.08)),
}


def _object_shape_for(obj_type: str) -> ObjectShape:
    return _OBJECT_SHAPES.get(obj_type, ('box', (0.04, 0.04, 0.04)))


# Fingertip rigid-body link indices (index, thumb, middle, ring), matching the
# convention already used in leap_hand_grasp.py's contact/proximity code.
FINGERTIP_LOCAL_IDS = [4, 8, 12, 16]

# z_mode='local' shape-family ids (parsed from the object_type_list name prefix).
_SHAPE_FAMILY_ID = {'cuboid': 0, 'box': 0, 'cylinder': 1, 'sphere': 2, 'cone': 3, 'capsule': 4}

# Procedural primitive categories _setup_object_info discovers via
# <assets>/<category>/<subset>/*.urdf (tools/gen_primitive_objects.py).
# Adding a new category (e.g. a future 'pyramid') means adding it here AND to
# _SHAPE_FAMILY_ID/_closest_point_and_normal_local if it needs z_mode=local.
_PRIMITIVE_FAMILY_CATEGORIES = ['cuboid', 'cylinder', 'sphere', 'cone', 'capsule']


class LeapHandRot(VecTaskRot):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=None, force_render=None):
        # ── 1. Pre-super-init: config must be parsed before VecTask.__init__ ─────
        self.cfg = cfg
        self.env_cfg = self.cfg["env"]
        self.set_defaults()
        self._setup_domain_rand_cfg(self.env_cfg['randomization'])
        self._setup_priv_option_cfg(self.env_cfg['privInfo'])
        self._setup_object_info(self.env_cfg['object'])
        self._setup_reward_cfg(self.env_cfg['reward'])
        self.base_obj_scale = self.env_cfg['baseObjScale']
        self.save_init_pose = self.env_cfg['genGrasps']
        self.aggregate_mode = self.env_cfg['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.env_cfg['reset_height_threshold']
        self.grasp_cache_name = self.env_cfg['grasp_cache_name']
        self.evaluate = self.cfg['on_evaluation']
        self._setup_z_channel()  # oracle shape prior z (Stage-1 C / Stage-2); grows numObservations pre-super

        # ── 2. Sim creation (calls _create_envs internally) ──────────────────────
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless)
        if self.z_dim > 0:
            self._build_z_features()  # static per-env z vector (needs env_object_type_id + obj_scales)

        # ── 3. Post-super-init: viewer, timing ───────────────────────────────────
        self.record_video = self.env_cfg.get('record_video', False)
        if self.record_video:
            import atexit
            self._init_video_writer()
            atexit.register(self._close_video_writer)

        self.debug_viz = self.env_cfg['enableDebugVis']
        self.max_episode_length = self.env_cfg['episodeLength']
        self.dt = self.sim_params.dt
        self.control_dt = self.sim_params.dt * self.control_freq_inv

        if self.viewer:
            self.default_cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            self.default_cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, self.default_cam_pos, self.default_cam_target)

        # ── 4. IsaacGym tensor acquisition ───────────────────────────────────────
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor        = self.gym.acquire_dof_state_tensor(self.sim)
        dof_force_tensor        = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_tensor       = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces      = self.gym.acquire_net_contact_force_tensor(self.sim)

        # Shared sim-level views (hand + object live here)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.dof_state         = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.contact_forces    = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.num_bodies        = self.rigid_body_states.shape[1]

        # ── 5. Hand joint buffers ────────────────────────────────────────────────
        self.leap_hand_default_dof_pos = torch.zeros(self.num_leap_hand_dofs, dtype=torch.float, device=self.device)
        self.leap_hand_dof_state       = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_leap_hand_dofs]
        self.leap_hand_dof_pos         = self.leap_hand_dof_state[..., 0]
        self.leap_hand_dof_vel         = self.leap_hand_dof_state[..., 1]
        self.torques                   = gymtorch.wrap_tensor(dof_force_tensor).view(-1, self.num_leap_hand_dofs)

        self.global_counter      = 0
        self.prev_global_counter = 0
        self._refresh_gym()
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets       = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets        = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.actions            = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_torques       = self.torques.clone()
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.init_pose_buf      = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain

        # ── 6. Object state buffers ──────────────────────────────────────────────
        self.object_rpy              = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel_finite_diff = torch.zeros((self.num_envs, 3), device=self.device)
        self.rot_axis_buf            = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.object_init_pose_buf    = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        self.previous_object_rot     = torch.zeros((self.num_envs, 4), device=self.device)

        # ── 7. Random-force / perturbation buffers ───────────────────────────────
        self.force_scale              = self.env_cfg.get('forceScale', 0.0)
        self.random_force_prob_scalar = self.env_cfg.get('randomForceProbScalar', 0.0)
        self.force_decay              = to_torch(self.env_cfg.get('forceDecay', 0.99), dtype=torch.float, device=self.device)
        self.force_decay_interval     = self.env_cfg.get('forceDecayInterval', 0.08)
        self.rb_forces                = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.early_termination_buf    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── 8. Grasp cache ───────────────────────────────────────────────────────
        # Rows are 23 cols (16 hand DoF + 7 obj pose) or 24 cols (+ object_type_id).
        # 24-col caches are object-aware: at reset each env samples only rows saved
        # for the object instance it holds (cache_rows_by_obj). Legacy 23-col caches
        # (single-object families: ball/cube/hammer) keep the old any-row behavior.
        self.cache_rows_by_obj = {}
        if self.randomize_scale and self.scale_list_init:
            self.saved_grasping_states = {}
            for s in self.randomize_scale_list:
                print("loading from cache path:", f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy')
                arr = np.load(f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy')
                self.saved_grasping_states[str(s)] = torch.from_numpy(arr).float().to(self.device)
                if arr.shape[1] >= 24:
                    ids = arr[:, -1].astype(np.int64)  # obj id is always the last column
                    self.cache_rows_by_obj[str(s)] = {
                        int(oid): np.flatnonzero(ids == oid) for oid in np.unique(ids)
                    }
                    counts = {oid: len(rows) for oid, rows in self.cache_rows_by_obj[str(s)].items()}
                    missing = [t for t in range(len(self.object_type_list)) if t not in counts]
                    print(f"  object-aware cache: rows per object id {counts}"
                          + (f" — NO ROWS for object ids {missing} (those envs fall back to any row)"
                             if missing else ""))
        else:
            assert self.save_init_pose

        self.resample_randomizations(None)

        # ── 9. Statistics / debug ────────────────────────────────────────────────
        self.env_timeout_counter = to_torch(np.zeros((len(self.envs)))).long().to(self.device)
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000
        self.object_angvel_finite_diff_ep_buf = deque(maxlen=1000)
        self.object_angvel_finite_diff_mean   = torch.zeros(self.num_envs, device=self.device)
        self.setup_keyboard_events()

        if "actions_mask" in self.env_cfg:
            self.actions_mask = torch.tensor(self.env_cfg["actions_mask"], device=self.device)[None, :]
        else:
            self.actions_mask = torch.ones((1, self.num_leap_hand_dofs), device=self.device)

        if self.debug_viz:
            self.setup_plot()

        if "debug" in self.env_cfg:
            self.obs_list = []
            self.target_list = []
            if "record" in self.env_cfg["debug"]:
                self.record_duration = int(self.env_cfg["debug"]["record"]["duration"] / self.control_dt)
            if "actions_file" in self.env_cfg["debug"]:
                self.actions_list = torch.from_numpy(np.load(self.env_cfg["debug"]["actions_file"])).cuda()
                self.record_duration = self.actions_list.shape[0]

        # ── 10. Readable buffer inventory ─────────────────────────────────────────
        # self.hand and self.obj alias the tensors above — use them as a map of
        # what state this class owns without hunting through __init__.
        # Runtime methods still use the self.* names; migrate incrementally.
        import types
        self.hand = types.SimpleNamespace(
            # live views into IsaacGym DOF state tensor
            dof_pos          = self.leap_hand_dof_pos,
            dof_vel          = self.leap_hand_dof_vel,
            dof_state        = self.leap_hand_dof_state,
            torques          = self.torques,
            # control
            cur_targets      = self.cur_targets,
            prev_targets     = self.prev_targets,
            p_gain           = self.p_gain,
            d_gain           = self.d_gain,
            actions          = self.actions,
            # limits / defaults
            default_dof_pos  = self.leap_hand_default_dof_pos,
            dof_lower_limits = self.leap_hand_dof_lower_limits,
            dof_upper_limits = self.leap_hand_dof_upper_limits,
            # misc
            dof_vel_fd       = self.dof_vel_finite_diff,
            last_torques     = self.last_torques,
            rb_forces        = self.rb_forces,
            init_pose        = self.init_pose_buf,
        )
        self.obj = types.SimpleNamespace(
            # orientation tracking
            rpy              = self.object_rpy,
            angvel_fd        = self.object_angvel_finite_diff,
            prev_rot         = self.previous_object_rot,
            rot_axis         = self.rot_axis_buf,
            # per-env randomisation state
            scales           = self.obj_scales,
            friction         = self.object_friction_buf,
            # reset state
            init_state       = self.object_init_state,
            init_pose        = self.object_init_pose_buf,
            # IsaacGym index maps
            indices          = self.object_indices,
            rb_handles       = self.object_rb_handles,
        )

        # ── 11. Rerun visualizer ──────────────────────────────────────────────────
        rr_cfg = self.env_cfg.get('rerun', {})
        self._rerun_vis: RerunVisualizer | None = None
        asset_root   = Path(__file__).parent.parent.parent
        self.hand_urdf_path    = asset_root / self.env_cfg['asset']['handAsset']
        object_shape = _object_shape_for(self.env_cfg['object']['type'])

        # Rerun records ONE env per window, so to view a healthy sample of objects
        # we ROTATE the observed env across windows (round-robin) instead of
        # perturbing the sim RNG (which would break training reproducibility and
        # still only stream one object per run). Default: one env per distinct
        # object id (covers every instance held in the run). Overrides:
        # rerun.env_indices=[...] for an explicit set, or sample_per_object=false
        # + env_idx=N for a single fixed env.
        ids_np = self.env_object_type_id.cpu().numpy()
        if rr_cfg.get('env_indices', None) is not None:
            self._rr_env_indices = [int(i) for i in rr_cfg['env_indices']]
        elif rr_cfg.get('sample_per_object', True):
            self._rr_env_indices = [int(np.where(ids_np == o)[0][0])
                                    for o in range(len(self.object_type_list))
                                    if (ids_np == o).any()]
        else:
            self._rr_env_indices = [int(rr_cfg.get('env_idx', 0))]

        # Per-slot metadata: render each observed env's ACTUAL object instance at
        # its own actor scale. Rendering obj_0's mesh for every env (the old bug)
        # shows an imposter shape — fingers appear to grip mid-air around a
        # floating wrong-shaped object.
        observed = []
        for idx in self._rr_env_indices:
            oid   = int(self.env_object_type_id[idx].item())
            otype = self.object_type_list[oid]
            if self.randomize_scale and self.obj_scales.numel() == self.num_envs:
                oscale = float(self.obj_scales[idx].item())
            else:
                oscale = float(self.base_obj_scale)
            observed.append(dict(env_idx=idx, object_id=oid, object_type=otype,
                                 object_urdf=asset_root / self.asset_files_dict[otype],
                                 object_scale=oscale))

        if rr_cfg.get('enabled', False):
            hand_handle  = self.gym.find_actor_handle(self.envs[0], 'hand')
            link_names   = self.gym.get_actor_rigid_body_names(self.envs[0], hand_handle)
            print(f"[rerun] streaming {len(observed)} env(s), one per window (round-robin):")
            for s in observed:
                print(f"[rerun]   env {s['env_idx']:5d} → '{s['object_type']}' "
                      f"(id {s['object_id']}) @ scale {s['object_scale']:.3f}")
            self._rerun_vis = RerunVisualizer(rr_cfg, self.hand_urdf_path, link_names,
                                              object_shape, observed)

    def set_camera(self, position, lookat):
        """ 
        Set camera position and direction
        """

        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, self.envs[self.lookat_id], cam_pos, cam_target)

    def lookat(self, i):
        look_at_pos = self.hand_pos[i, :].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    def render(self):
        super().render()

        if self.viewer:
            if not self.free_cam:
                self.lookat(self.lookat_id)
            
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if not self.free_cam:
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs
                        self.lookat(self.lookat_id)

                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    
                    if self.free_cam:
                        self.gym.viewer_camera_look_at(self.viewer, None, self.default_cam_pos, self.default_cam_target)

    def setup_keyboard_events(self):
        self.lookat_id = 0
        self.free_cam = False
        self.lookat_vec = torch.tensor([0.4, -0.2, 0.1], requires_grad=False, device=self.device)

        if self.viewer is None:
            return
        
        # subscribe to keyboard shortcuts
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_F, "free_cam")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")

    def resample_randomizations(self, env_ids):
        if "joint_noise" not in self.env_cfg["randomization"]:
            return

        self.joint_noise_cfg = self.env_cfg["randomization"]["joint_noise"]

        if env_ids is None:
            self.joint_noise_iid_scale = torch.zeros((self.num_envs, self.num_leap_hand_dofs), device=self.device)
            self.joint_noise_constant_offset = torch.zeros((self.num_envs, self.num_leap_hand_dofs), device=self.device)
            self.joint_noise_outlier_scale = torch.zeros((self.num_envs, self.num_leap_hand_dofs), device=self.device)
            self.joint_noise_outlier_rate = torch.zeros((self.num_envs, self.num_leap_hand_dofs), device=self.device)
            env_ids = torch.arange(self.num_envs, device=self.device)

        if "iid" in self.joint_noise_cfg:
            low, high = self.joint_noise_cfg["iid"]["scale_range"]
            self.joint_noise_iid_scale[env_ids] = torch.rand((env_ids.shape[0], self.num_leap_hand_dofs), device=self.device) * (high - low) + low
            self.joint_noise_iid_type = self.joint_noise_cfg["iid"]["type"]

        if "constant_offset" in self.joint_noise_cfg:
            low, high = self.joint_noise_cfg["constant_offset"]["range"]
            self.joint_noise_constant_offset[env_ids] = torch.rand((env_ids.shape[0], self.num_leap_hand_dofs), device=self.device) * (high - low) + low

        if "outlier" in self.joint_noise_cfg:
            low, high = self.joint_noise_cfg["outlier"]["scale_range"]
            self.joint_noise_outlier_scale[env_ids] = torch.rand((env_ids.shape[0], self.num_leap_hand_dofs), device=self.device) * (high - low) + low
            
            low, high = self.joint_noise_cfg["outlier"]["rate_range"]
            self.joint_noise_outlier_rate[env_ids] = torch.rand((env_ids.shape[0], self.num_leap_hand_dofs), device=self.device) * (high - low) + low
            
            self.joint_noise_outlier_type = self.joint_noise_cfg["outlier"]["type"]

    def setup_plot(self):   
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-20, 20)
        self.ydata = deque(maxlen=100) # Plot 5 seconds of data
        self.ydata2 = deque(maxlen=100)
        (self.ln,) = self.ax.plot(range(len(self.ydata)), list(self.ydata), animated=True)
        (self.ln2,) = self.ax.plot(range(len(self.ydata2)), list(self.ydata2), animated=True)
        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)

    def set_defaults(self):
        if "record_video" not in self.env_cfg:
            self.env_cfg["record_video"] = False
        if "video_dir" not in self.env_cfg:
            self.env_cfg["video_dir"] = "videos"
        if self.env_cfg["record_video"]:
            # Env.__init__ reads cfg["enableCameraSensors"] (top-level, not cfg["env"])
            # to decide whether to force graphics_device_id to -1.  Set it here so the
            # graphics device stays valid when we need camera sensors.
            self.cfg["enableCameraSensors"] = True

        if "include_pd_gains" not in self.env_cfg:
            self.env_cfg["include_pd_gains"] = False

        if "include_friction_coefficient" not in self.env_cfg:
            self.env_cfg["include_friction_coefficient"] = False

        if "include_obj_scales" not in self.env_cfg:
            self.env_cfg["include_obj_scales"] = False

        if "leap_hand_start_z" not in self.env_cfg:
            self.env_cfg["leap_hand_start_z"] = 0.5
        
        if "grasp_dof_search_radius" not in self.env_cfg:
            self.env_cfg["grasp_dof_search_radius"] = 0.25

        if "obs_mask" not in self.env_cfg:
            self.env_cfg["obs_mask"] = None

        if "include_targets" not in self.env_cfg:
            self.env_cfg["include_targets"] = True
        
        if "include_obj_pose" not in self.env_cfg:
            self.env_cfg["include_obj_pose"] = False

        if "include_history" not in self.env_cfg:
            self.env_cfg["include_history"] = True

        if "joint_limits" not in self.env_cfg["randomization"]:
            self.env_cfg["randomization"]["joint_limits"] = 0

        if "mask_body_collision" not in self.env_cfg:
            self.env_cfg["mask_body_collision"] = {}        
    
        if "disable_actions" not in self.env_cfg:
            self.env_cfg["disable_actions"] = False

        if "disable_gravity" not in self.env_cfg:
            self.env_cfg["disable_gravity"] = False

        if "disable_object_collision" not in self.env_cfg:
            self.env_cfg["disable_object_collision"] = False

        if "disable_resets" not in self.env_cfg:
            self.env_cfg["disable_resets"] = False

        if "disable_self_collision" not in self.env_cfg:
            self.env_cfg["disable_self_collision"] = False

        if "rotation_axis" not in self.env_cfg:
            self.rotation_axis = torch.tensor([0., 0., 1.])
        else:
            self.rotation_axis = torch.tensor(self.env_cfg["rotation_axis"])

        # Multiple rigid shapes correspond to a rigid body, the indices can be found using get_asset_rigid_body_shape_indices
        self.body_shape_indices = [ 
            ( 0 ,  17 ),
            ( 17 ,  1 ),
            ( 18 ,  5 ),
            ( 23 ,  5 ),
            ( 28 ,  3 ),
            ( 31 ,  1 ),
            ( 32 ,  4 ),
            ( 36 ,  9 ),
            ( 45 ,  3 ),
            ( 48 ,  1 ),
            ( 49 ,  5 ),
            ( 54 ,  5 ),
            ( 59 ,  3 ),
            ( 62 ,  1 ),
            ( 63 ,  5 ),
            ( 68 ,  5 ),
            ( 73 ,  3 )
        ]

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # ── Asset loading ─────────────────────────────────────────────────────
        asset_root = Path(__file__).parent.parent.parent
        self.hand_asset, self.object_asset_list = create_leap_assets(
            self.gym, self.sim, asset_root, self.env_cfg,
            self.body_shape_indices, self.object_type_list, self.asset_files_dict,
        )

        # ── DOF properties (limits + randomised PD gains) ─────────────────────
        self.num_leap_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        leap_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.leap_hand_dof_lower_limits = []
        self.leap_hand_dof_upper_limits = []
        for i in range(self.num_leap_hand_dofs):
            self.leap_hand_dof_lower_limits.append(leap_hand_dof_props['lower'][i])
            self.leap_hand_dof_upper_limits.append(leap_hand_dof_props['upper'][i])
            leap_hand_dof_props['effort'][i]    = 0.5
            leap_hand_dof_props['stiffness'][i] = self.env_cfg['controller']['pgain']
            leap_hand_dof_props['damping'][i]   = self.env_cfg['controller']['dgain']
            leap_hand_dof_props['friction'][i]  = 0.01
            leap_hand_dof_props['armature'][i]  = 0.001

        self.leap_hand_dof_lower_limits = to_torch(self.leap_hand_dof_lower_limits, device=self.device)
        self.leap_hand_dof_upper_limits = to_torch(self.leap_hand_dof_upper_limits, device=self.device)
        # per-env limit jitter
        self.leap_hand_dof_lower_limits = self.leap_hand_dof_lower_limits.repeat((self.num_envs, 1))
        self.leap_hand_dof_lower_limits += (2 * torch.rand_like(self.leap_hand_dof_lower_limits) - 1) * self.env_cfg["randomization"]["joint_limits"]
        self.leap_hand_dof_upper_limits = self.leap_hand_dof_upper_limits.repeat((self.num_envs, 1))
        self.leap_hand_dof_upper_limits += (2 * torch.rand_like(self.leap_hand_dof_upper_limits) - 1) * self.env_cfg["randomization"]["joint_limits"]

        # ── Initial poses ─────────────────────────────────────────────────────
        hand_pose, obj_pose = self._init_object_pose()

        # ── Aggregate bookkeeping ─────────────────────────────────────────────
        self.num_leap_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_leap_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_leap_hand_bodies + 2
        max_agg_shapes = self.num_leap_hand_shapes + 2

        leap_hand_rb_count  = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.object_rb_handles = list(range(leap_hand_rb_count, leap_hand_rb_count + 1))

        self.envs               = []
        self.object_init_state  = []
        self.hand_indices       = []
        self.object_indices     = []
        self.obj_scales         = []
        self.object_friction_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

        # Per-env object identity (index into object_type_list). Grasp caches are
        # per-object: rows are saved with this id (col 24) and restored only into
        # envs holding the same object. object_ids_from_cache forces env i to hold
        # the object of cache row i — used by tools/filter_grasp_caches.py so the
        # row↔env mapping is object-correct.
        env_object_type_id = []
        forced_ids = None
        if "object_ids_from_cache" in self.env_cfg:
            _cache_arr = np.load(self.env_cfg["object_ids_from_cache"])
            if _cache_arr.shape[1] >= 24:
                forced_ids = _cache_arr[:, -1].astype(np.int64)  # id = last column
            else:
                print(f"[object_ids_from_cache] {self.env_cfg['object_ids_from_cache']} "
                      f"has no object-id column (23-col legacy cache); ignoring.")

        # object_id_whitelist restricts which instances of the family envs may
        # hold (e.g. [1] → every env holds cuboid_1: single-object control runs).
        # Ids stay family-relative, so cache row matching keeps working.
        id_whitelist = self.env_cfg.get("object_id_whitelist", None)
        if id_whitelist is not None:
            id_whitelist = [int(x) for x in id_whitelist]
            assert all(0 <= x < len(self.object_type_list) for x in id_whitelist), \
                f"object_id_whitelist {id_whitelist} out of range for {self.object_type_list}"
            print(f"[object_id_whitelist] envs restricted to ids {id_whitelist} "
                  f"({[self.object_type_list[x] for x in id_whitelist]})")

        # ── Per-env creation loop ─────────────────────────────────────────────
        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # Hand actor
            # collision filter = -1 to use asset collision filters from the URDF loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, leap_hand_dof_props)
            self.hand_indices.append(self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM))

            # Object actor
            if forced_ids is not None:
                object_type_id = int(forced_ids[i % len(forced_ids)])
            elif id_whitelist is not None:
                object_type_id = id_whitelist[i % len(id_whitelist)]  # round-robin: exact balance
            else:
                object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            env_object_type_id.append(int(object_type_id))
            collision_group = -(i + 2) if self.env_cfg["disable_object_collision"] else i
            object_handle = self.gym.create_actor(
                env_ptr, self.object_asset_list[object_type_id], obj_pose, 'object', collision_group, 0, 0,
            )
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            self.object_indices.append(self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM))

            # Per-actor domain randomisation
            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                # ±jitter around the list value (upstream hardcoded 0.025). Pinnable
                # via scale_list_jitter=0 so cache replays restore the EXACT scale
                # the grasp was settled at — a cage grasp restored ±2.5% either
                # wedges (PhysX pops it) or sags loose onto the palm.
                jitter = float(self.env_cfg.get("scale_list_jitter", 0.025))
                obj_scale = np.random.uniform(
                    self.randomize_scale_list[i % num_scales] - jitter,
                    self.randomize_scale_list[i % num_scales] + jitter,
                )
                if "randomize_scale_factor" in self.env_cfg:
                    obj_scale *= np.random.uniform(*self.env_cfg["randomize_scale_factor"])
                self.obj_scales.append(obj_scale)
            # print("env_ptr, object_handle, obj_scale?", env_ptr, object_handle, obj_scale)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)

            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                prop[0].com.x = np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)
                prop[0].com.y = np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)
                prop[0].com.z = np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                for actor_handle in (hand_actor, object_handle):
                    props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
                    for p in props:
                        p.friction = rand_friction
                    self.gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, props)
                obj_friction = rand_friction
            self.object_friction_buf[i] = obj_friction

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        # ── Post-loop: convert index lists to tensors ─────────────────────────
        self.env_object_type_id = to_torch(env_object_type_id, dtype=torch.long, device=self.device)
        # [obj-sampling diag] ground-truth per-object env distribution. If every
        # env holds the same id this print collapses to one bucket — that would be
        # a real sampling bug. A ~even spread confirms diverse loading; Rerun still
        # only streams env `rerun.env_idx` (default 0), so a single run's replay
        # shows just that one env's object (fixed by seed) — not all of them.
        _hist = {self.object_type_list[o]: int((self.env_object_type_id == o).sum().item())
                 for o in range(len(self.object_type_list))}
        print(f"[obj-sampling diag] {self.num_envs} envs → per-object counts: {_hist}")
        self.obj_scales         = torch.tensor(self.obj_scales, device=self.device)
        self.object_init_state  = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles  = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices       = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices     = to_torch(self.object_indices, dtype=torch.long, device=self.device)

        # ── Video camera (env 0 only, if recording enabled) ───────────────────
        if self.env_cfg.get('record_video', False):
            cam_props = gymapi.CameraProperties()
            cam_props.width  = 640
            cam_props.height = 480
            self.cam_handle = self.gym.create_camera_sensor(self.envs[0], cam_props)
            # Env 0: hand base at (0,0,0.5), object at ~(−0.03, 0.04, 0.57).
            self.gym.set_camera_location(
                self.cam_handle, self.envs[0],
                gymapi.Vec3(0.5, 0.0, 0.9),
                gymapi.Vec3(0.0, 0.0, 0.58),
            )
    
    def reset_idx(self, env_ids):
        if self.env_cfg.get("skip_mass_props", False):
            # Diagnostic: skip ALL per-actor property calls in reset (both
            # branches touch get_actor_rigid_body_properties, which interferes
            # with tensor-API state applies on the GPU pipeline).
            pass
        elif self.randomize_mass:
            lower, upper = self.randomize_mass_lower, self.randomize_mass_upper

            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                for p in prop:
                    p.mass = np.random.uniform(lower, upper)
                self.gym.set_actor_rigid_body_properties(env, handle, prop)
        else:
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        self.resample_randomizations(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            
            if "sampled_pose_idx" in self.env_cfg:
                sampled_pose_idx = np.ones(len(s_ids), dtype=np.int32) * self.env_cfg["sampled_pose_idx"]
            elif self.env_cfg.get("sequential_pose_idx", False):
                # Deterministic env↔row mapping (env i always restores cache row
                # i mod len). Used by tools/filter_grasp_caches.py to attribute
                # a drop to the specific cache row that caused it (pair with
                # object_ids_from_cache so env i also holds row i's object).
                sampled_pose_idx = s_ids.cpu().numpy() % self.saved_grasping_states[scale_key].shape[0]
            elif scale_key in self.cache_rows_by_obj:
                # Object-aware cache: each env samples only rows saved for the
                # object instance it holds. A grasp is object-specific — restoring
                # a rod grasp onto a puck leaves the fingers wrapping air.
                pools = self.cache_rows_by_obj[scale_key]
                env_oids = self.env_object_type_id[s_ids].cpu().numpy()
                sampled_pose_idx = np.empty(len(s_ids), dtype=np.int64)
                for oid in np.unique(env_oids):
                    m = env_oids == oid
                    pool = pools.get(int(oid))
                    if pool is not None and len(pool) > 0:
                        sampled_pose_idx[m] = np.random.choice(pool, size=int(m.sum()))
                    else:  # no rows for this object — fall back to any row
                        sampled_pose_idx[m] = np.random.randint(
                            self.saved_grasping_states[scale_key].shape[0], size=int(m.sum()))
            else:
                sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))

            sp = self.saved_grasping_states[scale_key][sampled_pose_idx]
            if sp.shape[1] >= 40:
                # v4 cache: [16 dof pos | 16 PD targets | 7 obj pose | 1 obj id].
                # Restoring the TARGETS re-arms the grip: the PD error between the
                # deflected finger positions and their held targets is the squeeze
                # force. Restoring positions alone (targets := positions) leaves
                # zero PD error → zero grip → force-closure grasps drop the object.
                hand_pos, hand_tgt, obj_pose = sp[:, :16], sp[:, 16:32], sp[:, 32:39]
            else:
                # Legacy 23/24-col cache: positions double as targets. Sufficient
                # only for palm/geometry-supported grasps.
                hand_pos, hand_tgt, obj_pose = sp[:, :16], sp[:, :16], sp[:, 16:23]
            self.root_state_tensor[self.object_indices[s_ids], :7] = obj_pose
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0

            self.leap_hand_dof_pos[s_ids, :] = hand_pos
            self.leap_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, :self.num_leap_hand_dofs] = hand_tgt
            self.cur_targets[s_ids, :self.num_leap_hand_dofs] = hand_tgt
            self.init_pose_buf[s_ids, :] = hand_pos.clone()
            self.object_init_pose_buf[s_ids, :] = obj_pose.clone()

        # FULL-tensor applies, NOT the *_indexed variants: calling
        # get/set_actor_rigid_body_properties (the mass-randomization loop above)
        # in the same step silently breaks set_actor_root_state_tensor_indexed on
        # the GPU pipeline — the staged rows never reach the sim, so cache
        # restores no-op'd (objects stayed at creation pose) in every GPU run
        # while CPU-pipeline runs restored fine. The full-tensor calls are immune
        # (minimal repro: tools/probe_minimal_setter.py gpu massset). Semantics
        # are equivalent: the tensors hold refreshed current state for non-reset
        # envs and the staged restore for reset ones.
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.prev_targets))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
        # The applies above are POISONED on the GPU pipeline whenever
        # get/set_actor_rigid_body_properties ran this step (the mass loop at the
        # top of this method — both branches!): the staged state never reaches
        # the sim, so cache restores silently no-op'd in every GPU run ever
        # (objects stayed at creation pose) while CPU-pipeline runs were fine.
        # Repro: tools/probe_minimal_setter.py gpu massset, and
        # tools/probe_restore_orientation.py DIRECT ±skip_mass_props.
        # Fix: re-apply in update_low_level_control at substep 1 of the next
        # step — the first simulate clears the poison, and no property calls or
        # refreshes run in between, so the tensors still hold the staged rows.
        self._reapply_countdown = 2
        self._reapply_obj_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self._reapply_hand_indices = self.hand_indices[env_ids].to(torch.int32)

        mask = self.progress_buf[env_ids] > 0
        done_vals = self.object_angvel_finite_diff_mean[env_ids][mask]
        self.object_angvel_finite_diff_ep_buf.extend(list(done_vals))
        # per-category angvel accumulation (eval diagnostic: reveals whether an
        # easy family — spheres — is masking a hard one in the mix aggregate)
        if "print_object_angvel" in self.env_cfg:
            if not hasattr(self, "_angvel_cat_acc"):
                self._env_family = [self.object_type_list[int(o)].rsplit('_', 1)[0]
                                    for o in self.env_object_type_id.cpu().numpy()]
                self._angvel_cat_acc = {}
            for e, v in zip(env_ids[mask].cpu().numpy(), done_vals.detach().cpu().numpy()):
                fam = self._env_family[int(e)]
                s, c = self._angvel_cat_acc.get(fam, (0.0, 0))
                self._angvel_cat_acc[fam] = (s + float(v), c + 1)
        self.object_angvel_finite_diff_mean[env_ids] = 0

        if "print_object_angvel" in self.env_cfg and len(self.object_angvel_finite_diff_ep_buf) > 0:
            overall = sum(self.object_angvel_finite_diff_ep_buf) / len(self.object_angvel_finite_diff_ep_buf)
            percat = {f: round(s / c, 4) for f, (s, c) in sorted(self._angvel_cat_acc.items())}
            print("mean object angvel: ", overall, " | per-category:", percat)

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
        
    def get_joint_noise(self):
        tensor = torch.zeros_like(self.leap_hand_dof_pos)

        if "joint_noise" not in self.env_cfg["randomization"]:
            return tensor

        if not self.joint_noise_cfg["add_noise"]:
            return tensor

        if "iid" in self.joint_noise_cfg:
            if self.joint_noise_iid_type == "gaussian":
                tensor = tensor + torch.randn_like(tensor) * self.joint_noise_iid_scale
            elif self.joint_noise_iid_type == "uniform":
                tensor = tensor + (2 * torch.rand(tensor) - 1) * self.joint_noise_iid_scale
            
        if "constant_offset" in self.joint_noise_cfg:
            tensor = tensor + self.joint_noise_constant_offset

        if "outlier" in self.joint_noise_cfg:
            outlier_noise_prob = self.joint_noise_outlier_rate * self.control_dt 
            outlier_mask = torch.rand_like(outlier_noise_prob) <= outlier_noise_prob
            
            if self.joint_noise_outlier_type == "gaussian":
                tensor = tensor + torch.randn_like(tensor) * self.joint_noise_outlier_scale * outlier_mask
            elif self.joint_noise_outlier_type == "uniform":
                tensor = tensor + (2 * torch.rand(tensor) - 1) * self.joint_noise_outlier_scale * outlier_mask

        return tensor

    def compute_observations(self):
        # NO _refresh_gym here. post_physics_step already refreshed at its top,
        # and this method runs AFTER reset_idx: refreshing would overwrite the
        # rows reset_idx just staged, which must survive untouched until the
        # substep-1 re-apply in update_low_level_control (the fix for the GPU
        # property-call poison — see reset_idx).
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = self.get_joint_noise()
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.leap_hand_dof_pos, self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits
        ).clone().unsqueeze(1)

        self.cur_obs_buf_noisy = cur_obs_buf.squeeze(1).clone()
        self.cur_obs_buf_clean = unscale(
            self.leap_hand_dof_pos, self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits
        ).clone()

        if hasattr(self, "obs_list"):
            self.obs_list.append(cur_obs_buf[0].clone())
            self.target_list.append(self.cur_targets[0].clone().squeeze())

            if self.global_counter == self.record_duration - 1:
                self.obs_list = torch.stack(self.obs_list, dim=0)
                self.obs_list = self.obs_list.cpu().numpy()

                self.target_list = torch.stack(self.target_list, dim=0)
                self.target_list = self.target_list.cpu().numpy()

                if "actions_file" in self.env_cfg["debug"]:
                    actions_file = os.path.basename(self.env_cfg["debug"]["actions_file"])
                    folder = os.path.dirname(self.env_cfg["debug"]["actions_file"])
                    suffix = "_".join(actions_file.split("_")[1:])
                    joints_file = os.path.join(folder, "joints_sim_{}".format(suffix)) 
                    target_file = os.path.join(folder, "targets_sim_{}".format(suffix))
                else:
                    suffix = self.env_cfg["debug"]["record"]["suffix"]
                    joints_file = "debug/joints_sim_{}.npy".format(suffix)
                    target_file = "debug/targets_sim_{}.npy".format(suffix)

                np.save(joints_file, self.obs_list)
                np.save(target_file, self.target_list) 
                exit()

        cur_tar_buf = self.cur_targets[:, None]
        
        if self.env_cfg["include_targets"]:
            cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)

        if self.env_cfg["include_obj_pose"]:
            cur_obs_buf = torch.cat([
                cur_obs_buf, 
                self.object_pos.unsqueeze(1), 
                self.object_rpy.unsqueeze(1)
            ], dim=-1)

        if self.env_cfg["include_obj_scales"]:
            cur_obs_buf = torch.cat([
                cur_obs_buf, 
                self.obj_scales.unsqueeze(1).unsqueeze(1), 
            ], dim=-1)
        
        if self.env_cfg["include_pd_gains"]:
            cur_obs_buf = torch.cat([
                cur_obs_buf, 
                self.p_gain.unsqueeze(1), 
                self.d_gain.unsqueeze(1)
            ], dim=-1)
        
        if self.env_cfg["include_friction_coefficient"]:
            cur_obs_buf = torch.cat([
                cur_obs_buf,
                self.object_friction_buf.unsqueeze(1).unsqueeze(1)
            ], dim=-1)

        if "phase_period" in self.env_cfg:
            cur_obs_buf = torch.cat([cur_obs_buf, self.phase[:, None]], dim=-1)

        # Oracle shape prior z (Stage-1 C / Stage-2). z0/z1 are static per episode;
        # 'local' is recomputed fresh every call (see _current_z). Either way it
        # rides the obs history like include_obj_scales (appended per-step, then
        # stacked across the 3-step window). z_dim==0 (z_mode=none) is a no-op →
        # identical to the proprio baseline.
        if self.z_dim > 0:
            self._z_obs_start = cur_obs_buf.shape[-1]
            z_now = self._current_z()
            cur_obs_buf = torch.cat([cur_obs_buf, z_now[:, None, :]], dim=-1)

        if self.env_cfg["include_history"]:
            at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

            # refill the initialized buffers
            self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
                self.leap_hand_dof_pos[at_reset_env_ids], self.leap_hand_dof_lower_limits[at_reset_env_ids],
                self.leap_hand_dof_upper_limits[at_reset_env_ids]
            ).clone().unsqueeze(1)

            if self.env_cfg["include_targets"]:
                self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.leap_hand_dof_pos[at_reset_env_ids].unsqueeze(1)

            # keep z correct in the freshly-reset history frames — z encodes the
            # object's shape/contact state, so stale z for the first 3 steps would
            # feed the wrong value at episode start. Reuses z_now (computed once
            # above, not recomputed) — for 'local' this is "current config repeated
            # 3x," the only sensible fill since there's no meaningful pre-episode value.
            if self.z_dim > 0:
                zs = self._z_obs_start
                self.obs_buf_lag_history[at_reset_env_ids, :, zs:zs + self.z_dim] = \
                    z_now[at_reset_env_ids].unsqueeze(1)
            
            t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone() # attach three timesteps of history

            self.obs_buf[:, :t_buf.shape[1]] = t_buf

            # self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
            self.at_reset_buf[at_reset_env_ids] = 0
        else:
            self.obs_buf = cur_obs_buf.clone().squeeze(1)

        if self.env_cfg["obs_mask"] is not None:
            self.obs_buf = self.obs_buf * torch.tensor(self.env_cfg["obs_mask"], device=self.device)[None, :]

    def compute_reward(self, actions):
        self.rot_axis_buf[:, -1] = -1
        # pose diff penalty
        pose_diff_penalty = ((self.leap_hand_dof_pos - self.init_pose_buf) ** 2).sum(-1)
        # work and torque penalty
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2
        obj_linv_pscale = self.object_linvel_penalty_scale
        pose_diff_pscale = self.pose_diff_penalty_scale
        torque_pscale = self.torque_penalty_scale
        work_pscale = self.work_penalty_scale

        self.rew_buf[:], log_r_reward, olv_penalty = compute_hand_reward(
            self.object_linvel, obj_linv_pscale,
            self.object_angvel, self.rot_axis_buf, self.rotate_reward_scale,
            self.angvel_clip_max, self.angvel_clip_min,
            pose_diff_penalty, pose_diff_pscale,
            torque_penalty, torque_pscale,
            work_penalty, work_pscale,
        )

        if "additional_rewards" in self.env_cfg:
            for reward_name, reward_scale in self.env_cfg["additional_rewards"].items():
                reward_value = eval("self.reward_{}()".format(reward_name)) * reward_scale
                self.extras["reward_{}".format(reward_name)] = reward_value.mean()
                self.rew_buf += reward_value

        self.reset_buf[:] = self.check_termination(self.object_pos)
        
        if self.env_cfg["disable_resets"]:
            # only consider ep length and early termination
            self.reset_buf = self.progress_buf >= self.max_episode_length 

        self.reset_buf = self.reset_buf | self.early_termination_buf

        self.extras['rotation_reward'] = log_r_reward.mean()
        self.extras['object_linvel_penalty'] = olv_penalty.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = self.object_angvel[:, 0].mean()
        self.extras['pitch'] = self.object_angvel[:, 1].mean()
        self.extras['yaw'] = self.object_angvel[:, 2].mean()
        self.extras['yaw_finite_diff'] = self.object_angvel_finite_diff[:, 2].mean()

        if self.evaluate:
            finished_episode_mask = self.reset_buf == 1
            self.stat_sum_rewards += self.rew_buf.sum()
            self.stat_sum_rotate_rewards += log_r_reward.sum()
            self.stat_sum_torques += self.torques.abs().sum()
            self.stat_sum_obj_linvel += (self.object_linvel ** 2).sum(-1).sum()
            self.stat_sum_episode_length += (self.reset_buf == 0).sum()
            self.env_evaluated += (self.reset_buf == 1).sum()
            self.env_timeout_counter[finished_episode_mask] += 1
            info = f'progress {self.env_evaluated} / {self.max_evaluate_envs} | ' \
                   f'reward: {self.stat_sum_rewards / self.env_evaluated:.2f} | ' \
                   f'eps length: {self.stat_sum_episode_length / self.env_evaluated:.2f} | ' \
                   f'rotate reward: {self.stat_sum_rotate_rewards / self.env_evaluated:.2f} | ' \
                   f'lin vel (x100): {self.stat_sum_obj_linvel * 100 / self.stat_sum_episode_length:.4f} | ' \
                   f'command torque: {self.stat_sum_torques / self.stat_sum_episode_length:.2f}'
            if self.env_evaluated >= self.max_evaluate_envs:
                exit()
    
    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self.early_termination_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
                
            self.plot_callback()

        if self.record_video:
            self._capture_frame()

        if self._rerun_vis is not None:
            idxs = self._rr_env_indices
            rb   = self.rigid_body_states[idxs, :self.num_leap_hand_bodies].cpu().numpy()
            opos = self.object_pos[idxs].cpu().numpy()
            orot = self.object_rot[idxs].cpu().numpy()
            tgt  = self.cur_targets[idxs, :self.num_leap_hand_dofs].cpu().numpy()
            dof  = self.leap_hand_dof_pos[idxs].cpu().numpy()
            rst  = self.reset_buf[idxs].cpu().numpy()
            linv = self.object_linvel[idxs].norm(dim=-1).cpu().numpy()
            angv = self.object_angvel[idxs].norm(dim=-1).cpu().numpy()
            # One frame per observed env; the visualizer selects the active slot
            # per window (round-robin). Cheap: len(idxs) is small (≤ #objects).
            self._rerun_vis.tick([RerunFrame(
                rb_states=rb[k], obj_pos=opos[k], obj_rot=orot[k],
                targets=tgt[k], dof_pos=dof[k], reset=bool(rst[k]),
                linvel_mag=float(linv[k]), angvel_mag=float(angv[k]),
            ) for k in range(len(idxs))])

        # Optional synergy-analysis dump (Stage 4): accumulate the 16-DoF joint
        # trajectory + per-step yaw for offline eigengrasp PCA / limit-cycle work.
        # One-shot: writes an .npz after dump_dof_steps steps, then stops. Lazy-init
        # so no __init__ change; eval exits cleanly so no atexit fragility.
        dump_path = self.env_cfg.get('dump_dof_traj', None)
        if dump_path is not None:
            if not hasattr(self, '_dump_dof_buf'):
                self._dump_dof_buf, self._dump_yaw_buf = [], []
                self._dump_dof_max = int(self.env_cfg.get('dump_dof_steps', 400))
            if self._dump_dof_buf is not None:
                self._dump_dof_buf.append(self.leap_hand_dof_pos.detach().cpu().numpy().copy())
                self._dump_yaw_buf.append(self.object_angvel_finite_diff[:, 2].detach().cpu().numpy().copy())
                if len(self._dump_dof_buf) >= self._dump_dof_max:
                    np.savez(dump_path, dof=np.stack(self._dump_dof_buf),
                             yaw=np.stack(self._dump_yaw_buf))
                    print(f"[synergy] dumped {len(self._dump_dof_buf)} steps x {self.num_envs} envs -> {dump_path}")
                    self._dump_dof_buf = None   # one-shot

    def _init_video_writer(self):
        """
        NOTE: this may still be broken, we did not have sufficient time to test it out.
        in the end i may resort to another third party package like rerun.
        """
        import subprocess
        import shutil
        import datetime
        import os
        if shutil.which('ffmpeg') is None:
            print('[Video] ffmpeg not found — disabling video capture')
            self.record_video = False
            return
        video_dir = self.env_cfg['video_dir']
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        task_name = self.cfg.get('name', 'task')
        self._video_path = os.path.join(video_dir, f'{task_name}_{timestamp}.mp4')
        self._video_proc = subprocess.Popen(
            ['ffmpeg', '-y',
             '-f', 'rawvideo', '-vcodec', 'rawvideo',
             '-s', '640x480', '-pix_fmt', 'rgba', '-r', '20',
             '-i', 'pipe:',
             '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', self._video_path],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._video_frame_count = 0
        print(f'[Video] Recording to {self._video_path}')

    def _capture_frame(self):
        self.gym.render_all_camera_sensors(self.sim)
        frame = self.gym.get_camera_image(
            self.sim, self.envs[0], self.cam_handle, gymapi.IMAGE_COLOR
        )
        self._video_proc.stdin.write(frame.tobytes())
        self._video_frame_count += 1

    def _close_video_writer(self):
        if hasattr(self, '_video_proc') and self._video_proc is not None:
            self._video_proc.stdin.close()
            self._video_proc.wait()
            print(f'[Video] Saved {self._video_frame_count} frames → {self._video_path}')
            self._video_proc = None


    def plot_callback(self):
        self.fig.canvas.restore_region(self.bg)

        # self.ydata.append(self.object_rpy[0, 2].item())
        self.ydata.append(self.object_angvel_finite_diff[0, 2].item())
        self.ydata2.append(self.object_rpy[0, 2].item())

        self.ln.set_ydata(list(self.ydata))
        self.ln.set_xdata(range(len(self.ydata)))

        self.ln2.set_ydata(list(self.ydata2))
        self.ln2.set_xdata(range(len(self.ydata2)))

        self.ax.draw_artist(self.ln)
        self.ax.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        self.global_counter += 1
        
        if hasattr(self, "actions_list"):
            actions = self.actions_list[self.global_counter-1].repeat((self.num_envs, 1))

        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions.clone().to(self.device)
        self.actions *= self.actions_mask

        targets = self.prev_targets + 1 / 24 * self.actions
        self.cur_targets[:] = tensor_clamp(targets, self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits)
        
        # Code for debugging joint angles
        # self.cur_targets = torch.zeros_like(self.cur_targets)
        # self.cur_targets[:, 5] = math.sin(self.global_counter / 20 * 2 * math.pi / 2)
        # self.cur_targets = scale(self.cur_targets, self.leap_hand_dof_upper_limits, self.leap_hand_dof_lower_limits)
        
        self.prev_targets[:] = self.cur_targets.clone()

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def reset(self):
        super().reset()
        return self.obs_dict
    
    def construct_sim_to_real_transformation(self):
        self.sim_dof_order = self.gym.get_actor_dof_names(self.envs[0], 0)
        self.sim_dof_order = [int(x) for x in self.sim_dof_order]
        self.real_dof_order = list(range(16))
        self.sim_to_real_indices = [] # Value at i is the location of ith real index in the sim list

        for x in self.real_dof_order:
            self.sim_to_real_indices.append(self.sim_dof_order.index(x))
        
        self.real_to_sim_indices = []

        for x in self.sim_dof_order:
            self.real_to_sim_indices.append(self.real_dof_order.index(x))
        
        import pdb; pdb.set_trace()
        assert(self.sim_to_real_indices == self.env_cfg["sim_to_real_indices"])
        assert(self.real_to_sim_indices == self.env_cfg["real_to_sim_indices"])

    def real_to_sim(self, values):
        if not hasattr(self, "sim_dof_order"):
            self.construct_sim_to_real_transformation()

        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_dof_order"):
            self.construct_sim_to_real_transformation()
        
        return values[:, self.sim_to_real_indices]

    def update_low_level_control(self):
        # NO tensor refresh here in position-control mode: the staged reset rows
        # must survive in the tensors until the substep-1 re-apply below (the fix
        # for the GPU property-call poison — see reset_idx). Per-step consumers
        # all get fresh tensors from post_physics_step's _refresh_gym.
        if self.torque_control:
            self._refresh_gym()  # torque PD genuinely needs per-substep state
        if os.getenv("RVIZ") is None and not self.env_cfg["disable_actions"]:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        # Re-apply the state staged by the last reset_idx. Per-actor property
        # calls (mass loop in reset_idx, obj-mass reads in pre_physics' force
        # block) poison the GPU pipeline: EVERY tensor-API set issued between
        # the property call and the next gym.simulate is silently dropped.
        # That first simulate clears the poison, so re-issuing the sets on
        # substep 1 (after substep 0's simulate) is what makes them stick.
        # Without this, cache restores silently no-op'd (objects stayed at
        # creation pose) in every GPU run ever.
        # Repro/verify: tools/probe_minimal_setter.py, probe_restore_orientation.py.
        cd = getattr(self, "_reapply_countdown", 0)
        if cd:
            self._reapply_countdown = cd - 1
            if self._reapply_countdown == 0:
                # Substep 1, not 0: the poison clears at the FIRST simulate after
                # the property calls; an apply issued before that is dropped.
                # Indexed (reset envs only) so non-reset envs aren't rewound.
                self.gym.set_dof_state_tensor_indexed(
                    self.sim, gymtorch.unwrap_tensor(self.dof_state),
                    gymtorch.unwrap_tensor(self._reapply_hand_indices),
                    len(self._reapply_hand_indices))
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                    gymtorch.unwrap_tensor(self._reapply_obj_indices),
                    len(self._reapply_obj_indices))

    def check_termination(self, object_pos):
        resets = torch.logical_or(
            torch.less(object_pos[:, -1], self.reset_z_threshold),
            torch.greater_equal(self.progress_buf, self.max_episode_length),
        )

        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.hand_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        if self.prev_global_counter != self.global_counter: # This is required since sometimes _refresh_gym is called multiple times within same step
            new_object_roll, new_object_pitch, new_object_yaw = euler_from_quaternion(self.object_rot)
            new_object_rpy = torch.stack((new_object_roll, new_object_pitch, new_object_yaw), dim=1) 
            delta_counter = self.global_counter - self.prev_global_counter
            self.object_rpy = new_object_rpy
            self.prev_global_counter = self.global_counter
            
            dr, dp, dy = euler_from_quaternion(quat_mul(self.object_rot, quat_conjugate(self.previous_object_rot)))
            self.object_angvel_finite_diff = torch.stack([dr, dp, dy], dim=-1)
            self.object_angvel_finite_diff /= (self.control_dt * delta_counter)
            self.previous_object_rot = self.object_rot.clone() 

        if "phase_period" in self.env_cfg:
            omega = 2 * math.pi / self.env_cfg["phase_period"]
            phase_angle = (self.progress_buf - 1) * self.control_dt * omega
            self.phase = torch.stack([torch.sin(phase_angle), torch.cos(phase_angle)], dim=-1)

    def _setup_domain_rand_cfg(self, rand_cfg):
        self.randomize_mass = rand_cfg['randomizeMass']
        self.randomize_mass_lower = rand_cfg['randomizeMassLower']
        self.randomize_mass_upper = rand_cfg['randomizeMassUpper']
        self.randomize_com = rand_cfg['randomizeCOM']
        self.randomize_com_lower = rand_cfg['randomizeCOMLower']
        self.randomize_com_upper = rand_cfg['randomizeCOMUpper']
        self.randomize_friction = rand_cfg['randomizeFriction']
        self.randomize_friction_lower = rand_cfg['randomizeFrictionLower']
        self.randomize_friction_upper = rand_cfg['randomizeFrictionUpper']
        self.randomize_scale = rand_cfg['randomizeScale']
        self.scale_list_init = rand_cfg['scaleListInit']
        self.randomize_scale_list = rand_cfg['randomizeScaleList']
        self.randomize_scale_lower = rand_cfg['randomizeScaleLower']
        self.randomize_scale_upper = rand_cfg['randomizeScaleUpper']
        self.randomize_pd_gains = rand_cfg['randomizePDGains']
        self.randomize_p_gain_lower = rand_cfg['randomizePGainLower']
        self.randomize_p_gain_upper = rand_cfg['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_cfg['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_cfg['randomizeDGainUpper']

    def _setup_priv_option_cfg(self, p_cfg):
        self.enable_priv_obj_position = p_cfg['enableObjPos']
        self.enable_priv_obj_mass = p_cfg['enableObjMass']
        self.enable_priv_obj_scale = p_cfg['enableObjScale']
        self.enable_priv_obj_com = p_cfg['enableObjCOM']
        self.enable_priv_obj_friction = p_cfg['enableObjFriction']

    def _setup_z_channel(self):
        """Oracle shape-prior `z` appended to the policy observation.

        z_mode: 'none' (=proprio-only baseline B), 'z0' (coarse geometry:
        [scale, scaled bbox extents (3), bbox fill fraction] = 5-d), 'z1' (z0 +
        8-d analytic shape), or 'local' (16-d: per-fingertip signed distance +
        surface normal to the object, in WORLD frame — see _compute_local_z).
        z0/z1 are static per episode and ride the obs history like the existing
        include_obj_scales channel (replicated identically across the 3-step
        stack). 'local' is DYNAMIC — recomputed every step from the current
        hand/object configuration, so the 3-step stack shows 3 different real
        values (a finite-difference-like signal for free), not 3 copies of one.
        Either way numObservations grows by 3*z_dim.
        """
        self.z_mode = self.env_cfg.get('z_mode', 'none')
        # z0 (5-d coarse geometry) or z1 (8-d = z0 + analytic shape: inertia-eigen
        # ratios + normalized SA/V). z1 is NESTED (z0 held fixed, shape is the delta).
        # 'local' (16-d) is a separate, DYNAMIC axis — see docstring above and
        # docs/research-plan-object-generalization.md Stage 2 ("local vs. global").
        self.z_dim = {'none': 0, 'z0': 5, 'z1': 8, 'local': 16}.get(self.z_mode, 0)
        # z_shuffle (control, docs/HANDOFF.md §4.2): permute the geometry->object
        # map so C gets the CORRECT z0 values but attached to the WRONG objects (a
        # consistent-but-wrong shape prior). If shuffled ~= correct, the policy is
        # using z as an arbitrary per-object tag, not as geometry -> the id/few-
        # samples regime. false/0 = off; any truthy int = permutation seed.
        zs_cfg = self.env_cfg.get('z_shuffle', False)
        self.z_shuffle_seed = int(zs_cfg) if (zs_cfg is not False and zs_cfg is not None) else None
        if self.z_shuffle_seed == 0:
            self.z_shuffle_seed = None
        if self.z_dim == 0:
            return
        from leapsim.utils.rerun_vis import _load_object_mesh
        import trimesh
        asset_root = Path(__file__).parent.parent.parent
        self._obj_geom = {}
        self._obj_shape1 = {}  # z1 analytic shape: [inertia r2, r3, normalized SA/V]
        # z_mode='local' shape params (computed for every object type regardless
        # of z_mode — cheap, and lets z_mode be switched without re-deriving
        # this): family id (0 box/1 cylinder/2 sphere) parsed from the name
        # prefix, and UNSCALED canonical (hx,hy,hz)|(r,half_len,0)|(r,0,0) per
        # family, derived from the mesh bbox (see _compute_local_z for the
        # closed-form closest-point formulas that consume these).
        self._local_family = {}
        self._local_params = {}
        for tid, tname in enumerate(self.object_type_list):
            fam = _SHAPE_FAMILY_ID.get(tname.split('_')[0], 0)
            self._local_family[tid] = fam
            urdf = asset_root / self.asset_files_dict[tname]
            loaded = _load_object_mesh(urdf)
            if loaded is None:
                self._obj_geom[tid] = (np.array([0.07, 0.07, 0.07], dtype=np.float32), 1.0)
                self._obj_shape1[tid] = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # cube-like
                self._local_params[tid] = np.array([0.035, 0.035, 0.035], dtype=np.float32)
                print(f"[z geom] {tname}: mesh load FAILED — using fallback box")
                continue
            verts, faces = loaded
            ext = (verts.max(axis=0) - verts.min(axis=0)).astype(np.float32)
            bbox_vol = float(ext.prod()) or 1.0
            # cylinder/cone/capsule URDFs are authored with the axis along
            # local Z, bbox-centered (see tools/gen_primitive_objects.py) ->
            # ext = (2r, 2r, length|height|(L+2r)).
            if fam == 0:      # box: half-extents directly
                self._local_params[tid] = (ext / 2.0).astype(np.float32)
            elif fam == 1:    # cylinder: (radius, half_length, unused)
                r = float((ext[0] + ext[1]) / 4.0)
                self._local_params[tid] = np.array([r, float(ext[2] / 2.0), 0.0], dtype=np.float32)
            elif fam == 2:    # sphere: (radius, unused, unused)
                r = float(ext.mean() / 2.0)
                self._local_params[tid] = np.array([r, 0.0, 0.0], dtype=np.float32)
            elif fam == 3:    # cone: (base radius, full height, unused). base
                # at bbox z=-h/2, apex at bbox z=+h/2 (matches _make_mesh).
                r = float((ext[0] + ext[1]) / 4.0)
                self._local_params[tid] = np.array([r, float(ext[2]), 0.0], dtype=np.float32)
            else:             # fam == 4, capsule: (radius, half cyl-segment length, unused).
                # total bbox z-extent = L + 2r -> half_L = ext_z/2 - r.
                r = float((ext[0] + ext[1]) / 4.0)
                half_l = float(ext[2]) / 2.0 - r
                self._local_params[tid] = np.array([r, half_l, 0.0], dtype=np.float32)
            mesh = None
            try:
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                vol = abs(float(mesh.volume)) or bbox_vol
            except Exception:
                vol = bbox_vol
            fill = float(np.clip(vol / bbox_vol, 0.0, 1.0))
            self._obj_geom[tid] = (ext, fill)
            # z1 analytic descriptors (scale- & density-invariant → pure shape):
            #   inertia-eigenvalue ratios λ2/λ1, λ3/λ1 (elongation/flatness), and
            #   SA/V normalized by V^(2/3) then /6 (compactness; sphere~0.81, cube 1.0).
            r2 = r3 = 1.0
            sav = 1.0
            if mesh is not None:
                try:
                    lam = np.sort(np.linalg.eigvalsh(mesh.moment_inertia))[::-1]  # λ1≥λ2≥λ3
                    if lam[0] > 1e-12:
                        r2, r3 = float(lam[1] / lam[0]), float(lam[2] / lam[0])
                    area = float(mesh.area)
                    if vol > 1e-12 and area > 0:
                        sav = float(area / (vol ** (2.0 / 3.0)) / 6.0)
                except Exception:
                    pass
            self._obj_shape1[tid] = np.array(
                [np.clip(r2, 0, 1), np.clip(r3, 0, 1), np.clip(sav, 0, 5)], dtype=np.float32)
            extra = f" shape1=[{r2:.3f},{r3:.3f},{sav:.3f}]" if self.z_dim == 8 else ""
            print(f"[z geom] {tname}: bbox={ext.round(4).tolist()} fill={fill:.3f}{extra}")
        self._num_obs_base = int(self.env_cfg["numObservations"])
        self.cfg["env"]["numObservations"] = self._num_obs_base + 3 * self.z_dim
        print(f"[{self.z_mode}] z_mode={self.z_mode} z_dim={self.z_dim} "
              f"numObservations {self._num_obs_base} -> {self.cfg['env']['numObservations']}")

    def _build_z_features(self):
        """Static per-env z vector = [scale, scaled bbox extents (3), fill] (z0).

        With z_shuffle set, geometry is looked up through a fixed derangement of the
        object ids actually present in the envs, so each env receives the correct z0
        of a DIFFERENT object (a consistent but wrong shape->object map). Scale stays
        the env's own value (it is not shuffled — the control targets the shape map).

        z_mode='local' takes a DIFFERENT path (_build_local_z_params below): there
        is no static z0_features vector for 'local' — only the per-env shape
        family/params needed to compute the per-STEP feature in _compute_local_z.
        """
        if self.z_mode == 'local':
            self._build_local_z_params()
            return
        ids = self.env_object_type_id.cpu().numpy()
        geom_id = ids  # by default env i uses its own object's geometry
        if self.z_shuffle_seed is not None:
            present = np.unique(ids)
            rng = np.random.RandomState(self.z_shuffle_seed)
            perm = present.copy()
            rng.shuffle(perm)
            # break any fixed points so the map is guaranteed wrong (needs >=2 ids)
            if len(present) >= 2:
                for k in range(len(present)):
                    if perm[k] == present[k]:
                        j = (k + 1) % len(present)
                        perm[k], perm[j] = perm[j], perm[k]
            id2perm = {int(o): int(p) for o, p in zip(present, perm)}
            geom_id = np.array([id2perm[int(o)] for o in ids])
            names = self.object_type_list
            print(f"[z0 shuffle seed={self.z_shuffle_seed}] geometry->object map "
                  f"(shape of A given to B): " +
                  ", ".join(f"{names[int(o)]}<-{names[int(p)]}" for o, p in id2perm.items()))

        ext = torch.zeros((self.num_envs, 3), device=self.device)
        fill = torch.zeros((self.num_envs, 1), device=self.device)
        for i in range(self.num_envs):
            e, f = self._obj_geom[int(geom_id[i])]
            ext[i] = torch.tensor(e, device=self.device)
            fill[i, 0] = f
        scl = self.obj_scales.to(self.device).float().unsqueeze(1)
        parts = [scl, ext * scl, fill]                      # z0 (5-d)
        if self.z_dim == 8:                                 # z1: append analytic shape (3-d)
            sh = torch.zeros((self.num_envs, 3), device=self.device)
            for i in range(self.num_envs):
                sh[i] = torch.tensor(self._obj_shape1[int(geom_id[i])], device=self.device)
            parts.append(sh)
        self.z0_features = torch.cat(parts, dim=1).float()
        assert self.z0_features.shape[1] == self.z_dim, \
            f"z features dim {self.z0_features.shape[1]} != z_dim {self.z_dim}"
        print(f"[{self.z_mode}] built z for {self.num_envs} envs, dim {self.z_dim}"
              f"{' (SHUFFLED)' if self.z_shuffle_seed is not None else ''}; "
              f"env0 z={self.z0_features[0].cpu().numpy().round(4).tolist()}")

    def _build_local_z_params(self):
        """Static per-env shape family + canonical (unscaled) params for
        z_mode='local'. z_shuffle applies here too (same geometry->object
        derangement as z0/z1), for the same reason: a future shuffled-local
        control needs it wired up identically to the static rungs.
        """
        ids = self.env_object_type_id.cpu().numpy()
        geom_id = ids
        if self.z_shuffle_seed is not None:
            present = np.unique(ids)
            rng = np.random.RandomState(self.z_shuffle_seed)
            perm = present.copy()
            rng.shuffle(perm)
            if len(present) >= 2:
                for k in range(len(present)):
                    if perm[k] == present[k]:
                        j = (k + 1) % len(present)
                        perm[k], perm[j] = perm[j], perm[k]
            id2perm = {int(o): int(p) for o, p in zip(present, perm)}
            geom_id = np.array([id2perm[int(o)] for o in ids])

        family = np.array([self._local_family[int(g)] for g in geom_id], dtype=np.int64)
        params = np.stack([self._local_params[int(g)] for g in geom_id], axis=0)
        self._env_shape_family = torch.tensor(family, device=self.device, dtype=torch.long)
        self._env_shape_params = torch.tensor(params, device=self.device, dtype=torch.float32)
        # INTERPRETABILITY (default off): counterfactually tell the policy the object
        # is a DIFFERENT shape than it really is — the local analogue of "drive object
        # A with object B's z". local_cf_family in {0=box,1=cyl,2=sphere}; optional
        # local_cf_params=[a,b,c] overrides the canonical half-extents/(r,half_len)/r.
        # The physical object is unchanged; only the feature's geometry is swapped.
        cf_fam = self.env_cfg.get('local_cf_family', None)
        if cf_fam is not None:
            self._env_shape_family[:] = int(cf_fam)
            cf_par = self.env_cfg.get('local_cf_params', None)
            if cf_par is not None:
                self._env_shape_params[:] = torch.tensor(
                    [float(x) for x in cf_par], device=self.device, dtype=torch.float32)
            print(f"[local CF] COUNTERFACTUAL: every env told family={int(cf_fam)} "
                  f"params={self._env_shape_params[0].cpu().numpy().round(4).tolist()}")
        print(f"[local] built shape family/params for {self.num_envs} envs"
              f"{' (SHUFFLED)' if self.z_shuffle_seed is not None else ''}; "
              f"env0 family={int(self._env_shape_family[0])} params="
              f"{self._env_shape_params[0].cpu().numpy().round(4).tolist()}")

    @staticmethod
    def _closest_point_and_normal_local(p, family, params):
        """Closed-form signed distance + outward unit normal from query points
        `p` to the surface of a box/cylinder/sphere/cone/capsule, all in the
        OBJECT-LOCAL, UNSCALED (canonical) frame. Fully vectorized (no python
        loop) so this is cheap enough to call every step for all envs.

        p:      [N,4,3] query points (per env, per fingertip) in local frame.
        family: [N] long, 0=box 1=cylinder 2=sphere 3=cone 4=capsule.
        params: [N,3] float, meaning depends on family:
                  box      -> (hx, hy, hz)            half-extents
                  cylinder -> (r, half_len, unused)    axis = local Z
                  sphere   -> (r, unused, unused)
                  cone     -> (base radius R, full height h, unused)
                              base at local z=-h/2, apex at z=+h/2, axis Z
                  capsule  -> (r, half cyl-segment length, unused)  axis Z

        Returns (dist [N,4], normal [N,4,3]) — dist is SIGNED (negative =
        penetrating). The "inside" branches (fingertip past the surface) use an
        approximate nearest-face/axis fallback since exact interior projection
        isn't needed for a contact-proximity feature — what matters is behaving
        sanely near and outside the surface, which is where fingertips live
        during normal grasping/rotation.
        """
        eps = 1e-8
        device = p.device
        params_b = params[:, None, :].expand(-1, 4, -1)   # [N,4,3]
        family_b = family[:, None].expand(-1, 4)           # [N,4]

        # ---- box ----
        half = params_b.clamp_min(eps)
        cp_box = torch.clamp(p, -half, half)
        diff_box = p - cp_box
        dist_box = diff_box.norm(dim=-1)
        outside_box = dist_box > eps
        ratio = p.abs() / half
        axis = ratio.argmax(dim=-1)                                    # [N,4]
        inside_normal_box = torch.nn.functional.one_hot(axis, 3).to(p.dtype) * torch.sign(p)
        normal_box = torch.where(
            outside_box.unsqueeze(-1),
            diff_box / dist_box.clamp_min(eps).unsqueeze(-1),
            inside_normal_box,
        )
        pen_box = (half - p.abs()).amin(dim=-1)                        # >0 if inside
        signed_dist_box = torch.where(outside_box, dist_box, -pen_box)

        # ---- sphere ----
        r_sph = params_b[..., 0].clamp_min(eps)
        pn = p.norm(dim=-1)
        default_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=p.dtype)
        normal_sphere = torch.where(
            (pn > eps).unsqueeze(-1), p / pn.clamp_min(eps).unsqueeze(-1), default_up.expand_as(p),
        )
        signed_dist_sphere = pn - r_sph

        # ---- cylinder (axis = local Z) ----
        r_cyl = params_b[..., 0].clamp_min(eps)
        hl_cyl = params_b[..., 1].clamp_min(eps)
        xy = p[..., :2]
        r_xy = xy.norm(dim=-1)
        default_radial = torch.tensor([1.0, 0.0], device=device, dtype=p.dtype)
        radial_dir = torch.where(
            (r_xy > eps).unsqueeze(-1), xy / r_xy.clamp_min(eps).unsqueeze(-1), default_radial.expand_as(xy),
        )
        z = p[..., 2]
        outside_cap = z.abs() > hl_cyl
        outside_side = r_xy > r_cyl
        cp_z = torch.where(outside_cap, torch.sign(z) * hl_cyl, z)
        cp_xy = torch.where(outside_side.unsqueeze(-1), radial_dir * r_cyl.unsqueeze(-1), xy)
        cp_cyl = torch.cat([cp_xy, cp_z.unsqueeze(-1)], dim=-1)
        diff_cyl = p - cp_cyl
        dist_cyl = diff_cyl.norm(dim=-1)
        inside_cyl = (~outside_cap) & (~outside_side)
        radial_pen = r_cyl - r_xy                                       # >0 if inside radially
        cap_pen = hl_cyl - z.abs()                                      # >0 if inside axially
        use_radial = radial_pen < cap_pen
        zero2 = torch.zeros_like(xy)
        inside_normal_cyl = torch.where(
            use_radial.unsqueeze(-1),
            torch.cat([radial_dir, zero2[..., :1]], dim=-1),
            torch.cat([zero2, torch.sign(z).unsqueeze(-1)], dim=-1),
        )
        normal_cyl = torch.where(
            (dist_cyl > eps).unsqueeze(-1),
            diff_cyl / dist_cyl.clamp_min(eps).unsqueeze(-1),
            inside_normal_cyl,
        )
        signed_dist_cyl = torch.where(inside_cyl, -torch.minimum(radial_pen, cap_pen), dist_cyl)

        # ---- capsule (axis = local Z; central segment + radius r) ----
        # Simplest of all five: SDF(p) = |p - clamp_to_segment(p)| - r. The sign
        # falls out for free (negative once inside the tube), no separate
        # inside/outside branch needed.
        r_cap = params_b[..., 0].clamp_min(eps)
        hl_cap = params_b[..., 1].clamp_min(eps)             # half cyl-segment length
        z_cap = p[..., 2]
        z_clamped = torch.clamp(z_cap, -hl_cap, hl_cap)
        seg_pt = torch.cat([torch.zeros_like(p[..., :2]), z_clamped.unsqueeze(-1)], dim=-1)
        diff_cap = p - seg_pt
        dist_axis_cap = diff_cap.norm(dim=-1)
        default_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=p.dtype)
        normal_cap = torch.where(
            (dist_axis_cap > eps).unsqueeze(-1),
            diff_cap / dist_axis_cap.clamp_min(eps).unsqueeze(-1),
            default_up.expand_as(p),
        )
        signed_dist_cap = dist_axis_cap - r_cap

        # ---- cone (axis = local Z; base radius R at z=-h/2, apex at z=+h/2) ----
        # Rotationally symmetric -> work in the 2D (r,z) cross-section: closest
        # point is on either the slant segment (apex -> base rim) or the base
        # disk's edge-on segment (z=-h/2, r in [0,R]); take whichever is nearer,
        # then lift back to 3D using the query's own azimuthal direction.
        R_cone = params_b[..., 0].clamp_min(eps)
        h_cone = params_b[..., 1].clamp_min(eps)
        half_h = h_cone / 2.0
        xy_c = p[..., :2]
        r_c = xy_c.norm(dim=-1)
        default_radial = torch.tensor([1.0, 0.0], device=device, dtype=p.dtype)
        radial_dir_c = torch.where(
            (r_c > eps).unsqueeze(-1), xy_c / r_c.clamp_min(eps).unsqueeze(-1), default_radial.expand_as(xy_c),
        )
        z_c = p[..., 2]

        apex_r = torch.zeros_like(R_cone)
        seg_dr, seg_dz = R_cone - apex_r, (-half_h) - half_h        # base_r - apex_r, base_z - apex_z
        seg_len_sq = (seg_dr * seg_dr + seg_dz * seg_dz).clamp_min(eps)
        t = ((r_c - apex_r) * seg_dr + (z_c - half_h) * seg_dz) / seg_len_sq
        t = t.clamp(0.0, 1.0)
        slant_r = apex_r + t * seg_dr
        slant_z = half_h + t * seg_dz
        dist_slant = torch.sqrt((r_c - slant_r) ** 2 + (z_c - slant_z) ** 2 + eps)

        base_cp_r = torch.clamp(r_c, torch.zeros_like(R_cone), R_cone)
        base_cp_z = -half_h
        dist_base = torch.sqrt((r_c - base_cp_r) ** 2 + (z_c - base_cp_z) ** 2 + eps)

        use_slant = dist_slant <= dist_base
        closest_r = torch.where(use_slant, slant_r, base_cp_r)
        closest_z = torch.where(use_slant, slant_z, base_cp_z)
        unsigned_dist_cone = torch.where(use_slant, dist_slant, dist_base)

        R_at_z = R_cone * (half_h - z_c) / h_cone                    # cone radius at height z_c
        inside_cone = (z_c >= -half_h) & (z_c <= half_h) & (r_c <= R_at_z)

        closest_3d_cone = torch.cat([radial_dir_c * closest_r.unsqueeze(-1), closest_z.unsqueeze(-1)], dim=-1)
        diff_cone = p - closest_3d_cone
        diff_norm_cone = diff_cone.norm(dim=-1)
        outward_cone = diff_cone / diff_norm_cone.clamp_min(eps).unsqueeze(-1)
        # Inside: diff points p<-closest (into the solid); flip it so the
        # fallback still reads as an "outward-ish" escape direction, same
        # convention as every other shape's inside branch above.
        normal_cone = torch.where(inside_cone.unsqueeze(-1), -outward_cone, outward_cone)
        signed_dist_cone = torch.where(inside_cone, -unsigned_dist_cone, unsigned_dist_cone)

        # ---- select per env family ----
        is_box = (family_b == 0).unsqueeze(-1)
        is_cyl = (family_b == 1).unsqueeze(-1)
        is_cone = (family_b == 3).unsqueeze(-1)
        is_cap = (family_b == 4).unsqueeze(-1)
        normal = torch.where(
            is_box, normal_box, torch.where(
                is_cyl, normal_cyl, torch.where(
                    is_cone, normal_cone, torch.where(
                        is_cap, normal_cap, normal_sphere))))
        dist = torch.where(
            family_b == 0, signed_dist_box, torch.where(
                family_b == 1, signed_dist_cyl, torch.where(
                    family_b == 3, signed_dist_cone, torch.where(
                        family_b == 4, signed_dist_cap, signed_dist_sphere))))
        return dist, normal

    def _compute_local_z(self):
        """Per-step DYNAMIC local feature: for each of the 4 fingertips, signed
        distance + outward surface normal to the object, in WORLD units/frame.
        16-d = 4 fingertips * (1 distance + 3 normal). See
        docs/research-plan-object-generalization.md Stage 2 ("local vs global")
        for why this exists — the DexRepNet++-style counterpart to z0/z1's
        static, global whole-object descriptors.
        """
        finger_pos = self.rigid_body_states[:, FINGERTIP_LOCAL_IDS, :3]           # [N,4,3] world
        obj_p = self.object_pos.unsqueeze(1).expand(-1, 4, -1)                     # [N,4,3]
        obj_q = self.object_rot.unsqueeze(1).expand(-1, 4, -1)                     # [N,4,4] xyzw
        local_p = quat_rotate_inverse(obj_q.reshape(-1, 4), (finger_pos - obj_p).reshape(-1, 3))
        obj_scales = self.obj_scales.to(self.device).float()                       # [N]
        scales = obj_scales.view(-1, 1, 1).expand(-1, 4, 1).reshape(-1, 1)
        local_p_unscaled = (local_p / scales.clamp_min(1e-6)).reshape(self.num_envs, 4, 3)

        dist_unscaled, normal_local = self._closest_point_and_normal_local(
            local_p_unscaled, self._env_shape_family, self._env_shape_params)

        scale_env = obj_scales.view(-1, 1)                                         # [N,1]
        dist_world = dist_unscaled * scale_env                                     # [N,4]
        normal_world = quat_apply(
            self.object_rot.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
            normal_local.reshape(-1, 3),
        ).reshape(self.num_envs, 4, 3)

        feat = torch.cat([dist_world.unsqueeze(-1), normal_world], dim=-1).reshape(self.num_envs, 16)
        # INTERPRETABILITY ablation (default off): 'zero' kills the feature (does the
        # policy rely on it?); 'noise' adds gaussian noise of the given std.
        abl = self.env_cfg.get('local_ablate', None)
        if abl == 'zero':
            feat = torch.zeros_like(feat)
        elif abl == 'noise':
            feat = feat + torch.randn_like(feat) * float(self.env_cfg.get('local_ablate_std', 0.02))
        return feat

    def _current_z(self):
        """Dispatch: static z0/z1 read the precomputed per-env tensor; 'local'
        recomputes fresh every call (every step) from the current configuration.
        """
        if self.z_mode == 'local':
            return self._compute_local_z()
        return self.z0_features

    def _setup_object_info(self, o_cfg):
        self.object_type = o_cfg['type']
        raw_prob = o_cfg['sampleProb']
        primitive_list = self.object_type.split('+')
        # one sampleProb entry per '+'-joined family; default to a uniform split so
        # multi-family mixes (Stage-1 B/C) don't need it spelled out every run.
        if len(raw_prob) != len(primitive_list):
            raw_prob = [1.0 / len(primitive_list)] * len(primitive_list)
        assert abs(sum(raw_prob) - 1.0) < 1e-6, \
            f"sampleProb must sum to 1, got {raw_prob} (sum {sum(raw_prob)})"

        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'simple_tennis_ball': 'assets/ball.urdf',
            'cube':               'assets/cube.urdf',
            'hammer':             'assets/hammer.urdf',
        }
        # Procedural primitive families (tools/gen_primitive_objects.py) — one
        # <assets>/<category>/<subset>/*.urdf glob per category sharing this
        # exact pattern. Generalized from 3 near-identical copy-pasted
        # branches (cuboid/cylinder/sphere) so a new category (cone, capsule,
        # ...) only needs adding here, not a 4th near-duplicate block.
        for p_id, prim in enumerate(primitive_list):
            category = next((c for c in _PRIMITIVE_FAMILY_CATEGORIES if c in prim), None)
            if category is not None:
                subset_name = self.object_type.split('_')[-1]
                paths = sorted(glob(f'../assets/{category}/{subset_name}/*.urdf'))
                names = [f'{category}_{i}' for i in range(len(paths))]
                self.object_type_list += names
                for i, path in enumerate(paths):
                    self.asset_files_dict[f'{category}_{i}'] = path.replace('../assets/', 'assets/')
                self.object_type_prob += [raw_prob[p_id] / max(len(names), 1) for _ in names]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(self.object_type_list)
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _setup_reward_cfg(self, r_cfg):
        self.angvel_clip_min = r_cfg['angvelClipMin']
        self.angvel_clip_max = r_cfg['angvelClipMax']
        self.rotate_reward_scale = r_cfg['rotateRewardScale']
        self.object_linvel_penalty_scale = r_cfg['objLinvelPenaltyScale']
        self.pose_diff_penalty_scale = r_cfg['poseDiffPenaltyScale']
        self.torque_penalty_scale = r_cfg['torquePenaltyScale']
        self.work_penalty_scale = r_cfg['workPenaltyScale']

    def _init_object_pose(self):
        return init_object_pose(self.env_cfg, self.save_init_pose, self.grasp_cache_name)

    def reward_rotate_finite_diff(self):
        min_angvel = self.env_cfg["reward"]["angvelClipMin"]
        max_angvel = self.env_cfg["reward"]["angvelClipMax"]

        reward = torch.clip(self.object_angvel_finite_diff[:, 2], min=min_angvel, max=max_angvel)

        N = self.progress_buf  
        self.object_angvel_finite_diff_mean = N * self.object_angvel_finite_diff_mean / (N  + 1) + reward / (N + 1) 

        return reward

    def reward_object_fallen(self):
        return torch.less(self.object_pos[:, -1], self.reset_z_threshold).float()

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_hand_dof_lower_limits).squeeze().cpu().numpy()
        sim_max = self.sim_to_real(self.leap_hand_dof_upper_limits).squeeze().cpu().numpy()
        
        return sim_min, sim_max

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)

        return joints
    
    def LEAPhand_to_LEAPsim(self, joints):
        joints = np.array(joints)
        ret_joints = joints - 3.14159
        
        return ret_joints

def compute_hand_reward(
    object_linvel, object_linvel_penalty_scale: float,
    object_angvel, rotation_axis, rotate_reward_scale: float,
    angvel_clip_max: float, angvel_clip_min: float,
    pose_diff_penalty, pose_diff_penalty_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
):
    rotate_reward_cond = (rotation_axis[:, -1] != 0).float()
    vec_dot = (object_angvel * rotation_axis).sum(-1)
    rotate_reward = torch.clip(vec_dot, max=angvel_clip_max, min=angvel_clip_min)
    rotate_reward = rotate_reward_scale * rotate_reward * rotate_reward_cond
    object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)

    reward = rotate_reward
    # Distance from the hand to the object
    reward = reward + object_linvel_penalty * object_linvel_penalty_scale
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    return reward, rotate_reward, object_linvel_penalty

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower)/(upper - lower)

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
