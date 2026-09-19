# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/HaozhiQi/hora/blob/main/hora/tasks/leap_hand_grasp.py
# --------------------------------------------------------

import atexit
import yaml
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    torch_rand_float, quat_from_angle_axis, quat_mul, tensor_clamp, to_torch,
    quat_apply, quat_rotate_inverse,
)
from leapsim.tasks.leap_hand_rot import LeapHandRot
from leapsim.utils.rerun_vis import _load_object_mesh
from pathlib import Path
import trimesh


class LeapHandGrasp(LeapHandRot):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=None, force_render=None):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless)
        # Rows: 16 DoF positions + 16 PD targets + 7 object pose + 1 object_type_id.
        # Targets matter as much as positions: during settle the object deflects
        # the fingers off their held targets, and that PD error IS the grip force
        # (1-4 N). Restoring positions without targets leaves zero PD error → zero
        # squeeze → force-closure grasps drop their object on restore. The id lets
        # LeapHandRot restore rows only into envs holding the same object instance.
        self.saved_grasping_states = torch.zeros((0, 40), dtype=torch.float, device=self.device)

        if "canonical_pose" in cfg["env"]:
            self.canonical_pose = cfg["env"]["canonical_pose"]
        else:
            self.canonical_pose = [0.69,-0.52,1.0,-0.35,1,1.1,0.6,0.3,0.77,0.0,1.0,-0.35,0.73,0.22,1.0,-0.35]

        if "num_contact_fingers" in cfg["env"]:
            self.num_contact_fingers = cfg["env"]["num_contact_fingers"]
        else:
            self.num_contact_fingers = 2

        if "finger_dist_threshold" in cfg["env"]:
            self.finger_dist_threshold = cfg["env"]["finger_dist_threshold"]
        else:
            self.finger_dist_threshold = 0.1

        if "grasp_cache_len" not in self.cfg["env"]:
            self.cfg["env"]["grasp_cache_len"] = 5e4

        # Force window: [min, max]. Below min → not touching. Above max → penetration
        # artifact (PhysX corrects interpenetration with huge single-step impulses at
        # reset). Only forces in this window count toward the contact criterion.
        self.min_contact_force: float = float(cfg["env"].get("min_contact_force", 0.5))
        self.max_contact_force: float = float(cfg["env"].get("max_contact_force", 50.0))

        # Contact-criterion grace period: the N-finger contact condition is waived
        # for the first `contact_grace_steps` steps of every episode so the object
        # (which spawns airborne above the palm) can fall and settle before contact
        # is demanded. Without this, num_contact_fingers > 0 kills every episode at
        # step 1 — the hammer runs' ~0.002% yield.
        self.contact_grace_steps: int = int(cfg["env"].get("contact_grace_steps", 0))

        # Palm/proximal-contact exclusion: a grip the search certifies must be
        # load-bearing through the DISTAL finger segments. If the object presses on
        # any proximal hand body with more than this force [N], the episode fails
        # (grace-gated like the finger condition). Set well below the object's
        # weight (~0.3-0.7 N) so neither the palm nor the curled proximal phalanges
        # can be a support surface; < 0 disables (upstream in-palm behavior).
        # Palm-only exclusion is not enough: wide cuboids bridge across the curled
        # mcp/pip segments with the palm itself untouched — a palm rest in all but
        # name. Rotation training is unaffected — the policy may still use the palm.
        self.max_palm_contact_force: float = float(cfg["env"].get("max_palm_contact_force", -1.0))
        # Hand bodies treated as "proximal" (see robot.urdf link order):
        #   0 palm_lower · 1/5/9 mcp_joint · 2/6/10 pip · 13 pip_4 · 14 thumb_pip
        # Allowed load path: dip (3/7/11), thumb_dip (15), fingertips (4/8/12/16).
        self.palm_contact_body_ids = set(
            cfg["env"].get("palm_contact_bodies", [0, 1, 2, 5, 6, 9, 10, 13, 14]))

        # ── Per-episode stat accumulators (reset in reset_idx) ──────────────────
        N, F = self.num_envs, 4
        self._ep_force_max    = torch.zeros((N, F), dtype=torch.float32, device=self.device)
        self._ep_force_spike  = torch.zeros((N, F), dtype=torch.float32, device=self.device)
        self._ep_contact_steps = torch.zeros((N, F), dtype=torch.int32,   device=self.device)
        self._ep_dist_to_obj_center_min     = torch.full((N, F), 9999.0, dtype=torch.float32, device=self.device)
        self._ep_palm_force_max = torch.zeros((N,), dtype=torch.float32, device=self.device)

        # ── Global accumulators for the final stats YAML ─────────────────────────
        self._gs_ep_count        = 0
        self._gs_success_count   = 0
        self._gs_force_max       = []
        self._gs_spike_rate      = []
        self._gs_contact_fingers = []
        self._gs_dist_min        = []
        self._gs_palm_force      = []

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # ── Hill-climbing state (per env) ──────────────────────────────────────
        canon_t = torch.tensor(self.canonical_pose, dtype=torch.float32, device=self.device)
        self.best_pose_per_env    = canon_t.unsqueeze(0).repeat(N, 1)      # [N, 16]
        self.best_score_per_env   = torch.full((N,), -float('inf'), device=self.device)
        self.last_attempted_pose  = canon_t.unsqueeze(0).repeat(N, 1)      # [N, 16]
        # End-of-episode surface distance per fingertip, refreshed every reset.
        self._last_surface_dist   = torch.zeros((N, F), dtype=torch.float32, device=self.device)
        # Surface distance at the moment best_pose was promoted (monotonic-ish).
        self._best_surf_per_env   = torch.full((N, F), 9999.0, dtype=torch.float32, device=self.device)

        # ── Object meshes for surface-distance queries ─────────────────────────
        # One mesh per object instance in the family: each env's fitness must be
        # scored against the object that env actually holds, not object 0's mesh.
        asset_root = Path(__file__).parent.parent.parent
        self.obj_proximity_by_type = []
        for obj_type in self.object_type_list:
            verts, faces = _load_object_mesh(asset_root / self.asset_files_dict[obj_type])
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            # ProximityQuery caches a KD-tree internally; reuse across episodes.
            self.obj_proximity_by_type.append(trimesh.proximity.ProximityQuery(mesh))

        # Per-fingertip pad point in LINK-LOCAL frame. NOT in mesh frame — each
        # fingertip link has a <visual><origin xyz rpy> that transforms the STL
        # into the link's frame, and you have to apply that to the mesh-frame
        # pad coordinate to get the actual link-frame coordinate.
        #
        # Index/thumb/middle (link ids 4, 8, 12) all use fingertip.stl with the
        # same URDF origin xyz=(0.013, -0.006, 0.015), rpy=(π, 0, 0). Mesh-frame
        # pad ≈ (-0.014, +0.043, 0). Applying Rx(π) flips Y and Z: → (-0.014,
        # -0.043, 0). Translating: → (-0.001, -0.049, 0.015).
        #
        # Ring fingertip (link id 16) uses thumb_fingertip.stl, a different mesh
        # with bounds offset way out at y∈[-0.141,-0.069], with URDF translation
        # (0.063, 0.078, 0.049) and identity rotation. Mesh-frame tip ≈ -Y
        # extreme: in link frame ≈ (0.004, -0.063, -0.014).  ⚠ NOT VERIFIED ⚠
        self.fingertip_tip_local = torch.tensor([
            [-0.001, -0.049,  0.015],   # 0: index  fingertip
            [-0.001, -0.049,  0.015],   # 1: thumb  fingertip_2
            [-0.001, -0.049,  0.015],   # 2: middle fingertip_3
            [ 0.004, -0.063, -0.014],   # 3: ring   thumb_fingertip  ⚠ unverified ⚠
        ], dtype=torch.float32, device=self.device)                       # [4, 3]

        # Per-env scale tensor. When randomizeScale is False (typical for grasp
        # gen), parent's _create_envs leaves self.obj_scales as an empty tensor,
        # so fall back to base_obj_scale uniformly across envs.
        scales_src = torch.as_tensor(self.obj_scales, dtype=torch.float32, device=self.device)
        if scales_src.numel() == self.num_envs:
            self._obj_scales_t = scales_src
        else:
            self._obj_scales_t = torch.full(
                (self.num_envs,), float(self.base_obj_scale),
                dtype=torch.float32, device=self.device,
            )                                                              # [N]

        # Flush whatever's in the cache on any process exit (Ctrl+C, crash, etc.)
        # so partial runs aren't thrown away. The exit() at cache-full also calls
        # _save_cache so the threshold-reached path still works.
        atexit.register(self._save_cache_partial)



    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _cache_path(self) -> str:
        return f'cache/{self.grasp_cache_name}_grasp_50k_s{str(self.base_obj_scale).replace(".", "")}.npy'

    def _stats_path(self) -> str:
        return self._cache_path().replace('.npy', '_stats.yaml')

    def _save_cache_partial(self) -> None:
        """Write whatever grasps we have right now to disk. Safe to call
        repeatedly. Called from reset_idx every time a success arrives, AND
        from atexit on any process termination, AND when the cache fills.
        """
        if not hasattr(self, 'saved_grasping_states'):
            return  # init failed early; nothing to save
        n = int(self.saved_grasping_states.shape[0])
        if n == 0:
            return
        # Trim to the configured cache_len if we somehow overshot (we shouldn't
        # in normal flow, but defensive).
        target = int(self.cfg["env"]["grasp_cache_len"])
        rows   = self.saved_grasping_states[:target] if n > target else self.saved_grasping_states
        np.save(self._cache_path(), rows.cpu().numpy())
        self._dump_stats_yaml()
        print(f'[cache] flushed {rows.shape[0]} grasps to {self._cache_path()}')

    def _dump_stats_yaml(self) -> None:
        ep = max(1, self._gs_ep_count)
        stats = {
            'total_episodes':         self._gs_ep_count,
            'successful_grasps':      self._gs_success_count,
            'success_rate_pct':       round(100.0 * self._gs_success_count / ep, 2),
            'contact_criterion': {
                'min_contact_force_N':    self.min_contact_force,
                'max_contact_force_N':    self.max_contact_force,
                'num_contact_fingers':    int(self.num_contact_fingers),
                'finger_dist_threshold_m': self.finger_dist_threshold,
                'contact_grace_steps':    int(self.contact_grace_steps),
                'max_palm_contact_force_N': float(self.max_palm_contact_force),
                'palm_contact_bodies':    sorted(int(b) for b in self.palm_contact_body_ids),
            },
            'force': {
                'mean_ep_max_valid_N': round(float(np.mean(self._gs_force_max))  if self._gs_force_max  else 0.0, 3),
                'mean_spike_rate_pct': round(float(np.mean(self._gs_spike_rate)) * 100 if self._gs_spike_rate else 0.0, 2),
                'mean_ep_max_palm_N':  round(float(np.mean(self._gs_palm_force)) if self._gs_palm_force else 0.0, 3),
            },
            'contact': {
                'mean_fingers_at_ep_end': round(float(np.mean(self._gs_contact_fingers)) if self._gs_contact_fingers else 0.0, 3),
            },
            'distance': {
                'mean_min_fingertip_dist_m': round(float(np.mean(self._gs_dist_min)) if self._gs_dist_min else 0.0, 4),
            },
        }
        with open(self._stats_path(), 'w') as f:
            yaml.safe_dump(stats, f, default_flow_style=False)
        print(f'[grasp_stats] wrote {self._stats_path()}')

    # ── Reset ─────────────────────────────────────────────────────────────────────

    def reset_idx(self, env_ids):
        if self.randomize_mass:
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

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_leap_hand_dofs * 2 + 5), device=self.device)
        self.rb_forces[env_ids, :, :] = 0.0

        # ── End-of-episode surface distance for these envs (BEFORE any reset) ──
        # Refreshes self._last_surface_dist[env_ids] so _episode_fitness sees it.
        self._last_surface_dist[env_ids] = self._compute_fingertip_surface_distance(env_ids)

        success = self.progress_buf[env_ids] == self.max_episode_length
        # Targets = last_attempted_pose, NOT cur_targets: with disable_actions the
        # sim holds the attempt pose pushed at reset for the whole episode, but
        # pre_physics_step still integrates the player's random actions into the
        # cur_targets TENSOR (a drifting random walk the sim never sees). Saving
        # cur_targets stores garbage targets that rip the grip open on restore.
        all_states = torch.cat([
            self.leap_hand_dof_pos,
            self.last_attempted_pose,
            self.root_state_tensor[self.object_indices, :7],
            self.env_object_type_id.float().unsqueeze(1),
        ], dim=1)
        self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success]])

        # ── HILL CLIMB: promote this attempt if it beat the env's previous best ─
        new_score = self._episode_fitness(env_ids)                            # [n]
        improved  = new_score > self.best_score_per_env[env_ids]
        if improved.any():
            imp_eids = env_ids[improved]
            self.best_pose_per_env[imp_eids]  = self.last_attempted_pose[imp_eids]
            self.best_score_per_env[imp_eids] = new_score[improved]
            # Snapshot surface distance for the pose that just became best.
            self._best_surf_per_env[imp_eids] = self._last_surface_dist[imp_eids]

        # ── Episode-end stats ──────────────────────────────────────────────────
        self._gs_ep_count      += len(env_ids)
        self._gs_success_count += int(success.sum().item())

        ep_force  = self._ep_force_max[env_ids]     # [n, 4]
        ep_spike  = self._ep_force_spike[env_ids]   # [n, 4]
        ep_steps  = self._ep_contact_steps[env_ids] # [n, 4]
        ep_dist_to_obj_center   = self._ep_dist_to_obj_center_min[env_ids]      # [n, 4]

        mean_fingers = (ep_steps > 0).float().sum(dim=-1).mean().item()
        mean_force   = ep_force.max(dim=-1).values.mean().item()
        spike_rate   = ((ep_spike > self.max_contact_force) & (ep_steps > 0)).float().mean().item()
        # Mask the 9999 init sentinel: the very first reset batch (all envs, before
        # any physics step) otherwise poisons the running mean — one 9998 across
        # ~570 batches is how "mean_min_fingertip_dist_m: 17.6" happens.
        _valid_dist  = ep_dist_to_obj_center[ep_dist_to_obj_center < 9000.0]
        mean_dist    = _valid_dist.mean().item() if _valid_dist.numel() > 0 else float('nan')
        mean_palm    = self._ep_palm_force_max[env_ids].mean().item()
        # Two surface-distance metrics:
        #   surf_now:  this batch's noisy attempts (high variance, can wander)
        #   surf_best: averaged BEST-pose surf across all envs (the real signal)
        mean_surf_now  = self._last_surface_dist[env_ids].mean().item()
        valid_best_surf = self._best_surf_per_env[self._best_surf_per_env < 9000]
        mean_surf_best = valid_best_surf.mean().item() if valid_best_surf.numel() > 0 else float('nan')
        # Best-score across ALL envs — should monotonically increase.
        valid_best   = self.best_score_per_env[torch.isfinite(self.best_score_per_env)]
        best_score   = valid_best.max().item() if valid_best.numel() > 0 else float('nan')

        self._gs_force_max.append(mean_force)
        self._gs_spike_rate.append(spike_rate)
        self._gs_contact_fingers.append(mean_fingers)
        if not np.isnan(mean_dist):
            self._gs_dist_min.append(mean_dist)
        self._gs_palm_force.append(mean_palm)

        sr = 100.0 * self._gs_success_count / max(1, self._gs_ep_count)
        print(
            f'cache={self.saved_grasping_states.shape[0]:<6d} | '
            f'ep={self._gs_ep_count:<7d} | '
            f'success={self._gs_success_count} ({sr:.1f}%) | '
            f'fingers={mean_fingers:.2f}/4 | '
            f'force={mean_force:.2f}N | '
            f'palm={mean_palm:.2f}N | '
            f'spike={100*spike_rate:.1f}% | '
            f'surface_now={mean_surf_now*1000:.1f}mm | '
            f'surface_best={mean_surf_best*1000:.1f}mm | '
            f'best_fit={best_score:.2f}'
        )

        # ── Periodic flush ─────────────────────────────────────────────────────
        # atexit only fires on CLEAN exit; under `uv run` SIGINT/SIGQUIT are set to
        # SIG_IGN in the child and SIGTERM/SIGKILL skip atexit, so a stopped gen
        # loses its whole in-memory cache. Flush every 50 grasps so partial progress
        # always survives any stop method.
        _bucket = self.saved_grasping_states.shape[0] // 50
        if _bucket > getattr(self, '_last_flush_bucket', 0):
            self._last_flush_bucket = _bucket
            self._save_cache_partial()

        # ── Exit when target reached ───────────────────────────────────────────
        # atexit hook in __init__ also flushes on clean exit / Ctrl+C.
        if len(self.saved_grasping_states) >= self.cfg["env"]["grasp_cache_len"]:
            print(f'[cache] reached target {self.cfg["env"]["grasp_cache_len"]} grasps — exiting.')
            exit()

        # ── Reset per-episode accumulators for these envs ──────────────────────
        self._ep_palm_force_max[env_ids] = 0.0
        self._ep_force_max[env_ids]     = 0.0
        self._ep_force_spike[env_ids]   = 0.0
        self._ep_contact_steps[env_ids] = 0
        self._ep_dist_to_obj_center_min[env_ids]      = 9999.0

        # reset object — clone preserves the configured initial orientation
        # (cols 3:7) from object_init_state, set by override_object_init_rot.
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        # FULL-tensor apply, not *_indexed: get/set_actor_rigid_body_properties in
        # the same step (mass randomization above) silently breaks the INDEXED
        # root-state apply on the GPU pipeline (see leap_hand_rot.reset_idx and
        # tools/probe_minimal_setter.py gpu massset). Gen currently runs the CPU
        # pipeline where indexed works, but keep the immune form everywhere.
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))

        # Sample next attempt around each env's *best* pose (not canonical_pose).
        # On the very first reset, best_pose_per_env is initialized to canonical,
        # so this reduces to the old behavior for the first round.
        base = self.best_pose_per_env[env_ids]                                # [n, 16]
        pos = base + self.cfg["env"]["grasp_dof_search_radius"] * rand_floats[:, 5:5 + self.num_leap_hand_dofs]
        pos = tensor_clamp(pos, self.leap_hand_dof_lower_limits[env_ids], self.leap_hand_dof_upper_limits[env_ids])

        # Remember THIS attempt so the next reset_idx can promote it on improve.
        self.last_attempted_pose[env_ids] = pos.clone()

        self.leap_hand_dof_pos[env_ids, :] = pos
        self.leap_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, :self.num_leap_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_leap_hand_dofs] = pos

        # Full-tensor applies (see root-state comment above): immune to the
        # property-setter interference; tensors hold current state for non-reset
        # envs and the new attempt pose for reset ones.
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.prev_targets))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        # GPU-pipeline poison guard: re-apply at substep 1 of the next step
        # (see leap_hand_rot.reset_idx / update_low_level_control).
        self._reapply_countdown = 2
        self._reapply_obj_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self._reapply_hand_indices = self.hand_indices[env_ids].to(torch.int32)

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    # ── Reward / contact criterion ────────────────────────────────────────────────

    def _compute_fingertip_surface_distance(self, env_ids):
        """Signed distance from each fingertip's pad point to the object surface.
        Negative = inside the mesh (penetration), 0 = on surface, positive = away.

        Returns: [n_envs_in_query, 4] torch tensor on self.device.
        """
        # Pull fingertip world transforms for just the envs being queried.
        n     = len(env_ids)
        finger_pos  = self.rigid_body_states[env_ids][:, [4, 8, 12, 16], :3]   # [n, 4, 3]
        finger_quat = self.rigid_body_states[env_ids][:, [4, 8, 12, 16], 3:7]  # [n, 4, 4] xyzw
        # print('finger_pos?', finger_pos, 'finger_quat?', finger_quat)

        # Transform fingertip-local pad point → world frame, per fingertip per env.
        # fingertip_tip_local is now [4, 3] (one pad point per fingertip type),
        # broadcast across n envs.
        tip_local_b = self.fingertip_tip_local.view(1, 4, 3).expand(n, 4, 3)
        tip_world   = quat_apply(
            finger_quat.reshape(-1, 4),
            tip_local_b.reshape(-1, 3),
        ).reshape(n, 4, 3) + finger_pos                                         # [n, 4, 3]
        # print('tip_world = apply quat + finger_pos?', tip_world)

        # Transform world → object-local (the mesh lives in object-local frame).
        obj_p = self.object_pos[env_ids].unsqueeze(1)                     # [n, 1, 3]
        obj_q = self.object_rot[env_ids].unsqueeze(1).expand(-1, 4, -1)   # [n, 4, 4]
        tip_local = quat_rotate_inverse(
            obj_q.reshape(-1, 4),
            (tip_world - obj_p).reshape(-1, 3),
        ).reshape(n, 4, 3)                                                # [n, 4, 3]
        # print('tip_local?', tip_local)

        # Un-scale the query point so it matches the unit-scale stored mesh.
        # randomizeScale applies via set_actor_scale which is a uniform scale,
        # so dividing the point position by env scale is exact.
        scales = self._obj_scales_t[env_ids].view(-1, 1, 1)               # [n, 1, 1]
        tip_local_unscaled = tip_local / scales

        # Query trimesh — CPU, batched per object type. Use UNSIGNED distance
        # because the hammer URDF (cylinder + box concatenated) is not a
        # closed manifold, so signed_distance gives unreliable signs at the
        # cylinder-box junction. Penetration is detected separately via the
        # contact-force tensor (large forces ⇒ PhysX is depenetrating).
        # Each env is queried against the mesh of the object IT holds.
        pts_np = tip_local_unscaled.reshape(-1, 3).cpu().numpy()      # [n*4, 3]
        env_oids = self.env_object_type_id[env_ids].cpu().numpy()     # [n]
        dists_np = np.empty(pts_np.shape[0], dtype=np.float64)
        for oid in np.unique(env_oids):
            pt_mask = np.repeat(env_oids == oid, 4)
            _, d, _ = trimesh.proximity.closest_point(
                self.obj_proximity_by_type[int(oid)]._mesh, pts_np[pt_mask])
            dists_np[pt_mask] = d

        # Convert back to world units (multiply by scale).
        dists = torch.from_numpy(dists_np).to(self.device, dtype=torch.float32)
        dists = dists.reshape(n, 4) * scales.squeeze(-1)                  # [n, 4], always ≥ 0
        if self._gs_ep_count < 10:  # only print early, not every reset
            print(f'[tip-verify] env0 tip_world per finger:')
            for fi in range(4):
                print(f'  finger {fi}: {tip_world[0, fi].cpu().numpy().round(4)}')
            print(f'[tip-verify] env0 link  pos  per finger:')
            for fi in range(4):
                print(f'  finger {fi}: {finger_pos[0, fi].cpu().numpy().round(4)}')
        # print('dists?', dists)
        return dists

    def _episode_fitness(self, env_ids):
        """Score = how well did this episode's attempt 'grasp' the object?
        Higher = better. Aggregates per-episode stats + end-state surface distance.

        Magnitude budget (rough, per fingertip, for an "almost contacting" grip):
          proximity term: -50 × 0.03 m × 4 fingers = -6.0   ← dominant for placement
          contact term  :  3 N × 4 fingers         = +12.0  ← rewards real touch
          duration term :  0.5 × 4 fingers         = +2.0   ← rewards stable hold
          spike penalty :  -1 to -4                          ← penalizes penetration
          palm penalty  :  -4 × palm N (≤2.5)      = 0..-10  ← steers away from palm rests
        Proximity has to be expensive enough that a "1 finger jabbing, others
        flailing" pose loses to a "4 fingers near surface, light touch" pose.
        """
        # Unsigned distance from each fingertip pad to the nearest object surface
        # point. Always ≥ 0. Penetration is captured separately by spike_pen.
        surf = self._last_surface_dist[env_ids]                           # [n, 4]
        proximity = -(surf.sum(-1)) * 50.0                                # 50/m → 1cm closer = +0.5

        contact     = self._ep_force_max[env_ids].clamp(max=10).sum(-1)
        contact_dur = self._ep_contact_steps[env_ids].float().sum(-1) / self.max_episode_length
        spike_pen   = -(self._ep_force_spike[env_ids] > self.max_contact_force).float().sum(-1)
        # Palm penalty only when the exclusion is active: a resting object bears
        # ~its weight (0.3-0.7 N) on the palm → -1.2..-2.8, comparable to losing a
        # fingertip's proximity credit, so the hill-climb prefers fingertip cages.
        palm_pen = torch.zeros_like(spike_pen)
        if self.max_palm_contact_force >= 0.0:
            palm_pen = -self._ep_palm_force_max[env_ids].clamp(max=2.5)

        return proximity + 1.0 * contact + 4.0 * contact_dur + 2.0 * spike_pen + 4.0 * palm_pen



    def compute_reward(self, actions):
        # def list_intersect(li, hash_num):
        #     # 17 is the object index
        #     # 4, 8, 12, 16 are fingertip index
        #     # return number of contact with obj_id
        #     obj_id = 17
        #     query_list = [obj_id * hash_num + 4, obj_id * hash_num + 8, obj_id * hash_num + 12, obj_id * hash_num + 16]
        #     return len(np.intersect1d(query_list, li))
        # assert self.device == 'cpu'
        # contacts = [self.gym.get_env_rigid_contacts(env) for env in self.envs]
        # print("contacts for env 0?", contacts[0].shape, contacts[0])
        # contact_list = [list_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000) for contact in contacts]
        # print("contact_list for env 0?", contact_list)
        # contact_condition = to_torch(contact_list, device=self.device)

        obj_pos: torch.Tensor    = self.rigid_body_states[:, [-1], :3]
        finger_pos: torch.Tensor = self.rigid_body_states[:, [4, 8, 12, 16], :3]

        fingertip_local_ids = {4, 8, 12, 16}
        obj_local_id = 17

        # Per-fingertip force from the object only.
        # fingertip_obj_force: force within the valid window [min, max] — used for criterion.
        # fingertip_raw_force: raw accumulated force — used to detect penetration spikes.
        # Uses get_env_rigid_contacts because body indices are local (0..N-1) in the
        # per-env API. The sim-wide get_rigid_contacts uses opaque handles, not indices.
        # Force magnitude comes from the contact's own 'lambda' field (per-contact
        # normal force). get_env_rigid_contact_forces is PER-RIGID-BODY (net force,
        # one Vec3 per body) — zipping it against the per-contact-pair array
        # attributed arbitrary bodies' net forces to fingertip contacts.
        fingertip_obj_force: torch.Tensor = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        fingertip_raw_force: torch.Tensor = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        palm_obj_force: torch.Tensor = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        fingertip_idx_to_col = {4: 0, 8: 1, 12: 2, 16: 3}
        proximal_ids = self.palm_contact_body_ids  # palm + mcp/pip segments + thumb base

        for env_idx, env in enumerate(self.envs):
            contacts: np.ndarray = self.gym.get_env_rigid_contacts(env)
            for c in contacts:
                local0: int = int(c['body0'])
                local1: int = int(c['body1'])
                is_ft_obj = (local0 == obj_local_id and local1 in fingertip_local_ids)
                is_obj_ft = (local1 == obj_local_id and local0 in fingertip_local_ids)
                if is_ft_obj or is_obj_ft:
                    finger_local: int = local1 if is_ft_obj else local0
                    col: int = fingertip_idx_to_col[finger_local]
                    raw_mag: float = abs(float(c['lambda']))
                    fingertip_raw_force[env_idx, col] += raw_mag
                    if raw_mag <= self.max_contact_force:
                        fingertip_obj_force[env_idx, col] += raw_mag
                elif (local0 == obj_local_id and local1 in proximal_ids) or \
                     (local1 == obj_local_id and local0 in proximal_ids):
                    palm_obj_force[env_idx] += abs(float(c['lambda']))

        # ── Update per-episode accumulators ───────────────────────────────────
        self._ep_force_max    = torch.maximum(self._ep_force_max,   fingertip_obj_force)
        self._ep_force_spike  = torch.maximum(self._ep_force_spike, fingertip_raw_force)
        contacted_this_step: torch.Tensor = fingertip_obj_force > self.min_contact_force
        self._ep_contact_steps += contacted_this_step.int()
        dists: torch.Tensor = torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1))  # [N, 4]
        self._ep_dist_to_obj_center_min = torch.minimum(self._ep_dist_to_obj_center_min, dists)
        self._ep_palm_force_max = torch.maximum(self._ep_palm_force_max, palm_obj_force)

        # ── Success criterion ──────────────────────────────────────────────────
        # 1) All fingertips are within finger_dist_threshold of the object centre
        cond1: torch.Tensor = (dists < self.finger_dist_threshold).all(-1)
        # 2) At least num_contact_fingers fingertips exert force within the valid window.
        #    Waived during the settle grace period (object is still falling onto the palm).
        contacting_count: torch.Tensor = contacted_this_step.sum(dim=-1)
        cond2: torch.Tensor = contacting_count >= self.num_contact_fingers
        if self.contact_grace_steps > 0:
            cond2 = torch.logical_or(cond2, self.progress_buf < self.contact_grace_steps)
        # 3) Object has not fallen below the drop threshold
        cond3: torch.Tensor = torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold)

        cond = cond1.float() * cond2.float() * cond3.float()

        # 4) Object is not resting on the palm: palm-object force stays under the
        #    limit once the grace period ends. Fingertip-borne grips only.
        if self.max_palm_contact_force >= 0.0:
            cond4 = palm_obj_force <= self.max_palm_contact_force
            if self.contact_grace_steps > 0:
                cond4 = torch.logical_or(cond4, self.progress_buf < self.contact_grace_steps)
            cond = cond * cond4.float()

        self.reset_buf[cond < 1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
