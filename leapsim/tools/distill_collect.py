"""distill_collect.py — Phase B: collect behaviour-cloning data from a specialist.

Rolls ONE proprio specialist (z_mode=none, GRU) out on its own object inside a
z_mode=z1 environment, and records ``(z1-augmented obs, specialist mean action)``
pairs. The specialist only ever sees the proprio slice (we strip z before
feeding it — see distill_lib.strip_z); the z1 features are captured in the stored
observation so the STUDENT can later learn to map [proprio, z] -> specialist
action. z is static per object, so one z_mode=z1 env yields both the specialist's
native proprio input AND the student's target obs with zero reconstruction.

The env is driven by the specialist's STOCHASTIC (sampled) action so the visited
state distribution matches how the specialist actually operates and is evaluated
(deterministic eval under-reads this config — see the stage0 memo); the recorded
BC TARGET is the specialist's MEAN action mu (a clean, noise-free regression
target whose conditional mean equals the sampled-action mean).

Run from src/LEAP_Hand_Sim/leapsim. Example (CPU smoke):

  uv run python tools/distill_collect.py task=LeapHandRot headless=true \
    wandb_activate=false log_to_sheet=false pipeline=cpu sim_device=cpu \
    rl_device=cpu graphics_device_id=-1 num_envs=16 \
    task.env.object.type=cuboid_train +task.env.object_id_whitelist=[0] \
    task.env.grasp_cache_name=leap_hand_in_cuboid_train task.env.z_mode=z1 \
    task.env.randomization.randomizeMass=False \
    task.env.randomization.randomizeCOM=False \
    task.env.randomization.randomizeFriction=False \
    task.env.randomization.randomizeScaleList=[1.0] +task.env.scale_list_jitter=0 \
    checkpoint=runs/<specialist>/nn/<specialist>.pth \
    +collect.out=tools/distill_data/smoke_cuboid0.npz +collect.max_pairs=3000
"""
import isaacgym  # noqa: F401  (MUST precede torch — gymdeps import-order constraint)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make tools/ importable

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from leapsim.utils.reformat import omegaconf_to_dict
import leapsim  # registers OmegaConf resolvers; provides make()
import distill_lib as D


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig) -> None:
    teacher_ckpt = cfg.checkpoint
    assert teacher_ckpt, "pass checkpoint=<specialist .pth>"
    teacher_ckpt = hydra.utils.to_absolute_path(teacher_ckpt)
    out = OmegaConf.select(cfg, "collect.out")
    assert out, "pass +collect.out=<path.npz>"
    out = hydra.utils.to_absolute_path(out)
    max_pairs = int(OmegaConf.select(cfg, "collect.max_pairs") or 150000)
    max_steps = int(OmegaConf.select(cfg, "collect.max_steps") or 2000)
    warmup = int(OmegaConf.select(cfg, "collect.warmup") or 0)
    drive_det = bool(OmegaConf.select(cfg, "collect.drive_deterministic") or False)

    device = cfg.rl_device
    env = leapsim.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs, cfg.sim_device,
                       cfg.rl_device, cfg.graphics_device_id, cfg.headless,
                       cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg)
    num_envs = env.num_envs
    z_dim = env.z_dim
    assert z_dim > 0, "collect in a z_mode env (z1) so the student obs carries z"
    base_per_frame = env._num_obs_base // D.NFRAMES
    proprio_dim = D.NFRAMES * base_per_frame
    full_dim = env.num_obs
    print(f"[collect] num_envs={num_envs} z_mode={env.z_mode} z_dim={z_dim} "
          f"proprio_dim={proprio_dim} full_dim={full_dim}")

    # ── Build + load the (recurrent) specialist. It consumes the 102-d proprio
    #    slice; input_shape is proprio_dim, not the full z1 obs. ────────────────
    params = omegaconf_to_dict(cfg.train.params)
    teacher = D.build_a2c_model(params, obs_dim=proprio_dim, actions_num=env.num_actions,
                                num_seqs=num_envs, device=device,
                                normalize_input=True, normalize_value=True)
    ckpt = D.load_model_weights(teacher, teacher_ckpt, device)
    teacher.eval()
    teacher_logstd = ckpt["model"]["a2c_network.sigma"].detach().cpu().numpy().astype(np.float32)
    rms_in = teacher.running_mean_std.running_mean.numel()
    assert rms_in == proprio_dim, (
        f"specialist input {rms_in} != proprio_dim {proprio_dim} — is this a z_mode=none specialist?")
    print(f"[collect] loaded specialist {os.path.basename(teacher_ckpt)}; "
          f"teacher sigma(mean)={np.exp(teacher_logstd).mean():.3f}")

    # ── Reset ALL envs from the grasp cache, then validate the z layout. ───────
    env.reset_buf[:] = 1
    obs = env.reset()["obs"].to(device)
    assert obs.shape[1] == full_dim
    # the z block of frame 0 must equal the static per-env z features
    zblk = obs[:, base_per_frame:base_per_frame + z_dim]
    zref = env.z0_features.to(device)
    if not torch.allclose(zblk, zref, atol=1e-4):
        # tolerate clip: z may be clamped by clipObservations; check strip instead
        print(f"[collect][warn] z block != z0_features (max diff "
              f"{(zblk - zref).abs().max().item():.4f}); continuing (clip/scale).")
    # strip round-trips to proprio width
    assert D.strip_z(obs, z_dim, base_per_frame).shape[1] == proprio_dim

    states = D.init_rnn_states(teacher, num_envs, device)

    obs_chunks, mu_chunks = [], []
    n_pairs = 0
    n_eps = 0
    ids_present = np.unique(env.env_object_type_id.cpu().numpy()).tolist()
    for step in range(max_steps):
        proprio = D.strip_z(obs, z_dim, base_per_frame)
        mu, sampled, states = D.teacher_forward(teacher, proprio, states)
        drive = mu if drive_det else sampled
        if step >= warmup:
            obs_chunks.append(obs.detach().cpu().clone())
            mu_chunks.append(mu.detach().cpu().clone())
            n_pairs += obs.shape[0]
        obs, _, done, _ = env.step(torch.clamp(drive, -1.0, 1.0))
        obs = obs["obs"].to(device)
        D.reset_done_states(states, done)
        n_eps += int(done.sum().item())
        if n_pairs >= max_pairs:
            break

    obs_all = torch.cat(obs_chunks, dim=0)[:max_pairs].numpy().astype(np.float32)
    mu_all = torch.cat(mu_chunks, dim=0)[:max_pairs].numpy().astype(np.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, obs=obs_all, mu=mu_all,
        z_mode=str(env.z_mode), z_dim=np.int64(z_dim),
        base_per_frame=np.int64(base_per_frame), nframes=np.int64(D.NFRAMES),
        proprio_dim=np.int64(proprio_dim), full_dim=np.int64(full_dim),
        object_ids=np.array(ids_present, dtype=np.int64),
        object_type=str(cfg.task.env.object.type),
        teacher_logstd=teacher_logstd,
        teacher_ckpt=os.path.basename(teacher_ckpt),
    )
    print(f"[collect] wrote {obs_all.shape[0]} pairs (obs {obs_all.shape[1]}-d, "
          f"mu {mu_all.shape[1]}-d) over ~{n_eps} episodes -> {out}")


if __name__ == "__main__":
    main()
