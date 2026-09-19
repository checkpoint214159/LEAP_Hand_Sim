#!/usr/bin/env python3
"""probe_z1_analysis.py — probing the crosscat_C_z1 checkpoint (no retraining).

One Hydra-app process per condition (same pattern as train.py / run_crosscat.sh's
eval_one — the only IsaacGym construction path proven reliable in this
environment; a prior version of this tool built the env by hand
(OmegaConf.load + manual vecenv/env_configurations registration, bypassing
@hydra.main's bootstrap) and segfaulted (NULL deref inside
libcarb.gym.plugin.so) on every attempt after the first, regardless of GPU
load — train.py itself and plain CLI eval commands never once crashed. Don't
go back to the manual-construction approach without understanding that first.

Answers, using ONLY the already-trained runs/crosscat_C_z1_2026-07-28_07-45-42
checkpoint (cross-category: trained on cylinder+sphere, evaluated zero-shot on
cuboid_train):

  1. Saliency  — does the policy's first read of the observation (the GRU's
     input-hidden weight matrix) and the local input-gradient of the mean
     action actually depend on the z1-specific columns (inertia ratios r2,r3
     + normalized SA/V), or does it ignore them?
  2. Ablation  — masking each z1 sub-group to 0 at eval time, which piece
     costs the most zero-shot performance / brings back the launch-failure
     mode?
  3. Force-vs-shape regression — across the 4 zero-shot cuboid_train objects
     (which vary in r2/r3/SA-V), does commanded force / launch rate actually
     correlate with those specific geometric numbers?

Architecture note: the checkpoint is RECURRENT (GRU, hidden=256, before_mlp,
layer_norm) — see a2c_network.rnn.rnn.weight_ih_l0 (768,126) in the state
dict. So "first layer" for saliency purposes is the GRU's input-hidden
matrix, not the downstream MLP (which reads the GRU's hidden state, not raw
obs). z rides the 3-step obs history (numObservations 126 = 3*42; each
42-wide per-step block ends with the z_dim=8 z1 vector, replicated
identically across the 3 blocks) — see leap_hand_rot.py _setup_z_channel /
compute_observations. Column indices are read live from the task object
(task._z_obs_start, task.z_dim), never hand-derived.

Run each condition as its own process from src/LEAP_Hand_Sim/leapsim, via
tools/run_probe_z1.sh (drives all conditions + aggregates), OR by hand:

  uv run python tools/probe_z1_analysis.py test=true \\
    checkpoint=runs/crosscat_C_z1_2026-07-28_07-45-42/nn/crosscat_C_z1.pth \\
    num_envs=256 task.env.object.type=cuboid_train \\
    task.env.grasp_cache_name=leap_hand_in_cuboid_train task.env.z_mode=z1 \\
    +probe.tag=baseline +probe.steps=220 +probe.zmask=[] +probe.do_item1=true \\
    +probe.out=tools/logs/probe_z1_cond_baseline.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import isaacgym  # noqa: F401 — must precede torch
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

import leapsim  # noqa: F401 — registers OmegaConf resolvers
from leapsim.utils.reformat import omegaconf_to_dict
from leapsim.utils.env_creator import EnvCreator
from leapsim.utils.rerun_algo_observer import RerunAlgoObserver
from leapsim.utils.rlgames_utils import RLGPUEnv
from leapsim.runner import CustomRunner
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import _restore

LEAPSIM_DIR = Path(__file__).resolve().parent.parent

# z1 sub-group column indices WITHIN the 8-d z vector (see _build_z_features):
#   [0]=scale [1:4]=bbox ext (3) [4]=fill  -> z0 (5-d)
#   [5]=r2 (lambda2/lambda1)  [6]=r3 (lambda3/lambda1)  [7]=normalized SA/V
Z0_IDX = list(range(0, 5))
RATIO_IDX = [5, 6]
SAV_IDX = [7]


def rollout(player, steps, snapshot_steps=()):
    """Drive the env with the loaded policy; log per-step diagnostics.

    snapshot_steps: timesteps at which to save (obs, rnn_states) BEFORE
    get_action is called, for the later gradient-saliency pass (so the
    hidden state matches the obs — a real, not synthetic, (state, obs) pair).
    """
    env = player.env  # BasePlayer.create_env() returns the raw task directly
    task = env
    obses = player.env_reset(env)
    player.get_batch_size(obses, 1)
    if player.is_rnn:
        player.init_rnn()

    max_ep_len = float(task.max_episode_length)
    prev_progress = task.progress_buf.clone()

    log = {
        'peak_torque': [], 'linvel': [], 'done': [], 'obj_type': [],
        'ep_len_at_done': [],
    }
    snaps = {}
    for t in range(steps):
        if t in snapshot_steps:
            snaps[t] = (obses.detach().clone(),
                        [s.detach().clone() for s in player.states] if player.is_rnn else None)

        action = player.get_action(obses, is_determenistic=False)
        obses, r, done, info = player.env_step(env, action)

        log['peak_torque'].append(task.torques.detach().abs().max(dim=-1).values.cpu())
        log['linvel'].append(task.object_linvel.detach().norm(dim=-1).cpu())
        log['done'].append(done.cpu())
        log['obj_type'].append(task.env_object_type_id.detach().cpu())
        ep_len = torch.where(done.cpu().bool(), prev_progress.cpu().float() + 1,
                              torch.full_like(prev_progress.cpu().float(), float('nan')))
        log['ep_len_at_done'].append(ep_len)
        prev_progress = task.progress_buf.clone()

        if player.is_rnn:
            done_idx = done.nonzero(as_tuple=False)
            if len(done_idx) > 0:
                for s in player.states:
                    s[:, done_idx, :] = s[:, done_idx, :] * 0.0

    log = {k: torch.stack(v, dim=0) for k, v in log.items()}  # [T, N]
    return task, log, snaps, max_ep_len


def item1_weight_norms(task, ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
    w = sd['a2c_network.rnn.rnn.weight_ih_l0']  # [3*hidden, numObservations]
    col_norm = w.norm(dim=0)  # [numObservations]

    z_dim = task.z_dim
    n_obs = col_norm.shape[0]
    frame = n_obs // 3
    z_start = frame - z_dim
    blocks = [col_norm[b * frame: (b + 1) * frame] for b in range(3)]

    out = {}
    for b, block in enumerate(blocks):
        proprio = block[:z_start]
        z = block[z_start:]
        out[f'frame{b}_proprio_mean_colnorm'] = proprio.mean().item()
        out[f'frame{b}_proprio_max_colnorm'] = proprio.max().item()
        out[f'frame{b}_z0_colnorms'] = z[Z0_IDX].tolist()
        out[f'frame{b}_ratio_colnorms'] = z[RATIO_IDX].tolist()
        out[f'frame{b}_sav_colnorm'] = z[SAV_IDX].tolist()
    return out


def item1_gradient_saliency(player, task, snaps):
    device = next(player.model.parameters()).device
    z_dim = task.z_dim
    z_start = task._z_obs_start

    # cuDNN's fused GRU backward refuses to run against an eval-mode forward
    # (no reserve buffer saved), so the GRU leaf alone needs train() for the
    # backward pass. Flipping the WHOLE model to train() would also flip
    # RunningMeanStd's forward branch, which recomputes running_mean/var from
    # the current batch and folds that batch-dependent statistic into the
    # normalization — cross-sample coupling that would invalidate the
    # sum-then-backward per-sample-gradient trick below. The GRU has no
    # children of its own, so toggling just this leaf changes nothing else.
    gru = player.model.a2c_network.rnn.rnn
    gru.train()

    # ModelA2CContinuousLogStd.Network.norm_obs() wraps the RunningMeanStd
    # call in `with torch.no_grad()`, severing the autograd edge from the raw
    # observation to mu entirely (by design). Bypass model.norm_obs() and
    # replicate its normalization formula by hand from the frozen, loaded
    # running_mean/var buffers, then call a2c_network directly.
    rms = player.model.running_mean_std
    mean = rms.running_mean.float().to(device)
    var = rms.running_var.float().to(device)
    eps = rms.epsilon

    grads = []
    for t, (obs_t, states_t) in snaps.items():
        obs_t = obs_t.to(device).clone().requires_grad_(True)
        normed = torch.clamp((obs_t - mean) / torch.sqrt(var + eps), min=-5.0, max=5.0)
        input_dict = {'obs': normed, 'rnn_states': states_t, 'seq_length': 1}
        mu, logstd, value, states = player.model.a2c_network(input_dict)
        mu.sum().backward()
        grads.append(obs_t.grad.detach().abs().cpu())  # [N, numObs]
    gru.eval()

    g = torch.cat(grads, dim=0)  # [snapshots*N, numObs]
    frame_width = g.shape[1] // 3
    z_grad_total = sum(g[:, b * frame_width + z_start: b * frame_width + z_start + z_dim] for b in range(3))
    proprio_grad = torch.cat([
        g[:, b * frame_width: b * frame_width + z_start] for b in range(3)
    ], dim=1)

    return {
        'proprio_mean_abs_grad': proprio_grad.mean().item(),
        'proprio_max_abs_grad': proprio_grad.max(dim=1).values.mean().item(),
        'z0_mean_abs_grad': z_grad_total[:, Z0_IDX].mean(dim=0).tolist(),
        'ratio_mean_abs_grad': z_grad_total[:, RATIO_IDX].mean(dim=0).tolist(),
        'sav_mean_abs_grad': z_grad_total[:, SAV_IDX].mean(dim=0).tolist(),
        'n_samples': g.shape[0],
    }


def summarize_rollout(task, log, max_ep_len, label):
    obj_ids = log['obj_type'][0]  # constant across time, [N]
    names = task.object_type_list
    per_obj = {}
    for oid in torch.unique(obj_ids).tolist():
        mask_env = (obj_ids == oid)
        ep_len = log['ep_len_at_done'][:, mask_env]
        finished = ep_len[~torch.isnan(ep_len)]
        early_frac = float('nan')
        if finished.numel() > 0:
            early_frac = (finished < (max_ep_len - 0.5)).float().mean().item()
        peak_linvel = log['linvel'][:, mask_env].max(dim=0).values
        peak_torque = log['peak_torque'][:, mask_env].max(dim=0).values
        shp = task._obj_shape1.get(int(oid))
        per_obj[names[int(oid)]] = {
            'n_episodes_finished': int(finished.numel()),
            'early_termination_frac': early_frac,
            'mean_ep_len': finished.mean().item() if finished.numel() > 0 else float('nan'),
            'mean_peak_linvel': peak_linvel.mean().item(),
            'mean_peak_torque': peak_torque.mean().item(),
            'r2': float(shp[0]) if shp is not None else None,
            'r3': float(shp[1]) if shp is not None else None,
            'sav': float(shp[2]) if shp is not None else None,
        }
    print(f"--- {label} ---")
    for name, d in per_obj.items():
        print(f"  {name:12s} n={d['n_episodes_finished']:4d} early_term={d['early_termination_frac']:.3f} "
              f"ep_len={d['mean_ep_len']:.1f} peak_linvel={d['mean_peak_linvel']:.3f} "
              f"peak_torque={d['mean_peak_torque']:.3f}  r2={d['r2']} r3={d['r3']} sav={d['sav']}")
    return per_obj


def item3_force_regression(per_obj_baseline):
    """Simple cross-object correlation: does r2/r3/sav predict launch rate / peak force?"""
    rows = [(v['r2'], v['r3'], v['sav'], v['early_termination_frac'], v['mean_peak_linvel'], v['mean_peak_torque'])
            for v in per_obj_baseline.values() if v['r2'] is not None]
    if len(rows) < 3:
        return {'note': 'too few distinct objects for correlation'}
    arr = np.array(rows, dtype=np.float64)
    r2, r3, sav, early, linvel, torque = arr.T
    out = {}
    for fname, fvals in [('r2', r2), ('r3', r3), ('sav', sav)]:
        for tname, tvals in [('early_termination_frac', early), ('peak_linvel', linvel), ('peak_torque', torque)]:
            if np.std(fvals) < 1e-8 or np.std(tvals) < 1e-8:
                corr = float('nan')
            else:
                corr = float(np.corrcoef(fvals, tvals)[0, 1])
            out[f'corr({fname},{tname})'] = corr
    return out


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig) -> None:
    probe = cfg.probe
    tag = probe.tag
    steps = int(probe.steps)
    zmask = list(probe.get('zmask', []) or [])
    do_item1 = bool(probe.get('do_item1', False))
    out_path = Path(probe.out)

    run_name = f"probe_z1_{tag}"
    vecenv.register(
        'RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': EnvCreator(cfg, run_name),
    })

    runner = CustomRunner(RerunAlgoObserver(output_dir=str(LEAPSIM_DIR / "tools/logs/probe_rerun")))
    runner.load(omegaconf_to_dict(cfg.train))
    runner.reset()
    player = runner.create_player()
    _restore(player, {'checkpoint': cfg.checkpoint, 'sigma': None})
    player.reset()

    task = player.env
    print(f"[{tag}] z_mode={task.z_mode} z_dim={task.z_dim} "
          f"numObservations={task.cfg['env']['numObservations']} "
          f"z_obs_start={getattr(task, '_z_obs_start', None)} max_episode_length={task.max_episode_length}")

    if zmask:
        with torch.no_grad():
            task.z0_features[:, zmask] = 0.0
        print(f"[{tag}] zeroed z1 indices {zmask}; env0 z now = "
              f"{task.z0_features[0].cpu().numpy().round(4).tolist()}")
    else:
        print(f"[{tag}] env0 z1 = {task.z0_features[0].cpu().numpy().round(4).tolist()} "
              f"object_type_list={task.object_type_list}")

    snap_steps = []
    if do_item1:
        snap_steps = sorted(set(min(steps - 1, s) for s in (10, steps // 3, 2 * steps // 3, steps - 5)))
    task_ref, log, snaps, max_ep_len = rollout(player, steps, snapshot_steps=snap_steps)

    results = {'per_object': summarize_rollout(task_ref, log, max_ep_len, tag)}

    if do_item1:
        print(f"\n=== [{tag}] ITEM 1a: first-layer (GRU input-hidden) weight-column norms ===")
        results['item1_weights'] = item1_weight_norms(task_ref, cfg.checkpoint)
        print(json.dumps(results['item1_weights'], indent=2))

        print(f"\n=== [{tag}] ITEM 1b: input-gradient saliency of mean action w.r.t. z1 ===")
        results['item1_gradient'] = item1_gradient_saliency(player, task_ref, snaps)
        print(json.dumps(results['item1_gradient'], indent=2))

        print(f"\n=== [{tag}] ITEM 3: force/launch-rate vs geometric feature correlation ===")
        results['item3_correlations'] = item3_force_regression(results['per_object'])
        print(json.dumps(results['item3_correlations'], indent=2))

    out_path = out_path if out_path.is_absolute() else LEAPSIM_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[{tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
