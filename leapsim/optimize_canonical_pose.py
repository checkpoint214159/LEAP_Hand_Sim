# --------------------------------------------------------
# Cross-Entropy Method optimizer for the LEAP-hand grasp canonical pose.
#
# Mirrors train.py's Hydra setup so all the usual task/env overrides work
# (task=LeapHandGrasp, task.env.object.type=hammer, etc.). Instead of
# launching rl-games, builds the task directly and iterates a CEM loop
# over the 16-DOF canonical_pose, using the existing grasp-gen pipeline's
# success count as the fitness signal.
#
# Usage from leapsim/:
#   uv run python optimize_canonical_pose.py task=LeapHandGrasp \
#       task.env.object.type=hammer task.env.baseObjScale=1.0 \
#       task.env.grasp_cache_name=leap_hand_in_palm_hammer \
#       pipeline=cpu task.env.numEnvs=4096 \
#       'task.env.override_object_init_rot=[0,0,1.5708]' \
#       +pose_optim.iterations=15
# --------------------------------------------------------

import isaacgym  # must precede any torch import; see CLAUDE.md  # noqa: F401

from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from leapsim.tasks import isaacgym_task_map
from leapsim.utils.reformat import omegaconf_to_dict
from leapsim.utils.utils import set_np_formatting, set_seed


# CEM hyperparameters — override via Hydra CLI, e.g. `+pose_optim.iterations=30`.
DEFAULTS = dict(
    iterations=20,        # CEM rounds
    candidates=8,         # samples per round
    elite=2,              # top-k used to fit the next distribution
    sigma_init=0.3,       # per-joint std dev (radians)
    sigma_floor=0.05,     # minimum sigma so the distribution doesn't collapse
    steps_per_eval=55,    # one full grasp episode + small margin
    seed_pose_jitter=0.0, # additive jitter on the seed pose before round 1
    output='cache/canonical_pose_optimized.yaml',
)


def _read_optim_cfg(cfg: DictConfig) -> dict:
    OmegaConf.set_struct(cfg, False)
    user_cfg = cfg.get('pose_optim', {}) or {}
    return {k: type(v)(user_cfg.get(k, v)) for k, v in DEFAULTS.items()}


def _force_reset_all(task) -> None:
    env_ids = torch.arange(task.num_envs, device=task.device)
    # Zero progress so the next 50 steps form one clean episode and the
    # success check inside reset_idx (progress == max_episode_length) is
    # not falsely true on this manual call.
    task.progress_buf[env_ids] = 0
    task.reset_idx(env_ids)


def _eval_pose(task, pose: np.ndarray, steps: int) -> int:
    task.canonical_pose = pose.tolist()
    task.saved_grasping_states = torch.zeros((0, 23), dtype=torch.float, device=task.device)
    _force_reset_all(task)
    actions = torch.zeros((task.num_envs, task.num_actions), device=task.device)
    for _ in range(steps):
        task.step(actions)
    return int(task.saved_grasping_states.shape[0])


@hydra.main(config_name="config", config_path="./cfg")
def main(cfg: DictConfig) -> None:
    set_np_formatting()
    set_seed(cfg.seed)
    optim = _read_optim_cfg(cfg)

    # The grasp task auto-saves the cache and calls exit() when
    # grasp_cache_len is reached. Push it past anything we'd produce so
    # the loop stays in our hands.
    cfg.task.env.grasp_cache_len = int(1e9)
    cfg.task.env.genGrasps = True
    # Rerun would burn disk for hundreds of evaluations — turn it off.
    if hasattr(cfg.task.env, 'rerun'):
        cfg.task.env.rerun.enabled = False

    cfg_dict = omegaconf_to_dict(cfg.task)
    cfg_dict['env']['numEnvs'] = int(cfg.task.env.numEnvs)

    task = isaacgym_task_map[cfg.task_name](
        cfg=cfg_dict,
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    seed_pose = np.array(task.canonical_pose, dtype=np.float32)
    n_dof = seed_pose.shape[0]
    if optim['seed_pose_jitter'] > 0:
        seed_pose = seed_pose + np.random.normal(0.0, optim['seed_pose_jitter'], size=n_dof).astype(np.float32)

    mean = seed_pose.copy()
    sigma = np.full(n_dof, optim['sigma_init'], dtype=np.float32)

    print(f"[optim] task={cfg.task_name} num_envs={task.num_envs} device={task.device}")
    print(f"[optim] seed pose: {seed_pose.tolist()}")
    print(f"[optim] iter={optim['iterations']} K={optim['candidates']} elite={optim['elite']} sigma_init={optim['sigma_init']}")

    # Baseline: how well does the seed pose itself do? Useful sanity check.
    baseline_fit = _eval_pose(task, seed_pose, optim['steps_per_eval'])
    print(f"[optim] baseline (seed pose) successes/episode = {baseline_fit}/{task.num_envs}")

    best_pose, best_fit = seed_pose.copy(), baseline_fit
    history = []

    for it in range(optim['iterations']):
        # Always include the current mean so progress is monotonic in best-fit.
        candidates = np.random.normal(mean, sigma, size=(optim['candidates'], n_dof)).astype(np.float32)
        candidates[0] = mean
        fitnesses = np.zeros(optim['candidates'], dtype=np.int32)

        for k in range(optim['candidates']):
            fitnesses[k] = _eval_pose(task, candidates[k], optim['steps_per_eval'])

        order = np.argsort(-fitnesses)
        elite = candidates[order[:optim['elite']]]
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), optim['sigma_floor'])

        top_k = int(fitnesses.max())
        top_idx = int(np.argmax(fitnesses))
        if top_k > best_fit:
            best_fit = top_k
            best_pose = candidates[top_idx].copy()

        rate_pct = 100.0 * top_k / task.num_envs
        print(f"[optim] iter {it+1:02d}/{optim['iterations']}: best_this_round={top_k} ({rate_pct:.1f}%) "
              f"mean_fit={fitnesses.mean():.1f} sigma_mean={sigma.mean():.3f}")
        history.append({'iter': it + 1, 'best': top_k, 'mean': float(fitnesses.mean())})

    print()
    print(f"[optim] best overall: {best_fit}/{task.num_envs} successes "
          f"({100.0 * best_fit / task.num_envs:.1f}%) — baseline was {baseline_fit}")
    print("[optim] best pose:")
    print(yaml.safe_dump({'canonical_pose': best_pose.tolist()}, default_flow_style=False))

    out_path = Path(optim['output'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.safe_dump(
            {
                'canonical_pose': best_pose.tolist(),
                'baseline_successes': baseline_fit,
                'best_successes': best_fit,
                'num_envs': task.num_envs,
                'history': history,
            },
            f,
            default_flow_style=False,
        )
    print(f"[optim] wrote {out_path}")


if __name__ == '__main__':
    main()
