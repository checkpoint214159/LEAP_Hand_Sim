"""env_creator.py — Callable env factory for rl-games registration.

rl-games calls env_creator() with no meaningful arguments. EnvCreator binds
the Hydra config and run name at construction so it satisfies that zero-arg
contract without needing a closure in train.py.

leapsim is imported inside __call__ to break the circular import that would
occur if it were imported at module level (env_creator lives inside leapsim).
"""
from __future__ import annotations

from typing import Any

import gym
from omegaconf import DictConfig


class EnvCreator:
    """Zero-argument callable that creates a leapsim vectorised environment.

    Args:
        cfg:      Full Hydra config (the DictConfig from @hydra.main).
        run_name: Experiment run name, used only for video output paths.
    """

    def __init__(self, cfg: DictConfig, run_name: str) -> None:
        self._cfg      = cfg
        self._run_name = run_name

    def __call__(self, **kwargs: Any) -> Any:
        import leapsim  # deferred — leapsim.__init__ registers OmegaConf resolvers
        cfg = self._cfg
        envs = leapsim.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{self._run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs
