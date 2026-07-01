# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import datetime
import isaacgym

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from leapsim.utils.reformat import omegaconf_to_dict, print_dict
from leapsim.utils.utils import set_np_formatting, set_seed
from leapsim.utils.rlgames_utils import RLGPUEnv
from leapsim.utils.rerun_algo_observer import RerunAlgoObserver
from leapsim.utils.env_creator import EnvCreator
from leapsim.runner import CustomRunner
from rl_games.common import env_configurations, vecenv
import shutil


@hydra.main(config_name="config", config_path="./cfg")
def main_with_cfg_overrides(cfg: DictConfig) -> None:
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.default_run_name}_{time_str}"

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device  = f'cuda:{rank}'

    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    if cfg.wandb_activate and rank == 0:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            resume="allow",
        )
        run_name = wandb.run.name

    runs_dir: Path      = Path(to_absolute_path(cfg.runs_dir))
    experiment_dir: Path = runs_dir / run_name
    rerun_dir: Path      = experiment_dir / "rerun"

    cfg.train.params.config.full_experiment_name = run_name

    if cfg.task.env.rerun.enabled:
        cfg.task.env.rerun.output_dir = str(rerun_dir)

    vecenv.register(
        'RLGPU',
        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': EnvCreator(cfg, run_name),
    })

    runner = CustomRunner(RerunAlgoObserver(output_dir=rerun_dir))
    runner.load(omegaconf_to_dict(cfg.train))
    runner.reset()

    # Snapshot configs alongside the run for reproducibility. The train config
    # filename is the actually-selected one (e.g. LeapHandRotPPO / ...SAC /
    # ...TD3), not a hardcoded PPO name.
    train_cfg_name: str = HydraConfig.get().runtime.choices["train"]
    leapsim_cfg_dir: Path = Path(__file__).parent / "cfg"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        leapsim_cfg_dir / "task"  / f"{cfg.task_name}.yaml",
        experiment_dir            / f"{cfg.task_name}.yaml",
    )
    shutil.copyfile(
        leapsim_cfg_dir / "train" / f"{train_cfg_name}.yaml",
        experiment_dir            / f"{train_cfg_name}.yaml",
    )
    with open(experiment_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    if cfg.wandb_activate and rank == 0:
        wandb.save(str(experiment_dir / "config.yaml"))
        wandb.save(str(experiment_dir / f"{cfg.task_name}.yaml"))
        wandb.save(str(experiment_dir / f"{train_cfg_name}.yaml"))

    if cfg.multi_gpu:
        import horovod.torch as hvd
        rank = hvd.rank()
    else:
        rank = 0

    os.system("rm -rf ~/.isaacgym/vhacd")

    runner.run({
        'train':      not cfg.test,
        'play':       cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma':      None,
    })

    if cfg.wandb_activate and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main_with_cfg_overrides()
