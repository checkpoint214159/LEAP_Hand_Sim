# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# Thin SAC subclass that fixes where TensorBoard summaries are written.
#
# rl_games' stock SACAgent builds its SummaryWriter at
#   runs/<name>_<dd-HH-MM-SS>
# which ignores the `full_experiment_name` train.py assigns, so every SAC run
# spawns a stray, date-less duplicate directory containing only a tfevents file.
# PPO (rl_games a2c_common) and our TD3Agent both log into the run's own
# `summaries/` folder. This subclass redirects SAC to do the same and removes
# the stray directory the parent constructor created.

import os
import shutil

from torch.utils.tensorboard import SummaryWriter

from rl_games.algos_torch.sac_agent import SACAgent as _SACAgent


class SACAgent(_SACAgent):

    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        stray_logdir = self.writer.log_dir
        self.writer.close()
        self.writer = SummaryWriter(self.summaries_dir)

        # Drop the stray runs/<name>_<dd-HH-MM-SS> directory the parent created.
        if os.path.abspath(stray_logdir) != os.path.abspath(self.summaries_dir):
            shutil.rmtree(stray_logdir, ignore_errors=True)
