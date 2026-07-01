# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# TD3 inference player, mirroring rl_games' SACPlayer
# (rl_games/algos_torch/players.py). Runs the deterministic actor (no
# exploration noise) for evaluation / `test=true` rollouts.

import torch

from rl_games.algos_torch import torch_ext
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs


class TD3PlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max()),
        ]

        obs_shape = self.obs_shape
        self.normalize_input = self.config.get('normalize_input', False)
        config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': False,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.td3_network.actor.load_state_dict(checkpoint['actor'])
        self.model.td3_network.critic.load_state_dict(checkpoint['critic'])
        self.model.td3_network.critic_target.load_state_dict(checkpoint['critic_target'])
        if 'actor_target' in checkpoint:
            self.model.td3_network.actor_target.load_state_dict(checkpoint['actor_target'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def get_action(self, obs, is_determenistic=False):
        if self.has_batch_dimension is False:
            obs = unsqueeze_obs(obs)
        obs = self.model.norm_obs(obs)
        action = self.model.actor(obs)
        action = action.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension is False:
            action = torch.squeeze(action.detach())
        return action

    def reset(self):
        pass
