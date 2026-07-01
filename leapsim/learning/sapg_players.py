# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# SAPG inference player. At test time we run the leader policy (chunk 0): the
# observation is augmented with the fixed leader conditioning code before being
# fed to the shared backbone (mirrors SAPGAgent's obs conditioning).

import torch

from rl_games.algos_torch.players import PpoPlayerContinuous, rescale_actions
from rl_games.common.tr_helpers import unsqueeze_obs

from leapsim.learning.sapg_agent import _sinusoidal_encoding


class SAPGPlayerContinuous(PpoPlayerContinuous):
    def __init__(self, params):
        super().__init__(params)

        sapg_cfg = self.config.get('sapg', {})
        self.conditioning_dim = int(sapg_cfg.get('conditioning_dim', 32))

        # Leader = chunk 0; its genvec value is linspace(50, 0, M)[0] == 50.0.
        self.leader_code = _sinusoidal_encoding(
            torch.tensor([50.0], device=self.device), self.conditioning_dim)[0]

        # Rebuild the model with the enlarged (obs + conditioning) input.
        raw_obs_dim = self.obs_shape[0]
        aug_dim = raw_obs_dim + self.conditioning_dim
        self.obs_shape = (aug_dim,)
        config = {
            'actions_num': self.actions_num,
            'input_shape': (aug_dim,),
            'num_seqs': self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_determenistic=False):
        was_batched = self.has_batch_dimension
        if not was_batched:
            obs = unsqueeze_obs(obs)
        obs = torch.cat([obs, self.leader_code.expand(obs.shape[0], -1)], dim=1)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        current_action = mu if is_determenistic else action
        if not was_batched:
            current_action = torch.squeeze(current_action.detach())
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high,
                                   torch.clamp(current_action, -1.0, 1.0))
        return current_action
