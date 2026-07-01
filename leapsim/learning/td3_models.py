# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# TD3 model wrapper, mirroring rl_games' ModelSACContinuous
# (rl_games/algos_torch/models.py). Wraps the TD3Builder network and exposes the
# actor / actor_target / critic / critic_target plus input normalization
# (norm_obs) inherited from BaseModelNetwork.

from rl_games.algos_torch.models import BaseModel, BaseModelNetwork


class ModelTD3Continuous(BaseModel):

    def __init__(self, network):
        BaseModel.__init__(self, 'td3')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, td3_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.td3_network = td3_network

        def critic(self, obs, action):
            return self.td3_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.td3_network.critic_target(obs, action)

        def actor(self, obs):
            return self.td3_network.actor(obs)

        def actor_target(self, obs):
            return self.td3_network.actor_target(obs)

        def is_rnn(self):
            return False

        def forward(self, input_dict):
            obs = input_dict['obs']
            return self.actor(obs)
