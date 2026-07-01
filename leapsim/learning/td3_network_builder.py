# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# TD3 network builder. rl_games ships SAC but not TD3, so this lives locally and
# is registered in runner.py (CustomRunner), mirroring how the AMP builders are
# registered. It closely follows rl_games' SACBuilder
# (rl_games/algos_torch/network_builder.py): the twin-Q critic is reused
# verbatim (DoubleQCritic), the only structural difference is a *deterministic*
# tanh actor instead of SAC's squashed-Gaussian actor, plus a target actor.

import torch
from torch import nn

from rl_games.algos_torch.network_builder import NetworkBuilder, DoubleQCritic


class DeterministicActor(NetworkBuilder.BaseNetwork):
    """Deterministic MLP policy that maps observations to actions in [-1, 1]."""

    def __init__(self, output_dim, **mlp_args):
        super().__init__()

        self.trunk = self._build_mlp(**mlp_args)
        last_layer = list(self.trunk.children())[-2].out_features
        self.trunk = nn.Sequential(
            *list(self.trunk.children()),
            nn.Linear(last_layer, output_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.trunk(obs)


class TD3Builder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = TD3Builder.Network(self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            obs_dim = kwargs.pop('obs_dim')
            action_dim = kwargs.pop('action_dim')
            self.num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            actor_mlp_args = {
                'input_size': obs_dim,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer,
            }

            critic_mlp_args = {
                'input_size': obs_dim + action_dim,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer,
            }

            print("Building TD3 Actor")
            self.actor = self._build_actor(action_dim, **actor_mlp_args)
            self.actor_target = self._build_actor(action_dim, **actor_mlp_args)

            if self.separate:
                print("Building TD3 Critic")
                self.critic = self._build_critic(1, **critic_mlp_args)
                print("Building TD3 Critic Target")
                self.critic_target = self._build_critic(1, **critic_mlp_args)

            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            # Sync targets *after* initialization so they start identical to the
            # live networks (the init loop above touches both copies).
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

        def _build_critic(self, output_dim, **mlp_args):
            return DoubleQCritic(output_dim, **mlp_args)

        def _build_actor(self, output_dim, **mlp_args):
            return DeterministicActor(output_dim, **mlp_args)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            return self.actor(obs)

        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.separate = params.get('separate', True)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
