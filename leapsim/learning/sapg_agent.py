# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# SAPG: Split and Aggregate Policy Gradients (Singla, Agarwal, Pathak, ICML 2024;
# arXiv 2407.20230, code github.com/jayeshs999/sapg). On-policy algorithm for the
# massively-parallel-env regime where vanilla PPO saturates.
#
# This is a re-derivation of the *core* SAPG (ir_type=none) onto the installed
# rl_games 1.5.2 A2CAgent — registered locally in runner.py like AMP/SAC/TD3,
# rather than adopting the authors' rl_games-1.6.1 fork. The per-chunk entropy
# intrinsic-reward stratification is deliberately NOT implemented here (follow-up).
#
# Method (Algorithm 1, §4.3-4.4):
#   * Split num_actors into M chunks. A fixed per-chunk conditioning code (a
#     sinusoidal encoding) is appended to observations, so the single shared
#     backbone realizes M behaviorally-distinct policies pi_j.
#   * Leader = chunk 0 (fixed). Loss = sum_j OnPolicyPPO(D_j)  [every chunk trains
#     on its own data]  +  OffPolicyPPO(D_1'), where D_1' is follower transitions
#     replayed under the leader's conditioning code with TD(0) targets from the
#     leader's critic.
#   * The off-policy term needs no special loss: PPO's clipped ratio
#     exp(old_neglogp - new_neglogp) = pi_1 / pi_j (old_neglogp is the follower's
#     stored behaviour log-prob, new is the leader's) IS the clipped importance
#     weight. So calc_gradients is inherited from A2CAgent unchanged; all SAPG
#     logic lives in obs conditioning (env_step/env_reset) + dataset augmentation
#     (prepare_dataset).
#
# NOTE: first cut is feed-forward (MLP) only; RNN aggregation is a follow-up.

import gym
import numpy as np
import torch
from torch import optim

from rl_games.algos_torch import a2c_continuous
from rl_games.common.a2c_common import swap_and_flatten01


def _sinusoidal_encoding(values: torch.Tensor, dim: int, n: float = 100.0) -> torch.Tensor:
    """dim-dimensional sinusoidal encoding of a 1-D tensor of scalars.

    Mirrors the authors' create_sinusoidal_encoding (rl_games/common/custom_utils.py).
    """
    assert dim % 2 == 0, "conditioning_dim must be even"
    denom = n ** (2 * torch.arange(dim // 2, dtype=torch.float32, device=values.device) / dim)
    return torch.cat([torch.sin(values.unsqueeze(-1) / denom),
                      torch.cos(values.unsqueeze(-1) / denom)], dim=-1)


class SAPGAgent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, params):
        # Builds model at the raw obs shape, optimizer, dataset, etc.
        super().__init__(base_name, params)

        sapg_cfg = self.config.get('sapg', {})
        self.num_chunks = int(sapg_cfg.get('num_chunks', 6))
        self.conditioning_dim = int(sapg_cfg.get('conditioning_dim', 32))
        # number of follower chunks the leader replays per update
        self.off_policy_ratio = int(sapg_cfg.get('off_policy_ratio', self.num_chunks - 1))
        self.shuffle_augmented = bool(sapg_cfg.get('shuffle_augmented', True))

        assert self.num_actors % self.num_chunks == 0, (
            f"num_actors ({self.num_actors}) must be divisible by num_chunks "
            f"({self.num_chunks}); set num_envs accordingly.")
        if self.is_rnn:
            raise NotImplementedError(
                "SAPG first cut is MLP-only; disable the network.rnn block in the "
                "SAPG train config. RNN aggregation is a planned follow-up.")

        self.chunk_size = self.num_actors // self.num_chunks
        self.off_policy_ratio = min(self.off_policy_ratio, self.num_chunks - 1)

        # Fixed per-chunk conditioning codes. genvec spreads chunks apart on a
        # scalar axis (50..0); the sinusoidal encoding turns each into a distinct
        # D-dim code. Codes vary across chunks, so input-normalization preserves
        # them (non-zero variance across the batch).
        block_of_env = torch.arange(self.num_chunks, device=self.ppo_device).repeat_interleave(self.chunk_size)
        genvec = torch.linspace(50.0, 0.0, self.num_chunks, device=self.ppo_device)[block_of_env]
        self.coef_embd = _sinusoidal_encoding(genvec, self.conditioning_dim)   # [num_actors, D]
        self.leader_code = self.coef_embd[0].clone()                           # block 0 = leader

        # Rebuild the model with the enlarged input (obs + conditioning code) and
        # a matching optimizer. The standard actor_critic / continuous_a2c_logstd
        # network handles the larger input; no custom network is needed.
        raw_obs_dim = self.obs_shape[0]
        aug_dim = raw_obs_dim + self.conditioning_dim
        aug_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(aug_dim,), dtype=np.float32)
        self.observation_space = aug_space
        self.env_info['observation_space'] = aug_space   # so ExperienceBuffer sizes obses to aug_dim
        self.obs_shape = (aug_dim,)

        build_config = {
            'actions_num': self.actions_num,
            'input_shape': (aug_dim,),
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.init_rnn_from_model(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr),
                                    eps=1e-08, weight_decay=self.weight_decay)
        if self.normalize_value:
            self.value_mean_std = self.model.value_mean_std

    # ------------------------------------------------------------------ #
    # observation conditioning: append the per-env chunk code to obs
    # ------------------------------------------------------------------ #
    def _append_code(self, obs):
        obs['obs'] = torch.cat([obs['obs'], self.coef_embd], dim=1)
        return obs

    def env_step(self, actions):
        obs, rewards, dones, infos = super().env_step(actions)
        return self._append_code(obs), rewards, dones, infos

    def env_reset(self):
        return self._append_code(super().env_reset())

    # ------------------------------------------------------------------ #
    # dataset augmentation: leader gets followers' replayed experience
    # ------------------------------------------------------------------ #
    def prepare_dataset(self, batch_dict):
        augmented = self._augment_with_followers(batch_dict)

        # Resize the PPO dataset to the (larger) augmented batch. Any remainder
        # not divisible by minibatch_size is dropped (leader on-policy data is
        # placed first, so only some trailing off-policy samples are unused).
        self.dataset.batch_size = augmented['returns'].shape[0]
        self.dataset.length = max(1, self.dataset.batch_size // self.dataset.minibatch_size)

        if getattr(self, 'writer', None) is not None and hasattr(self, 'last_off_policy_frac'):
            self.writer.add_scalar('sapg/off_policy_frac', self.last_off_policy_frac, self.frame)

        super().prepare_dataset(augmented)

    def _augment_with_followers(self, batch_dict):
        if self.off_policy_ratio <= 0:
            return batch_dict

        H, bs = self.horizon_length, self.chunk_size
        gamma = self.gamma
        D = self.conditioning_dim

        # mb tensors held by the experience buffer after play_steps: [H, num_actors, ...]
        mb_obses = self.experience_buffer.tensor_dict['obses']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_dones = self.experience_buffer.tensor_dict['dones'].float()
        last_obs = self.obs['obs']

        follower_blocks = np.random.choice(range(1, self.num_chunks),
                                           self.off_policy_ratio, replace=False)

        # On-policy keys to carry through (env-major flattened, [num_actors*H, ...]).
        flat_keys = ['obses', 'actions', 'neglogpacs', 'values', 'mus', 'sigmas', 'dones', 'returns']
        out = {k: [batch_dict[k]] for k in flat_keys if k in batch_dict}
        out_mask = [torch.zeros(batch_dict['returns'].shape[0], dtype=torch.bool, device=self.ppo_device)]

        self.set_eval()
        for j in follower_blocks:
            env_ids = torch.arange(j * bs, (j + 1) * bs, device=self.ppo_device)
            row = slice(int(j) * bs * H, (int(j) + 1) * bs * H)  # env-major flattened slice

            # Relabel this block's observations with the leader's conditioning code,
            # then recompute leader-critic values for TD(0) targets.
            obs_blk = mb_obses[:, env_ids, :].clone()
            obs_blk[:, :, -D:] = self.leader_code
            last_blk = last_obs[env_ids].clone()
            last_blk[:, -D:] = self.leader_code

            with torch.no_grad():
                v = self.get_values({'obs': obs_blk.reshape(-1, obs_blk.shape[-1])}).reshape(H, bs, -1)
                v_last = self.get_values({'obs': last_blk})
            v_next = torch.cat([v[1:], v_last.unsqueeze(0)], dim=0)
            rew_blk = mb_rewards[:, env_ids, :]
            done_blk = mb_dones[:, env_ids]
            returns_blk = rew_blk + gamma * v_next * (1.0 - done_blk).unsqueeze(-1)

            out['obses'].append(swap_and_flatten01(obs_blk))
            out['values'].append(swap_and_flatten01(v))
            out['returns'].append(swap_and_flatten01(returns_blk))
            for k in ['actions', 'neglogpacs', 'mus', 'sigmas', 'dones']:
                out[k].append(batch_dict[k][row])
            out_mask.append(torch.ones(bs * H, dtype=torch.bool, device=self.ppo_device))

        augmented = {k: torch.cat(v, dim=0) for k, v in out.items()}
        augmented['off_policy_mask'] = torch.cat(out_mask, dim=0)
        for k in batch_dict:
            if k not in augmented:
                augmented[k] = batch_dict[k]

        if self.shuffle_augmented:
            perm = torch.randperm(augmented['returns'].shape[0], device=self.ppo_device)
            for k in flat_keys:
                augmented[k] = augmented[k][perm]
            augmented['off_policy_mask'] = augmented['off_policy_mask'][perm]

        # Diagnostic: fraction of the leader's batch that is off-policy.
        self.last_off_policy_frac = augmented['off_policy_mask'].float().mean().item()
        return augmented
