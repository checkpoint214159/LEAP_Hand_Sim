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
# Both feed-forward and recurrent (GRU/LSTM) policies are supported. The RNN
# path caches per-step hidden states during rollout (play_steps_rnn override) so
# the leader's value re-estimate over replayed follower transitions is a batched
# one-step forward; followers' sequence-start hidden states seed the leader's
# replay, matching the authors' rl_games fork.

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
    # recurrent rollout: cache the per-step hidden states so the leader's
    # value re-estimate over replayed follower transitions is a batched
    # one-step forward (rather than a sequential re-roll).
    # ------------------------------------------------------------------ #
    def play_steps_rnn(self):
        self._rnn_step_states = []
        self._capture_rnn = True
        batch_dict = super().play_steps_rnn()
        self._capture_rnn = False
        # [num_rnn_tensors] each stacked to [H, *state_shape]
        self._rnn_state_buffer = [
            torch.stack([step[i] for step in self._rnn_step_states], dim=0)
            for i in range(len(self._rnn_step_states[0]))
        ]
        return batch_dict

    def get_action_values(self, obs):
        # self.rnn_states here is h_n (the state that will process obs_n).
        if getattr(self, '_capture_rnn', False):
            self._rnn_step_states.append([s.detach().clone() for s in self.rnn_states])
        return super().get_action_values(obs)

    # ------------------------------------------------------------------ #
    # dataset augmentation: leader gets followers' replayed experience
    # ------------------------------------------------------------------ #
    def prepare_dataset(self, batch_dict):
        augmented = self._augment_with_followers(batch_dict)
        # The follower value re-estimate runs the model in eval mode; restore the
        # train mode that base train_epoch set before prepare_dataset (a GRU
        # backward pass requires training mode).
        self.set_train()

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

        H, bs, D = self.horizon_length, self.chunk_size, self.conditioning_dim

        # mb tensors held by the experience buffer after play_steps: [H, num_actors, ...]
        mb_obses = self.experience_buffer.tensor_dict['obses']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_dones = self.experience_buffer.tensor_dict['dones'].float()
        last_obs = self.obs['obs']
        num_seqs = H // self.seq_len

        follower_blocks = np.random.choice(range(1, self.num_chunks),
                                           self.off_policy_ratio, replace=False)

        # On-policy keys to carry through (env-major flattened, [num_actors*H, ...]).
        flat_keys = [k for k in ['obses', 'actions', 'neglogpacs', 'values',
                                 'mus', 'sigmas', 'dones', 'returns'] if k in batch_dict]
        out = {k: [batch_dict[k]] for k in flat_keys}
        out_mask = [torch.zeros(batch_dict['returns'].shape[0], dtype=torch.bool, device=self.ppo_device)]
        # rnn_states are per-game [layers, num_games, hidden]; concatenate follower games.
        out_rnn = [[s] for s in batch_dict['rnn_states']] if self.is_rnn else None

        self.set_eval()
        for j in follower_blocks:
            j = int(j)
            env_ids = torch.arange(j * bs, (j + 1) * bs, device=self.ppo_device)
            row = slice(j * bs * H, (j + 1) * bs * H)  # env-major flattened slice

            # Relabel this block's observations with the leader's conditioning code,
            # then recompute the leader critic's values for TD(0) targets.
            obs_blk = mb_obses[:, env_ids, :].clone()          # [H, bs, aug]
            obs_blk[:, :, -D:] = self.leader_code
            last_blk = last_obs[env_ids].clone()               # [bs, aug]
            last_blk[:, -D:] = self.leader_code

            v, v_last = self._leader_values(obs_blk, last_blk, env_ids)   # [H,bs,1], [bs,1]
            v_next = torch.cat([v[1:], v_last.unsqueeze(0)], dim=0)
            returns_blk = mb_rewards[:, env_ids, :] + self.gamma * v_next * (1.0 - mb_dones[:, env_ids]).unsqueeze(-1)

            out['obses'].append(swap_and_flatten01(obs_blk))
            out['values'].append(swap_and_flatten01(v))
            out['returns'].append(swap_and_flatten01(returns_blk))
            for k in ['actions', 'neglogpacs', 'mus', 'sigmas', 'dones']:
                out[k].append(batch_dict[k][row])
            out_mask.append(torch.ones(bs * H, dtype=torch.bool, device=self.ppo_device))
            if self.is_rnn:
                gslice = slice(j * bs * num_seqs, (j + 1) * bs * num_seqs)
                for i, s in enumerate(batch_dict['rnn_states']):
                    out_rnn[i].append(s[:, gslice, :])

        augmented = {k: torch.cat(v, dim=0) for k, v in out.items()}
        augmented['off_policy_mask'] = torch.cat(out_mask, dim=0)
        if self.is_rnn:
            augmented['rnn_states'] = [torch.cat(s, dim=1) for s in out_rnn]
        for k in batch_dict:
            if k not in augmented:
                augmented[k] = batch_dict[k]

        if self.shuffle_augmented:
            self._shuffle_augmented(augmented, flat_keys)

        # Diagnostic: fraction of the leader's batch that is off-policy.
        self.last_off_policy_frac = augmented['off_policy_mask'].float().mean().item()
        return augmented

    def _leader_values(self, obs_blk, last_blk, env_ids):
        """Leader-critic values for a follower block. obs_blk [H, bs, aug],
        last_blk [bs, aug]. Returns (v [H, bs, 1], v_last [bs, 1])."""
        H, bs = obs_blk.shape[0], obs_blk.shape[1]
        with torch.no_grad():
            if not self.is_rnn:
                v = self._model_values(obs_blk.reshape(-1, obs_blk.shape[-1]), None).reshape(H, bs, -1)
                v_last = self._model_values(last_blk, None)
                return v, v_last

            # Recurrent: use the cached per-step hidden states (h_t that generated
            # obs_t), sliced to this block, as one-step conditioning contexts.
            # buffer[i]: [H, layers, num_actors, hidden] -> [layers, bs*H, hidden] (env-major)
            step_states = [buf[:, :, env_ids, :].permute(1, 2, 0, 3).reshape(buf.shape[1], bs * H, buf.shape[-1])
                           for buf in self._rnn_state_buffer]
            obs_flat = swap_and_flatten01(obs_blk)                       # [bs*H, aug] env-major
            v_flat = self._model_values(obs_flat, step_states)          # [bs*H, 1]
            v = v_flat.reshape(bs, H, -1).permute(1, 0, 2).contiguous()  # [H, bs, 1]
            # last step bootstrap uses the post-rollout hidden state for these envs
            last_states = [s[:, env_ids, :] for s in self.rnn_states]
            v_last = self._model_values(last_blk, last_states)          # [bs, 1]
            return v, v_last

    def _model_values(self, obs, rnn_states, chunk=8192):
        """One-step value estimates for [N, aug] obs under optional [layers, N, hidden]
        rnn_states, chunked to bound memory."""
        self.model.eval()
        outs = []
        for s in range(0, obs.shape[0], chunk):
            e = s + chunk
            input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs': self._preproc_obs(obs[s:e]),
                'rnn_states': [r[:, s:e, :] for r in rnn_states] if rnn_states is not None else None,
            }
            outs.append(self.model(input_dict)['values'])
        return torch.cat(outs, dim=0)

    def _shuffle_augmented(self, augmented, flat_keys):
        if self.is_rnn:
            seq = self.seq_len
            num_games = augmented['returns'].shape[0] // seq
            perm_g = torch.randperm(num_games, device=self.ppo_device)
            flat_perm = (perm_g.view(-1, 1) * seq
                         + torch.arange(seq, device=self.ppo_device).view(1, -1)).reshape(-1)
            for k in flat_keys:
                augmented[k] = augmented[k][flat_perm]
            augmented['off_policy_mask'] = augmented['off_policy_mask'][flat_perm]
            augmented['rnn_states'] = [s[:, perm_g, :].contiguous() for s in augmented['rnn_states']]
        else:
            perm = torch.randperm(augmented['returns'].shape[0], device=self.ppo_device)
            for k in flat_keys:
                augmented[k] = augmented[k][perm]
            augmented['off_policy_mask'] = augmented['off_policy_mask'][perm]
