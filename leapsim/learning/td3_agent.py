# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# Twin Delayed DDPG (TD3, Fujimoto et al. 2018). rl_games ships SAC but not TD3,
# so this agent lives locally and is registered in runner.py (CustomRunner).
# It is adapted from rl_games' SACAgent (rl_games/algos_torch/sac_agent.py):
# the env interaction loop, replay buffer, observation normalization and
# checkpoint plumbing are reused almost verbatim. TD3-specific differences vs
# SAC:
#   * deterministic tanh actor (no entropy / temperature alpha)
#   * Gaussian exploration noise added to actions during data collection
#   * target-policy smoothing: clipped noise on the target actor's next action
#   * delayed policy updates: actor + target nets updated every policy_freq
#     critic updates

import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import experience, vecenv
from rl_games.interfaces.base_algorithm import BaseAlgorithm


class TD3Agent(BaseAlgorithm):

    def __init__(self, base_name, params):
        self.config = config = params['config']
        print(config)

        self.load_networks(params)
        self.base_init(base_name, config)

        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = config["critic_tau"]
        self.actor_tau = config.get("actor_tau", self.critic_tau)
        self.batch_size = config["batch_size"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.normalize_input = config.get("normalize_input", False)
        self.max_env_steps = config.get("max_env_steps", 1000)

        # TD3 exploration / target-smoothing / delayed-update hyperparameters.
        self.expl_noise = config.get("expl_noise", 0.1)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_freq = config.get("policy_freq", 2)

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.action_range = [
            float(action_space.low.min()),
            float(action_space.high.max()),
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.Adam(
            self.model.td3_network.actor.parameters(),
            lr=float(self.config['actor_lr']),
            betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizer = torch.optim.Adam(
            self.model.td3_network.critic.parameters(),
            lr=float(self.config["critic_lr"]),
            betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.replay_buffer = experience.VectorizedReplayBuffer(
            self.env_info['observation_space'].shape,
            self.env_info['action_space'].shape,
            self.replay_buffer_size,
            self._device)

        self.step = 0
        self.total_it = 0  # counts critic updates, drives delayed policy updates
        self.algo_observer = config['features']['observer']

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self._device = config.get('device', 'cuda:0')

        # temporary for Isaac gym compatibility
        self.ppo_device = self._device
        print('Env info:')
        print(self.env_info)

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.is_train = config.get('is_train', True)

        self.c_loss = nn.MSELoss()

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', 1e6)

        self.network = config['network']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs = None

        self.frame = 0
        self.update_time = 0
        self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0

        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        self.train_dir = config.get('train_dir', 'runs')

        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter(self.summaries_dir)
        print("Run Directory:", self.experiment_name)

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)

    @property
    def device(self):
        return self._device

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def get_weights(self):
        state = {
            'actor': self.model.td3_network.actor.state_dict(),
            'actor_target': self.model.td3_network.actor_target.state_dict(),
            'critic': self.model.td3_network.critic.state_dict(),
            'critic_target': self.model.td3_network.critic_target.state_dict(),
        }
        if self.normalize_input:
            state['running_mean_std'] = self.model.running_mean_std.state_dict()
        return state

    def get_full_state_weights(self):
        state = self.get_weights()
        state['step'] = self.step
        state['total_it'] = self.total_it
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        return state

    def set_weights(self, weights):
        self.model.td3_network.actor.load_state_dict(weights['actor'])
        self.model.td3_network.actor_target.load_state_dict(weights['actor_target'])
        self.model.td3_network.critic.load_state_dict(weights['critic'])
        self.model.td3_network.critic_target.load_state_dict(weights['critic_target'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.step = weights.get('step', 0)
        self.total_it = weights.get('total_it', 0)
        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            # Target-policy smoothing: clipped noise on the target action.
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.model.actor_target(next_obs) + noise).clamp(*self.action_range)

            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.gamma * target_Q)
            target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.model.critic(obs, action)
        critic1_loss = self.c_loss(current_Q1, target_Q)
        critic2_loss = self.c_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach(), critic1_loss.detach(), critic2_loss.detach()

    def update_actor(self, obs):
        for p in self.model.td3_network.critic.parameters():
            p.requires_grad = False

        action = self.model.actor(obs)
        actor_Q1, _ = self.model.critic(obs, action)
        actor_loss = -actor_Q1.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.model.td3_network.critic.parameters():
            p.requires_grad = True

        return actor_loss.detach()

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, step):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
        not_done = ~done

        obs = self.preproc_obs(obs)
        next_obs = self.preproc_obs(next_obs)

        self.total_it += 1
        critic_loss, critic1_loss, critic2_loss = self.update_critic(
            obs, action, reward, next_obs, not_done)

        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            actor_loss = self.update_actor(obs)
            self.soft_update_params(
                self.model.td3_network.critic, self.model.td3_network.critic_target, self.critic_tau)
            self.soft_update_params(
                self.model.td3_network.actor, self.model.td3_network.actor_target, self.actor_tau)

        return actor_loss, critic1_loss, critic2_loss

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        obs = self.model.norm_obs(obs)
        return obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert self.observation_space.dtype != np.int8
            obs = torch.FloatTensor(obs).to(self._device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:
            upd_obs = {'obs': upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        self.step += self.num_actors
        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos
        else:
            return (torch.from_numpy(obs).to(self._device),
                    torch.from_numpy(rewards).to(self._device),
                    torch.from_numpy(dones).to(self._device), infos)

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        return obs

    def act(self, obs, action_dim, sample=False):
        obs = self.preproc_obs(obs)
        action = self.model.actor(obs)
        if sample:
            noise = torch.randn_like(action) * self.expl_noise
            action = action + noise
        action = action.clamp(*self.action_range)
        assert action.ndim == 2
        return action

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def play_steps(self, random_exploration=False):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        critic1_losses = []
        critic2_losses = []

        obs = self.obs
        for s in range(self.num_steps_per_episode):
            self.set_eval()
            if random_exploration:
                action = torch.rand(
                    (self.num_actors, *self.env_info["action_space"].shape),
                    device=self._device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.time()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_end = time.time()

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            if isinstance(obs, dict):
                obs = obs['obs']
            if isinstance(next_obs, dict):
                next_obs = next_obs['obs']

            rewards = self.rewards_shaper(rewards)
            self.replay_buffer.add(
                obs, action, torch.unsqueeze(rewards, 1), next_obs, torch.unsqueeze(dones, 1))

            self.obs = obs = next_obs.clone()

            if not random_exploration:
                self.set_train()
                update_time_start = time.time()
                actor_loss, critic1_loss, critic2_loss = self.update(self.epoch_num)
                update_time_end = time.time()
                update_time = update_time_end - update_time_start

                if actor_loss is not None:
                    actor_losses.append(actor_loss)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
            else:
                update_time = 0

            total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_losses, critic1_losses, critic2_losses

    def train_epoch(self):
        random_exploration = self.epoch_num < self.num_warmup_steps
        return self.play_steps(random_exploration)

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        self.last_mean_rewards = -100500
        total_time = 0
        self.frame = 0
        self.obs = self.env_reset()

        while True:
            self.epoch_num += 1
            (step_time, play_time, update_time, epoch_total_time,
             actor_losses, critic1_losses, critic2_losses) = self.train_epoch()

            total_time += epoch_total_time

            curr_frames = self.num_frames_per_epoch
            self.frame += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time

            if self.print_stats:
                print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} '
                      f'fps total: {fps_total:.0f} epoch: {self.epoch_num}/{self.max_epochs}')

            self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
            self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
            self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
            self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
            self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
            self.writer.add_scalar('performance/step_time', step_time, self.frame)

            if self.epoch_num >= self.num_warmup_steps:
                self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), self.frame)
                self.writer.add_scalar('losses/c2_loss', torch_ext.mean_list(critic2_losses).item(), self.frame)
                if len(actor_losses) > 0:
                    self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), self.frame)

            self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
            self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                self.writer.add_scalar('rewards/step', mean_rewards, self.frame)
                self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, self.frame)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)
                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, self.config['name']))
                    if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                        print('Network won!')
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        return self.last_mean_rewards, self.epoch_num

            if self.epoch_num >= self.max_epochs:
                self.save(os.path.join(
                    self.nn_dir,
                    'last_' + self.config['name'] + 'ep' + str(self.epoch_num) + 'rew' + str(self.last_mean_rewards)))
                print('MAX EPOCHS NUM!')
                return self.last_mean_rewards, self.epoch_num
