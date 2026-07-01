"""runner.py — Custom rl-games Runner subclass.

Subclasses Runner so we have a stable extension point for overriding
run_train / run_play without touching upstream rl-games code. AMP builder
registration lives here so train.py stays thin.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from rl_games.torch_runner import Runner, _override_sigma, _restore
from rl_games.algos_torch import model_builder
from rl_games.common.algo_observer import AlgoObserver

from leapsim.learning import amp_continuous, amp_models, amp_network_builder, amp_players
from leapsim.learning import td3_agent, td3_models, td3_network_builder, td3_players
from leapsim.learning import sac_agent
from leapsim.learning import sapg_agent, sapg_players

logger = logging.getLogger(__name__)


class CustomRunner(Runner):
    """Runner with AMP builders pre-registered and overridable run hooks."""

    def __init__(self, algo_observer: AlgoObserver) -> None:
        super().__init__(algo_observer)
        self.algo_factory.register_builder(
            'amp_continuous', lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        self.player_factory.register_builder(
            'amp_continuous', lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model(
            'continuous_amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        model_builder.register_network(
            'amp', lambda **kwargs: amp_network_builder.AMPBuilder())

        # Override rl_games' stock SAC with our subclass so it logs into the
        # run's summaries/ dir instead of spawning a stray run directory.
        self.algo_factory.register_builder(
            'sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))

        # TD3 is not shipped by rl_games, so register it locally.
        self.algo_factory.register_builder(
            'td3', lambda **kwargs: td3_agent.TD3Agent(**kwargs))
        self.player_factory.register_builder(
            'td3', lambda **kwargs: td3_players.TD3PlayerContinuous(**kwargs))
        model_builder.register_model(
            'td3', lambda network, **kwargs: td3_models.ModelTD3Continuous(network))
        model_builder.register_network(
            'td3', lambda **kwargs: td3_network_builder.TD3Builder())

        # SAPG re-derived on rl_games 1.5.2 (split + leader-follower aggregation).
        # It reuses the stock continuous_a2c_logstd model / actor_critic network
        # (enlarged input), so only the algo and player need registering.
        self.algo_factory.register_builder(
            'sapg', lambda **kwargs: sapg_agent.SAPGAgent(**kwargs))
        self.player_factory.register_builder(
            'sapg', lambda **kwargs: sapg_players.SAPGPlayerContinuous(**kwargs))

    def run_train(self, args: Dict[str, Any]) -> None:
        logger.info("======= TRAINING COMMENCED =======")
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        self._announce_checkpoint(args, role="train")
        _restore(agent, args)
        _override_sigma(agent, args)
        agent.train()

    def run_play(self, args: Dict[str, Any]) -> None:
        logger.info("======= PLAY COMMENCED =======")
        player = self.create_player()
        self._announce_checkpoint(args, role="play")
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    @staticmethod
    def _announce_checkpoint(args: Dict[str, Any], role: str) -> None:
        ckpt = args.get('checkpoint')
        if ckpt:
            print(f"[CustomRunner] {role}: loading checkpoint from {ckpt}")
        else:
            print(f"[CustomRunner] {role}: starting from scratch (no checkpoint)")
