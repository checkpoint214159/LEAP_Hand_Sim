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
