# --------------------------------------------------------
# local_z_network_builder.py — tiny jointly-trained encoder on z_mode='local'.
#
# Tests the RMA (Kumar et al. 2021)/HORA (Qi et al. 2022) mechanism: a small
# MLP encoder mapping raw privileged features -> a compact code, trained
# END-TO-END with the RL policy (gradients from the task loss shape the code),
# as opposed to a frozen hand-designed feature the input layer can only
# reweight. See docs/research-plan-object-generalization.md Stage 2, "tiny
# jointly-trained encoder on the local feature" — the cheap alternative to
# building a full z2 point-cloud/pretraining pipeline.
#
# Mechanically: subclasses rl_games' A2CBuilder/A2CBuilder.Network exactly
# like learning/amp_network_builder.py does, but instead of adding an
# auxiliary discriminator head, it PREPROCESSES the observation before
# delegating to the parent forward() — splits each of the 3 stacked history
# frames into [proprio(34) | raw_local_z(16)], runs raw_local_z through a
# small trainable nn.Sequential, and reassembles [proprio(34) | z_code(k)]
# before handing off to the stock GRU+MLP (which is sized for the SMALLER,
# post-encoder width — see __init__).
#
# IMPORTANT: only valid with task.env.z_mode=local (raw_z_dim=16). Using it
# with z0/z1/none will fail the width assertion in __init__ by design — this
# builder is specifically for testing the local-vs-global x learned-vs-frozen
# 2x2 from the plan doc, not a general-purpose z encoder.
# --------------------------------------------------------

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn

# Must match leap_hand_rot.py's z-channel layout (34 proprio dims/frame:
# 16 dof pos + 16 PD targets + 2 phase) and tools/distill_lib.py's
# BASE_PER_FRAME/NFRAMES constants — same convention, kept here to avoid a
# tools/ <-> learning/ cross-import.
BASE_PER_FRAME = 34
NFRAMES = 3


class LocalZBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, name, **kwargs):
        # A2CBuilder.build() hardcodes `A2CBuilder.Network(...)` by literal
        # class name rather than `type(self).Network`/`self.Network` — without
        # this override (mirrors amp_network_builder.py's AMPBuilder.build),
        # inheriting it silently constructs the PARENT's plain Network and
        # this class's z_encoder is never built at all.
        return LocalZBuilder.Network(self.params, **kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            local_z_cfg = params.get('local_z', {})
            self._raw_z_dim = int(local_z_cfg.get('raw_z_dim', 16))
            self._encoded_dim = int(local_z_cfg.get('encoded_dim', 8))
            hidden = list(local_z_cfg.get('hidden', [16]))
            self._frame_width = BASE_PER_FRAME + self._raw_z_dim
            self._compressed_frame_width = BASE_PER_FRAME + self._encoded_dim

            raw_input_shape = kwargs['input_shape']
            raw_width = raw_input_shape[0]
            expected = NFRAMES * self._frame_width
            assert raw_width == expected, (
                f"LocalZBuilder: obs width {raw_width} != expected {expected} "
                f"({NFRAMES} frames * ({BASE_PER_FRAME}+{self._raw_z_dim})). "
                f"This builder only makes sense with task.env.z_mode=local — "
                f"check the task config and local_z.raw_z_dim.")

            # Build the parent GRU+MLP sized for the COMPRESSED width — the
            # rest of A2CBuilder.Network.__init__ (rnn/actor_mlp/mu/value
            # sizing) reads kwargs['input_shape'], so this is the one hook
            # point needed to make the whole downstream network smaller.
            compressed_kwargs = dict(kwargs)
            compressed_kwargs['input_shape'] = (NFRAMES * self._compressed_frame_width,)
            super().__init__(params, **compressed_kwargs)

            enc_layers = []
            in_size = self._raw_z_dim
            for h in hidden:
                enc_layers += [nn.Linear(in_size, h), nn.ELU()]
                in_size = h
            enc_layers.append(nn.Linear(in_size, self._encoded_dim))
            self.z_encoder = nn.Sequential(*enc_layers)
            for m in self.z_encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias)

        def load(self, params):
            super().load(params)
            self._local_z_params = params.get('local_z', {})

        def _encode_obs(self, obs):
            """[B, NFRAMES*(34+raw_z_dim)] -> [B, NFRAMES*(34+encoded_dim)],
            per-frame raw local-z replaced by its jointly-trained encoding.
            `obs` here is ALREADY normalized (RunningMeanStd runs in the
            outer ModelA2CContinuousLogStd wrapper, sized to the TRUE raw
            observation width, before this network ever sees it) — the
            encoder therefore learns on normalized raw features, same as
            every other weight in the network.
            """
            frames = []
            for f in range(NFRAMES):
                start = f * self._frame_width
                proprio = obs[:, start:start + BASE_PER_FRAME]
                raw_z = obs[:, start + BASE_PER_FRAME:start + self._frame_width]
                z_enc = self.z_encoder(raw_z)
                frames.append(torch.cat([proprio, z_enc], dim=-1))
            return torch.cat(frames, dim=-1)

        def forward(self, obs_dict):
            obs_dict = dict(obs_dict)
            obs_dict['obs'] = self._encode_obs(obs_dict['obs'])
            return super().forward(obs_dict)
