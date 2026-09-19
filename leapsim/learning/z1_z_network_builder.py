# --------------------------------------------------------
# z1_z_network_builder.py — tiny jointly-trained encoder on z_mode='z1'.
#
# The LEARNED-GLOBAL control cell of the plan's 2x2 = {static-global,
# dynamic-local} x {frozen, learned}. It is an EXACT analogue of
# learning/local_z_network_builder.py (LocalZBuilder) EXCEPT it consumes the
# 8-d STATIC/GLOBAL z1 sub-vector of the observation (whole-object analytic
# shape: z0 scalars + inertia-eigen ratios + normalized SA/V) instead of the
# 16-d DYNAMIC/LOCAL per-fingertip feature.
#
# Why it exists (docs/research-plan-object-generalization.md Stage 2): the
# first positive result was learned-LOCAL (LocalZBuilder on z_mode=local)
# beating proprio zero-shot, AND the jump was raw-local 0.090 -> learned-local
# 0.133 with IDENTICAL features -> the jointly-trained encoder (RMA/HORA axis),
# not locality, may be the active ingredient. This builder puts the SAME kind
# of small MLP encoder on the GLOBAL z1 feature to decide:
#   learned-global HELPS  -> encoder/joint-training is what matters (locality
#                            incidental);
#   learned-global does NOT help (~frozen z1 0.097) -> BOTH locality AND the
#                            encoder are needed.
#
# Mechanically identical to LocalZBuilder: subclasses A2CBuilder /
# A2CBuilder.Network, preprocesses the observation before delegating to the
# parent forward() — splits each of the 3 stacked history frames into
# [proprio(34) | raw_z1(8)], runs raw_z1 through a small trainable
# nn.Sequential, and reassembles [proprio(34) | z_code(k)] before handing off
# to the stock GRU+MLP (sized for the post-encoder width — see __init__).
#
# IMPORTANT: only valid with task.env.z_mode=z1 (raw_z_dim=8). Using it with
# none/z0/local will fail the width assertion in __init__ by design.
# --------------------------------------------------------

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn

# Must match leap_hand_rot.py's z-channel layout (34 proprio dims/frame:
# 16 dof pos + 16 PD targets + 2 phase) — same convention as
# local_z_network_builder.py, kept here to avoid a cross-import.
BASE_PER_FRAME = 34
NFRAMES = 3


class Z1EncBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, name, **kwargs):
        # A2CBuilder.build() hardcodes `A2CBuilder.Network(...)` by literal
        # class name rather than `type(self).Network`/`self.Network` — without
        # this override (mirrors amp_network_builder.py / local_z_network_
        # builder.py), inheriting it silently constructs the PARENT's plain
        # Network and this class's z_encoder is never built at all, so the
        # encoder gets no gradient. This override is load-bearing.
        return Z1EncBuilder.Network(self.params, **kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            z1_cfg = params.get('z1_z', {})
            self._raw_z_dim = int(z1_cfg.get('raw_z_dim', 8))
            self._encoded_dim = int(z1_cfg.get('encoded_dim', 8))
            hidden = list(z1_cfg.get('hidden', [16]))
            self._frame_width = BASE_PER_FRAME + self._raw_z_dim
            self._compressed_frame_width = BASE_PER_FRAME + self._encoded_dim

            raw_input_shape = kwargs['input_shape']
            raw_width = raw_input_shape[0]
            expected = NFRAMES * self._frame_width
            assert raw_width == expected, (
                f"Z1EncBuilder: obs width {raw_width} != expected {expected} "
                f"({NFRAMES} frames * ({BASE_PER_FRAME}+{self._raw_z_dim})). "
                f"This builder only makes sense with task.env.z_mode=z1 — "
                f"check the task config and z1_z.raw_z_dim.")

            # Build the parent GRU+MLP sized for the POST-ENCODER width — the
            # rest of A2CBuilder.Network.__init__ (rnn/actor_mlp/mu/value
            # sizing) reads kwargs['input_shape'], so this is the one hook
            # point needed to size the whole downstream network.
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
            self._z1_z_params = params.get('z1_z', {})

        def _encode_obs(self, obs):
            """[B, NFRAMES*(34+raw_z_dim)] -> [B, NFRAMES*(34+encoded_dim)],
            per-frame raw z1 replaced by its jointly-trained encoding.
            `obs` here is ALREADY normalized (RunningMeanStd runs in the outer
            ModelA2CContinuousLogStd wrapper, sized to the TRUE raw observation
            width, before this network ever sees it) — the encoder therefore
            learns on normalized raw features, same as every other weight.
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
