"""distill_lib.py — shared helpers for the policy-distillation harness.

The harness tests whether an oracle shape prior ``z`` (concatenated to the
observation) becomes usable when the policy is trained with DENSE supervision
(behaviour cloning from per-object specialists) instead of sparse PPO reward.
See tools/run_distill.sh for the full narrative and the exact commands.

This module holds the pieces reused across the three phases so the layout logic
lives in exactly one place:

  * observation-layout constants + ``strip_z`` (recover the proprio-only obs the
    z_mode=none specialist expects, from a z_mode=z1 observation),
  * a thin wrapper around rl_games' own model builder so the student and the
    teacher are built through the SAME code path that ``test=true`` uses — the
    saved student checkpoint therefore loads back through the stock rl_games
    player without any hand-rolled state-dict surgery,
  * GRU hidden-state init / per-env reset copied verbatim from
    rl_games.common.player.BasePlayer so a recurrent teacher rolls out exactly
    as it did under PPO.

IMPORTANT: this module imports torch (via rl_games) but NEVER isaacgym, so the
pure-torch BC trainer (distill_bc_train.py) can import it without a sim. Any
entry point that also needs the sim must ``import isaacgym`` BEFORE importing
this module (the gymdeps import-order constraint).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


# ── Observation layout ────────────────────────────────────────────────────────
# LeapHandRot stacks NFRAMES history frames (obs_buf_lag_history[:, -3:]). Each
# frame is [16 dof pos | 16 PD targets | 2 phase] = 34 proprio dims, then the
# oracle z is appended AFTER phase (see compute_observations), so a z_mode frame
# is 34 + z_dim wide and the full obs is NFRAMES*(34 + z_dim):
#   none -> 102,  z0 -> 117,  z1 -> 126.
NFRAMES = 3
BASE_PER_FRAME = 34          # proprio dims per frame (dof16 + tgt16 + phase2)
Z_DIMS = {"none": 0, "z0": 5, "z1": 8}


def keep_z(obs_full: torch.Tensor, z_dim_full: int, keep: int,
           base_per_frame: int = BASE_PER_FRAME, nframes: int = NFRAMES) -> torch.Tensor:
    """Down-project a z-augmented obs to a SMALLER z layout.

    Each history frame is ``[proprio(base_per_frame) | z(z_dim_full)]`` and the
    z-block is NESTED: z1's z-block is ``[z0(5) | shape1(3)]`` (see
    _build_z_features). So keeping the first ``keep`` z-dims of every frame — and
    dropping the rest — yields exactly what a lower-z-mode env would emit:
      keep=0 -> z_mode=none (102-d),  keep=5 -> z_mode=z0 (117-d),
      keep=8 -> z_mode=z1 (unchanged, 126-d).
    z is static per episode, so this slice is byte-for-byte the lower env's obs.
    """
    assert 0 <= keep <= z_dim_full, f"keep_z: keep {keep} not in [0, {z_dim_full}]"
    fpf = base_per_frame + z_dim_full
    assert obs_full.shape[-1] == nframes * fpf, (
        f"keep_z: obs width {obs_full.shape[-1]} != {nframes}*({base_per_frame}+{z_dim_full})")
    if keep == z_dim_full:
        return obs_full
    width = base_per_frame + keep
    parts = [obs_full[..., f * fpf: f * fpf + width] for f in range(nframes)]
    return torch.cat(parts, dim=-1)


def strip_z(obs_full: torch.Tensor, z_dim: int,
            base_per_frame: int = BASE_PER_FRAME, nframes: int = NFRAMES) -> torch.Tensor:
    """Proprio-only (z_mode=none) obs from a z-augmented one — keep_z with keep=0.

    Used at collection time so the z_mode=none specialist (trained on 102-d
    proprio) is consumable while we simultaneously record the full z-augmented
    obs for the student."""
    return keep_z(obs_full, z_dim, 0, base_per_frame, nframes)


# ── rl_games model construction (single source of truth) ──────────────────────
def build_a2c_model(params: dict, obs_dim: int, actions_num: int, num_seqs: int,
                    device: str, normalize_input: bool = True,
                    normalize_value: bool = True) -> torch.nn.Module:
    """Build a ModelA2CContinuousLogStd.Network via rl_games' own builder.

    ``params`` is the ``params:`` block of a train yaml (needs at least the
    ``model`` and ``network`` sub-dicts). A network dict WITH an ``rnn`` key
    yields the recurrent teacher; WITHOUT it yields the feed-forward MLP student.
    Because the student is built here and saved with ``model.state_dict()``, the
    stock player rebuilds an identical module from LeapHandRotBC.yaml and loads
    it with zero key mismatches.
    """
    from rl_games.algos_torch import model_builder
    model_factory = model_builder.ModelBuilder().load(params)
    model = model_factory.build({
        "actions_num": actions_num,
        "input_shape": (obs_dim,),
        "num_seqs": num_seqs,
        "value_size": 1,
        "normalize_value": normalize_value,
        "normalize_input": normalize_input,
    })
    return model.to(device)


def load_model_weights(model: torch.nn.Module, ckpt_path: str, device: str) -> dict:
    """Load an rl_games checkpoint's ``model`` blob (net + running_mean_std +
    value_mean_std, all inside one state_dict) into ``model``. Returns the raw
    checkpoint dict so callers can read e.g. the teacher's sigma."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


# ── GRU hidden-state handling (mirrors rl_games BasePlayer) ────────────────────
def init_rnn_states(model: torch.nn.Module, batch_size: int, device: str
                    ) -> Optional[List[torch.Tensor]]:
    if not model.is_rnn():
        return None
    default = model.get_default_rnn_state()
    return [torch.zeros((s.size(0), batch_size, s.size(2)), dtype=torch.float32,
                        device=device) for s in default]


def reset_done_states(states: Optional[List[torch.Tensor]], done: torch.Tensor) -> None:
    """Zero the recurrent state of envs that terminated this step (fresh episode
    starts with a zeroed GRU state, exactly as the player does)."""
    if states is None:
        return
    idx = done.nonzero(as_tuple=False).flatten()
    if idx.numel() > 0:
        for s in states:
            s[:, idx, :] = 0.0


def teacher_forward(model, obs_proprio, states):
    """One inference step of a (recurrent) specialist. Returns (mu, sampled,
    new_states). ``obs_proprio`` is RAW (unnormalised) — the model normalises
    internally via its loaded running_mean_std."""
    input_dict = {"is_train": False, "prev_actions": None,
                  "obs": obs_proprio, "rnn_states": states}
    with torch.no_grad():
        res = model(input_dict)
    return res["mus"], res["actions"], res["rnn_states"]
