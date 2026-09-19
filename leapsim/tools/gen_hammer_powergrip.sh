#!/usr/bin/env bash
# gen_hammer_powergrip.sh — regenerate the hammer grasp cache as a REAL POWER GRIP
# (handle across the palm, world Y), fixing the two bugs that made the old cache a
# loose handshake:
#   1. criterion: old script set `finger_dist_threshold_m` (unread key → default
#      0.1 m = anything counts). Here: the REAL `finger_dist_threshold=0.03`.
#   2. seed: old script used an open "settling" canonical_pose that let the handle
#      roll to X. Here: the hand-tuned POWER-GRIP seed (fingers pre-curled to cage
#      a Y handle) so grasp-gen searches around an actual cage.
# Fresh gen writes v4 40-col (PD targets restored → firm grip). Orientation is
# horizontal (init_rot [0,0,pi/2]) so world-Z reward yaws the hammer's HEAD around
# the vertical axis (rotate-about-global-Z-in-a-power-grip), NOT a handle-axis spin.
# CPU pipeline. New cache name so the current cache stays intact for comparison.
set -o pipefail
cd "$(dirname "$0")/.."

SCALE="${SCALE:-1.0}"                     # -> _s10, matches training's randomizeScaleList=[1.0]
CACHE="${CACHE:-leap_hand_in_hammer_pg}"  # new name; training will point here once verified
NENVS="${NENVS:-1024}"
POWERGRIP_SEED="[0.0,-1.35,2.02,-0.52,1.30,1.91,1.28,1.37,-0.10,0.33,1.90,0.42,-0.10,1.27,2.03,0.4]"

echo "gen power-grip cache: cache/${CACHE}_grasp_50k_s${SCALE/./}.npy   (nenvs=$NENVS scale=$SCALE)"
uv run python -u train.py task=LeapHandGrasp test=true pipeline=cpu wandb_activate=false \
  task.env.baseObjScale=$SCALE task.env.grasp_cache_name=$CACHE \
  train.params.config.player.games_num=5000000 task.env.episodeLength=150 \
  task.env.numEnvs=$NENVS \
  task.env.object.type='hammer' task.env.disable_actions=True \
  task.env.override_object_init_rot=[0,0,1.5708] \
  task.env.override_object_init_z=0.61 task.env.override_object_init_x=-0.03 task.env.override_object_init_y=0.05 \
  task.env.grasp_dof_search_radius=0.15 \
  "task.env.canonical_pose=$POWERGRIP_SEED" \
  task.env.num_contact_fingers=3 \
  +task.env.finger_dist_threshold=0.03 \
  +task.env.min_contact_force=1.0 \
  2>&1 | tee tools/logs/gen_hammer_pg.log
