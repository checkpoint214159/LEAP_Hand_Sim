#!/usr/bin/env bash
# gen_hammer_firm.sh — regenerate the hammer cache the RELIABLE way (settling seed
# that actually converges) but as v4 40-col, so restore re-arms the grip with PD
# targets (squeeze -> friction -> damps the free-roll the legacy 23-col cache had).
# This does NOT change the grip TYPE (still a horizontal resting grip, handle along
# the fingers) — it makes it FIRM. Motion under world-Z reward = head orbits vertical.
# Correctly-named finger_dist_threshold (old script's `_m` suffix was an unread key).
# CPU pipeline; new cache name so the old cache stays for comparison.
set -o pipefail
cd "$(dirname "$0")/.."

SCALE="${SCALE:-1.0}"
CACHE="${CACHE:-leap_hand_in_hammer_firm}"
NENVS="${NENVS:-1024}"
DIST="${DIST:-0.06}"                       # firmer than the old effective 0.1m default, still converges
NFING="${NFING:-3}"
SETTLE_SEED="[0.69,-0.52,1.0,-0.35,1,1.1,0.6,0.3,0.77,0.0,1.0,-0.35,0.73,0.22,1.0,-0.35]"

echo "gen FIRM cache: cache/${CACHE}_grasp_50k_s${SCALE/./}.npy  (nenvs=$NENVS dist=$DIST nfing=$NFING)"
# Run python in the BACKGROUND and record its PID. The cache flush is atexit-only
# (leap_hand_grasp.py), so it must be stopped with SIGINT (-> KeyboardInterrupt ->
# atexit flush), NOT SIGKILL. To stop+flush:  kill -INT $(cat tools/gen_hammer_firm.pid)
uv run python -u train.py task=LeapHandGrasp test=true pipeline=cpu wandb_activate=false \
  task.env.baseObjScale=$SCALE task.env.grasp_cache_name=$CACHE \
  train.params.config.player.games_num=5000000 task.env.episodeLength=150 \
  task.env.numEnvs=$NENVS \
  task.env.object.type='hammer' task.env.disable_actions=True \
  task.env.override_object_init_rot=[0,0,1.5708] \
  task.env.override_object_init_z=0.61 task.env.override_object_init_x=-0.03 task.env.override_object_init_y=0.05 \
  task.env.grasp_dof_search_radius=0.15 \
  "task.env.canonical_pose=$SETTLE_SEED" \
  task.env.num_contact_fingers=$NFING \
  +task.env.finger_dist_threshold=$DIST \
  +task.env.min_contact_force=1.0 \
  > tools/logs/gen_hammer_firm.log 2>&1 &
PID=$!
echo "$PID" > tools/gen_hammer_firm.pid
echo "gen PID=$PID  (flush+stop with:  kill -INT $PID)"
wait "$PID"
