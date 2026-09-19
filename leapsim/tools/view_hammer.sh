#!/usr/bin/env bash
# view_hammer.sh — watch the CURRENT trained hammer policy in Rerun, so the grip
# orientation + rotation axis can be SEEN. Reproduces the training run's exact
# hammer setup (init pose, canonical_pose, ±0.5 angvel clip), evals the best
# checkpoint, and records Rerun windows. CPU pipeline → coexists with the GPU
# training job still running. Output: runs/hammer_view_*/rerun/*.rrd (open on Windows).
set -o pipefail
cd "$(dirname "$0")/.."

# newest hammer_rot run + its best checkpoint
RD=$(ls -dt runs/hammer_rot_* 2>/dev/null | head -1)
CK="${RD}/nn/hammer_rot.pth"
[ -f "$CK" ] || { echo "no hammer checkpoint at $CK"; exit 1; }
echo "policy   : $CK  (from $RD)"

CANON="[0.69,-0.52,1.0,-0.35,1,1.1,0.6,0.3,0.77,0.0,1.0,-0.35,0.73,0.22,1.0,-0.35]"

uv run python -u train.py task=LeapHandRot test=true checkpoint="$CK" \
  headless=true wandb_activate=false log_to_sheet=false pipeline=cpu num_envs=16 \
  experiment=hammer_view \
  task.env.object.type=hammer task.env.grasp_cache_name=leap_hand_in_hammer \
  task.env.z_mode=none \
  task.env.override_object_init_z=0.61 task.env.override_object_init_x=-0.03 \
  task.env.override_object_init_y=0.05 task.env.override_object_init_rot=[0,0,1.5708] \
  task.env.canonical_pose="$CANON" \
  task.env.reward.angvelClipMin=-0.5 task.env.reward.angvelClipMax=0.5 \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 \
  +task.env.print_object_angvel=true \
  task.env.rerun.enabled=true task.env.rerun.sample_per_object=true \
  task.env.rerun.window_length_steps=400 task.env.rerun.record_every_n_steps=400 \
  train.params.config.player.games_num=64 train.params.config.player.deterministic=False \
  2>&1 | tee tools/logs/hammer_view.log

WD=$(ls -dt runs/hammer_view_* 2>/dev/null | head -1)
echo "=== DONE ==="
echo "mean angvel_z : $(grep -oE 'mean object angvel:  tensor\([0-9.]+' tools/logs/hammer_view.log | tail -1 | grep -oE '[0-9.]+$')"
echo "rerun .rrd    : $WD/rerun/    <- open this on Windows"
