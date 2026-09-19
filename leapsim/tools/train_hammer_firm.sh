#!/usr/bin/env bash
# train_hammer_firm.sh — retrain hammer in-hand rotation on the FIRM v4 power-grip
# cache (leap_hand_in_hammer_firm: handle across the palm / world Y, PD targets ->
# no free-roll). Exact cmd.txt recipe, only the cache name + warm-start + experiment
# differ. Warm-starts from the general LeapHand base policy for faster convergence.
set -o pipefail
cd "$(dirname "$0")/.."

ITERS="${ITERS:-1200}"
EXP="${EXP:-hammer_firm}"
CACHE="${CACHE:-leap_hand_in_hammer_firm}"

echo "train $EXP on cache=$CACHE for $ITERS iters (warm-start ../weights/LeapHand.pth)"
uv run python -u train.py headless=true wandb_activate=false \
  experiment="$EXP" max_iterations=$ITERS \
  checkpoint=../weights/LeapHand.pth \
  task.env.rerun.enabled=true \
  task.env.rerun.record_every_n_steps=2000 task.env.rerun.window_length_steps=400 \
  task.env.object.type=hammer \
  task.env.grasp_cache_name="$CACHE" \
  task.env.numEnvs=4096 \
  task.env.randomization.randomizeScaleList=[1.0] \
  task.env.reward.objLinvelPenaltyScale=-0.1 \
  task.env.override_object_init_rot=[0,0,1.5708] \
  task.env.override_object_init_z=0.61 \
  task.env.override_object_init_x=-0.03 \
  task.env.override_object_init_y=0.05 \
  'task.env.canonical_pose=[0.69,-0.52,1.0,-0.35,1,1.1,0.6,0.3,0.77,0.0,1.0,-0.35,0.73,0.22,1.0,-0.35]' \
  task.env.reward.angvelClipMin=-0.5 \
  task.env.reward.angvelClipMax=0.5 \
  task.env.reward.workPenaltyScale=-0.3 \
  task.env.reward.poseDiffPenaltyScale=-0.03 \
  train.params.config.learning_rate=1e-3 \
  train.params.config.entropy_coef=0.001 \
  train.params.config.horizon_length=64 \
  > tools/logs/train_hammer_firm.log 2>&1 &
PID=$!
echo "$PID" > tools/train_hammer_firm.pid
echo "train PID=$PID  log=tools/logs/train_hammer_firm.log"
wait "$PID"
