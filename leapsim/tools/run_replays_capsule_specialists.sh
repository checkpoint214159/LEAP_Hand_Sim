#!/usr/bin/env bash
# run_replays_capsule_specialists.sh — Rerun replays of the FIVE capsule specialists
# (each policy trained + tested on the capsule) rotating the capsule, so the 2x2
# ordering (Table 5) can be seen: B, z0, z1 (global) vs local-frozen, local-learned.
# GPU pipeline so it coexists with the CPU cross-category replay job.
set -o pipefail
cd "$(dirname "$0")/.."
MAN=tools/logs/replays_capsulespec_manifest.tsv; : > "$MAN"
POLICIES=(
  "B|specialist_capsule_B|none|LeapHandRotPPO"
  "z0|specialist_capsule_z0|z0|LeapHandRotPPO"
  "z1|specialist_capsule_z1|z1|LeapHandRotPPO"
  "localfrozen|specialist_capsule_localfrozen|local|LeapHandRotPPO"
  "locallearned|specialist_capsule_locallearned|local|LeapHandRotPPOLocalZ"
)
COMMON="headless=true wandb_activate=false log_to_sheet=false pipeline=gpu \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 num_envs=16 +task.env.print_object_angvel=true \
  task.env.object.type=capsule_train task.env.grasp_cache_name=leap_hand_in_capsule_train \
  task.env.rerun.enabled=true task.env.rerun.sample_per_object=true \
  task.env.rerun.window_length_steps=400 task.env.rerun.record_every_n_steps=400 \
  train.params.config.player.games_num=48 train.params.config.player.deterministic=False"

for p in "${POLICIES[@]}"; do
  IFS='|' read -r pol exp zmode yaml <<< "$p"
  rd="$(ls -dt runs/${exp}_2* 2>/dev/null | head -1)"; ck="$rd/nn/${exp}.pth"
  [ -f "$ck" ] || { echo "SKIP $exp (no ckpt)"; continue; }
  echo "=== REPLAY capsule-specialist $pol ==="
  uv run python -u train.py task=LeapHandRot train=$yaml test=true $COMMON \
    checkpoint="$ck" task.env.z_mode="$zmode" experiment="replay_capsulespec_${pol}" \
    > "tools/logs/replay_capsulespec_${pol}.log" 2>&1
  wd="$(ls -dt runs/replay_capsulespec_${pol}_* 2>/dev/null | head -1)"
  av="$(grep -oE 'mean object angvel:  tensor\(-?[0-9.]+' tools/logs/replay_capsulespec_${pol}.log | tail -1 | grep -oE '\-?[0-9.]+$')"
  printf '%s\t%s\t%s\n' "$pol" "${av:-?}" "$wd/rerun/" >> "$MAN"
  echo "   capsule/$pol angvel=${av:-?}  rerun=$wd/rerun/"
done
echo "=== CAPSULE-SPECIALIST REPLAYS DONE ==="; column -t "$MAN"
