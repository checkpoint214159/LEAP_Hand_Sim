#!/usr/bin/env bash
# run_replays.sh — Rerun replays of the three policy types (B proprio, C_z1 global
# prior, learned-local) on ALL five object categories (cuboid, cylinder, sphere, and
# the two novel ones, cone + capsule), for side-by-side visual comparison. Same three
# cross-category checkpoints throughout, so it shows one B / one C / one local policy
# behaving across every shape. Records one .rrd per (policy, object). CPU pipeline.
set -o pipefail
cd "$(dirname "$0")/.."
MAN=tools/logs/replays_manifest.tsv; : > "$MAN"
POLICIES=(
  "B|crosscat_B|none|LeapHandRotPPO"
  "Cz1|crosscat_C_z1|z1|LeapHandRotPPO"
  "Clocal|localz_learned_s42|local|LeapHandRotPPOLocalZ"
)
OBJECTS=(
  "cuboid|cuboid_train|leap_hand_in_cuboid_train"
  "cylinder|cylinder_train|leap_hand_in_cylinder_train"
  "sphere|sphere_train|leap_hand_in_sphere_train"
  "cone|cone_train|leap_hand_in_cone_train"
  "capsule|capsule_train|leap_hand_in_capsule_train"
)
COMMON="headless=true wandb_activate=false log_to_sheet=false pipeline=cpu \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 num_envs=16 +task.env.print_object_angvel=true \
  task.env.rerun.enabled=true task.env.rerun.sample_per_object=true \
  task.env.rerun.window_length_steps=400 task.env.rerun.record_every_n_steps=400 \
  train.params.config.player.games_num=48 train.params.config.player.deterministic=False"

for p in "${POLICIES[@]}"; do
  IFS='|' read -r pol exp zmode yaml <<< "$p"
  rd="$(ls -dt runs/${exp}_2* 2>/dev/null | head -1)"; ck="$rd/nn/${exp}.pth"
  [ -f "$ck" ] || { echo "SKIP $exp (no ckpt)"; continue; }
  for o in "${OBJECTS[@]}"; do
    IFS='|' read -r obj ot ca <<< "$o"
    echo "=== REPLAY $pol on $obj ==="
    uv run python -u train.py task=LeapHandRot train=$yaml test=true $COMMON \
      checkpoint="$ck" task.env.object.type="$ot" task.env.grasp_cache_name="$ca" task.env.z_mode="$zmode" \
      experiment="replay_${pol}_${obj}" > "tools/logs/replay_${pol}_${obj}.log" 2>&1
    wd="$(ls -dt runs/replay_${pol}_${obj}_* 2>/dev/null | head -1)"
    av="$(grep -oE 'mean object angvel:  tensor\(-?[0-9.]+' tools/logs/replay_${pol}_${obj}.log | tail -1 | grep -oE '\-?[0-9.]+$')"
    printf '%s\t%s\t%s\t%s\n' "$pol" "$obj" "${av:-?}" "$wd/rerun/" >> "$MAN"
    echo "   $pol/$obj angvel=${av:-?}  rerun=$wd/rerun/"
  done
done
echo "=== REPLAYS DONE ==="; column -t "$MAN"
