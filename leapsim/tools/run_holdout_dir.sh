#!/usr/bin/env bash
# run_holdout_dir.sh — cross-category holdout in ANY direction (which class is held out).
#
# We already did cyl+sph -> cuboid (cuboid EXTRAPOLATES the shape descriptors -> hardest
# case for a prior). This runs the other directions to map interpolation vs extrapolation:
#   cuboid+sph -> CYLINDER  (held-out class INTERPOLATES between the two -> easiest for a prior)
#   cuboid+cyl -> SPHERE    (held-out extrapolates low)
# Trains B (proprio) and C.z1 for each direction, evals zero-shot (held-out class) + in-dist.
# Same controlled regime as run_crosscat.sh: scale fixed 1.0, dynamics OFF.
#
# Usage (each arg = label:train_type:train_cache:heldout_type:heldout_cache):
#   tools/run_holdout_dir.sh \
#     "cyl:cuboid_train+sphere_train:leap_hand_in_cubsph_train:cylinder_train:leap_hand_in_cylinder_train" \
#     "sph:cuboid_train+cylinder_train:leap_hand_in_cubcyl_train:sphere_train:leap_hand_in_sphere_train"
set -o pipefail   # deliberately NOT -e/-u (one failed eval must not kill the whole run)
cd "$(dirname "$0")/.."
SUMMARY=tools/logs/holdout_dirs_summary.tsv
mkdir -p tools/logs
ENVS=4096; ITERS=1500
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 task.env.rerun.enabled=false"

rundir () {  # $1=label $2=train_type $3=train_cache $4=heldout_type $5=heldout_cache
  local dl="$1" tt="$2" tc="$3" ht="$4" hc="$5"
  local pol tag zmode exp rd spec lab rest ot ca av
  for pol in "B|none" "Cz1|z1"; do
    tag="${pol%%|*}"; zmode="${pol#*|}"; exp="crosscat_${dl}_${tag}"
    rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
    if [ -n "$rd" ] && [ -f "${rd}/nn/${exp}.pth" ]; then
      echo "=== SKIP TRAIN $exp (checkpoint exists) ==="
    else
      echo "=== TRAIN $exp (train=$tt z=$zmode) ==="
      PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot $COMMON \
        experiment="$exp" num_envs=$ENVS max_iterations=$ITERS \
        task.env.object.type="$tt" task.env.grasp_cache_name="$tc" \
        task.env.z_mode="$zmode" > "tools/logs/${exp}.log" 2>&1
      rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
    fi
    for spec in "zeroshot|${ht}|${hc}" "indist|${tt}|${tc}"; do
      lab="${spec%%|*}"; rest="${spec#*|}"; ot="${rest%%|*}"; ca="${rest##*|}"
      uv run python -u train.py task=LeapHandRot test=true $COMMON \
        checkpoint="${rd}/nn/${exp}.pth" num_envs=384 \
        task.env.object.type="$ot" task.env.grasp_cache_name="$ca" task.env.z_mode="$zmode" \
        +task.env.print_object_angvel=true train.params.config.player.games_num=768 \
        train.params.config.player.deterministic=False \
        > "tools/logs/evalhd_${exp}_${lab}.log" 2>&1
      av="$(grep -oE 'mean object angvel:  tensor\([0-9.]+' "tools/logs/evalhd_${exp}_${lab}.log" | tail -1 | grep -oE '[0-9.]+$')"
      printf '%s\t%s\t%s\t%s\n' "$dl" "$exp" "$lab" "$av" >> "$SUMMARY"
      echo "   $exp $lab angvel=$av"
    done
  done
}

for d in "$@"; do
  IFS=':' read -r dl tt tc ht hc <<< "$d"
  rundir "$dl" "$tt" "$tc" "$ht" "$hc"
done
echo "=== HOLDOUT DIRECTIONS COMPLETE ==="; column -t "$SUMMARY" 2>/dev/null
