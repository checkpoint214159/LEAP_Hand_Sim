#!/usr/bin/env bash
# run_crosscat_seeds.sh — seed-replicate the cross-category B vs C·z1 result.
#
# The n=1 finding: on the zero-shot cuboid, C·z1 (0.096) > B (0.085) > C·z0 (0.064).
# This trains B (proprio) and C·z1 (richer analytic prior) for extra seeds and evals
# each on the zero-shot cuboid + in-distribution, to confirm the z1>B ordering is
# robust and not seed luck. z0 stays n=1 (already known to hurt).
#
# Usage (from src/LEAP_Hand_Sim/leapsim):  tools/run_crosscat_seeds.sh 43 44
# Result summary appended to tools/logs/crosscat_seeds_summary.tsv
set -o pipefail   # NOT -e/-u: one failed eval must not kill the whole multi-run script
cd "$(dirname "$0")/.."
SUMMARY=tools/logs/crosscat_seeds_summary.tsv
mkdir -p tools/logs
ENVS=4096; ITERS=1500
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 task.env.rerun.enabled=false"
TRAIN_TYPE="cylinder_train+sphere_train"; TRAIN_CACHE="leap_hand_in_cylsph_train"

run () {  # $1=exp $2=z_mode $3=seed
  local exp="$1" zmode="$2" seed="$3"
  local rd; rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
  if [ -n "$rd" ] && [ -f "${rd}/nn/${exp}.pth" ]; then
    echo "=== SKIP TRAIN $exp (checkpoint already exists) ==="
  else
    echo "=== TRAIN $exp (z=$zmode seed=$seed) ==="
    PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot $COMMON \
      experiment="$exp" seed="$seed" num_envs=$ENVS max_iterations=$ITERS \
      task.env.object.type="$TRAIN_TYPE" task.env.grasp_cache_name="$TRAIN_CACHE" \
      task.env.z_mode="$zmode" > "tools/logs/${exp}.log" 2>&1
    rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
  fi
  # NOTE: assign each var on its own line — `local a=.. b=${a..}` evaluates b before a is bound.
  local spec lab r ot ca av
  for spec in "cuboid|cuboid_train|leap_hand_in_cuboid_train" \
              "indist|${TRAIN_TYPE}|${TRAIN_CACHE}"; do
    lab="${spec%%|*}"; r="${spec#*|}"; ot="${r%%|*}"; ca="${r##*|}"
    uv run python -u train.py task=LeapHandRot test=true $COMMON \
      checkpoint="${rd}/nn/${exp}.pth" num_envs=384 \
      task.env.object.type="$ot" task.env.grasp_cache_name="$ca" task.env.z_mode="$zmode" \
      +task.env.print_object_angvel=true train.params.config.player.games_num=768 \
      train.params.config.player.deterministic=False \
      > "tools/logs/evalcc_${exp}_${lab}.log" 2>&1
    av="$(grep -oE 'mean object angvel:  tensor\([0-9.]+' "tools/logs/evalcc_${exp}_${lab}.log" | tail -1 | grep -oE '[0-9.]+$')"
    printf '%s\t%s\t%s\t%s\t%s\n' "$exp" "$zmode" "$seed" "$lab" "$av" >> "$SUMMARY"
    echo "   $exp $lab angvel=$av"
  done
}

for s in "$@"; do
  run "crosscat_B_s${s}"    none "$s"
  run "crosscat_C_z1_s${s}" z1   "$s"
done
echo "=== SEED REPLICATION COMPLETE ==="; column -t "$SUMMARY"
