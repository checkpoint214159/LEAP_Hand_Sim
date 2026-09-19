#!/usr/bin/env bash
# run_holdout_dir_seeds.sh — seed-replicate B and C.z1 in the cyl/sph holdout
# directions (crosscat_cyl_*/crosscat_sph_* currently n=1, default seed only).
# Companion to run_crosscat_seeds.sh (does this for the cuboid direction) and
# run_holdout_localz.sh (does this for learned-local).
#
# Usage: tools/run_holdout_dir_seeds.sh <seeds_csv> <dirspec> [<dirspec> ...]
#   dirspec = label:train_type:train_cache:heldout_type:heldout_cache
set -o pipefail   # deliberately NOT -e/-u (one failed eval must not kill the whole run)
cd "$(dirname "$0")/.."
SUMMARY=tools/logs/holdout_dir_seeds_summary.tsv
mkdir -p tools/logs
ENVS=4096; ITERS=1500
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 task.env.rerun.enabled=false"

SEEDS="$1"; shift
for d in "$@"; do
  IFS=':' read -r dl tt tc ht hc <<< "$d"
  for seed in ${SEEDS//,/ }; do
    for pol in "B|none" "Cz1|z1"; do
      tag="${pol%%|*}"; zmode="${pol#*|}"
      exp="crosscat_${dl}_${tag}_s${seed}"
      rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
      if [ -n "$rd" ] && [ -f "${rd}/nn/${exp}.pth" ]; then
        echo "=== SKIP TRAIN $exp (checkpoint exists) ==="
      else
        echo "=== TRAIN $exp (train=$tt z=$zmode seed=$seed) ==="
        PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot $COMMON \
          experiment="$exp" seed="$seed" num_envs=$ENVS max_iterations=$ITERS \
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
          > "tools/logs/evalhds_${exp}_${lab}.log" 2>&1
        av="$(grep -oE 'mean object angvel:  tensor\([0-9.]+' "tools/logs/evalhds_${exp}_${lab}.log" | tail -1 | grep -oE '[0-9.]+$')"
        printf '%s\t%s\t%s\t%s\t%s\n' "$dl" "$exp" "$seed" "$lab" "$av" >> "$SUMMARY"
        echo "   $exp $lab angvel=$av"
      done
    done
  done
done
echo "=== HOLDOUT-DIR SEED REPLICATION COMPLETE ==="; column -t "$SUMMARY" 2>/dev/null
