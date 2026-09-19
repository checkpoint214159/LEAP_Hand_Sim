#!/usr/bin/env bash
# run_holdout_localz.sh — test the WORKING z (learned-local: z_mode=local +
# LeapHandRotPPOLocalZ encoder) in the holdout directions it was NEVER tested in.
# The +28% result is only for cyl+sph -> cuboid. This runs the other two:
#   cyl : cuboid+sphere   -> zero-shot CYLINDER   (proprio B on record = 0.110)
#   sph : cuboid+cylinder -> zero-shot SPHERE     (proprio B on record = 0.159)
# Trains learned-local for N seeds per direction, evals zero-shot(heldout)+in-dist.
# Same controlled regime as run_holdout_dir.sh (scale 1.0, dynamics OFF).
#
# Usage:  tools/run_holdout_localz.sh <seed1,seed2,...> <dirspec> [<dirspec> ...]
#   dirspec = label:train_type:train_cache:heldout_type:heldout_cache
set -o pipefail
cd "$(dirname "$0")/.."
SUMMARY=tools/logs/holdout_localz_summary.tsv
mkdir -p tools/logs
ENVS=4096; ITERS=1500
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 task.env.rerun.enabled=false"
YAML=LeapHandRotPPOLocalZ

SEEDS="$1"; shift
for d in "$@"; do
  IFS=':' read -r dl tt tc ht hc <<< "$d"
  for seed in ${SEEDS//,/ }; do
    exp="hdlocalz_${dl}_s${seed}"
    rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
    if [ -n "$rd" ] && [ -f "${rd}/nn/${exp}.pth" ]; then
      echo "=== SKIP TRAIN $exp (checkpoint exists) ==="
    else
      echo "=== TRAIN $exp (learned-local, train=$tt, seed=$seed) ==="
      PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot train=$YAML $COMMON \
        experiment="$exp" seed="$seed" num_envs=$ENVS max_iterations=$ITERS \
        task.env.object.type="$tt" task.env.grasp_cache_name="$tc" \
        task.env.z_mode=local > "tools/logs/${exp}.log" 2>&1
      rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
    fi
    for spec in "zeroshot|${ht}|${hc}" "indist|${tt}|${tc}"; do
      lab="${spec%%|*}"; rest="${spec#*|}"; ot="${rest%%|*}"; ca="${rest##*|}"
      uv run python -u train.py task=LeapHandRot train=$YAML test=true $COMMON \
        checkpoint="${rd}/nn/${exp}.pth" num_envs=384 \
        task.env.object.type="$ot" task.env.grasp_cache_name="$ca" task.env.z_mode=local \
        +task.env.print_object_angvel=true train.params.config.player.games_num=768 \
        train.params.config.player.deterministic=False \
        > "tools/logs/evalhdlz_${exp}_${lab}.log" 2>&1
      av="$(grep -oE 'mean object angvel:  tensor\([0-9.]+' "tools/logs/evalhdlz_${exp}_${lab}.log" | tail -1 | grep -oE '[0-9.]+$')"
      printf '%s\t%s\t%s\t%s\n' "$dl" "$exp" "$lab" "$av" >> "$SUMMARY"
      echo "   $exp $lab angvel=$av"
    done
  done
done
echo "=== HOLDOUT-LOCALZ COMPLETE ==="; column -t "$SUMMARY" 2>/dev/null
