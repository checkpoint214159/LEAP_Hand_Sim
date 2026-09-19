#!/usr/bin/env bash
# run_specialist_new_category.sh — single-category specialist sanity matrix
# (B / z0 / z1 / local-frozen / local-learned) for a newly-added object family,
# in-distribution only. Uses the SAME controlled regime as the crosscat/
# holdout-dir generalization experiments (scale fixed 1.0, dynamics
# randomization off) so results are directly comparable once this category is
# folded into the generalist mix. Purpose: confirm the new category trains
# sanely across every z-condition before committing it to a bigger run.
#
# Usage: tools/run_specialist_new_category.sh <train_type> <grasp_cache_name>
#   e.g.: tools/run_specialist_new_category.sh capsule_train leap_hand_in_capsule_train
set -o pipefail
cd "$(dirname "$0")/.."
SUMMARY=tools/logs/specialist_newcat_summary.tsv
mkdir -p tools/logs
ENVS=4096; ITERS=1500
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 task.env.rerun.enabled=false"

TT="$1"; TC="$2"
if [ -z "$TT" ] || [ -z "$TC" ]; then
  echo "usage: $0 <train_type> <grasp_cache_name>"; exit 1
fi
CATLABEL="${TT%_train}"; CATLABEL="${CATLABEL%_heldout}"

for cond in "B|none|LeapHandRotPPO" "z0|z0|LeapHandRotPPO" "z1|z1|LeapHandRotPPO" \
            "localfrozen|local|LeapHandRotPPO" "locallearned|local|LeapHandRotPPOLocalZ"; do
  tag="${cond%%|*}"; rest="${cond#*|}"; zmode="${rest%%|*}"; yaml="${rest##*|}"
  exp="specialist_${CATLABEL}_${tag}"
  rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
  if [ -n "$rd" ] && [ -f "${rd}/nn/${exp}.pth" ]; then
    echo "=== SKIP TRAIN $exp (checkpoint exists) ==="
  else
    echo "=== TRAIN $exp (z=$zmode yaml=$yaml) ==="
    PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot train=$yaml $COMMON \
      experiment="$exp" num_envs=$ENVS max_iterations=$ITERS \
      task.env.object.type="$TT" task.env.grasp_cache_name="$TC" \
      task.env.z_mode="$zmode" > "tools/logs/${exp}.log" 2>&1
    rd="$(ls -dt runs/${exp}_* 2>/dev/null | head -1)"
  fi
  uv run python -u train.py task=LeapHandRot train=$yaml test=true $COMMON \
    checkpoint="${rd}/nn/${exp}.pth" num_envs=384 \
    task.env.object.type="$TT" task.env.grasp_cache_name="$TC" task.env.z_mode="$zmode" \
    +task.env.print_object_angvel=true train.params.config.player.games_num=768 \
    train.params.config.player.deterministic=False \
    > "tools/logs/eval_${exp}.log" 2>&1
  av="$(grep -oE 'mean object angvel:  tensor\([0-9.]+' "tools/logs/eval_${exp}.log" | tail -1 | grep -oE '[0-9.]+$')"
  printf '%s\t%s\t%s\n' "$exp" "$tag" "$av" >> "$SUMMARY"
  echo "   $exp indist angvel=$av"
done
echo "=== SPECIALIST SANITY (${TT}) COMPLETE ==="; column -t "$SUMMARY" 2>/dev/null
