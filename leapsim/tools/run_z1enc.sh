#!/usr/bin/env bash
# run_z1enc.sh — the LEARNED-GLOBAL control (the missing 2x2 cell).
#
# Puts the SAME kind of tiny jointly-trained encoder as run_localz.sh's
# `learned` phase (Z1EncBuilder, train=LeapHandRotPPOZ1Enc), but on the STATIC/
# GLOBAL z1 feature (task.env.z_mode=z1, 8-d analytic whole-object shape)
# instead of the dynamic-local per-fingertip feature. Same cross-category
# protocol as run_localz.sh / run_crosscat.sh (train cylinder+sphere, zero-shot
# cuboid), so results drop straight onto the existing table:
#   B (proprio)                cuboid zero-shot 0.102  (seeds 42-44)
#   C.z1  static-global FROZEN cuboid zero-shot 0.097  (n=3)
#   local raw dynamic-local FROZEN            0.090
#   local LEARNED (LocalZBuilder)            0.131  (n=3) <- first positive
#   >>> z1 LEARNED (THIS)  static-global LEARNED  = the decider <<<
#     learned-global HELPS (~>=0.13) -> ENCODER is the active ingredient.
#     learned-global ~= frozen z1 (0.097) -> BOTH locality AND encoder needed.
#
# >=2 SEEDS MANDATORY (42, 43; ideally 44) — the z1 saga was an n=1 result that
# reversed under replication. This project reports nothing on <2 seeds.
#
# Usage (from src/LEAP_Hand_Sim/leapsim):
#   tools/run_z1enc.sh train 42
#   tools/run_z1enc.sh train 43
#   tools/run_z1enc.sh eval
#   tools/run_z1enc.sh all            # train 42,43,44 sequentially then eval
set -o pipefail
cd "$(dirname "$0")/.."

MANIFEST=tools/logs/z1enc_manifest.tsv
mkdir -p tools/logs

ENVS=4096
ITERS=1500
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False \
  task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False \
  task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 \
  task.env.rerun.enabled=true"

TRAIN_TYPE="cylinder_train+sphere_train"
TRAIN_CACHE="leap_hand_in_cylsph_train"

train_one () {  # $1=seed
  local seed="$1"
  local exp="z1enc_s${seed}"
  echo "=== TRAIN $exp (z_mode=z1 + Z1EncBuilder learned-global encoder, seed=$seed) ==="
  PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot train=LeapHandRotPPOZ1Enc $COMMON \
    experiment="$exp" seed="$seed" num_envs=$ENVS max_iterations=$ITERS \
    task.env.object.type="$TRAIN_TYPE" \
    task.env.grasp_cache_name="$TRAIN_CACHE" \
    task.env.z_mode=z1 \
    2>&1 | tee "tools/logs/${exp}.log"
  local rd
  rd="$(ls -dt runs/${exp}_* | head -1)"
  printf '%s\t%s\n' "$exp" "$rd" >> "$MANIFEST"
  echo "   -> $rd"
}

eval_one () {  # $1=exp $2=rundir $3=label $4=otype $5=cache
  local exp="$1" rd="$2" label="$3" otype="$4" cache="$5"
  echo "=== EVAL $exp / $label (obj=$otype) ==="
  uv run python -u train.py task=LeapHandRot test=true train=LeapHandRotPPOZ1Enc $COMMON \
    checkpoint="${rd}/nn/${exp}.pth" num_envs=384 \
    task.env.object.type="$otype" \
    task.env.grasp_cache_name="$cache" \
    task.env.z_mode=z1 \
    +task.env.print_object_angvel=true \
    train.params.config.player.games_num=768 \
    train.params.config.player.deterministic=False \
    2>&1 | tee "tools/logs/evalz1enc_${exp}_${label}.log"
}

run_eval_all () {
  while IFS=$'\t' read -r exp rd; do
    [ -z "${exp:-}" ] && continue
    eval_one "$exp" "$rd" cuboid_zeroshot cuboid_train                  leap_hand_in_cuboid_train
    eval_one "$exp" "$rd" indist_cylsph   "cylinder_train+sphere_train" "$TRAIN_CACHE"
  done < "$MANIFEST"
}

case "${1:-help}" in
  train)
    seed="${2:-42}"
    train_one "$seed" ;;
  eval)
    run_eval_all ;;
  all)
    train_one 42
    train_one 43
    train_one 44
    run_eval_all ;;
  *)
    sed -n '1,45p' "$0"; exit 0 ;;
esac
