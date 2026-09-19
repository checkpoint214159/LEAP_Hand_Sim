#!/usr/bin/env bash
# run_crosscat.sh — cross-category holdout (research-plan Stage 1, the sharp test).
#
# The density sweep showed that WITHIN the cuboid family, shape is not a
# generalization barrier (proprio B generalizes; oracle z0 adds nothing). The real
# observability challenge is CROSS-category: train on 2 families, zero-shot the
# unseen 3rd. Here we train on cylinder+sphere (round things) and zero-shot on
# CUBOID (flat faces, corners — the sharpest holdout). z0's fill fraction cleanly
# separates the classes (box 1.00 / cyl 0.78 / sph 0.52), so if a shape prior ever
# helps, it should help here.
#   H1: B (proprio) shows a big cuboid gap  (in-dist cyl/sph >> zero-shot cuboid).
#   H2: C (z0) closes it  (C cuboid angvel >> B cuboid angvel).
# Caveat we're testing: the held-out cuboid's z0 (fill=1.0) is OUT of the training
# z-distribution, so C only helps if the z->strategy map extrapolates to the box.
#
# Controlled: scale fixed 1.0 (no jitter), dynamics randomization OFF — isolate
# category (shape class) as the only variable. Same protocol as the density study.
#
# Prereqs (build once):
#   uv run python tools/build_mix_cache.py --subset train   --families cylinder sphere --out-name leap_hand_in_cylsph_train
#   uv run python tools/build_mix_cache.py --subset heldout --families cylinder sphere --out-name leap_hand_in_cylsph_heldout
#
# Usage (from src/LEAP_Hand_Sim/leapsim):
#   tools/run_crosscat.sh train      # B + C on cylinder+sphere
#   tools/run_crosscat.sh eval       # each on: zero-shot cuboid | in-dist cyl+sph | same-class heldout
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=tools/logs/crosscat_manifest.tsv
mkdir -p tools/logs
ENVS=4096; ITERS=1500
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False \
  task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False \
  task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 \
  task.env.rerun.enabled=true"

TRAIN_TYPE="cylinder_train+sphere_train"
TRAIN_CACHE="leap_hand_in_cylsph_train"

train_one () {  # $1=exp $2=z_mode
  local exp="$1" zmode="$2"
  echo "=== TRAIN $exp (train on $TRAIN_TYPE, z_mode=$zmode) ==="
  PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot $COMMON \
    experiment="$exp" num_envs=$ENVS max_iterations=$ITERS \
    task.env.object.type="$TRAIN_TYPE" \
    task.env.grasp_cache_name="$TRAIN_CACHE" \
    task.env.z_mode="$zmode" \
    2>&1 | tee "tools/logs/${exp}.log"
  local rd; rd="$(ls -dt runs/${exp}_* | head -1)"
  printf '%s\t%s\n' "$exp" "$rd" >> "$MANIFEST"
  echo "   -> $rd"
}

eval_one () {  # $1=exp $2=rundir $3=z_mode $4=label $5=otype $6=cache
  local exp="$1" rd="$2" zmode="$3" label="$4" otype="$5" cache="$6"
  echo "=== EVAL $exp / $label (obj=$otype z_mode=$zmode) ==="
  uv run python -u train.py task=LeapHandRot test=true $COMMON \
    checkpoint="${rd}/nn/${exp}.pth" num_envs=384 \
    task.env.object.type="$otype" \
    task.env.grasp_cache_name="$cache" \
    task.env.z_mode="$zmode" \
    +task.env.print_object_angvel=true \
    train.params.config.player.games_num=768 \
    train.params.config.player.deterministic=False \
    2>&1 | tee "tools/logs/evalcc_${exp}_${label}.log"
}

case "${1:-help}" in
  train)
    train_one crosscat_B none
    train_one crosscat_C z0 ;;
  z1)   # richer analytic shape prior (nested z0 + inertia ratios + SA/V)
    train_one crosscat_C_z1 z1 ;;
  eval)
    while IFS=$'\t' read -r exp rd; do
      [ -z "${exp:-}" ] && continue
      zmode=z0; case "$exp" in *_B) zmode=none;; *_z1) zmode=z1;; esac
      eval_one "$exp" "$rd" "$zmode" cuboid_zeroshot cuboid_train                      leap_hand_in_cuboid_train
      eval_one "$exp" "$rd" "$zmode" indist_cylsph   "cylinder_train+sphere_train"     "$TRAIN_CACHE"
      eval_one "$exp" "$rd" "$zmode" heldclass_cylsph "cylinder_heldout+sphere_heldout" leap_hand_in_cylsph_heldout
    done < "$MANIFEST" ;;
  *) sed -n '1,35p' "$0"; exit 0 ;;
esac
