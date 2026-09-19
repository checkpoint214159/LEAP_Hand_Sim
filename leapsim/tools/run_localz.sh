#!/usr/bin/env bash
# run_localz.sh — cross-category test of the z_mode=local prior (analytic
# per-fingertip distance+normal, DYNAMIC per step) and, optionally, the
# RMA/HORA-style tiny jointly-trained encoder on top of it.
#
# Same cross-category protocol as run_crosscat.sh (train cylinder+sphere,
# zero-shot cuboid) so results are directly comparable to the existing
# numbers in docs/research-plan-object-generalization.md Stage 2:
#   B (proprio)      cuboid zero-shot ~0.11   (seeds 42-44)
#   C·z0 (global)    cuboid zero-shot 0.064   (n=1, hurts)
#   C·z1 (global)    cuboid zero-shot ~0.096  (n=3, hurts vs B on average)
#
# Phases:
#   raw    — z_mode=local, STOCK actor_critic network (no learned encoder).
#            Run this FIRST — cheapest, isolates local-vs-global from
#            learned-vs-frozen. See the plan doc's priority order.
#   learned — z_mode=local + LocalZBuilder's tiny jointly-trained encoder
#            (train=LeapHandRotPPOLocalZ). Only run this if `raw` is
#            ambiguous and time remains — see the plan doc.
#   eval   — zero-shot cuboid + in-dist cyl/sph, for whichever manifest rows exist.
#
# >=2 SEEDS MANDATORY before trusting any result here — the z1 saga (this
# same doc, Stage 2) was an n=1 result that reversed under replication.
#
# Usage (from src/LEAP_Hand_Sim/leapsim):
#   tools/run_localz.sh raw [seed]        # default seed 42
#   tools/run_localz.sh raw 43
#   tools/run_localz.sh raw 44
#   tools/run_localz.sh learned [seed]
#   tools/run_localz.sh eval
set -o pipefail
cd "$(dirname "$0")/.."

MANIFEST=tools/logs/localz_manifest.tsv
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

train_one () {  # $1=exp $2=train_yaml_override(empty for default)
  local exp="$1" train_yaml="$2" seed="$3"
  local train_override=""
  [ -n "$train_yaml" ] && train_override="train=$train_yaml"
  echo "=== TRAIN $exp (z_mode=local, train_yaml=${train_yaml:-default}, seed=$seed) ==="
  PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot $train_override $COMMON \
    experiment="$exp" seed="$seed" num_envs=$ENVS max_iterations=$ITERS \
    task.env.object.type="$TRAIN_TYPE" \
    task.env.grasp_cache_name="$TRAIN_CACHE" \
    task.env.z_mode=local \
    2>&1 | tee "tools/logs/${exp}.log"
  local rd; rd="$(ls -dt runs/${exp}_* | head -1)"
  printf '%s\t%s\n' "$exp" "$rd" >> "$MANIFEST"
  echo "   -> $rd"
}

eval_one () {  # $1=exp $2=rundir $3=label $4=otype $5=cache
  local exp="$1" rd="$2" label="$3" otype="$4" cache="$5"
  # Must match the train_yaml used to PRODUCE this checkpoint — a "learned"
  # checkpoint's state_dict has z_encoder.* keys the stock actor_critic
  # network doesn't declare, so loading it through the wrong yaml fails on a
  # state_dict key mismatch, not a silent misread.
  local train_override=""
  case "$exp" in localz_learned_*) train_override="train=LeapHandRotPPOLocalZ" ;; esac
  echo "=== EVAL $exp / $label (obj=$otype, train_yaml=${train_override:-default}) ==="
  uv run python -u train.py task=LeapHandRot test=true $train_override $COMMON \
    checkpoint="${rd}/nn/${exp}.pth" num_envs=384 \
    task.env.object.type="$otype" \
    task.env.grasp_cache_name="$cache" \
    task.env.z_mode=local \
    +task.env.print_object_angvel=true \
    train.params.config.player.games_num=768 \
    train.params.config.player.deterministic=False \
    2>&1 | tee "tools/logs/evallocalz_${exp}_${label}.log"
}

case "${1:-help}" in
  raw)
    seed="${2:-42}"
    train_one "localz_raw_s${seed}" "" "$seed" ;;
  learned)
    seed="${2:-42}"
    train_one "localz_learned_s${seed}" "LeapHandRotPPOLocalZ" "$seed" ;;
  eval)
    while IFS=$'\t' read -r exp rd; do
      [ -z "${exp:-}" ] && continue
      eval_one "$exp" "$rd" cuboid_zeroshot cuboid_train                  leap_hand_in_cuboid_train
      eval_one "$exp" "$rd" indist_cylsph   "cylinder_train+sphere_train" "$TRAIN_CACHE"
    done < "$MANIFEST" ;;
  *)
    sed -n '1,40p' "$0"; exit 0 ;;
esac
