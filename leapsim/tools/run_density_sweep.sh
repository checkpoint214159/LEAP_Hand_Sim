#!/usr/bin/env bash
# run_density_sweep.sh — the shape-density sweep (docs/HANDOFF.md §4/§5, the crux).
#
# Breaks the Stage-1 confound ("oracle z0 buys nothing" is ambiguous between
# "shape is useless" and "12 shapes is far too few to learn a continuous
# geometry->strategy map"). We hold the held-out CUBOID set fixed and vary only the
# number of distinct TRAINING cuboids (N = 4, 16, 40, all nested in one cache), and
# watch whether C(z0)'s zero-shot angvel advantage over B(proprio) EMERGES with N.
#   emerges  -> shape is usable; the Stage-1 null was data-starvation.
#   stays 0  -> shape is genuinely uninformative at this complexity.
#
# Controlled variables (why the sweep is clean): scale fixed 1.0 (no jitter),
# volume fixed 250 cm^3, dynamics randomization OFF, so z0's only informative
# signal is the 3 box edge lengths. Density is a pure runtime knob via
# object_id_whitelist over the one Halton-nested densetrain cache.
#
# Prereqs: tools/gen_primitive_objects.py --dense  AND grasp caches
#   cache/leap_hand_in_cuboid_densetrain_grasp_50k_s10.npy
#   cache/leap_hand_in_cuboid_denseheldout_grasp_50k_s10.npy
#
# Usage (from src/LEAP_Hand_Sim/leapsim):
#   tools/run_density_sweep.sh train 4 16 40      # train B+C at each density
#   tools/run_density_sweep.sh shuffle 40         # z-shuffle control at N=40
#   tools/run_density_sweep.sh eval               # eval every run in the manifest
# Each phase is sequential; inspect Rerun + the [obj-sampling diag] per run.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> leapsim/

MANIFEST=tools/logs/density_sweep_manifest.tsv   # exp <tab> run_dir
mkdir -p tools/logs
ENVS=4096
ITERS=1500
# Fix scale (single-scale s10 cache; scale is not the shape variable) + dynamics OFF.
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False \
  task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False \
  task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 \
  task.env.rerun.enabled=true"

train_one () {  # $1=exp  $2=z_mode  $3=whitelist(csv)  $4..=extra
  local exp="$1" zmode="$2" wl="$3"; shift 3
  echo "=== TRAIN $exp (z_mode=$zmode whitelist=[$wl]) ==="
  PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot $COMMON \
    experiment="$exp" num_envs=$ENVS max_iterations=$ITERS \
    task.env.object.type=cuboid_densetrain \
    task.env.grasp_cache_name=leap_hand_in_cuboid_densetrain \
    task.env.z_mode="$zmode" \
    +task.env.object_id_whitelist="[$wl]" "$@" \
    2>&1 | tee "tools/logs/${exp}.log"
  local rundir; rundir="$(ls -dt runs/${exp}_* | head -1)"
  printf '%s\t%s\n' "$exp" "$rundir" >> "$MANIFEST"
  echo "   -> $rundir"
}

eval_one () {  # $1=exp $2=run_dir $3=z_mode $4=set(train|heldout) $5=whitelist(csv, train only) $6..=extra
  local exp="$1" rundir="$2" zmode="$3" set="$4" wl="${5:-}"; shift 5 || true
  local ckpt="${rundir}/nn/${exp}.pth"
  local otype cache wlarg=()
  if [ "$set" = heldout ]; then
    otype=cuboid_denseheldout; cache=leap_hand_in_cuboid_denseheldout
  else
    otype=cuboid_densetrain;   cache=leap_hand_in_cuboid_densetrain
    wlarg=(+task.env.object_id_whitelist="[$wl]")
  fi
  echo "=== EVAL $exp on $set (z_mode=$zmode) ckpt=$ckpt ==="
  uv run python -u train.py task=LeapHandRot test=true $COMMON \
    checkpoint="$ckpt" num_envs=384 \
    task.env.object.type="$otype" \
    task.env.grasp_cache_name="$cache" \
    task.env.z_mode="$zmode" "${wlarg[@]}" \
    +task.env.print_object_angvel=true \
    train.params.config.player.games_num=768 \
    train.params.config.player.deterministic=False "$@" \
    2>&1 | tee "tools/logs/eval_${exp}_${set}.log"
}

phase="${1:-help}"; shift || true
case "$phase" in
  train)
    for N in "$@"; do
      WL="$(seq -s, 0 $((N-1)))"
      train_one "dsweep_B_N${N}" none "$WL"
      train_one "dsweep_C_N${N}" z0   "$WL"
    done ;;
  shuffle)   # z-shuffle control: correct z0 values, wrong shape->object map
    for N in "$@"; do
      WL="$(seq -s, 0 $((N-1)))"
      train_one "dsweep_Cshuf_N${N}" z0 "$WL" +task.env.z_shuffle=1234
    done ;;
  eval)
    while IFS=$'\t' read -r exp rundir; do
      [ -z "${exp:-}" ] && continue
      zmode=z0; case "$exp" in *_B_*) zmode=none;; esac
      # shuffled-z run: eval with the SAME wrong map it trained on (else the policy
      # sees a z mapping it never learned). Its comparison is on the TRAIN set.
      shuf=(); case "$exp" in *_Cshuf_*) shuf=(+task.env.z_shuffle=1234);; esac
      N="$(echo "$exp" | sed -E 's/.*_N([0-9]+)$/\1/')"
      WL="$(seq -s, 0 $((N-1)))"
      case "$exp" in
        *_Cshuf_*) eval_one "$exp" "$rundir" "$zmode" train "$WL" "${shuf[@]}" ;;
        *) eval_one "$exp" "$rundir" "$zmode" heldout ""
           eval_one "$exp" "$rundir" "$zmode" train "$WL" ;;
      esac
    done < "$MANIFEST" ;;
  *)
    sed -n '1,40p' "$0"; exit 0 ;;
esac
