#!/usr/bin/env bash
# run_interp.sh - interpretability on the learned-LOCAL policy (Stage 4): does it
# actually USE the per-contact shape feature, and how? On the zero-shot cuboid:
#   * correct        - real cuboid feature (reference)
#   * cf_sphere/_cyl  - COUNTERFACTUAL: tell the policy the cuboid is a sphere/cylinder
#                       (local analogue of "drive object A with object B's z")
#   * ablate_zero     - kill the feature (reliance test)
#   * ablate_noise    - corrupt it
# Each records per-object angvel + Rerun windows so the gait can be SEEN to morph.
# CPU pipeline → coexists with any GPU job.
set -o pipefail
cd "$(dirname "$0")/.."
EXP="${EXP:-localz_learned_s42}"
OBJTYPE="${OBJTYPE:-cuboid_train}"        # physical object being rotated
CACHE="${CACHE:-leap_hand_in_cuboid_train}"
PREFIX="${PREFIX:-}"                       # tag prefix for non-cuboid runs, e.g. "cone_"
RD=$(ls -dt runs/${EXP}_* 2>/dev/null | head -1); CK="${RD}/nn/${EXP}.pth"
[ -f "$CK" ] || { echo "no learned-local checkpoint at $CK"; exit 1; }
echo "policy: $CK  object=$OBJTYPE"
COMMON="headless=true wandb_activate=false log_to_sheet=false pipeline=${PIPELINE:-cpu} \
  train=LeapHandRotPPOLocalZ test=true checkpoint=$CK \
  task.env.object.type=$OBJTYPE task.env.grasp_cache_name=$CACHE \
  task.env.z_mode=local \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 num_envs=32 \
  +task.env.print_object_angvel=true \
  task.env.rerun.enabled=true task.env.rerun.sample_per_object=true \
  task.env.rerun.window_length_steps=400 task.env.rerun.record_every_n_steps=400 \
  train.params.config.player.games_num=192 train.params.config.player.deterministic=False"

run () {  # $1=label  $2..=extra flags
  local label="${PREFIX}$1"; shift
  echo "=== INTERP $label ==="
  uv run python -u train.py task=LeapHandRot $COMMON experiment="interp_${label}" "$@" \
    > "tools/logs/interp_${label}.log" 2>&1
  local av; av=$(grep -oE "mean object angvel:  tensor\(-?[0-9.]+(e-?[0-9]+)?" "tools/logs/interp_${label}.log" | tail -1 | grep -oE "\-?[0-9.]+(e-?[0-9]+)?\$")
  local wd; wd=$(ls -dt runs/interp_${label}_* 2>/dev/null | head -1)
  echo "   $label angvel=${av:-?}  rerun=$wd/rerun/"
}

run correct                                                                     # reference
run cf_sphere    +task.env.local_cf_family=2 +task.env.local_cf_params=[0.045]   # "it's a sphere"
run cf_cylinder  +task.env.local_cf_family=1 +task.env.local_cf_params=[0.03,0.05,0.0]  # "it's a cylinder" (3 params: r,half_len,unused)
run ablate_zero  +task.env.local_ablate=zero                                    # feature killed
run ablate_noise +task.env.local_ablate=noise +task.env.local_ablate_std=0.05   # feature corrupted
echo "=== INTERP DONE ($OBJTYPE) ==="
printf "%-14s %s\n" "condition" "$OBJTYPE angvel"
for l in correct cf_sphere cf_cylinder ablate_zero ablate_noise; do
  ll="${PREFIX}$l"
  printf "  %-12s %s\n" "$ll" "$(grep -oE 'mean object angvel:  tensor\(-?[0-9.]+(e-?[0-9]+)?' tools/logs/interp_${ll}.log 2>/dev/null | tail -1 | grep -oE '\-?[0-9.]+(e-?[0-9]+)?$')"
done
