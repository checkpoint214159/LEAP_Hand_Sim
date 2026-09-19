#!/usr/bin/env bash
# run_synergy_dump.sh - Stage-4 SYNERGY PROJECTION, data collection.
# Roll out a policy on each object category and dump the 16-DoF joint trajectory
# (+per-step yaw) for offline eigengrasp-PCA / limit-cycle analysis. CPU pipeline.
# Generalized to any policy so we can contrast the WORKING policy (learned-local)
# against the proprio BASELINE B, which is what turns the synergy plots from
# descriptive into a test of the generalization hypothesis.
#
#   learned-local (default):
#     EXP=localz_learned_s42 bash tools/run_synergy_dump.sh
#   proprio B:
#     EXP=crosscat_B ZMODE=none TRAINYAML= PREFIX=B_synergy bash tools/run_synergy_dump.sh
set -o pipefail
cd "$(dirname "$0")/.."
EXP="${EXP:-localz_learned_s42}"
ZMODE="${ZMODE:-local}"
TRAINYAML="${TRAINYAML-LeapHandRotPPOLocalZ}"   # empty for B (stock actor_critic)
PREFIX="${PREFIX:-synergy}"
OUT="${OUT:-tools/logs/synergy_dumps}"
STEPS="${STEPS:-120}"
RD=$(ls -dt runs/${EXP}_* 2>/dev/null | head -1)
CK="${CK:-${RD}/nn/${EXP}.pth}"   # CK env overrides the glob (needed when ${EXP}_* is ambiguous)
[ -f "$CK" ] || { echo "no checkpoint $CK"; exit 1; }
mkdir -p "$OUT"
TRAINOV=""; [ -n "$TRAINYAML" ] && TRAINOV="train=$TRAINYAML"
echo "synergy dump: $CK  zmode=$ZMODE train=${TRAINYAML:-default} -> $OUT/${PREFIX}_*.npz (steps=$STEPS)"

COMMON="headless=true wandb_activate=false log_to_sheet=false pipeline=cpu \
  $TRAINOV test=true checkpoint=$CK task.env.z_mode=$ZMODE \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 num_envs=64 task.env.rerun.enabled=false \
  +task.env.print_object_angvel=true \
  train.params.config.player.games_num=64 train.params.config.player.deterministic=False"

dump () {  # $1=cat  $2=otype  $3=cache
  echo "=== SYNERGY dump $PREFIX $1 ($2) ==="
  uv run python -u train.py task=LeapHandRot $COMMON \
    task.env.object.type="$2" task.env.grasp_cache_name="$3" \
    "+task.env.dump_dof_traj=$OUT/${PREFIX}_$1.npz" +task.env.dump_dof_steps=$STEPS \
    experiment="syn_${PREFIX}_$1" > "tools/logs/synergy_${PREFIX}_$1.log" 2>&1
  echo "   $1 angvel=$(grep -oE 'mean object angvel:  tensor\(-?[0-9.]+(e-?[0-9]+)?' tools/logs/synergy_${PREFIX}_$1.log | tail -1 | grep -oE '\-?[0-9.]+(e-?[0-9]+)?$')  dump=$(ls -la $OUT/${PREFIX}_$1.npz 2>/dev/null | awk '{print $5}')B"
}

dump cuboid   cuboid_train   leap_hand_in_cuboid_train
dump cylinder cylinder_train leap_hand_in_cylinder_train
dump sphere   sphere_train   leap_hand_in_sphere_train
if [ "${EXTRA_NOVEL:-0}" = "1" ]; then   # cone/capsule: never in any training mix, use _train
  dump cone    cone_train     leap_hand_in_cone_train      # (heldout cache only exists @s095; train has all scales)
  dump capsule capsule_train  leap_hand_in_capsule_train
fi
echo "=== SYNERGY DUMP DONE ($PREFIX) ==="
