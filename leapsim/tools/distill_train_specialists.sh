#!/usr/bin/env bash
# distill_train_specialists.sh — Phase A of the policy-distillation harness.
#
# Trains ONE proprio-only (z_mode=none) specialist per TRAINING object: an expert
# on a single object, produced by whitelisting that instance onto every env
# (+task.env.object_id_whitelist=[i]). Regime matches the crosscat runs exactly
# (scale fixed 1.0 / no jitter, dynamics randomization OFF, entropy_coef=0, best
# checkpoint, Rerun on). These are the TEACHERS the student is distilled from.
#
# Train families: cylinder_train + sphere_train = 8 objects (4 each). This is the
# heavy GPU step (~8 * a crosscat run). The user launches it; do NOT run it while
# another GPU job is active. Reuses train.py verbatim.
#
# Knobs (env vars, defaults = full GPU regime):
#   ENVS=4096 ITERS=1500 PIPELINE=gpu SIMDEV=cuda:0 RLDEV=cuda:0 GDEV=0
#   MINIBATCH=32768 HORIZON=32   FAMILIES="cylinder sphere"  RERUN=true
# For a CPU smoke of the *command* (not a usable teacher):
#   ENVS=64 ITERS=3 PIPELINE=cpu SIMDEV=cpu RLDEV=cpu GDEV=-1 MINIBATCH=512 \
#     HORIZON=8 FAMILIES=cylinder RERUN=false tools/distill_train_specialists.sh 0
#
# Usage (from src/LEAP_Hand_Sim/leapsim):
#   tools/distill_train_specialists.sh            # all 8 (cyl 0-3, sph 0-3)
#   tools/distill_train_specialists.sh 0 1        # only ids 0,1 within each family
set -o pipefail
cd "$(dirname "$0")/.."

ENVS=${ENVS:-4096}; ITERS=${ITERS:-1500}
PIPELINE=${PIPELINE:-gpu}; SIMDEV=${SIMDEV:-cuda:0}; RLDEV=${RLDEV:-cuda:0}; GDEV=${GDEV:-0}
MINIBATCH=${MINIBATCH:-32768}; HORIZON=${HORIZON:-32}
FAMILIES=${FAMILIES:-"cylinder sphere"}
RERUN=${RERUN:-true}
IDS=("$@"); [ ${#IDS[@]} -eq 0 ] && IDS=(0 1 2 3)

mkdir -p tools/logs
MANIFEST=tools/logs/distill_specialists_manifest.tsv

for fam in $FAMILIES; do
  for i in "${IDS[@]}"; do
    short=${fam:0:3}          # cyl / sph
    exp="distill_spec_${short}${i}"
    existing=$(ls -dt runs/${exp}_*/nn/${exp}.pth 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
      echo "=== SKIP $exp (checkpoint exists: $existing) ==="
      printf '%s\t%s\n' "$exp" "$existing" >> "$MANIFEST"
      continue
    fi
    echo "=== TRAIN $exp : ${fam}_train id=$i (z_mode=none) ==="
    PYTHONUNBUFFERED=1 uv run python -u train.py task=LeapHandRot \
      headless=true wandb_activate=false log_to_sheet=false \
      pipeline="$PIPELINE" sim_device="$SIMDEV" rl_device="$RLDEV" graphics_device_id="$GDEV" \
      experiment="$exp" num_envs="$ENVS" max_iterations="$ITERS" \
      task.env.object.type="${fam}_train" \
      +task.env.object_id_whitelist=[$i] \
      task.env.grasp_cache_name="leap_hand_in_${fam}_train" \
      task.env.z_mode=none \
      task.env.randomization.randomizeMass=False \
      task.env.randomization.randomizeCOM=False \
      task.env.randomization.randomizeFriction=False \
      task.env.randomization.randomizeScaleList=[1.0] \
      +task.env.scale_list_jitter=0 \
      train.params.config.minibatch_size="$MINIBATCH" \
      train.params.config.horizon_length="$HORIZON" \
      task.env.rerun.enabled="$RERUN" \
      2>&1 | tee "tools/logs/${exp}.log"
    rd=$(ls -dt runs/${exp}_*/nn/${exp}.pth 2>/dev/null | head -1)
    printf '%s\t%s\n' "$exp" "$rd" >> "$MANIFEST"
    echo "   -> $rd"
  done
done
echo "=== specialists manifest: $MANIFEST ==="
