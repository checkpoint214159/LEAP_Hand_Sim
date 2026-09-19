#!/bin/bash
# Overall v5 data generation: size-equalized taxonomy, strict criterion
# (3 fingers >=1N, proximal <=0.1N, grace 15), 40-col object-aware caches.
set -u
cd "$(dirname "$0")/.."

# ── 0. archive all old caches and old gen/check runs ──
STAMP=$(date +%Y%m%d)
mkdir -p cache/pre_size_eq runs/archive_${STAMP}
mv cache/leap_hand_in_{cuboid,cylinder,sphere}_* cache/_grasp_gen_status.json cache/pre_size_eq/ 2>/dev/null
mv runs/graspgen_* runs/graspcheck_* runs/archive_${STAMP}/ 2>/dev/null
rm -rf ../assets/cuboid/pilot   # stale pre-size-eq scratch family

# ── 1. per family: generate all 5 scales, then graspcheck s1.0 ──
for FAM in cuboid_train cuboid_heldout cylinder_train cylinder_heldout sphere_train sphere_heldout; do
  uv run python tools/gen_grasp_caches.py --only $FAM --timeout 3600 \
    --rerun-every 150 --grace-steps 15 --max-palm-force 0.1

  uv run python train.py task=LeapHandRot test=true headless=true \
    wandb_activate=false log_to_sheet=false \
    experiment=graspcheck_${FAM}_s10 \
    task.env.object.type=$FAM \
    task.env.grasp_cache_name=leap_hand_in_$FAM \
    'task.env.randomization.randomizeScaleList=[1.0]' \
    +task.env.scale_list_jitter=0 \
    task.env.numEnvs=64 task.env.forceScale=0 \
    task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.051 \
    task.env.randomization.randomizeCOM=false task.env.randomization.randomizeFriction=false \
    task.env.randomization.randomizePDGains=false \
    +task.env.debug.actions_file=debug/actions_zero_long.npy \
    task.env.rerun.enabled=true task.env.rerun.record_every_n_steps=1 task.env.rerun.window_length_steps=400
done
echo "=== OVERALL GRASPGEN COMPLETE ==="
