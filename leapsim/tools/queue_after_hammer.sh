#!/usr/bin/env bash
# queue_after_hammer.sh — runs the Stage-4 interpretability queue the moment the
# hammer retrain frees the GPU (regardless of the hammer's outcome). Sequenced so
# nothing contends for the GPU:
#   1. wait for the hammer training PID to exit
#   2. hammer eval clip on the FIRM power-grip run (best ckpt, Rerun)
#   3. cross-seed interpretability: counterfactual + ablation on s43, s44 (s42 done)
#   4. continuous latent traversal on s42 (gait-morphs-with-shape curve)
# Everything on pipeline=gpu (fast, GPU is free by then).
set -o pipefail
cd "$(dirname "$0")/.."
Q=tools/logs/queue_after_hammer.log
echo "[$(date -u +%H:%M:%S)] queue armed" > "$Q"

# ---- 1. wait for the hammer to finish ------------------------------------------
PID=$(cat tools/train_hammer_firm.pid 2>/dev/null)
echo "[queue] waiting on hammer PID=$PID" | tee -a "$Q"
n=0
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do
  sleep 30; n=$((n+1)); [ $n -gt 400 ] && { echo "[queue] hammer wait timeout (>3.3h)" | tee -a "$Q"; break; }
done
echo "[queue] hammer finished; latest: $(grep -oE 'epoch: [0-9]+/[0-9]+' tools/logs/train_hammer_firm.log | tail -1)" | tee -a "$Q"
sleep 15   # let the GPU fully release

CANON="[0.69,-0.52,1.0,-0.35,1,1.1,0.6,0.3,0.77,0.0,1.0,-0.35,0.73,0.22,1.0,-0.35]"

# ---- 2. hammer eval clip (firm run, best ckpt) ---------------------------------
FRD=$(ls -dt runs/hammer_firm_* 2>/dev/null | head -1); FCK="$FRD/nn/hammer_firm.pth"
if [ -f "$FCK" ]; then
  echo "[queue] hammer eval clip: $FCK" | tee -a "$Q"
  uv run python -u train.py test=true headless=true wandb_activate=false pipeline=gpu \
    checkpoint="$FCK" experiment=hammer_firm_view \
    task.env.object.type=hammer task.env.grasp_cache_name=leap_hand_in_hammer_firm \
    task.env.numEnvs=16 task.env.randomization.randomizeScaleList=[1.0] \
    task.env.override_object_init_rot=[0,0,1.5708] task.env.override_object_init_z=0.61 \
    task.env.override_object_init_x=-0.03 task.env.override_object_init_y=0.05 \
    "task.env.canonical_pose=$CANON" \
    task.env.reward.angvelClipMin=-0.5 task.env.reward.angvelClipMax=0.5 \
    task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
    task.env.randomization.randomizeFriction=False +task.env.scale_list_jitter=0 \
    +task.env.print_object_angvel=true \
    task.env.rerun.enabled=true task.env.rerun.sample_per_object=true \
    task.env.rerun.window_length_steps=400 task.env.rerun.record_every_n_steps=400 \
    train.params.config.player.games_num=64 train.params.config.player.deterministic=False \
    > tools/logs/hammer_firm_view.log 2>&1
  echo "[queue] hammer clip angvel_z: $(grep -oE 'mean object angvel:  tensor\([0-9.]+' tools/logs/hammer_firm_view.log | tail -1 | grep -oE '[0-9.]+$')  rerun=$(ls -dt runs/hammer_firm_view_* 2>/dev/null | head -1)/rerun/" | tee -a "$Q"
else
  echo "[queue] no hammer_firm checkpoint found — skipping hammer clip" | tee -a "$Q"
fi

# ---- 3. cross-seed interpretability (s43, s44) ---------------------------------
for s in 43 44; do
  echo "[queue] cross-seed interp s$s" | tee -a "$Q"
  EXP=localz_learned_s$s PIPELINE=gpu bash tools/run_interp.sh >> "$Q" 2>&1
done

# ---- 4. continuous latent traversal (s42) --------------------------------------
echo "[queue] latent traversal s42" | tee -a "$Q"
EXP=localz_learned_s42 PIPELINE=gpu bash tools/run_interp_traversal.sh >> "$Q" 2>&1

echo "[$(date -u +%H:%M:%S)] QUEUE DONE" | tee -a "$Q"
