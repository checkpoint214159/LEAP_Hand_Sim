#!/usr/bin/env bash
# run_interp_traversal.sh — Stage-4 INTERVENTIONAL LATENT TRAVERSAL.
# Physical object fixed (zero-shot cuboid); sweep the shape the policy is TOLD it
# holds (local_cf_family=0 box, local_cf_params=[hx,hy,hz]) continuously along one
# half-extent. Record per-point angvel + a Rerun window, so the gait can be seen to
# morph smoothly with the (counterfactual) shape signal — the central mechanistic figure.
# Complements run_interp.sh's DISCRETE counterfactual with a CONTINUOUS curve.
set -o pipefail
cd "$(dirname "$0")/.."
EXP="${EXP:-localz_learned_s42}"
PIPELINE="${PIPELINE:-cpu}"
RD=$(ls -dt runs/${EXP}_* 2>/dev/null | head -1); CK="${RD}/nn/${EXP}.pth"
[ -f "$CK" ] || { echo "no learned-local checkpoint at $CK"; exit 1; }
echo "traversal policy: $CK  (pipeline=$PIPELINE)"

# swept axis: told half-extent hx, small->elongated; hy,hz held fixed
HXS=(0.018 0.026 0.034 0.042 0.050 0.058)
HY=0.030; HZ=0.030

COMMON="headless=true wandb_activate=false log_to_sheet=false pipeline=$PIPELINE \
  train=LeapHandRotPPOLocalZ test=true checkpoint=$CK \
  task.env.object.type=cuboid_train task.env.grasp_cache_name=leap_hand_in_cuboid_train \
  task.env.z_mode=local \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 num_envs=32 +task.env.print_object_angvel=true \
  +task.env.local_cf_family=0 \
  task.env.rerun.enabled=true task.env.rerun.sample_per_object=true \
  task.env.rerun.window_length_steps=400 task.env.rerun.record_every_n_steps=400 \
  train.params.config.player.games_num=160 train.params.config.player.deterministic=False"

step () {  # $1=hx
  local hx="$1"; local tag="hx${hx/./}"
  echo "=== TRAVERSAL told-box [$hx, $HY, $HZ] ==="
  uv run python -u train.py task=LeapHandRot $COMMON experiment="trav_${tag}" \
    "+task.env.local_cf_params=[$hx,$HY,$HZ]" > "tools/logs/trav_${tag}.log" 2>&1
  local av; av=$(grep -oE "mean object angvel:  tensor\([0-9.]+" "tools/logs/trav_${tag}.log" | tail -1 | grep -oE "[0-9.]+$")
  local wd; wd=$(ls -dt runs/trav_${tag}_* 2>/dev/null | head -1)
  echo "   hx=$hx  angvel=${av:-?}  rerun=$wd/rerun/"
}

for hx in "${HXS[@]}"; do step "$hx"; done

echo "=== TRAVERSAL DONE — angvel vs told half-extent hx ==="
printf "%-10s %s\n" "hx" "angvel"
for hx in "${HXS[@]}"; do
  tag="hx${hx/./}"
  printf "  %-8s %s\n" "$hx" "$(grep -oE 'mean object angvel:  tensor\([0-9.]+' tools/logs/trav_${tag}.log 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')"
done
