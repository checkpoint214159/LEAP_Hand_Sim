#!/usr/bin/env bash
# watch_replay_specialists.sh — as each Phase-A specialist finishes training (its
# line appears in the manifest), run a clean CPU-pipeline Rerun replay of its BEST
# checkpoint so the behavior can be eyeballed on Windows. CPU sim → coexists with
# the GPU specialist training. One 400-step window ≈ one episode of the specialist
# rotating its single object.
set -o pipefail
cd "$(dirname "$0")/.."
MAN=tools/logs/distill_specialists_manifest.tsv
mkdir -p tools/logs
replayed=0
echo "watching $MAN ..."
while [ "$replayed" -lt 8 ]; do
  n=$( [ -f "$MAN" ] && wc -l < "$MAN" || echo 0 ); n=${n:-0}
  while [ "$replayed" -lt "$n" ]; do
    line=$(sed -n "$((replayed+1))p" "$MAN")
    exp=$(printf '%s' "$line" | cut -f1)
    ckpt=$(printf '%s' "$line" | cut -f2)
    short=$(printf '%s' "$exp" | sed -E 's/distill_spec_([a-z]+)[0-9]+/\1/')
    id=$(printf '%s' "$exp" | sed -E 's/distill_spec_[a-z]+([0-9]+)/\1/')
    case "$short" in cyl) fam=cylinder;; sph) fam=sphere;; *) fam=unknown;; esac
    echo "=== REPLAY $exp ($fam id=$id) ==="
    uv run python -u train.py task=LeapHandRot test=true pipeline=cpu \
      headless=true wandb_activate=false log_to_sheet=false \
      experiment="replay_${exp}" checkpoint="$ckpt" \
      task.env.object.type="${fam}_train" task.env.grasp_cache_name="leap_hand_in_${fam}_train" \
      +task.env.object_id_whitelist=[$id] task.env.z_mode=none \
      task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
      task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
      +task.env.scale_list_jitter=0 num_envs=16 \
      task.env.rerun.enabled=true task.env.rerun.sample_per_object=true \
      task.env.rerun.window_length_steps=400 task.env.rerun.record_every_n_steps=400 \
      +task.env.print_object_angvel=true \
      train.params.config.player.games_num=64 train.params.config.player.deterministic=False \
      > "tools/logs/replay_${exp}.log" 2>&1
    wd=$(ls -dt runs/replay_${exp}_* 2>/dev/null | head -1)
    nwin=$(ls "$wd"/rerun/*.rrd 2>/dev/null | grep -vc scalars)
    av=$(grep -oE "mean object angvel:  tensor\([0-9.]+" "tools/logs/replay_${exp}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "   -> ${nwin:-0} windows | replay angvel=${av:-?} | $wd/rerun/"
    replayed=$((replayed+1))
  done
  sleep 60
done
echo "=== ALL 8 SPECIALIST REPLAYS DONE ==="
