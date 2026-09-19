#!/usr/bin/env bash
# run_probe_z1.sh — drives tools/probe_z1_analysis.py, one process per
# condition (see that file's docstring for why: a single long-lived process
# building multiple IsaacGym sims by hand segfaulted reliably; one @hydra.main
# process per condition, same pattern as train.py / run_crosscat.sh, has not).
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:-runs/crosscat_C_z1_2026-07-28_07-45-42/nn/crosscat_C_z1.pth}"
SHUFFLE_CKPT="runs/crosscat_C_z1_shuffle_2026-07-28_23-00-05/nn/crosscat_C_z1_shuffle.pth"
OBJTYPE="${OBJTYPE:-cuboid_train}"
CACHE="${CACHE:-leap_hand_in_cuboid_train}"
TAGSUF="${TAGSUF:-}"      # e.g. "_cone" to keep outputs distinct from the cuboid run
ENVS="${1:-256}"
STEPS="${2:-450}"  # max_episode_length=400 for this task config — need > that to see real episode completions
OUTDIR="tools/logs"
mkdir -p "$OUTDIR"

COMMON="test=true headless=true wandb_activate=false log_to_sheet=false pipeline=cpu \
  checkpoint=${CKPT} num_envs=${ENVS} \
  task.env.object.type=$OBJTYPE \
  task.env.grasp_cache_name=$CACHE \
  task.env.z_mode=z1 task.env.z_shuffle=false task.env.rerun.enabled=false \
  train.params.config.player.deterministic=False \
  train.params.config.player.games_num=1000000000 \
  +probe.steps=${STEPS}"

run_cond () {  # $1=tag  $2=zmask (hydra list literal, e.g. [] or [5,6])  $3=do_item1 (true/false)
  local tag="$1$TAGSUF" zmask="$2" item1="$3"
  echo "=== condition: $tag (obj=$OBJTYPE zmask=$zmask item1=$item1) ==="
  # rm -rf the VHACD cache is cheap insurance against a corrupted cache left
  # by any earlier crashed process; each condition re-decomposes in seconds.
  rm -rf ~/.isaacgym/vhacd
  uv run python -u tools/probe_z1_analysis.py $COMMON \
    +probe.tag="$tag" "+probe.zmask=$zmask" +probe.do_item1=$item1 \
    +probe.out="${OUTDIR}/probe_z1_cond_${tag}.json" \
    2>&1 | tee "${OUTDIR}/probe_z1_cond_${tag}.log" | tail -30
}

run_cond baseline               "[]"      true
run_cond zero_ratios_r2r3       "[5,6]"   false
run_cond zero_sav               "[7]"     false
run_cond zero_shape_extra_all   "[5,6,7]" false
run_cond zero_z0_keep_shape     "[0,1,2,3,4]" false

if [ -z "$TAGSUF" ]; then   # shuffle-ckpt control is specific to the original cuboid experiment
echo "=== condition: shuffle_ckpt (z1-shuffle-trained checkpoint, correct z at eval) ==="
rm -rf ~/.isaacgym/vhacd
uv run python -u tools/probe_z1_analysis.py test=true headless=true pipeline=cpu \
  wandb_activate=false log_to_sheet=false \
  checkpoint="${SHUFFLE_CKPT}" num_envs="${ENVS}" \
  task.env.object.type=cuboid_train \
  task.env.grasp_cache_name=leap_hand_in_cuboid_train \
  task.env.z_mode=z1 task.env.z_shuffle=false task.env.rerun.enabled=false \
  train.params.config.player.deterministic=False \
  train.params.config.player.games_num=1000000000 \
  +probe.steps="${STEPS}" +probe.tag=shuffle_ckpt "+probe.zmask=[]" +probe.do_item1=false \
  +probe.out="${OUTDIR}/probe_z1_cond_shuffle_ckpt.json" \
  2>&1 | tee "${OUTDIR}/probe_z1_cond_shuffle_ckpt.log" | tail -30
fi

echo "=== all conditions done; results in ${OUTDIR}/probe_z1_cond_*.json ==="
