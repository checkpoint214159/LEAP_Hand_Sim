#!/usr/bin/env bash
# run_eval_novel_categories.sh — zero-cost extension of the existing B/C_z1/
# learned-local checkpoints (already trained on some 2-of-{cuboid,cylinder,
# sphere} mix) to TWO genuinely novel categories neither saw in ANY training
# run: cone and capsule. No training, eval-only — reuses checkpoints already
# on disk. Tests whether "z1 fails / local-learned wins" holds beyond the
# 3 original categories.
#
# Usage: tools/run_eval_novel_categories.sh
#   (manifest is hardcoded below — edit CHECKPOINTS to add more)
set -o pipefail
cd "$(dirname "$0")/.."
SUMMARY=tools/logs/eval_novel_categories_summary.tsv
mkdir -p tools/logs
COMMON="headless=true wandb_activate=false log_to_sheet=false \
  task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] \
  +task.env.scale_list_jitter=0 task.env.rerun.enabled=false"

# direction|condition|seed|exp|zmode|yaml
CHECKPOINTS=(
  "cuboid|B|42|crosscat_B|none|LeapHandRotPPO"
  "cuboid|B|43|crosscat_B_s43|none|LeapHandRotPPO"
  "cuboid|B|44|crosscat_B_s44|none|LeapHandRotPPO"
  "cuboid|Cz1|42|crosscat_C_z1|z1|LeapHandRotPPO"
  "cuboid|Cz1|43|crosscat_C_z1_s43|z1|LeapHandRotPPO"
  "cuboid|Cz1|44|crosscat_C_z1_s44|z1|LeapHandRotPPO"
  "cuboid|Clocal|42|localz_learned_s42|local|LeapHandRotPPOLocalZ"
  "cuboid|Clocal|43|localz_learned_s43|local|LeapHandRotPPOLocalZ"
  "cuboid|Clocal|44|localz_learned_s44|local|LeapHandRotPPOLocalZ"
  "cyl|B|42|crosscat_cyl_B|none|LeapHandRotPPO"
  "cyl|B|43|crosscat_cyl_B_s43|none|LeapHandRotPPO"
  "cyl|B|44|crosscat_cyl_B_s44|none|LeapHandRotPPO"
  "cyl|Cz1|42|crosscat_cyl_Cz1|z1|LeapHandRotPPO"
  "cyl|Cz1|43|crosscat_cyl_Cz1_s43|z1|LeapHandRotPPO"
  "cyl|Cz1|44|crosscat_cyl_Cz1_s44|z1|LeapHandRotPPO"
  "cyl|Clocal|42|hdlocalz_cyl_s42|local|LeapHandRotPPOLocalZ"
  "cyl|Clocal|43|hdlocalz_cyl_s43|local|LeapHandRotPPOLocalZ"
  "sph|B|42|crosscat_sph_B|none|LeapHandRotPPO"
  "sph|B|43|crosscat_sph_B_s43|none|LeapHandRotPPO"
  "sph|B|44|crosscat_sph_B_s44|none|LeapHandRotPPO"
  "sph|Cz1|42|crosscat_sph_Cz1|z1|LeapHandRotPPO"
  "sph|Cz1|43|crosscat_sph_Cz1_s43|z1|LeapHandRotPPO"
  "sph|Cz1|44|crosscat_sph_Cz1_s44|z1|LeapHandRotPPO"
  "sph|Clocal|42|hdlocalz_sph_s42|local|LeapHandRotPPOLocalZ"
  "sph|Clocal|43|hdlocalz_sph_s43|local|LeapHandRotPPOLocalZ"
)
NOVEL_TARGETS=(
  "cone|cone_heldout|leap_hand_in_cone_heldout"
  "capsule|capsule_heldout|leap_hand_in_capsule_heldout"
)

for c in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r dl cond seed exp zmode yaml <<< "$c"
  rd="$(ls -dt runs/${exp}_2* 2>/dev/null | head -1)"   # `_2*` (year prefix) avoids
    # matching a longer sibling like crosscat_B_s43_2026... when exp=crosscat_B
  ckpt="${rd}/nn/${exp}.pth"
  if [ ! -f "$ckpt" ]; then
    echo "=== SKIP $exp (no checkpoint found) ==="
    continue
  fi
  for t in "${NOVEL_TARGETS[@]}"; do
    IFS='|' read -r tlabel ot ca <<< "$t"
    lab="novel_${tlabel}"
    logf="tools/logs/evalnov_${exp}_${tlabel}.log"
    if [ -f "$logf" ] && grep -q "mean object angvel" "$logf"; then
      echo "=== SKIP EVAL $exp on $tlabel (already evaluated) ==="
    else
      echo "=== EVAL $exp (dir=$dl cond=$cond seed=$seed) on $tlabel ==="
      uv run python -u train.py task=LeapHandRot train=$yaml test=true $COMMON \
        checkpoint="$ckpt" num_envs=384 task.env.randomization.randomizeScaleList=[0.95] \
        task.env.object.type="$ot" task.env.grasp_cache_name="$ca" task.env.z_mode="$zmode" \
        +task.env.print_object_angvel=true train.params.config.player.games_num=768 \
        train.params.config.player.deterministic=False \
        > "$logf" 2>&1
    fi
    av="$(grep -oE 'mean object angvel:  tensor\(-?[0-9.]+(e-?[0-9]+)?' "$logf" | tail -1 | grep -oE '\-?[0-9.]+(e-?[0-9]+)?$')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$dl" "$cond" "$seed" "$exp" "$tlabel" "$av" >> "$SUMMARY"
    echo "   $exp -> $tlabel angvel=$av"
  done
done
echo "=== NOVEL-CATEGORY EVAL SWEEP COMPLETE ==="; column -t "$SUMMARY" 2>/dev/null
