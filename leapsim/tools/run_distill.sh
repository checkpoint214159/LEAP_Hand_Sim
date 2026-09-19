#!/usr/bin/env bash
# run_distill.sh — orchestrate the cross-category policy-distillation experiment.
#
# HYPOTHESIS: the shape-prior null (oracle z buys nothing under PPO) may be a
# LEARNING-SIGNAL artifact — sparse RL never discovers z's subtle value. Dense
# supervision (behaviour cloning from per-object specialists) is the test: train
# per-object experts, then distil ONE z-conditioned student to reproduce the
# right expert's actions. If specialists behave differently and the student must
# read z to match them, this directly tests whether concat-z CAN drive shape-
# appropriate behaviour — injection unchanged, only the learning signal changes.
#
# Phases:
#   A  train 8 proprio specialists  (tools/distill_train_specialists.sh)   [GPU]
#   B  collect (obs_z1, specialist mu) pairs, one specialist per object    [CPU ok]
#   C  BC-train THREE students by MSE from the one collection: none / z0 / z1  [CPU]
#   D  eval each student: zero-shot cuboid + in-dist cyl/sph on the angvel harness
#
# Clean 3-way (none / z0 / z1) mirrors the RL B / C(z0) / C·z1 comparison — same
# concat-z injection, only the learning signal changes. Compare across the three
# students on zero-shot cuboid, and vs the RL numbers (B proprio ~0.10, C·z1
# ~0.096 rad/s zero-shot cuboid, n=3).
#
# Usage (from src/LEAP_Hand_Sim/leapsim):
#   tools/run_distill.sh smoke        # CPU end-to-end validation (existing cuboid teachers)
#   tools/run_distill.sh collect      # full: collect from the 8 trained specialists
#   tools/run_distill.sh bc           # full: BC the z1 student + none control
#   tools/run_distill.sh eval         # full: eval both, zero-shot cuboid + in-dist
#   tools/run_distill.sh full-help    # print the exact full-scale command block
set -o pipefail
cd "$(dirname "$0")/.."

DATA=tools/distill_data
mkdir -p "$DATA" tools/logs

CPU="pipeline=cpu sim_device=cpu rl_device=cpu graphics_device_id=-1"
DYN="task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False task.env.randomization.randomizeFriction=False task.env.randomization.randomizeScaleList=[1.0] +task.env.scale_list_jitter=0"
STD="task=LeapHandRot headless=true wandb_activate=false log_to_sheet=false"

find_ckpt () {  # $1 = run-name glob prefix ; echoes best (no-suffix) checkpoint
  ls -dt runs/${1}_*/nn/${1}.pth 2>/dev/null | head -1
}

case "${1:-help}" in

  # ── CPU end-to-end validation ────────────────────────────────────────────────
  # Reuses the existing GOOD cuboid single-object specialists (cuboid0, cuboid3 —
  # different shapes → different z1) as real teachers, so the whole B→C→D pipeline
  # is exercised with genuine experts and yields a real angvel. Mechanics check
  # (in-dist eval on the distilled objects), not the scientific result.
  smoke)
    C0=$(find_ckpt specialist_v2_cuboid0_solo_4k)
    C3=$(find_ckpt specialist_v2_cuboid3_solo_4k)
    echo "[smoke] teacher0=$C0"
    echo "[smoke] teacher3=$C3"
    [ -z "$C0" ] && { echo "no cuboid0 specialist checkpoint found"; exit 1; }
    NENV=64

    echo "### B: collect from cuboid0 (id 0) ###"
    uv run python tools/distill_collect.py $STD $CPU $DYN num_envs=$NENV \
      task.env.object.type=cuboid_train +task.env.object_id_whitelist=[0] \
      task.env.grasp_cache_name=leap_hand_in_cuboid_train task.env.z_mode=z1 \
      checkpoint="$C0" +collect.out=$DATA/smoke_cub0.npz \
      +collect.max_pairs=4000 +collect.max_steps=400 2>&1 | tee tools/logs/smoke_collect0.log || exit 1

    if [ -n "$C3" ]; then
      echo "### B: collect from cuboid3 (id 3) ###"
      uv run python tools/distill_collect.py $STD $CPU $DYN num_envs=$NENV \
        task.env.object.type=cuboid_train +task.env.object_id_whitelist=[3] \
        task.env.grasp_cache_name=leap_hand_in_cuboid_train task.env.z_mode=z1 \
        checkpoint="$C3" +collect.out=$DATA/smoke_cub3.npz \
        +collect.max_pairs=4000 +collect.max_steps=400 2>&1 | tee tools/logs/smoke_collect3.log || exit 1
    fi

    echo "### C: BC all THREE students (none / z0 / z1) from the same collection ###"
    for tag in none z0 z1; do
      uv run python tools/distill_bc_train.py --data "$DATA/smoke_cub*.npz" \
        --student-zmode $tag --out runs/distill_student_${tag}_smoke \
        --epochs 25 --batch 2048 --lr 1e-3 2>&1 | tee tools/logs/smoke_bc_${tag}.log || exit 1
    done

    echo "### D: eval all three (in-dist on the distilled cuboids 0,3) ###"
    for tag in none z0 z1; do
      uv run python tools/distill_eval.py $STD $CPU $DYN num_envs=$NENV \
        task.env.object.type=cuboid_train +task.env.object_id_whitelist=[0,3] \
        task.env.grasp_cache_name=leap_hand_in_cuboid_train task.env.z_mode=$tag \
        +task.env.print_object_angvel=true \
        checkpoint=runs/distill_student_${tag}_smoke/nn/distill_student_${tag}_smoke.pth \
        +eval.games=96 +eval.max_steps=1500 +eval.stochastic=true 2>&1 | tee tools/logs/smoke_eval_${tag}.log || exit 1
    done
    echo "[smoke] DONE — see the [eval RESULT] lines above."
    ;;

  # ── Full-scale Phase B: collect from the 8 trained specialists ───────────────
  collect)
    for fam in cylinder sphere; do
      for i in 0 1 2 3; do
        short=${fam:0:3}; ck=$(find_ckpt distill_spec_${short}${i})
        [ -z "$ck" ] && { echo "MISSING specialist distill_spec_${short}${i} — run Phase A first"; continue; }
        echo "### collect ${fam} id $i from $ck ###"
        uv run python tools/distill_collect.py $STD $DYN num_envs=384 \
          task.env.object.type=${fam}_train +task.env.object_id_whitelist=[$i] \
          task.env.grasp_cache_name=leap_hand_in_${fam}_train task.env.z_mode=z1 \
          checkpoint="$ck" +collect.out=$DATA/${short}${i}.npz \
          +collect.max_pairs=150000 +collect.max_steps=800 2>&1 | tee tools/logs/collect_${short}${i}.log
      done
    done ;;

  # ── Full-scale Phase C: BC all THREE students from the single collection ─────
  #    none / z0 / z1 — the clean 3-way mirroring the RL B / C(z0) / C·z1 study.
  bc)
    for tag in none z0 z1; do
      uv run python tools/distill_bc_train.py --data "$DATA/cyl*.npz" "$DATA/sph*.npz" \
        --student-zmode $tag --out runs/distill_student_${tag} --epochs 60 --batch 16384 --lr 1e-3 \
        2>&1 | tee tools/logs/bc_${tag}.log
    done ;;

  # ── Full-scale Phase D: eval zero-shot cuboid + in-dist cyl/sph ──────────────
  eval)
    for tag in none z0 z1; do
      zmode=$tag                       # student tag == env z_mode name
      ck=runs/distill_student_${tag}/nn/distill_student_${tag}.pth
      [ -e "$ck" ] || { echo "missing $ck — run 'bc' first"; continue; }
      # Stochastic eval (student sigma is calibrated to the teacher's), to match
      # how B/C were measured — the distilled student inherits the teacher's
      # stochastic-controller nature, so deterministic eval under-reads it the
      # same way it under-reads B/C. Drop +eval.stochastic=true for the (lower)
      # deterministic diagnostic.
      echo "### EVAL student=$tag  ZERO-SHOT cuboid ###"
      uv run python tools/distill_eval.py $STD $DYN num_envs=384 \
        task.env.object.type=cuboid_train \
        task.env.grasp_cache_name=leap_hand_in_cuboid_train task.env.z_mode=$zmode \
        +task.env.print_object_angvel=true checkpoint="$ck" +eval.games=768 \
        +eval.stochastic=true \
        2>&1 | tee tools/logs/eval_${tag}_cuboid_zeroshot.log
      echo "### EVAL student=$tag  IN-DIST cyl+sph ###"
      uv run python tools/distill_eval.py $STD $DYN num_envs=384 \
        task.env.object.type=cylinder_train+sphere_train \
        task.env.grasp_cache_name=leap_hand_in_cylsph_train task.env.z_mode=$zmode \
        +task.env.print_object_angvel=true checkpoint="$ck" +eval.games=768 \
        +eval.stochastic=true \
        2>&1 | tee tools/logs/eval_${tag}_indist.log
    done ;;

  full-help)
    cat <<'EOF'
FULL-SCALE RUN (user launches on the GPU when free; ~est below):
  # Phase A — 8 specialists (~ 8 * a crosscat run; RTX 4070 Ti S ~1.7h each at
  #           4096 envs/1500 iters => ~10-14h total). Resumable/skippable.
  tools/distill_train_specialists.sh
  # Phase B — collect BC data (GPU env stepping, no training; ~5-10 min/spec):
  tools/run_distill.sh collect
  # Phase C — BC the z1 student + none control (CPU, minutes):
  tools/run_distill.sh bc
  # Phase D — eval zero-shot cuboid + in-dist cyl/sph (GPU or CPU; ~few min each):
  tools/run_distill.sh eval
Read the [eval RESULT] lines; compare the THREE students (none / z0 / z1) on
ZERO-SHOT cuboid against B~0.10 / C·z1~0.096. A z-conditioned student (z0 or z1)
>> the none student on zero-shot cuboid would be the first positive signal that
dense supervision makes concat-z usable where sparse RL reward did not.
EOF
    ;;
  *) sed -n '1,40p' "$0"; exit 0 ;;
esac
