# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: 
# https://github.com/HaozhiQi/hora/blob/main/scripts/gen_grasp.sh
# --------------------------------------------------------


# ^^ thanks leap. anyway lemme uhhh inject uv into here
SCALE=$1
GRASP_CACHE=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

uv run python train.py task=LeapHandGrasp task.env.baseObjScale=$SCALE \
    task.env.grasp_cache_name=$GRASP_CACHE test=true pipeline=cpu test=true \
    train.params.config.player.games_num=10000000000000000 task.env.episodeLength=150 \
    task.env.numEnvs=4096 \
    wandb_activate=false task.env.object.type='hammer' \
    task.env.override_object_init_rot=[0,0,1.5708] \
    task.env.override_object_init_z=0.57 \
    task.env.override_object_init_x=-0.04 \
    task.env.override_object_init_y=0.02 \
    task.env.grasp_dof_search_radius=0.40 \
    task.env.canonical_pose=[0.0,-1.35,2.02,-0.52,1.30,1.91,1.28,1.37,-0.10,0.33,1.90,0.42,-0.10,1.27,2.03,0.4]

${EXTRA_ARGS}
