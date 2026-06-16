uv run python optimize_canonical_pose.py task=LeapHandGrasp \
    task.env.object.type=hammer task.env.baseObjScale=1.0 \
    task.env.grasp_cache_name=leap_hand_in_palm_hammer \
    train.params.config.player.games_num=1000000 \
    pipeline=cpu task.env.numEnvs=1024 \
    'task.env.override_object_init_rot=[0,0,1.5708]' \
    wandb_activate=false \
    task.env.override_object_init_z=0.61 \
    task.env.override_object_init_x=-0.04 \
    task.env.override_object_init_y=0.06 \
    task.env.grasp_dof_search_radius=0.4 \
    'task.env.canonical_pose=[0.69,-0.52,1.0,-0.35,1,1.1,0.6,0.3,0.77,0.0,1.0,-0.35,0.73,0.22,1.0,-0.35]' \
    +pose_optim.iterations=10


    
# 'task.env.canonical_pose=[0.0,-1.35,2.02,-0.52,1.30,1.91,1.28,1.37,-0.10,0.33,1.90,0.42,-0.10,1.27,2.03,0.41]' \