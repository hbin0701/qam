#!/bin/bash
# Run QAM on Task 2 with SPARSE (original) reward

MUJOCO_GL=egl python main.py \
    --run_group=task2_rewards \
    --agent=agents/qam.py \
    --tags=sparse \
    --seed=10001 \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=True \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=3.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --offline_steps=1000000 \
    --online_steps=500000 \
    --eval_interval=50000 \
    --log_interval=5000
