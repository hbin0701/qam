#!/bin/bash
# Run QAM on Triple-Play Task 2 with DENSE V4 reward (ONLINE-ONLY)

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

MUJOCO_GL=egl python main.py \
    --run_group=triple_task2_rewards_online \
    --project=project-qam-v5 \
    --run_name=TRIPLE_QAM_EDIT_ONLINE_V4.8_Seed10001 \
    --agent=agents/qam.py \
    --tags=QAM,dense_v4,online,triple \
    --seed=10001 \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --ogbench_dataset_dir=/rlwrld3/home/hyeonbin/.ogbench/data/cube-triple-play-3m \
    --sparse=False \
    --dense_reward_version=v4 \
    --terminal_bonus=50 \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=3.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --offline_steps=0 \
    --online_steps=500000 \
    --eval_interval=50000 \
    --log_interval=5000 \
    --dataset_replace_interval=0 \
    --video_episodes=1
