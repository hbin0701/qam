#!/bin/bash
# Run QAM on Triple-Play Task 2 with sparse reward

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

MUJOCO_GL=egl python main.py \
    --run_group=triple_task2_rewards \
    --project=project-v4 \
    --run_name=TRIPLE_QAM_EDIT_SPARSE_Seed10001 \
    --agent=agents/qam.py \
    --tags=QAM,sparse,triple \
    --seed=10001 \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --ogbench_dataset_dir=/rlwrld3/home/hyeonbin/.ogbench/data/cube-triple-play-3m \
    --sparse=True \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=3.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --offline_steps=1000000 \
    --online_steps=500000 \
    --eval_interval=50000 \
    --log_interval=5000 \
    --dataset_replace_interval=0 \
    --video_episodes=1
