#!/bin/bash
# Run QAM-EDIT on Single-Play Task 2 with SPARSE reward

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

MUJOCO_GL=egl python main.py \
    --run_group=single_task2_rewards \
    --project=project-qam-v5 \
    --run_name=SINGLE_QAM_EDIT_SPARSE.8_Seed10001 \
    --agent=agents/qam.py \
    --tags=QAM_EDIT,sparse,single \
    --seed=10001 \
    --env_name=cube-single-play-singletask-task2-v0 \
    --ogbench_dataset_dir=/home/hyeonbin/RL/data/data/cube-single-play-1m \
    --sparse=True \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=1.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --offline_steps=1000000 \
    --online_steps=500000 \
    --eval_interval=50000 \
    --log_interval=5000 \
    --dataset_replace_interval=0 \
    --video_episodes=1
