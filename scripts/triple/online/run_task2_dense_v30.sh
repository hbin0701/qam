#!/bin/bash
# Run QAM on Triple-Play Task 2 with DENSE V30 reward (ONLINE-ONLY)

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="/rlwrld3/home/hyeonbin/RL/ogbench:${PYTHONPATH:-}"

MUJOCO_GL=egl python main.py \
    --run_group=triple_task2_rewards_online \
    --project=0224_CUBE_TRIPLE \
    --run_name="[Triple, QAM-Edit, V30, Online]" \
    --agent=agents/qam.py \
    --tags=QAM,dense_v30,online,triple \
    --seed=10001 \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --ogbench_dataset_dir=/rlwrld3/home/hyeonbin/.ogbench/data/cube-triple-play-3m \
    --sparse=False \
    --dense_reward_version=v30 \
    --terminal_bonus=10 \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=3.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.1 \
    --offline_steps=0 \
    --online_steps=500000 \
    --eval_interval=10000 \
    --log_interval=10000 \
    --dataset_replace_interval=0 \
    --video_episodes=1
