#!/bin/bash

# Run RLPD on Double-Play Task 2 with DENSE V1 reward (ONLINE-ONLY).
# Single-seed run script (seed 10001), no Slurm.

set -euo pipefail

PROJECT_DIR=/rlwrld3/home/hyeonbin/RL/qam
cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p logs/slurm

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export PYTHONPATH="/rlwrld3/home/hyeonbin/RL/ogbench:${PYTHONPATH:-}"

SEED=10001
GPU_ID=0

CUDA_VISIBLE_DEVICES=${GPU_ID} MUJOCO_EGL_DEVICE_ID=${GPU_ID} MUJOCO_GL=egl python main.py     --run_group=double_task2_rewards_online     --project=CUBE_TASK_DOUBLE_ALGO     --run_name="[RLPD, Online, V1 (Sparse, Dense), SEED ${SEED}]"     --agent=agents/rlpd.py     --tags=RLPD,dense_v1,double,online     --seed=${SEED}     --env_name=cube-double-play-singletask-task2-v0     --cube_success_threshold=0.04     --max_episode_steps=200     --ogbench_dataset_dir=/rlwrld3/home/hyeonbin/.ogbench/data/cube-double-play-1m     --sparse=False     --dense_reward_version=v1     --terminal_bonus=20     --dense_shaping_lambda=20     --horizon_length=5     --utd_ratio=1     --agent.num_qs=10     --agent.discount=0.99     --agent.action_chunking=True     --agent.actor_hidden_dims='(512, 512, 512, 512)'     --agent.value_hidden_dims='(512, 512, 512, 512)'     --agent.batch_size=256     --agent.rho=0.5     --agent.bc_alpha=0.1     --offline_steps=0     --online_steps=500000     --start_training=1     --eval_interval=10000     --log_interval=5000     --dataset_replace_interval=0     --video_episodes=1
