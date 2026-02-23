#!/bin/bash
# Run QAM on Double-Play Task 2 with DENSE V29 reward (ONLINE-ONLY) from pretrained BC actor.

set -euo pipefail

# JAX Memory Management
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES}}"
export PYTHONPATH="/rlwrld3/home/hyeonbin/RL/ogbench:${PYTHONPATH:-}"

PRETRAINED_PATH="${PRETRAINED_PATH:-/rlwrld3/home/hyeonbin/RL/MeltFlow/exp/qam-reproduce/qam-bc/cube-double-play-singletask-task2-v0/ATQAM_BC_ONLY_atqam_cube_s10001_4f9d15df/params_1000000.pkl}"

MUJOCO_GL=egl python main.py \
    --run_group=double_task2_rewards_pretrained_online \
    --project=0219-double-tmp1 \
    --run_name="[Online, Pretrained] Double, V29, QAM" \
    --agent=agents/qam.py \
    --tags=QAM_EDIT,dense_v29,online,pretrained \
    --seed=10001 \
    --env_name=cube-double-play-singletask-task2-v0 \
    --cube_success_threshold=0.04 \
    --ogbench_dataset_dir=/rlwrld3/home/hyeonbin/.ogbench/data/cube-double-play-1m \
    --pretrained_actor_path="${PRETRAINED_PATH}" \
    --sparse=False \
    --dense_reward_version=v29 \
    --terminal_bonus=20 \
    --dense_shaping_lambda=20 \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=1.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --offline_steps=0 \
    --online_steps=500000 \
    --eval_interval=10000 \
    --log_interval=10000 \
    --dataset_replace_interval=0 \
    --video_episodes=1

