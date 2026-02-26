#!/bin/bash
#SBATCH --job-name=qam_dbl_t2_v30_10k
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/double_online_v30_ds10k_%j.out
#SBATCH --error=logs/slurm/double_online_v30_ds10k_%j.err

# Run QAM on Double-Play Task 2 with DENSE V30 reward (ONLINE-ONLY), using first 10k dataset samples.

set -euo pipefail

PROJECT_DIR=/rlwrld3/home/hyeonbin/RL/qam
cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p logs/slurm

# JAX memory management
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.30

export PYTHONPATH="/rlwrld3/home/hyeonbin/RL/ogbench:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
MUJOCO_GL=egl python main.py \
    --run_group=double_task2_rewards_online_dataset_size \
    --project=0219-double-tmp1 \
    --run_name="[Online] Double, V30, QAM, DS-10k" \
    --agent=agents/qam.py \
    --tags=QAM_EDIT,dense_v30,online,dataset_10k \
    --seed=10001 \
    --env_name=cube-double-play-singletask-task2-v0 \
    --cube_success_threshold=0.04 \
    --ogbench_dataset_dir=/rlwrld3/home/hyeonbin/.ogbench/data/cube-double-play-1m \
    --dataset_num_samples=10000 \
    --sparse=False \
    --dense_reward_version=v30 \
    --terminal_bonus=20 \
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
    --video_episodes=0
