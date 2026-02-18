#!/bin/bash
#SBATCH --job-name=qam_double_online_sparse_8
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/double_online_sparse_8_%j.out
#SBATCH --error=logs/slurm/double_online_sparse_8_%j.err

set -euo pipefail

PROJECT_DIR=/rlwrld3/home/hyeonbin/RL/qam
cd "$PROJECT_DIR"

source .venv/bin/activate
mkdir -p logs/slurm

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# One task per allocated GPU.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="/rlwrld3/home/hyeonbin/RL/ogbench:${PYTHONPATH:-}"

MUJOCO_GL=egl python main.py \
    --run_group=double_task2_rewards_online \
    --project=0218-tmp \
    --run_name=DOUBLE_QAM_EDIT_ONLINE_SPARSE.8_Seed10001 \
    --agent=agents/qam.py \
    --tags=QAM_EDIT,sparse,online \
    --seed=10001 \
    --env_name=cube-double-play-singletask-task2-v0 \
    --randomize_task_init_cube_pos=True \
    --ogbench_dataset_dir=/rlwrld3/home/hyeonbin/.ogbench/data/cube-double-play-1m \
    --sparse=True \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=1.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --offline_steps=0 \
    --online_steps=500000 \
    --eval_interval=50000 \
    --log_interval=5000 \
    --dataset_replace_interval=0 \
    --video_episodes=1
