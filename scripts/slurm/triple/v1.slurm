#!/bin/bash
#SBATCH --job-name=qam_triple_v1_7
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/triple_v1_7_%j.out
#SBATCH --error=logs/slurm/triple_v1_7_%j.err

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

bash scripts/triple/run_task2_dense_v1.sh
