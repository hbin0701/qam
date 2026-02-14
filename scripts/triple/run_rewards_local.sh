#!/bin/bash

# Project directory
PROJECT_DIR=/rlwrld3/home/hyeonbin/RL/qam
cd $PROJECT_DIR

# Activate virtual environment
source .venv/bin/activate

# Environment Setup
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=online
export PYTHONPATH=$PYTHONPATH:$(pwd)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# Create log directory
mkdir -p logs/local

echo "Starting local parallel training runs for Sparse, V1, V2, and V3 rewards..."
echo "Logs will be saved to logs/local/"

# Configuration
ENV_NAME="cube-triple-play-singletask-task2-v0"
DATASET_DIR="/rlwrld3/home/hyeonbin/.ogbench/data"
GROUP_NAME="qam_task2_reward_comparison_local_test"

# Run parameters for quick test (offline_steps=1000)
OFFLINE_STEPS=1000
ONLINE_STEPS=0
HORIZON=5

# GPU 0: Sparse
echo "Launching Sparse on GPU 0..."
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python main.py \
    --env_name=$ENV_NAME \
    --ogbench_dataset_dir=$DATASET_DIR \
    --run_group=$GROUP_NAME \
    --run_name="qam_task2_sparse_local" \
    --sparse=True \
    --agent.action_chunking=True \
    --horizon_length=$HORIZON \
    --offline_steps=$OFFLINE_STEPS \
    --online_steps=$ONLINE_STEPS \
    --save_interval=10000 \
    --eval_interval=0 \
    --log_interval=100 > logs/local/sparse.log 2>&1 &

# GPU 1: Dense V1
echo "Launching Dense V1 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python main.py \
    --env_name=$ENV_NAME \
    --ogbench_dataset_dir=$DATASET_DIR \
    --run_group=$GROUP_NAME \
    --run_name="qam_task2_v1_local" \
    --dense_reward_version=v1 \
    --agent.action_chunking=True \
    --horizon_length=$HORIZON \
    --offline_steps=$OFFLINE_STEPS \
    --online_steps=$ONLINE_STEPS \
    --save_interval=10000 \
    --eval_interval=0 \
    --log_interval=100 > logs/local/v1.log 2>&1 &

# GPU 0: Dense V2 (Shared)
echo "Launching Dense V2 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python main.py \
    --env_name=$ENV_NAME \
    --ogbench_dataset_dir=$DATASET_DIR \
    --run_group=$GROUP_NAME \
    --run_name="qam_task2_v2_local" \
    --dense_reward_version=v2 \
    --agent.action_chunking=True \
    --horizon_length=$HORIZON \
    --offline_steps=$OFFLINE_STEPS \
    --online_steps=$ONLINE_STEPS \
    --save_interval=10000 \
    --eval_interval=0 \
    --log_interval=100 > logs/local/v2.log 2>&1 &

# GPU 1: Dense V3 (Shared)
echo "Launching Dense V3 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python main.py \
    --env_name=$ENV_NAME \
    --ogbench_dataset_dir=$DATASET_DIR \
    --run_group=$GROUP_NAME \
    --run_name="qam_task2_v3_local" \
    --dense_reward_version=v3 \
    --agent.action_chunking=True \
    --horizon_length=$HORIZON \
    --offline_steps=$OFFLINE_STEPS \
    --online_steps=$ONLINE_STEPS \
    --save_interval=10000 \
    --eval_interval=0 \
    --log_interval=100 > logs/local/v3.log 2>&1 &

# Wait for all background processes to finish
wait
echo "All local training runs completed."
