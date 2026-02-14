#!/bin/bash
# Debug run: QAM on cube-double-play task2, GPU 0
# Downloads 1 shard (1M transitions) from the 100M dataset

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
export XLA_PYTHON_CLIENT_MEM_FRACTION=.40
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

# Create directories
mkdir -p logs/debug
DATASET_DIR="/rlwrld3/home/hyeonbin/.ogbench/data/cube-double-play-1m"
mkdir -p $DATASET_DIR

# Download 1 shard (1M transitions) if not already present
BASE_URL="https://rail.eecs.berkeley.edu/datasets/ogbench/cube-double-play-100m-v0"
TRAIN_FILE="cube-double-play-v0-000.npz"
VAL_FILE="cube-double-play-v0-000-val.npz"

echo "Downloading 1M double-play dataset (1 shard)..."
if [ ! -f "${DATASET_DIR}/${TRAIN_FILE}" ]; then
    echo "  Downloading ${TRAIN_FILE}..."
    wget -q --show-progress -O "${DATASET_DIR}/${TRAIN_FILE}" "${BASE_URL}/${TRAIN_FILE}"
else
    echo "  ${TRAIN_FILE} already exists, skipping."
fi
if [ ! -f "${DATASET_DIR}/${VAL_FILE}" ]; then
    echo "  Downloading ${VAL_FILE}..."
    wget -q --show-progress -O "${DATASET_DIR}/${VAL_FILE}" "${BASE_URL}/${VAL_FILE}"
else
    echo "  ${VAL_FILE} already exists, skipping."
fi
echo "Dataset download complete."

# Run training
ENV_NAME="cube-double-play-singletask-task2-v0"

echo ""
echo "Starting debug run on GPU 0: $ENV_NAME (original rewards, 1M dataset)"
echo "Log: logs/debug/double_play_task2.log"

python main.py \
    --env_name=$ENV_NAME \
    --ogbench_dataset_dir=$DATASET_DIR \
    --dataset_replace_interval=0 \
    --run_group="debug_double_play_task2" \
    --run_name="double_play_task2_original_debug_att2" \
    --sparse=False \
    --agent.inv_temp=1.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --agent.action_chunking=True \
    --horizon_length=5 \
    --offline_steps=1000000 \
    --online_steps=500000 \
    --save_interval=50000 \
    --eval_interval=50000 \
    --log_interval=1000 \
    --seed=42 \
    --video_episodes=1 \
    2>&1 | tee logs/debug/double_play_task2.log
