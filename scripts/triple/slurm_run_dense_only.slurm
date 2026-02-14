#!/bin/bash
#SBATCH --job-name=qam_task2_dense
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:3
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/qam_task2_dense_%j.out
#SBATCH --error=logs/slurm/qam_task2_dense_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hbin0701@kaist.ac.kr

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
mkdir -p logs/slurm

# Configuration
ENV_NAME="cube-triple-play-singletask-task2-v0"
DATASET_DIR="/rlwrld3/home/hyeonbin/.ogbench/data"
GROUP_NAME="qam_task2_reward_comparison_v1"
MAX_RETRIES=10
RETRY_WAIT=300  # 5 minutes

launch_training() {
    PIDS=()

    # GPU 0: Dense V1
    echo "Launching Dense V1 on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python main.py \
        --env_name=$ENV_NAME \
        --ogbench_dataset_dir=$DATASET_DIR \
        --run_group=$GROUP_NAME \
        --run_name="qam_task2_v1" \
        --dense_reward_version=v1 \
        --agent.action_chunking=True \
        --horizon_length=5 \
        --offline_steps=1000000 \
        --online_steps=500000 \
        --dataset_replace_interval=0 \
        --save_interval=50000 \
        --eval_interval=50000 \
        --log_interval=1000 &
    PIDS+=($!)

    sleep 30

    # GPU 1: Dense V2
    echo "Launching Dense V2 on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python main.py \
        --env_name=$ENV_NAME \
        --ogbench_dataset_dir=$DATASET_DIR \
        --run_group=$GROUP_NAME \
        --run_name="qam_task2_v2" \
        --dense_reward_version=v2 \
        --agent.action_chunking=True \
        --horizon_length=5 \
        --offline_steps=1000000 \
        --online_steps=500000 \
        --dataset_replace_interval=0 \
        --save_interval=50000 \
        --eval_interval=50000 \
        --log_interval=1000 &
    PIDS+=($!)

    sleep 30

    # GPU 2: Dense V3
    echo "Launching Dense V3 on GPU 2..."
    CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 python main.py \
        --env_name=$ENV_NAME \
        --ogbench_dataset_dir=$DATASET_DIR \
        --run_group=$GROUP_NAME \
        --run_name="qam_task2_v3" \
        --dense_reward_version=v3 \
        --agent.action_chunking=True \
        --horizon_length=5 \
        --offline_steps=1000000 \
        --online_steps=500000 \
        --dataset_replace_interval=0 \
        --save_interval=50000 \
        --eval_interval=50000 \
        --log_interval=1000 &
    PIDS+=($!)

    # Wait for any process to exit and check for failures
    FAILED=0
    for pid in "${PIDS[@]}"; do
        wait "$pid"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Process $pid failed with exit code $EXIT_CODE"
            FAILED=1
        fi
    done

    return $FAILED
}

kill_all_training() {
    echo "Killing all remaining training processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
        fi
    done
}

# Preflight check
echo "Running preflight check..."
python -c "
import jax
import jaxlib
print('JAX version:', jax.__version__)
print('jaxlib version:', jaxlib.__version__)
print('Devices:', jax.devices())
print('Preflight OK')
" || {
    echo 'PREFLIGHT FAILED - aborting'
    exit 1
}

# Retry loop
for attempt in $(seq 1 $MAX_RETRIES); do
    echo ""
    echo "=========================================="
    echo "  Attempt $attempt/$MAX_RETRIES  $(date)"
    echo "=========================================="
    echo ""

    launch_training
    RESULT=$?

    if [ $RESULT -eq 0 ]; then
        echo "All training runs completed successfully."
        exit 0
    fi

    # Something failed -- kill stragglers
    kill_all_training

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo ""
        echo "[RETRY] Training failed on attempt $attempt. Waiting ${RETRY_WAIT}s for you to fix things..."
        sleep $RETRY_WAIT
    else
        echo "[GIVE UP] All $MAX_RETRIES attempts failed."
        exit 1
    fi
done
