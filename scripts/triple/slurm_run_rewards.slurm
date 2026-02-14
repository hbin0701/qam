#!/bin/bash
#SBATCH --job-name=qam_task2_rewards
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/qam_task2_rewards_%j.out
#SBATCH --error=logs/slurm/qam_task2_rewards_%j.err
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

    # GPU 0: Sparse
    echo "Launching Sparse on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python main.py \
        --env_name=$ENV_NAME \
        --ogbench_dataset_dir=$DATASET_DIR \
        --run_group=$GROUP_NAME \
        --run_name="qam_task2_sparse" \
        --sparse=True \
        --agent.action_chunking=True \
        --horizon_length=5 \
        --offline_steps=1000000 \
        --online_steps=9000000 \
        --dataset_replace_interval=0 \
        --save_interval=500000 \
        --eval_interval=500000 \
        --log_interval=1000 &
    PIDS+=($!)

    sleep 30

    # GPU 1: Dense V1
    echo "Launching Dense V1 on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python main.py \
        --env_name=$ENV_NAME \
        --ogbench_dataset_dir=$DATASET_DIR \
        --run_group=$GROUP_NAME \
        --run_name="qam_task2_v1" \
        --dense_reward_version=v1 \
        --agent.action_chunking=True \
        --horizon_length=5 \
        --offline_steps=1000000 \
        --online_steps=9000000 \
        --dataset_replace_interval=0 \
        --save_interval=500000 \
        --eval_interval=500000 \
        --log_interval=1000 &
    PIDS+=($!)

    sleep 30

    # GPU 2: Dense V2
    echo "Launching Dense V2 on GPU 2..."
    CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 python main.py \
        --env_name=$ENV_NAME \
        --ogbench_dataset_dir=$DATASET_DIR \
        --run_group=$GROUP_NAME \
        --run_name="qam_task2_v2" \
        --dense_reward_version=v2 \
        --agent.action_chunking=True \
        --horizon_length=5 \
        --offline_steps=1000000 \
        --online_steps=9000000 \
        --dataset_replace_interval=0 \
        --save_interval=500000 \
        --eval_interval=500000 \
        --log_interval=1000 &
    PIDS+=($!)

    sleep 30

    # GPU 3: Dense V3
    echo "Launching Dense V3 on GPU 3..."
    CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 python main.py \
        --env_name=$ENV_NAME \
        --ogbench_dataset_dir=$DATASET_DIR \
        --run_group=$GROUP_NAME \
        --run_name="qam_task2_v3" \
        --dense_reward_version=v3 \
        --agent.action_chunking=True \
        --horizon_length=5 \
        --offline_steps=1000000 \
        --online_steps=9000000 \
        --dataset_replace_interval=0 \
        --save_interval=500000 \
        --eval_interval=500000 \
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
    python -c "
import smtplib
from email.mime.text import MIMEText
msg = MIMEText('Preflight check failed for job $SLURM_JOB_ID on $(hostname). Check logs/slurm/qam_task2_rewards_${SLURM_JOB_ID}.err')
msg['Subject'] = '[SLURM $SLURM_JOB_ID] PREFLIGHT FAILED on $(hostname)'
msg['From'] = 'slurm@$(hostname)'
msg['To'] = 'hbin0701@kaist.ac.kr'
try:
    smtplib.SMTP('localhost').sendmail(msg['From'], [msg['To']], msg.as_string())
except: pass
" 2>/dev/null
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
        echo "[RETRY] Training failed on attempt $attempt. Sending email and waiting ${RETRY_WAIT}s for you to fix things..."
        python -c "
import smtplib
from email.mime.text import MIMEText
msg = MIMEText('''Job $SLURM_JOB_ID on $(hostname) failed on attempt $attempt/$MAX_RETRIES.

Check the error log:
  logs/slurm/qam_task2_rewards_${SLURM_JOB_ID}.err

You have 5 minutes to fix the code before retry attempt $((attempt+1)).
Remaining retries: $((MAX_RETRIES - attempt))''')
msg['Subject'] = '[SLURM $SLURM_JOB_ID] Training FAILED - retrying in 5 min (attempt $attempt/$MAX_RETRIES)'
msg['From'] = 'slurm@$(hostname)'
msg['To'] = 'hbin0701@kaist.ac.kr'
try:
    smtplib.SMTP('localhost').sendmail(msg['From'], [msg['To']], msg.as_string())
except: pass
" 2>/dev/null
        sleep $RETRY_WAIT
    else
        echo "[GIVE UP] All $MAX_RETRIES attempts failed."
        python -c "
import smtplib
from email.mime.text import MIMEText
msg = MIMEText('Job $SLURM_JOB_ID on $(hostname) failed all $MAX_RETRIES attempts. Job is exiting.')
msg['Subject'] = '[SLURM $SLURM_JOB_ID] Training FAILED - all retries exhausted'
msg['From'] = 'slurm@$(hostname)'
msg['To'] = 'hbin0701@kaist.ac.kr'
try:
    smtplib.SMTP('localhost').sendmail(msg['From'], [msg['To']], msg.as_string())
except: pass
" 2>/dev/null
        exit 1
    fi
done
