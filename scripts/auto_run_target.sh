#!/bin/bash
# Auto-run target: single task launcher for double/task2 dense v1.

set -euo pipefail

MAX_RETRIES="${MAX_RETRIES:-5}"
WAIT_AFTER_JOB="${WAIT_AFTER_JOB:-20}"
PARTITION="${PARTITION:-debug}"
JOB_NAME="${JOB_NAME:-qam_auto_target}"
GPU_ID="${GPU_ID:-0}"
TARGET_SCRIPT="scripts/double/run_task2_dense_v1.sh"
TARGET_LOG="logs/auto_run/target_dense_v1.log"

echo "=== [TARGET] Checking for running ${JOB_NAME} SLURM jobs ==="
while squeue -u "$USER" -p "$PARTITION" -n "$JOB_NAME" --noheader 2>/dev/null | grep -q .; do
    echo "$(date): [TARGET] ${JOB_NAME} still running, checking again in 30s..."
    sleep 30
done

echo "[TARGET] No ${JOB_NAME} jobs found. Waiting ${WAIT_AFTER_JOB}s before requesting allocation..."
sleep "$WAIT_AFTER_JOB"

export PARTITION JOB_NAME GPU_ID TARGET_SCRIPT TARGET_LOG
for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo ""
    echo "=== [TARGET] Attempt $attempt/$MAX_RETRIES: Requesting interactive allocation (1 GPU) ==="

    python3 -u - <<'PYEOF'
import os
import pexpect
import sys

partition = os.environ["PARTITION"]
job_name = os.environ["JOB_NAME"]
gpu_id = os.environ["GPU_ID"]
target_script = os.environ["TARGET_SCRIPT"]
target_log = os.environ["TARGET_LOG"]

child = pexpect.spawn(
    f"srun --partition={partition} --job-name={job_name} --nodes=1 --gpus=1 --pty bash",
    encoding="utf-8",
    timeout=3600,
)
child.logfile_read = sys.stdout

child.expect(r"\$ ")
child.sendline("cd ~/RL/qam")

child.expect(r"\$ ")
child.sendline("source .venv/bin/activate")

child.expect(r"\$ ")
child.sendline("mkdir -p logs/auto_run")

launch_cmd = (
    f"CUDA_VISIBLE_DEVICES={gpu_id} MUJOCO_EGL_DEVICE_ID={gpu_id} "
    f"bash {target_script} > {target_log} 2>&1 &"
)
print(f"\\n=== [TARGET] Launching {target_script} on GPU {gpu_id} ===")
child.expect(r"\$ ")
child.sendline(launch_cmd)

child.expect(r"\$ ")
child.sendline("jobs -l")

print("\\n=== [TARGET] Waiting for target task ===")
child.expect(r"\$ ", timeout=172800)
child.sendline("wait")
child.expect(r"\$ ", timeout=172800)
print("\\n=== [TARGET] Target task finished ===")

child.sendline("exit")
child.expect(pexpect.EOF, timeout=30)
PYEOF

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[TARGET] Attempt $attempt succeeded."
    else
        echo "[TARGET] Attempt $attempt failed with exit code $exit_code."
    fi

    sleep 10
    while squeue -u "$USER" -p "$PARTITION" -n "$JOB_NAME" --noheader 2>/dev/null | grep -q .; do
        sleep 10
    done

    if [ $attempt -lt "$MAX_RETRIES" ]; then
        echo "[TARGET] Waiting ${WAIT_AFTER_JOB}s before next attempt..."
        sleep "$WAIT_AFTER_JOB"
    fi
done

echo "=== [TARGET] All $MAX_RETRIES attempts completed ==="
#!/bin/bash
# Auto-run target: single task launcher for double/task2 dense v1.

set -euo pipefail

MAX_RETRIES="${MAX_RETRIES:-5}"
WAIT_AFTER_JOB="${WAIT_AFTER_JOB:-20}"
PARTITION="${PARTITION:-debug}"
JOB_NAME="${JOB_NAME:-qam_auto_target}"
GPU_ID="${GPU_ID:-0}"
TARGET_SCRIPT="scripts/double/run_task2_dense_v1.sh"
TARGET_LOG="logs/auto_run/target_dense_v1.log"

echo "=== [TARGET] Checking for running ${JOB_NAME} SLURM jobs ==="
while squeue -u "$USER" -p "$PARTITION" -n "$JOB_NAME" --noheader 2>/dev/null | grep -q .; do
    echo "$(date): [TARGET] ${JOB_NAME} still running, checking again in 30s..."
    sleep 30
done

echo "[TARGET] No ${JOB_NAME} jobs found. Waiting ${WAIT_AFTER_JOB}s before requesting allocation..."
sleep "$WAIT_AFTER_JOB"

export PARTITION JOB_NAME GPU_ID TARGET_SCRIPT TARGET_LOG
for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo ""
    echo "=== [TARGET] Attempt $attempt/$MAX_RETRIES: Requesting interactive allocation (1 GPU) ==="

    python3 -u - <<'PYEOF'
import os
import pexpect
import sys

partition = os.environ["PARTITION"]
job_name = os.environ["JOB_NAME"]
gpu_id = os.environ["GPU_ID"]
target_script = os.environ["TARGET_SCRIPT"]
target_log = os.environ["TARGET_LOG"]

child = pexpect.spawn(
    f"srun --partition={partition} --job-name={job_name} --nodes=1 --gpus=1 --pty bash",
    encoding="utf-8",
    timeout=3600,
)
child.logfile_read = sys.stdout

child.expect(r"\$ ")
child.sendline("cd ~/RL/qam")

child.expect(r"\$ ")
child.sendline("source .venv/bin/activate")

child.expect(r"\$ ")
child.sendline("mkdir -p logs/auto_run")

launch_cmd = (
    f"CUDA_VISIBLE_DEVICES={gpu_id} MUJOCO_EGL_DEVICE_ID={gpu_id} "
    f"bash {target_script} > {target_log} 2>&1 &"
)
print(f"\\n=== [TARGET] Launching {target_script} on GPU {gpu_id} ===")
child.expect(r"\$ ")
child.sendline(launch_cmd)

child.expect(r"\$ ")
child.sendline("jobs -l")

print("\\n=== [TARGET] Waiting for target task ===")
child.expect(r"\$ ", timeout=172800)
child.sendline("wait")
child.expect(r"\$ ", timeout=172800)
print("\\n=== [TARGET] Target task finished ===")

child.sendline("exit")
child.expect(pexpect.EOF, timeout=30)
PYEOF

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[TARGET] Attempt $attempt succeeded."
    else
        echo "[TARGET] Attempt $attempt failed with exit code $exit_code."
    fi

    sleep 10
    while squeue -u "$USER" -p "$PARTITION" -n "$JOB_NAME" --noheader 2>/dev/null | grep -q .; do
        sleep 10
    done

    if [ $attempt -lt "$MAX_RETRIES" ]; then
        echo "[TARGET] Waiting ${WAIT_AFTER_JOB}s before next attempt..."
        sleep "$WAIT_AFTER_JOB"
    fi
done

echo "=== [TARGET] All $MAX_RETRIES attempts completed ==="
