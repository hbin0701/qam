#!/bin/bash
# Auto-run part 1: double task suite (7 runs), 2 GPUs per allocation.

set -euo pipefail

MAX_RETRIES=5
WAIT_AFTER_JOB=30
JOB_NAME="qam_auto_pt1"

echo "=== [PT1] Checking for running ${JOB_NAME} SLURM jobs ==="
while squeue -u "$USER" -p debug -n "$JOB_NAME" --noheader 2>/dev/null | grep -q .; do
    echo "$(date): [PT1] ${JOB_NAME} still running, checking again in 30s..."
    sleep 30
done

echo "[PT1] No ${JOB_NAME} jobs found. Waiting ${WAIT_AFTER_JOB}s before requesting allocation..."
sleep "$WAIT_AFTER_JOB"

for attempt in $(seq 1 $MAX_RETRIES); do
    echo ""
    echo "=== [PT1] Attempt $attempt/$MAX_RETRIES: Requesting interactive allocation (2 GPUs) ==="

    python3 -u - <<'PYEOF'
import pexpect
import sys

child = pexpect.spawn(
    'srun --partition=debug --job-name=qam_auto_pt1 --nodes=1 --gpus=2 --pty bash',
    encoding='utf-8',
    timeout=3600,
)
child.logfile_read = sys.stdout

child.expect(r'\$ ')
child.sendline('cd ~/RL/qam')

child.expect(r'\$ ')
child.sendline('source .venv/bin/activate')

cmds = [
    ('scripts/double/run_task2_dense_v1.sh', 0, 'logs/auto_run/pt1_v1.log'),
    ('scripts/double/run_task2_dense_v2.sh', 0, 'logs/auto_run/pt1_v2.log'),
    ('scripts/double/run_task2_dense_v3.sh', 0, 'logs/auto_run/pt1_v3.log'),
    ('scripts/double/run_task2_dense_v4.sh', 1, 'logs/auto_run/pt1_v4.log'),
    ('scripts/double/run_task2_dense_v5.sh', 1, 'logs/auto_run/pt1_v5.log'),
    ('scripts/double/run_task2_dense_v6.sh', 1, 'logs/auto_run/pt1_v6.log'),
    ('scripts/double/run_task2_sparse.sh',   1, 'logs/auto_run/pt1_sparse.log'),
]

child.expect(r'\$ ')
child.sendline('mkdir -p logs/auto_run')

for i, (script, gpu, logfile) in enumerate(cmds, start=1):
    launch_cmd = (
        f'XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 '
        f'CUDA_VISIBLE_DEVICES={gpu} MUJOCO_EGL_DEVICE_ID={gpu} '
        f'bash {script} > {logfile} 2>&1 &'
    )
    print(f"\\n=== [PT1] Launching task {i}/{len(cmds)} on GPU {gpu}: {script} ===")
    child.expect(r'\$ ')
    child.sendline(launch_cmd)

child.expect(r'\$ ')
child.sendline('jobs -l')

print('\\n=== [PT1] Waiting for all 7 background tasks ===')
child.expect(r'\$ ', timeout=172800)
child.sendline('wait')
child.expect(r'\$ ', timeout=172800)
print('\\n=== [PT1] All 7 tasks finished ===')
child.sendline('exit')
child.expect(pexpect.EOF, timeout=30)
PYEOF

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[PT1] Attempt $attempt succeeded."
    else
        echo "[PT1] Attempt $attempt failed with exit code $exit_code."
    fi

    sleep 10
    while squeue -u "$USER" -p debug -n "$JOB_NAME" --noheader 2>/dev/null | grep -q .; do
        sleep 10
    done

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "[PT1] Waiting ${WAIT_AFTER_JOB}s before next attempt..."
        sleep "$WAIT_AFTER_JOB"
    fi
done

echo "=== [PT1] All $MAX_RETRIES attempts completed ==="
