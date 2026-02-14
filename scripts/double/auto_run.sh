#!/bin/bash
# Auto-run script: wait for existing debug/bash job to finish, allocate interactive node, run experiments.
# Retries allocation up to 3 times.

MAX_RETRIES=3
WAIT_AFTER_JOB=30  # seconds to wait after job terminates

echo "=== Checking for running debug/bash SLURM jobs ==="
while squeue -u "$USER" -p debug -n bash --noheader 2>/dev/null | grep -q .; do
    echo "$(date): debug/bash job still running, checking again in 30s..."
    sleep 30
done
echo "No debug/bash jobs found. Waiting ${WAIT_AFTER_JOB}s before requesting allocation..."
sleep "$WAIT_AFTER_JOB"

for attempt in $(seq 1 $MAX_RETRIES); do
    echo ""
    echo "=== Attempt $attempt/$MAX_RETRIES: Requesting interactive allocation ==="

    python3 -u - <<'PYEOF'
import pexpect
import sys

child = pexpect.spawn('srun --partition=debug --nodes=1 --gpus=2 --pty bash',
                       encoding='utf-8', timeout=3600)
child.logfile_read = sys.stdout

# Wait for shell prompt
child.expect(r'\$ ')
child.sendline('cd ~/RL/qam')

child.expect(r'\$ ')
child.sendline('source .venv/bin/activate')

child.expect(r'\$ ')
child.sendline('bash scripts/double/run_tasks_all.sh')

# Wait for run_tasks_all.sh to finish (long timeout: 48 hours)
child.expect(r'\$ ', timeout=172800)
print('\n=== run_tasks_all.sh finished ===')

child.sendline('exit')
child.expect(pexpect.EOF, timeout=30)
PYEOF

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Attempt $attempt succeeded."
    else
        echo "Attempt $attempt failed with exit code $exit_code."
    fi

    # Wait for the job we just ran to fully clear from squeue
    sleep 10
    while squeue -u "$USER" -p debug -n bash --noheader 2>/dev/null | grep -q .; do
        sleep 10
    done

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "Waiting ${WAIT_AFTER_JOB}s before next attempt..."
        sleep "$WAIT_AFTER_JOB"
    fi
done

echo "=== All $MAX_RETRIES attempts completed ==="
