#!/bin/bash
# Orchestrator: launch both 2-GPU auto-runners in parallel.
# - PT1: double suite
# - PT2: triple suite

set -euo pipefail

bash scripts/auto_run_pt1.sh &
PID1=$!

bash scripts/auto_run_pt2.sh &
PID2=$!

echo "Started PT1 (pid=${PID1}) and PT2 (pid=${PID2}). Waiting for both..."

set +e
wait $PID1
RC1=$?
wait $PID2
RC2=$?
set -e

echo "PT1 exit code: ${RC1}"
echo "PT2 exit code: ${RC2}"

if [ $RC1 -ne 0 ] || [ $RC2 -ne 0 ]; then
  exit 1
fi

echo "Both PT1 and PT2 completed successfully."
