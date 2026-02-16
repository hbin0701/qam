#!/bin/bash
set -euo pipefail

while true; do
  bash scripts/single/online/run_tasks_all.sh
  sleep 5
done
